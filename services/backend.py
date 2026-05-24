"""Flask backend for MQTT sensor ingestion, LSTM forecasts, and SQLite logs."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import timedelta
from pathlib import Path

import pandas as pd
import paho.mqtt.client as mqtt
from flask import Flask, jsonify, request

from src.config import horizon_steps
from src.occupancy_predictor import LSTMOccupancyPredictor
from services import config


app = Flask(__name__)

_lock = threading.Lock()
_mqtt_connected = False
_rooms: dict[int, dict] = {}
_pending: dict[int, dict[str, dict]] = {}
_windows: dict[int, pd.DataFrame] = {}
_lstm = LSTMOccupancyPredictor.load(config.LSTM_CHECKPOINT)
_mqtt_client: mqtt.Client | None = None


def init_db() -> None:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema = Path(__file__).with_name("schema.sql").read_text()
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.executescript(schema)


def create_mqtt_client() -> mqtt.Client:
    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="hvac-backend",
        )
    else:
        client = mqtt.Client(client_id="hvac-backend")
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    return client


def on_connect(client, userdata, flags, reason_code, properties=None):
    global _mqtt_connected
    _mqtt_connected = True
    client.subscribe(config.MQTT_TOPIC_FILTER)
    print("MQTT connected", flush=True)


def on_disconnect(client, userdata, flags=None, reason_code=None, properties=None):
    global _mqtt_connected
    _mqtt_connected = False
    print("MQTT disconnected", flush=True)


def on_message(client, userdata, message):
    try:
        room_id, kind = _parse_topic(message.topic)
        payload = json.loads(message.payload.decode("utf-8"))
        _ingest_message(client, room_id, kind, payload)
    except Exception as exc:
        print(f"backend message error on {message.topic}: {exc}", flush=True)


def _parse_topic(topic: str) -> tuple[int, str]:
    parts = topic.split("/")
    if len(parts) != 4 or parts[0] != "building" or parts[1] != "room":
        raise ValueError(f"unexpected topic: {topic}")
    return int(parts[2]), parts[3]


def _ingest_message(client: mqtt.Client, room_id: int, kind: str, payload: dict) -> None:
    timestamp = payload["timestamp"]
    with _lock:
        pending = _pending.setdefault(room_id, {}).setdefault(timestamp, {"timestamp": timestamp})
        if kind == "occupancy":
            pending["occupancy_count"] = int(payload["occupancy_count"])
            print(f"received occupancy for room {room_id}", flush=True)
        elif kind == "temperature":
            pending["indoor_c"] = float(payload["indoor_c"])
            pending["outdoor_c"] = float(payload["outdoor_c"])
            print(f"received temperature for room {room_id}", flush=True)
        else:
            return

        if {"occupancy_count", "indoor_c", "outdoor_c"}.issubset(pending):
            record = _append_complete_record(room_id, pending)
            del _pending[room_id][timestamp]
            predicted_count = _predict_room(room_id)
            _rooms[room_id] = {
                "occupancy": record["occupancy_count"],
                "indoor_c": record["indoor_c"],
                "outdoor_c": record["outdoor_c"],
                "predicted_count": predicted_count,
                "last_update": timestamp,
            }
            _write_sensor(record)
            _write_prediction(timestamp, room_id, predicted_count)
            client.publish(
                f"building/room/{room_id}/prediction",
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "room_id": room_id,
                        "horizon_minutes": config.PREDICTOR_HORIZON_MINUTES,
                        "predicted_count": predicted_count,
                    }
                ),
                qos=0,
            )


def _append_complete_record(room_id: int, pending: dict) -> dict:
    record = {
        "timestamp": pending["timestamp"],
        "room_id": room_id,
        "occupancy_count": int(pending["occupancy_count"]),
        "occupancy": int(pending["occupancy_count"] > 0),
        "indoor_c": float(pending["indoor_c"]),
        "outdoor_c": float(pending["outdoor_c"]),
        "outdoor_temperature": float(pending["outdoor_c"]),
    }
    current = _windows.get(room_id, pd.DataFrame())
    updated = pd.concat([current, pd.DataFrame([record])], ignore_index=True)
    updated["timestamp"] = pd.to_datetime(updated["timestamp"])
    cutoff = updated["timestamp"].max() - timedelta(hours=config.ROLLING_WINDOW_HOURS)
    _windows[room_id] = updated[updated["timestamp"] >= cutoff].reset_index(drop=True)
    record["timestamp"] = pd.to_datetime(record["timestamp"]).isoformat()
    return record


def _predict_room(room_id: int) -> float:
    window = _windows.get(room_id, pd.DataFrame()).copy()
    if len(window) < _lstm.sequence_length:
        return 0.0

    window["timestamp"] = pd.to_datetime(window["timestamp"])
    current = window.iloc[-1].copy()
    future_rows = []
    for step in range(1, horizon_steps() + 1):
        future = current.copy()
        future["timestamp"] = current["timestamp"] + pd.Timedelta(minutes=5 * step)
        future["occupancy_count"] = 0
        future["occupancy"] = 0
        future_rows.append(future)
    forecast_frame = pd.concat([window, pd.DataFrame(future_rows)], ignore_index=True)
    forecast = _lstm.predict(forecast_frame, horizon_steps())
    predicted = forecast.iloc[len(window) - 1]
    return 0.0 if pd.isna(predicted) else float(predicted)


def _write_sensor(record: dict) -> None:
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO sensor_log(timestamp, room_id, occupancy_count, indoor_c, outdoor_c)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record["timestamp"],
                record["room_id"],
                record["occupancy_count"],
                record["indoor_c"],
                record["outdoor_c"],
            ),
        )


def _write_prediction(timestamp: str, room_id: int, predicted_count: float) -> None:
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO prediction_log(timestamp, room_id, horizon_minutes, predicted_count)
            VALUES (?, ?, ?, ?)
            """,
            (timestamp, room_id, config.PREDICTOR_HORIZON_MINUTES, predicted_count),
        )


@app.get("/api/state")
def api_state():
    with _lock:
        rooms = {str(room): state.copy() for room, state in _rooms.items()}
    return jsonify({"rooms": rooms})


@app.get("/api/history")
def api_history():
    room_id = int(request.args.get("room_id", config.DEFAULT_ROOM))
    hours = float(request.args.get("hours", config.ROLLING_WINDOW_HOURS))
    with sqlite3.connect(config.DB_PATH) as conn:
        query = """
        SELECT s.timestamp, s.room_id, s.occupancy_count, s.indoor_c, s.outdoor_c,
               p.predicted_count
        FROM sensor_log s
        LEFT JOIN prediction_log p
          ON s.timestamp = p.timestamp AND s.room_id = p.room_id
        WHERE s.room_id = ?
        ORDER BY s.timestamp
        """
        df = pd.read_sql_query(query, conn, params=(room_id,))
    if not df.empty:
        timestamps = pd.to_datetime(df["timestamp"])
        cutoff = timestamps.max() - timedelta(hours=hours)
        df = df[timestamps >= cutoff]
    return jsonify({"room_id": room_id, "history": df.to_dict(orient="records")})


@app.get("/api/health")
def api_health():
    with _lock:
        rooms_seen = sorted(_rooms.keys())
    with sqlite3.connect(config.DB_PATH) as conn:
        sensor_records = conn.execute("SELECT COUNT(*) FROM sensor_log").fetchone()[0]
        prediction_records = conn.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0]
    return jsonify(
        {
            "status": "ok",
            "mqtt_connected": _mqtt_connected,
            "rooms_seen": rooms_seen,
            "sensor_records": sensor_records,
            "prediction_records": prediction_records,
        }
    )


def start_mqtt() -> None:
    global _mqtt_client
    _mqtt_client = create_mqtt_client()
    _mqtt_client.connect(config.MQTT_HOST, config.MQTT_PORT, keepalive=60)
    _mqtt_client.loop_start()


def main() -> None:
    init_db()
    start_mqtt()
    app.run(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
