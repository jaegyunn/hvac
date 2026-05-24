"""Publish ROBOD room telemetry to MQTT as a BMS simulator."""

from __future__ import annotations

import argparse
import json
import time

import paho.mqtt.client as mqtt

from src.config import CONFIG
from src.data_loader import load_robod
from services.config import DEFAULT_ROOM, DEFAULT_SPEEDUP, MQTT_HOST, MQTT_PORT


def main() -> None:
    args = _parse_args()
    host, port = _parse_broker(args.broker)
    df = load_robod(rooms=[args.room]).sort_values("timestamp").reset_index(drop=True)

    client = _make_client()
    client.connect(host, port, keepalive=60)
    client.loop_start()
    print(f"BMS simulator connected to MQTT at {host}:{port}")
    print(f"Publishing Room {args.room} ROBOD telemetry at {args.speedup:g}x")

    sleep_seconds = (CONFIG["freq_minutes"] * 60) / args.speedup
    try:
        for row in df.itertuples(index=False):
            timestamp = row.timestamp.isoformat()
            occupancy_count = int(row.occupancy_count)
            occupancy_payload = {
                "timestamp": timestamp,
                "occupancy_count": occupancy_count,
                "occupancy_binary": int(occupancy_count > 0),
            }
            temperature_payload = {
                "timestamp": timestamp,
                "indoor_c": float(row.indoor_temperature_reference),
                "outdoor_c": float(row.outdoor_temperature),
            }
            client.publish(
                f"building/room/{args.room}/occupancy",
                json.dumps(occupancy_payload),
                qos=0,
            )
            client.publish(
                f"building/room/{args.room}/temperature",
                json.dumps(temperature_payload),
                qos=0,
            )
            print(f"published room {args.room} @ {timestamp}", flush=True)
            time.sleep(sleep_seconds)
    finally:
        client.loop_stop()
        client.disconnect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish ROBOD room data to MQTT.")
    parser.add_argument("--room", type=int, default=DEFAULT_ROOM)
    parser.add_argument("--speedup", type=float, default=DEFAULT_SPEEDUP)
    parser.add_argument("--broker", default=f"{MQTT_HOST}:{MQTT_PORT}")
    return parser.parse_args()


def _parse_broker(value: str) -> tuple[str, int]:
    if ":" not in value:
        return value, MQTT_PORT
    host, port = value.rsplit(":", 1)
    return host, int(port)


def _make_client() -> mqtt.Client:
    if hasattr(mqtt, "CallbackAPIVersion"):
        return mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="hvac-bms-simulator",
        )
    return mqtt.Client(client_id="hvac-bms-simulator")


if __name__ == "__main__":
    main()
