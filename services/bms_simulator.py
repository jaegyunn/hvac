"""Publish ROBOD room telemetry to MQTT as a BMS simulator.

Supports single- or multi-room replay. When --start-from-test is set,
the last 20% of each room (computed individually since rooms have
different lengths) is used.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import paho.mqtt.client as mqtt

from src.config import CONFIG
from src.data_loader import load_robod
from services.config import DEFAULT_ROOM, DEFAULT_SPEEDUP, MQTT_HOST, MQTT_PORT


def main() -> None:
    args = _parse_args()
    host, port = _parse_broker(args.broker)
    rooms = [int(r.strip()) for r in args.rooms.split(",") if r.strip()]
    if not rooms:
        raise ValueError("--rooms must include at least one room id")

    df = load_robod(rooms=rooms).sort_values(["timestamp", "room_id"]).reset_index(drop=True)

    if args.start_from_test:
        df = _filter_to_test_split(df, train_ratio=0.8)
        first_ts = df["timestamp"].iloc[0] if len(df) else "N/A"
        print(f"Starting from test split (per-room last 20%): {len(df)} rows, first ts={first_ts}")

    client = _make_client()
    client.connect(host, port, keepalive=60)
    client.loop_start()
    print(f"BMS simulator connected to MQTT at {host}:{port}")
    print(f"Publishing rooms {rooms} at {args.speedup:g}x speedup ({len(df)} total rows)")

    sleep_seconds = (CONFIG["freq_minutes"] * 60) / args.speedup
    try:
        for ts, group in df.groupby("timestamp", sort=False):
            timestamp_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            for row in group.itertuples(index=False):
                room_id = int(row.room_id)
                occupancy_count = int(row.occupancy_count)
                occupancy_payload = {
                    "timestamp": timestamp_str,
                    "occupancy_count": occupancy_count,
                    "occupancy_binary": int(occupancy_count > 0),
                }
                temperature_payload = {
                    "timestamp": timestamp_str,
                    "indoor_c": float(row.indoor_temperature_reference),
                    "outdoor_c": float(row.outdoor_temperature),
                }
                client.publish(
                    f"building/room/{room_id}/occupancy",
                    json.dumps(occupancy_payload),
                    qos=0,
                )
                client.publish(
                    f"building/room/{room_id}/temperature",
                    json.dumps(temperature_payload),
                    qos=0,
                )
            print(f"published {len(group)} room(s) @ {timestamp_str}", flush=True)
            time.sleep(sleep_seconds)
    finally:
        client.loop_stop()
        client.disconnect()


def _filter_to_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> pd.DataFrame:
    """Return per-room final (1 - train_ratio) test split.

    Different rooms in ROBOD have different lengths, so we compute each
    room's split point individually and keep only its tail.
    """
    timestamps = pd.to_datetime(df["timestamp"])
    test_mask = pd.Series(False, index=df.index)
    for room in sorted(df["room_id"].dropna().unique()):
        room_mask = df["room_id"] == room
        room_ts = timestamps.loc[room_mask].sort_values()
        split_idx = int(len(room_ts) * train_ratio)
        if split_idx >= len(room_ts):
            split_idx = len(room_ts) - 1
        train_end = room_ts.iloc[split_idx]
        test_mask |= room_mask & (timestamps >= train_end)
    return df.loc[test_mask].reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish ROBOD room data to MQTT.")
    parser.add_argument(
        "--rooms",
        default=str(DEFAULT_ROOM),
        help="Comma-separated ROBOD room ids (e.g. '2' or '1,2,3,4,5'). Default: '2'.",
    )
    parser.add_argument("--speedup", type=float, default=DEFAULT_SPEEDUP)
    parser.add_argument("--broker", default=f"{MQTT_HOST}:{MQTT_PORT}")
    parser.add_argument(
        "--start-from-test",
        action="store_true",
        help="Start replay from the test split (last 20%% of each room).",
    )
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
