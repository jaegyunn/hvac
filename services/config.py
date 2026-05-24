"""Service-layer configuration for MQTT, SQLite, and simulator timing."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MQTT_HOST = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_FILTER = "building/+/+/+"
DB_PATH = ROOT / "data" / "runtime.db"
DEFAULT_ROOM = 2
DEFAULT_SPEEDUP = 300.0
PREDICTOR_HORIZON_MINUTES = 120
LSTM_CHECKPOINT = ROOT / "models" / "lstm_robod_room2.pt"
ROLLING_WINDOW_HOURS = 24
