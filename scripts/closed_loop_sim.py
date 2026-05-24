#!/usr/bin/env python3
"""Headless closed-loop HVAC simulation for future Isaac Sim callback logic."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG, horizon_steps
from src.data_loader import load_robod
from src.hvac_controller import PredictiveController
from src.occupancy_predictor import LSTMOccupancyPredictor


MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


@dataclass
class ClosedLoopState:
    room_id: int
    indoor_temperature: float = CONFIG["initial_indoor_temp_c"]
    energy_used: float = 0.0
    violation_minutes: int = 0
    history: list[dict] = field(default_factory=list)


class ClosedLoopRunner:
    """Step-by-step closed-loop runner that mirrors a future physics callback."""

    def __init__(
        self,
        df: pd.DataFrame,
        lstm_model: LSTMOccupancyPredictor,
        controller_cls,
        room_id: int,
        horizon: int,
    ):
        self.df = (
            df[df["room_id"] == room_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        self.lstm = lstm_model
        self.controller = controller_cls()
        self.room_id = room_id
        self.horizon = horizon
        self.current_step = 0
        self.state = ClosedLoopState(room_id=room_id)

    def step(self) -> dict | None:
        """Advance one timestamp: read, predict, decide, update, and log."""
        if self.current_step >= len(self.df):
            return None

        row = self.df.iloc[self.current_step]
        occupancy_count = int(row["occupancy_count"])
        outdoor_temp = float(row["outdoor_temperature"])
        is_occupied = int(occupancy_count > 0)

        seq_len = self.lstm.sequence_length
        start = max(0, self.current_step - seq_len + 1)
        end = self.current_step + self.horizon + 1
        window = self.df.iloc[start:end].copy()
        predicted_count = 0.0
        if len(window) >= seq_len + self.horizon:
            forecast = self.lstm.predict(window, self.horizon)
            local_idx = self.current_step - start
            predicted_count = float(forecast.iloc[local_idx]) if local_idx < len(forecast) else 0.0
        if pd.isna(predicted_count):
            predicted_count = 0.0

        self.controller.indoor_temperature = self.state.indoor_temperature
        action = self.controller.choose_action(
            occupancy=is_occupied,
            outdoor_temperature=outdoor_temp,
            predicted_count=predicted_count,
        )
        new_temp = (
            self.state.indoor_temperature
            + CONFIG["thermal_a"] * action
            - CONFIG["thermal_b"] * (self.state.indoor_temperature - outdoor_temp)
        )
        self.state.energy_used += abs(action)
        if is_occupied and (new_temp < CONFIG["comfort_min_c"] or new_temp > CONFIG["comfort_max_c"]):
            self.state.violation_minutes += CONFIG["freq_minutes"]

        record = {
            "step": self.current_step,
            "timestamp": row["timestamp"],
            "room_id": self.room_id,
            "occupancy_count": occupancy_count,
            "occupancy": is_occupied,
            "outdoor_temperature": outdoor_temp,
            "indoor_before": self.state.indoor_temperature,
            "indoor_temperature": new_temp,
            "predicted_count": predicted_count,
            "action": action,
            "energy_cumulative": self.state.energy_used,
            "violations_cumulative": self.state.violation_minutes,
        }
        self.state.indoor_temperature = new_temp
        self.state.history.append(record)
        self.current_step += 1
        return record

    def run_all(self) -> pd.DataFrame:
        while self.step() is not None:
            pass
        return pd.DataFrame(self.state.history)


def _filter_to_test_set(df: pd.DataFrame, train_ratio: float = 0.8) -> pd.DataFrame:
    """Return the per-room final 20% test split."""
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


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_robod(rooms=[1, 2, 3, 4, 5])
    model_path = MODELS_DIR / "lstm_robod_multiroom.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {model_path}. Train it first with: "
            "python scripts/run_demo.py --data robod --rooms 1,2,3,4,5"
        )
    lstm_model = LSTMOccupancyPredictor.load(model_path)

    test_df = _filter_to_test_set(df)
    runner = ClosedLoopRunner(
        test_df,
        lstm_model=lstm_model,
        controller_cls=PredictiveController,
        room_id=2,
        horizon=horizon_steps(),
    )
    trajectory = runner.run_all()
    output_path = RESULTS_DIR / "closed_loop_trajectory.csv"
    trajectory.to_csv(output_path, index=False)

    print(trajectory.head(5).to_string(index=False))
    print(f"\nClosed-loop room: 2")
    print(f"Steps: {len(trajectory)}")
    print(f"Trajectory: {output_path.relative_to(ROOT)}")
    print(f"Energy used: {runner.state.energy_used:.1f}")
    print(f"Comfort violation minutes: {runner.state.violation_minutes}")


if __name__ == "__main__":
    main()
