#!/usr/bin/env python3
"""Evaluate PPO HVAC controller against Room 2 ROBOD test split."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import horizon_steps
from src.data_loader import load_robod
from src.hvac_controller import PredictiveController, ReactiveController, run_controller
from src.metrics import summarize
from src.occupancy_predictor import LSTMOccupancyPredictor
from src.rl_controller import RLController


def main() -> None:
    args = _parse_args()
    results_dir = ROOT / f"results/robod_room{args.room}"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_robod(rooms=[args.room])
    lstm = LSTMOccupancyPredictor.load(ROOT / "models" / "lstm_robod_multiroom.pt")
    forecast = lstm.predict(df, horizon_steps())

    test_mask = _test_mask(df)
    test_df = df.loc[test_mask].reset_index(drop=True)
    test_forecast = forecast.loc[test_mask].reset_index(drop=True)

    reactive_log = run_controller(ReactiveController(), test_df)
    predictive_log = run_controller(PredictiveController(), test_df, test_forecast)
    controller = RLController(model_path=args.model, df_provider=test_df)
    rl_log = run_controller(controller, test_df, test_forecast)

    reactive_log.to_csv(results_dir / "reactive_log.csv", index=False)
    predictive_log.to_csv(results_dir / "predictive_log.csv", index=False)
    rl_log.to_csv(results_dir / "rl_log.csv", index=False)

    reactive_row = summarize(reactive_log)
    predictive_row = summarize(predictive_log)
    rl_row = summarize(rl_log)
    rl_row["controller"] = "rl_ppo"

    metrics_path = results_dir / "metrics.csv"
    metrics = pd.DataFrame([reactive_row, predictive_row, rl_row])
    metrics.to_csv(metrics_path, index=False)

    print(pd.DataFrame([rl_row]).to_string(index=False))
    scores = metrics.set_index("controller")["combined_score"].to_dict()
    print(
        "RL combined_score="
        f"{scores.get('rl_ppo', float('nan')):.1f} "
        f"vs reactive={scores.get('reactive', float('nan')):.1f} "
        f"vs predictive={scores.get('predictive', float('nan')):.1f}"
    )


def _test_mask(df: pd.DataFrame, train_ratio: float = 0.8) -> pd.Series:
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
    return test_mask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO HVAC controller.")
    parser.add_argument("--room", type=int, default=2)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "ppo_room2.zip")
    return parser.parse_args()


if __name__ == "__main__":
    main()
