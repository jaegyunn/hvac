#!/usr/bin/env python3
"""Run the Phase 1 synthetic HVAC comparison demo."""

from __future__ import annotations

import json
import os
import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG, horizon_steps
from src.data_loader import load_robod, load_synthetic
from src.hvac_controller import PredictiveController, ReactiveController, run_controller
from src.metrics import compare, count_predictor_metrics, count_predictor_metrics_per_room
from src.occupancy_predictor import LSTMOccupancyPredictor, train, train_lstm


RESULTS_BASE_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"


def main() -> None:
    started = time.perf_counter()
    args = _parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    scenario, df, rooms = _load_data(args)
    results_dir = RESULTS_BASE_DIR / scenario
    results_dir.mkdir(parents=True, exist_ok=True)
    horizon = horizon_steps()
    train_mask, test_mask, split_label = _time_split(df)
    timestamps = pd.to_datetime(df["timestamp"])
    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())

    if args.data == "robod":
        print(f"Data source: {scenario} (rooms: {','.join(str(room) for room in rooms)})")
        print(f"Simulation room: {args.sim_room}")
    else:
        print(f"Data source: {scenario}")
    print(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"n_total: {len(df)}, n_train: {n_train}, n_test: {n_test}")
    print(f"Train: {n_train} samples / Test: {n_test} samples ({split_label})")

    baseline_model = train(df)
    baseline_count_forecast = baseline_model.predict_count(df, horizon)
    checkpoint_path = MODELS_DIR / f"lstm_{scenario}.pt"
    if args.load_model and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"--load-model requires {checkpoint_path}; train first by running without --load-model."
        )
    if args.load_model:
        lstm_model = LSTMOccupancyPredictor.load(checkpoint_path)
        print(f"✓ Loaded from {checkpoint_path.relative_to(ROOT)}")
    else:
        lstm_model = train_lstm(df, horizon)
        lstm_model.save(checkpoint_path)
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"✓ Trained and saved to {checkpoint_path.relative_to(ROOT)} ({size_mb:.2f} MB)")
    lstm_count_forecast = lstm_model.predict(df, horizon)

    sim_df, sim_lstm_count_forecast = _simulation_inputs(args, df, lstm_count_forecast)
    reactive_log = run_controller(ReactiveController(), sim_df)
    predictive_log = run_controller(PredictiveController(), sim_df, sim_lstm_count_forecast)
    metrics = compare(reactive_log, predictive_log)
    forecast_actual = df["occupancy_count"].shift(-horizon)
    predictor_metrics_df = pd.DataFrame(
        [
            {
                "model": "same_time_yesterday",
                **count_predictor_metrics(baseline_count_forecast[test_mask], forecast_actual[test_mask]),
            },
            {
                "model": "lstm",
                **count_predictor_metrics(lstm_count_forecast[test_mask], forecast_actual[test_mask]),
            },
        ]
    )
    if "room_id" in df.columns:
        per_room_metrics_df = count_predictor_metrics_per_room(
            lstm_count_forecast[test_mask],
            forecast_actual[test_mask],
            df.loc[test_mask, "room_id"],
        )
    else:
        per_room_metrics_df = pd.DataFrame()

    reactive_log.to_csv(results_dir / "reactive_log.csv", index=False)
    predictive_log.to_csv(results_dir / "predictive_log.csv", index=False)
    metrics.to_csv(results_dir / "metrics.csv", index=False)
    predictor_metrics_df.to_csv(results_dir / "predictor_metrics.csv", index=False)
    per_room_metrics_df.to_csv(results_dir / "predictor_metrics_per_room.csv", index=False)
    forecast_output = pd.DataFrame({"timestamp": df["timestamp"]})
    if "room_id" in df.columns:
        forecast_output["room_id"] = df["room_id"]
    forecast_output["actual_count_at_horizon"] = forecast_actual
    forecast_output["same_time_yesterday_count"] = baseline_count_forecast
    forecast_output["lstm_count"] = lstm_count_forecast
    forecast_output.to_csv(results_dir / "predictor_forecasts.csv", index=False)
    (results_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    (results_dir / "notes.txt").write_text(
        "PredictiveController uses the LSTM count forecast and preconditions using its configured threshold.\n"
        "LSTM trains on the first 80% of timestamps and validates/tests on the final 20% for this v1 demo.\n"
    )
    _plot_comparison(sim_df, reactive_log, predictive_log, results_dir / "comparison.png")

    print("\nSmart Building HVAC Phase 1 Demo")
    print("--------------------------------")
    print(metrics.to_string(index=False))
    print("\nOccupancy count predictor metrics")
    print("---------------------------------")
    print(predictor_metrics_df.to_string(index=False))
    if not per_room_metrics_df.empty:
        lstm_row = predictor_metrics_df[predictor_metrics_df["model"] == "lstm"].iloc[0]
        per_room_mae = " ".join(
            f"R{int(row.room_id)}={row.mae:.2f}" for row in per_room_metrics_df.itertuples(index=False)
        )
        print(f"\nLSTM test MAE: {lstm_row.mae:.2f} (overall), per-room: {per_room_mae}")
        print("\nLSTM per-room predictor metrics")
        print("-------------------------------")
        print(per_room_metrics_df.to_string(index=False))
    print("\nLower combined_score is better; it weights energy plus occupied comfort violation minutes.")
    print(f"\nWrote results to: {results_dir}")
    print(f"Total time: {time.perf_counter() - started:.2f}s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 1 synthetic HVAC comparison demo.")
    parser.add_argument(
        "--data",
        choices=["synthetic", "robod"],
        default="robod",
        help="Dataset to use for the demo.",
    )
    parser.add_argument(
        "--rooms",
        default="1,2,3,4,5",
        help="Comma-separated ROBOD room list, used only with --data robod.",
    )
    parser.add_argument(
        "--sim-room",
        type=int,
        default=2,
        help="ROBOD room number to use for HVAC simulation.",
    )
    parser.add_argument(
        "--room",
        type=int,
        default=None,
        help="Alias for --sim-room.",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load the LSTM checkpoint and skip training.",
    )
    args = parser.parse_args()
    if args.room is not None:
        args.sim_room = args.room
    return args


def _load_data(args: argparse.Namespace) -> tuple[str, pd.DataFrame, list[int]]:
    if args.data == "synthetic":
        return "synthetic", load_synthetic(CONFIG["simulation_days"], CONFIG["freq_minutes"]), []

    rooms = [int(room.strip()) for room in args.rooms.split(",") if room.strip()]
    if not rooms:
        raise ValueError("--rooms must include at least one room id")
    scenario = f"robod_room{rooms[0]}" if len(rooms) == 1 else "robod_multiroom"
    return scenario, load_robod(rooms=rooms), rooms


def _time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.Series, pd.Series, str]:
    timestamps = pd.to_datetime(df["timestamp"])
    if "room_id" in df.columns:
        train_mask = pd.Series(False, index=df.index)
        test_mask = pd.Series(False, index=df.index)
        for room in sorted(df["room_id"].dropna().unique()):
            room_mask = df["room_id"] == room
            room_ts = timestamps.loc[room_mask].sort_values()
            split_idx = int(len(room_ts) * train_ratio)
            if split_idx >= len(room_ts):
                split_idx = len(room_ts) - 1
            train_end = room_ts.iloc[split_idx]
            train_mask |= room_mask & (timestamps < train_end)
            test_mask |= room_mask & (timestamps >= train_end)
        return train_mask, test_mask, "per-room 80/20"

    sorted_ts = timestamps.sort_values().reset_index(drop=True)
    split_idx = int(len(sorted_ts) * train_ratio)
    if split_idx >= len(sorted_ts):
        split_idx = len(sorted_ts) - 1
    train_end = sorted_ts.iloc[split_idx]
    train_mask = timestamps < train_end
    return train_mask, timestamps >= train_end, "80/20"


def _simulation_inputs(
    args: argparse.Namespace,
    df: pd.DataFrame,
    lstm_count_forecast: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    if args.data != "robod":
        return df.reset_index(drop=True), lstm_count_forecast.reset_index(drop=True)

    sim_mask = df["room_id"] == args.sim_room
    if not sim_mask.any():
        raise ValueError(f"Simulation room {args.sim_room} is not present in loaded rooms")
    return (
        df.loc[sim_mask].reset_index(drop=True),
        lstm_count_forecast.loc[sim_mask].reset_index(drop=True),
    )


def _plot_comparison(
    df,
    reactive_log,
    predictive_log,
    path: Path,
) -> None:
    window = CONFIG["simulation_days"] * 24 * 60 // CONFIG["freq_minutes"]
    window = min(window, 3 * 24 * 60 // CONFIG["freq_minutes"])
    plot_df = df.tail(window)
    reactive = reactive_log.tail(window)
    predictive = predictive_log.tail(window)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Reactive vs Predictive HVAC Control")

    axes[0].plot(plot_df["timestamp"], plot_df["outdoor_temperature"], color="tab:blue", label="Outdoor")
    axes[0].plot(reactive["timestamp"], reactive["indoor_temperature"], color="tab:orange", label="Reactive indoor")
    axes[0].plot(predictive["timestamp"], predictive["indoor_temperature"], color="tab:green", label="Predictive indoor")
    axes[0].axhspan(CONFIG["comfort_min_c"], CONFIG["comfort_max_c"], color="tab:green", alpha=0.12, label="Comfort band")
    axes[0].set_ylabel("Temp (C)")
    axes[0].legend(loc="upper left", ncols=2)

    axes[1].step(plot_df["timestamp"], plot_df["occupancy"], where="post", color="black", label="Actual occupancy")
    axes[1].plot(plot_df["timestamp"], plot_df["occupancy_count"], color="tab:blue", alpha=0.7, label="Actual count")
    axes[1].plot(predictive["timestamp"], predictive["predicted_count"], color="tab:purple", alpha=0.8, label="LSTM predicted count")
    axes[1].axhline(3, color="tab:red", linestyle="--", linewidth=1, label="Precondition threshold")
    axes[1].set_ylabel("Occupancy")
    axes[1].legend(loc="upper left")

    axes[2].step(reactive["timestamp"], reactive["action"], where="post", color="tab:orange", label="Reactive action")
    axes[2].step(predictive["timestamp"], predictive["action"], where="post", color="tab:green", alpha=0.8, label="Predictive action")
    axes[2].set_ylabel("HVAC action")
    axes[2].set_xlabel("Time")
    axes[2].set_yticks([-1, 0, 1], ["Cool", "Off", "Heat"])
    axes[2].legend(loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
