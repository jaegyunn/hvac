#!/usr/bin/env python3
"""Run the Phase 1 synthetic HVAC comparison demo."""

from __future__ import annotations

import json
import os
import sys
import argparse
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
from src.metrics import compare, count_predictor_metrics
from src.occupancy_predictor import LSTMOccupancyPredictor, train, train_lstm


RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"


def main() -> None:
    args = _parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    scenario, df = _load_data(args)
    horizon = horizon_steps()
    train_end, test_mask = _time_split(df)
    timestamps = pd.to_datetime(df["timestamp"])
    n_train = int((timestamps < train_end).sum())
    n_test = int(test_mask.sum())

    print(f"Data source: {scenario}")
    print(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"n_total: {len(df)}, n_train: {n_train}, n_test: {n_test}")
    print(f"Train: {timestamps.min()} to {train_end}")
    print(f"Test:  {train_end} to {timestamps.max()}")

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

    reactive_log = run_controller(ReactiveController(), df)
    predictive_log = run_controller(PredictiveController(), df, lstm_count_forecast)
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

    reactive_log.to_csv(RESULTS_DIR / "reactive_log.csv", index=False)
    predictive_log.to_csv(RESULTS_DIR / "predictive_log.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    predictor_metrics_df.to_csv(RESULTS_DIR / "predictor_metrics.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "actual_count_at_horizon": forecast_actual,
            "same_time_yesterday_count": baseline_count_forecast,
            "lstm_count": lstm_count_forecast,
        }
    ).to_csv(RESULTS_DIR / "predictor_forecasts.csv", index=False)
    (RESULTS_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    (RESULTS_DIR / "notes.txt").write_text(
        "PredictiveController uses the LSTM count forecast and preconditions when predicted_count > 3.\n"
        "LSTM trains on the first 80% of timestamps and validates/tests on the final 20% for this v1 demo.\n"
    )
    _plot_comparison(df, reactive_log, predictive_log, RESULTS_DIR / "comparison.png")

    print("\nSmart Building HVAC Phase 1 Demo")
    print("--------------------------------")
    print(metrics.to_string(index=False))
    print("\nOccupancy count predictor metrics")
    print("---------------------------------")
    print(predictor_metrics_df.to_string(index=False))
    print("\nLower combined_score is better; it weights energy plus occupied comfort violation minutes.")
    print(f"\nWrote results to: {RESULTS_DIR}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 1 synthetic HVAC comparison demo.")
    parser.add_argument(
        "--data",
        choices=["synthetic", "robod"],
        default="robod",
        help="Dataset to use for the demo.",
    )
    parser.add_argument(
        "--room",
        type=int,
        default=2,
        help="ROBOD room number, used only with --data robod.",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load the LSTM checkpoint and skip training.",
    )
    return parser.parse_args()


def _load_data(args: argparse.Namespace) -> tuple[str, pd.DataFrame]:
    if args.data == "synthetic":
        return "synthetic", load_synthetic(CONFIG["simulation_days"], CONFIG["freq_minutes"])

    scenario = f"robod_room{args.room}"
    return scenario, load_robod(room=args.room)


def _time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.Timestamp, pd.Series]:
    timestamps = pd.to_datetime(df["timestamp"])
    sorted_ts = timestamps.sort_values().reset_index(drop=True)
    split_idx = int(len(sorted_ts) * train_ratio)
    train_end = sorted_ts.iloc[split_idx]
    return train_end, timestamps >= train_end


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
