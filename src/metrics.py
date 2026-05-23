"""Metrics for comparing HVAC controller logs."""

from __future__ import annotations

import pandas as pd

from .config import CONFIG


def summarize(log: pd.DataFrame, freq_minutes: int = CONFIG["freq_minutes"]) -> dict:
    """Compute energy, comfort violations, and combined score for one run."""
    occupied = log["occupancy"] == 1
    outside_band = (
        (log["indoor_temperature"] < CONFIG["comfort_min_c"])
        | (log["indoor_temperature"] > CONFIG["comfort_max_c"])
    )
    violation_minutes = int((occupied & outside_band).sum() * freq_minutes)
    energy_used = float(log["action"].abs().sum())
    score = (
        CONFIG["energy_weight"] * energy_used
        + CONFIG["comfort_weight"] * violation_minutes
    )

    return {
        "controller": str(log["controller"].iloc[0]),
        "energy_used": energy_used,
        "comfort_violation_minutes": violation_minutes,
        "combined_score": round(score, 3),
    }


def compare(reactive_log: pd.DataFrame, predictive_log: pd.DataFrame) -> pd.DataFrame:
    """Return a comparison table for the two controller logs."""
    return pd.DataFrame(
        [
            summarize(reactive_log),
            summarize(predictive_log),
        ]
    )


def predictor_accuracy(forecast: pd.Series, actual: pd.Series) -> dict:
    """Compute binary predictor precision, recall, F1, and overall accuracy."""
    aligned = pd.DataFrame({"forecast": forecast, "actual": actual}).dropna()
    predicted = aligned["forecast"].astype(int)
    truth = aligned["actual"].astype(int)

    true_positive = int(((predicted == 1) & (truth == 1)).sum())
    false_positive = int(((predicted == 1) & (truth == 0)).sum())
    false_negative = int(((predicted == 0) & (truth == 1)).sum())
    correct = int((predicted == truth).sum())
    total = int(len(truth))

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = correct / total if total else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "n": total,
    }


def count_predictor_metrics(forecast: pd.Series, actual: pd.Series) -> dict:
    """Compute count regression metrics plus binary-equivalent F1."""
    aligned = pd.DataFrame({"forecast": forecast, "actual": actual}).dropna()
    forecast_values = aligned["forecast"].astype(float)
    actual_values = aligned["actual"].astype(float)
    errors = forecast_values - actual_values
    mae = float(errors.abs().mean()) if len(aligned) else 0.0
    rmse = float((errors.pow(2).mean()) ** 0.5) if len(aligned) else 0.0

    binary = predictor_accuracy((forecast_values > 0).astype(int), (actual_values > 0).astype(int))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "binary_f1": binary["f1"],
        "binary_precision": binary["precision"],
        "binary_recall": binary["recall"],
        "binary_accuracy": binary["accuracy"],
        "n": int(len(aligned)),
    }
