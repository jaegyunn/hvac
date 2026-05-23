"""Synthetic data generation and CSV loading for the HVAC demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG


DATA_DIR = Path("data")


def _csv_path(days: int, freq_minutes: int) -> Path:
    return DATA_DIR / f"synthetic_{days}d_{freq_minutes}min.csv"


def generate_synthetic(days: int, freq_minutes: int) -> pd.DataFrame:
    """Generate deterministic synthetic occupancy and weather data."""
    rng = np.random.default_rng(CONFIG["random_seed"])
    periods = days * 24 * 60 // freq_minutes
    index = pd.date_range(
        "2026-01-05 00:00:00",
        periods=periods,
        freq=f"{freq_minutes}min",
        name="timestamp",
    )

    minutes = index.hour * 60 + index.minute
    weekday = index.dayofweek < 5
    work_hours = (minutes >= 9 * 60) & (minutes < 18 * 60)
    occupancy = (weekday & work_hours).astype(int)

    gap_mask = (occupancy == 1) & (rng.random(periods) < 0.04)
    occupancy[gap_mask] = 0
    occasional_after_hours = (
        weekday
        & (occupancy == 0)
        & (((minutes >= 8 * 60) & (minutes < 9 * 60)) | ((minutes >= 18 * 60) & (minutes < 19 * 60)))
        & (rng.random(periods) < 0.05)
    )
    occupancy[occasional_after_hours] = 1
    main_lecture = occupancy.astype(bool) & weekday & work_hours
    after_hours = occupancy.astype(bool) & ~main_lecture
    occupancy_count = np.zeros(periods, dtype=int)
    occupancy_count[main_lecture] = rng.poisson(15, int(main_lecture.sum()))
    occupancy_count[after_hours] = rng.poisson(2, int(after_hours.sum()))

    outdoor_mid = (CONFIG["outdoor_min_c"] + CONFIG["outdoor_max_c"]) / 2
    outdoor_amp = (CONFIG["outdoor_max_c"] - CONFIG["outdoor_min_c"]) / 2
    day_fraction = (minutes / (24 * 60)).to_numpy()
    daily_cycle = np.sin(2 * np.pi * (day_fraction - 0.25))
    outdoor_temperature = outdoor_mid + outdoor_amp * daily_cycle + rng.normal(0.0, 0.6, periods)

    indoor_temperature = np.full(periods, CONFIG["initial_indoor_temp_c"])

    return pd.DataFrame(
        {
            "timestamp": index,
            "occupancy": occupancy.astype(int),
            "occupancy_count": occupancy_count,
            "outdoor_temperature": outdoor_temperature.round(3),
            "indoor_temperature": indoor_temperature,
        }
    )


def load_synthetic(days: int, freq_minutes: int) -> pd.DataFrame:
    """Load cached synthetic data, or generate and save it under data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _csv_path(days, freq_minutes)
    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if "occupancy_count" in df.columns:
            return df

    df = generate_synthetic(days, freq_minutes)
    df.to_csv(path, index=False)
    return df


def load_robod(
    rooms: list[int] | int = 2,
    start_hour: int = 6,
    end_hour: int = 22,
    exclude_break: bool = True,
    **legacy_kwargs,
) -> pd.DataFrame:
    """Load one or more ROBOD rooms, concatenated with room_id column.

    Returns columns: timestamp, room_id, occupancy, occupancy_count,
    outdoor_temperature, indoor_temperature_reference.
    """
    if "room" in legacy_kwargs:
        rooms = legacy_kwargs.pop("room")
    if legacy_kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(legacy_kwargs)}")
    if isinstance(rooms, int):
        rooms = [rooms]

    parts = []
    for room in rooms:
        path = DATA_DIR / "raw" / "ROBOD" / "SupplementaryData" / f"combined_Room{room}.csv"
        if not path.exists():
            raise FileNotFoundError(f"ROBOD room file not found: {path}")

        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Singapore").dt.tz_localize(None)
        df = df.sort_values("timestamp").reset_index(drop=True)

        hour = df["timestamp"].dt.hour
        df = df[(hour >= start_hour) & (hour < end_hour)]
        df = df[df["timestamp"].dt.date != pd.Timestamp("2021-11-04").date()]
        if exclude_break:
            break_start = pd.Timestamp("2021-12-05")
            break_end = pd.Timestamp("2021-12-24")
            df = df[~((df["timestamp"] >= break_start) & (df["timestamp"] < break_end))]

        part = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "room_id": room,
                "occupancy_count": df["occupant_count [number]"].fillna(0).astype(int),
                "outdoor_temperature": df["dry_bulb_temp [Celsius]"].astype(float),
                "indoor_temperature_reference": df["air_temperature [Celsius]"].astype(float),
            }
        )
        part["occupancy"] = (part["occupancy_count"] > 0).astype(int)
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    out = out[
        [
            "timestamp",
            "room_id",
            "occupancy",
            "occupancy_count",
            "outdoor_temperature",
            "indoor_temperature_reference",
        ]
    ]
    return out.reset_index(drop=True)
