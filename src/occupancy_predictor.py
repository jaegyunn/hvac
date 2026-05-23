"""Occupancy predictors for the HVAC demo."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import CONFIG


class SameTimeYesterdayPredictor:
    """Predict occupancy at the target time from the same slot on prior days."""

    def __init__(self, df: pd.DataFrame):
        # V1 fits and evaluates on the same data; there is no holdout split yet.
        self.freq_minutes = _infer_freq_minutes(df)
        prepared = _with_time_features(df)
        self.slot_rates = (
            prepared.groupby(["dayofweek", "slot"])["occupancy"]
            .mean()
            .to_dict()
        )
        self.overall_rate = float(prepared["occupancy"].mean())

    def predict(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        prepared = _with_time_features(df)
        steps_per_day = 24 * 60 // self.freq_minutes
        same_time_yesterday = prepared["occupancy"].shift(steps_per_day - horizon)

        target_time = prepared["timestamp"] + pd.to_timedelta(
            horizon * self.freq_minutes,
            unit="min",
        )
        target_day = target_time.dt.dayofweek
        target_slot = (target_time.dt.hour * 60 + target_time.dt.minute) // self.freq_minutes
        schedule_forecast = [
            self.slot_rates.get((int(day), int(slot)), self.overall_rate)
            for day, slot in zip(target_day, target_slot)
        ]

        forecast = same_time_yesterday.fillna(pd.Series(schedule_forecast, index=df.index))
        return (forecast >= 0.5).astype(int).rename("predicted_occupancy")

    def predict_count(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """Predict count from the same slot on the prior day, with schedule fallback."""
        prepared = _with_time_features(df)
        steps_per_day = 24 * 60 // self.freq_minutes
        same_time_yesterday = prepared["occupancy_count"].shift(steps_per_day - horizon)
        slot_counts = prepared.groupby(["dayofweek", "slot"])["occupancy_count"].mean().to_dict()
        overall_count = float(prepared["occupancy_count"].mean())

        target_time = prepared["timestamp"] + pd.to_timedelta(
            horizon * self.freq_minutes,
            unit="min",
        )
        target_day = target_time.dt.dayofweek
        target_slot = (target_time.dt.hour * 60 + target_time.dt.minute) // self.freq_minutes
        schedule_forecast = [
            slot_counts.get((int(day), int(slot)), overall_count)
            for day, slot in zip(target_day, target_slot)
        ]

        forecast = same_time_yesterday.fillna(pd.Series(schedule_forecast, index=df.index))
        return forecast.clip(lower=0).rename("baseline_count_forecast")


class _LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        return self.head(outputs[:, -1, :]).squeeze(-1)


class LSTMOccupancyPredictor:
    """Predict future occupancy count with a compact LSTM regressor."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        sequence_length: int = 12,
        max_epochs: int = 50,
        patience: int = 6,
        train_ratio: float = 0.8,
        random_seed: int = CONFIG["random_seed"],
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.max_epochs = max_epochs
        self.patience = patience
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.model: _LSTMRegressor | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.target_mean = 0.0
        self.target_std = 1.0
        self.freq_minutes = CONFIG["freq_minutes"]
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_end_timestamp: pd.Timestamp | None = None

    def train(self, df: pd.DataFrame, horizon: int) -> "LSTMOccupancyPredictor":
        _seed_everything(self.random_seed)
        self.freq_minutes = _infer_freq_minutes(df)
        features = _count_features(df)
        target = df["occupancy_count"].shift(-horizon).to_numpy(dtype=np.float32)
        room_ids = df["room_id"].to_numpy() if "room_id" in df.columns else None
        timestamps = pd.to_datetime(df["timestamp"])
        sorted_ts = timestamps.sort_values().reset_index(drop=True)
        split_idx = int(len(sorted_ts) * self.train_ratio)
        train_end = sorted_ts.iloc[split_idx]
        self.train_end_timestamp = train_end
        train_mask = timestamps < train_end

        train_indices = _sequence_end_indices(features, target, train_mask.to_numpy(), self.sequence_length, room_ids)
        val_indices = _sequence_end_indices(features, target, ~train_mask.to_numpy(), self.sequence_length, room_ids)
        x_train_raw, y_train_raw = _make_sequences(features, target, train_indices, self.sequence_length)
        x_val_raw, y_val_raw = _make_sequences(features, target, val_indices, self.sequence_length)

        self.feature_mean = x_train_raw.reshape(-1, x_train_raw.shape[-1]).mean(axis=0)
        self.feature_std = x_train_raw.reshape(-1, x_train_raw.shape[-1]).std(axis=0)
        self.feature_std[self.feature_std == 0] = 1.0
        self.target_mean = float(y_train_raw.mean())
        self.target_std = float(y_train_raw.std()) or 1.0

        x_train = self._scale_features(x_train_raw)
        y_train = self._scale_target(y_train_raw)
        x_val = self._scale_features(x_val_raw)
        y_val = self._scale_target(y_val_raw)

        self.model = _LSTMRegressor(features.shape[1], self.hidden_size, self.num_layers)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        loss_fn = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
            batch_size=64,
            shuffle=True,
        )

        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")
        stale_epochs = 0
        for _ in range(self.max_epochs):
            self.model.train()
            batch_losses = []
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            self.model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model(torch.tensor(x_val)), torch.tensor(y_val)).item())
            self.train_losses.append(float(np.mean(batch_losses)))
            self.val_losses.append(val_loss)

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        return self

    def predict(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        if self.model is None:
            raise RuntimeError("LSTMOccupancyPredictor must be trained before prediction.")

        features = _count_features(df)
        target = np.zeros(len(df), dtype=np.float32)
        room_ids = df["room_id"].to_numpy() if "room_id" in df.columns else None
        indices = _sequence_end_indices(
            features,
            target,
            np.ones(len(df), dtype=bool),
            self.sequence_length,
            room_ids,
        )
        indices = indices[indices < len(df) - horizon]
        sequences, _ = _make_sequences(features, target, indices, self.sequence_length)
        predictions = np.full(len(df), np.nan, dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            scaled = self.model(torch.tensor(self._scale_features(sequences))).numpy()
        predictions[indices] = self._unscale_target(scaled)
        return pd.Series(np.clip(predictions, 0, None), index=df.index, name="lstm_count_forecast")

    def _scale_features(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.feature_mean) / self.feature_std).astype(np.float32)

    def _scale_target(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.target_mean) / self.target_std).astype(np.float32)

    def _unscale_target(self, values: np.ndarray) -> np.ndarray:
        return values * self.target_std + self.target_mean

    def save(self, path: Path | str) -> None:
        """Save model weights + normalization stats + config."""
        if self.model is None:
            raise RuntimeError("LSTMOccupancyPredictor must be trained before saving.")
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("LSTMOccupancyPredictor normalization stats are missing.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "target_mean": self.target_mean,
                "target_std": self.target_std,
                "config": {
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "sequence_length": self.sequence_length,
                    "freq_minutes": self.freq_minutes,
                    "train_ratio": self.train_ratio,
                    "train_end_timestamp": self.train_end_timestamp,
                    "input_size": self.feature_mean.shape[0],
                },
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "LSTMOccupancyPredictor":
        """Reconstruct predictor from checkpoint."""
        checkpoint = torch.load(Path(path), weights_only=False)
        config = checkpoint["config"]
        predictor = cls(
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            sequence_length=config["sequence_length"],
            train_ratio=config.get("train_ratio", 0.8),
        )
        predictor.freq_minutes = config["freq_minutes"]
        predictor.train_end_timestamp = config.get("train_end_timestamp")
        predictor.feature_mean = checkpoint["feature_mean"]
        predictor.feature_std = checkpoint["feature_std"]
        predictor.target_mean = checkpoint["target_mean"]
        predictor.target_std = checkpoint["target_std"]
        predictor.model = _LSTMRegressor(
            config["input_size"],
            config["hidden_size"],
            config["num_layers"],
        )
        predictor.model.load_state_dict(checkpoint["state_dict"])
        predictor.model.eval()
        predictor.train_losses = checkpoint.get("train_losses", [])
        predictor.val_losses = checkpoint.get("val_losses", [])
        return predictor


def train(df: pd.DataFrame) -> SameTimeYesterdayPredictor:
    """Train the v1 baseline predictor."""
    return SameTimeYesterdayPredictor(df)


def predict(df: pd.DataFrame, horizon: int, model: SameTimeYesterdayPredictor | None = None) -> pd.Series:
    """Predict occupancy horizon steps ahead with a trained or ad-hoc model."""
    predictor = model or train(df)
    return predictor.predict(df, horizon)


def train_lstm(df: pd.DataFrame, horizon: int) -> LSTMOccupancyPredictor:
    """Train the LSTM occupancy count predictor."""
    return LSTMOccupancyPredictor().train(df, horizon)


def _with_time_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared["dayofweek"] = prepared["timestamp"].dt.dayofweek
    prepared["slot"] = (
        prepared["timestamp"].dt.hour * 60 + prepared["timestamp"].dt.minute
    ) // _infer_freq_minutes(prepared)
    return prepared


def _infer_freq_minutes(df: pd.DataFrame) -> int:
    timestamps = pd.to_datetime(df["timestamp"])
    if len(timestamps) < 2:
        return CONFIG["freq_minutes"]
    delta = timestamps.iloc[1] - timestamps.iloc[0]
    return int(delta.total_seconds() // 60)


def _count_features(df: pd.DataFrame) -> np.ndarray:
    prepared = df.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    hour = prepared["timestamp"].dt.hour + prepared["timestamp"].dt.minute / 60
    prepared["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    prepared["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    day_onehot = pd.get_dummies(prepared["timestamp"].dt.dayofweek, prefix="dow")
    day_onehot = day_onehot.reindex(columns=[f"dow_{idx}" for idx in range(7)], fill_value=0)
    if "room_id" in prepared.columns:
        room_onehot = pd.get_dummies(prepared["room_id"], prefix="room")
    else:
        room_onehot = pd.DataFrame(index=prepared.index)
    room_onehot = room_onehot.reindex(columns=[f"room_{room}" for room in range(1, 6)], fill_value=0)
    features = pd.concat(
        [
            prepared[["hour_sin", "hour_cos"]],
            day_onehot,
            room_onehot,
            prepared[["outdoor_temperature"]],
        ],
        axis=1,
    )
    return features.to_numpy(dtype=np.float32)


def _sequence_end_indices(
    features: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    sequence_length: int,
    room_ids: np.ndarray | None = None,
) -> np.ndarray:
    indices = []
    for idx in range(sequence_length - 1, len(features)):
        start = idx - sequence_length + 1
        if not np.isfinite(target[idx]):
            continue
        if not mask[start : idx + 1].all():
            continue
        if room_ids is not None:
            if not (room_ids[start : idx + 1] == room_ids[idx]).all():
                continue
        indices.append(idx)
    return np.array(indices, dtype=int)


def _make_sequences(
    features: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequences = np.stack([features[idx - sequence_length + 1 : idx + 1] for idx in indices])
    targets = target[indices]
    return sequences.astype(np.float32), targets.astype(np.float32)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
