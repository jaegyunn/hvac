"""PPO-backed HVAC controller matching the rule-based controller interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.hvac_controller import BaseController


ROOT = Path(__file__).resolve().parents[1]


class RLController(BaseController):
    """Controller wrapper for a trained PPO policy.

    `choose_action` accepts optional `occupancy_count` and `timestamp` kwargs.
    When used with `run_controller`, pass `df_provider` to recover those values
    by step index; otherwise occupancy is used as an occupancy_count proxy.
    """

    name: str = "rl_ppo"

    def __init__(self, model_path: Path = ROOT / "models" / "ppo_room2.zip", df_provider=None):
        super().__init__()
        from stable_baselines3 import PPO

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"PPO model not found: {self.model_path}")
        self._model = PPO.load(self.model_path)
        self.df_provider = df_provider.reset_index(drop=True) if df_provider is not None else None
        self._step_index = 0
        self._action_map = {0: -1, 1: 0, 2: 1}

    def choose_action(
        self,
        occupancy: int,
        outdoor_temperature: float,
        predicted_count: float = 0.0,
        **kwargs,
    ) -> int:
        occupancy_count = kwargs.get("occupancy_count")
        timestamp = kwargs.get("timestamp")
        if self.df_provider is not None and self._step_index < len(self.df_provider):
            row = self.df_provider.iloc[self._step_index]
            occupancy_count = row.get("occupancy_count", occupancy_count)
            timestamp = row.get("timestamp", timestamp)

        if occupancy_count is None:
            occupancy_count = occupancy
        hour_sin, hour_cos = self._time_features(timestamp)
        obs = np.array(
            [
                self.indoor_temperature,
                float(outdoor_temperature),
                float(occupancy_count),
                float(predicted_count),
                hour_sin,
                hour_cos,
            ],
            dtype=np.float32,
        )
        action_int, _ = self._model.predict(obs, deterministic=True)
        self._step_index += 1
        return self._action_map[int(action_int)]

    def _time_features(self, timestamp) -> tuple[float, float]:
        if timestamp is not None:
            ts = pd.to_datetime(timestamp)
            hour = ts.hour + ts.minute / 60.0
        else:
            hour = (self._step_index * CONFIG["freq_minutes"] / 60.0) % 24.0
        return (
            float(np.sin(2 * np.pi * hour / 24)),
            float(np.cos(2 * np.pi * hour / 24)),
        )
