"""Gymnasium environment for HVAC controller RL training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.config import CONFIG


class HVACRoomEnv(gym.Env):
    """Single-room HVAC environment backed by ROBOD data and LSTM forecasts."""

    metadata = {"render_modes": []}

    def __init__(self, df, forecast, episode_length=288, train_ratio: float = 0.8):
        """
        Args:
          df: pandas DataFrame for a single room with columns:
              timestamp, occupancy, occupancy_count, outdoor_temperature
          forecast: pd.Series of LSTM count forecast aligned with df index
                    (NaN where forecast unavailable, e.g., early rows)
          episode_length: max steps per episode (default 288 = 24h at 5-min)
          train_ratio: front fraction of df that reset() may sample from
        """
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.forecast = forecast.reset_index(drop=True)
        self.episode_length = episode_length
        self.train_max_start = max(1, int(train_ratio * len(self.df)) - episode_length)

        self.observation_space = spaces.Box(
            low=np.array([10.0, 10.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([40.0, 40.0, 50.0, 50.0, 1.0, 1.0], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(3)
        self._action_map = {0: -1, 1: 0, 2: 1}  # 0=cool, 1=off, 2=heat

        self.indoor_temp = CONFIG["initial_indoor_temp_c"]
        self.start_idx = 0
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.start_idx = int(self.np_random.integers(0, self.train_max_start))
        self.current_step = 0
        self.indoor_temp = CONFIG["initial_indoor_temp_c"]
        return self._get_obs(), {}

    def step(self, action):
        env_action = self._action_map[int(action)]
        row_idx = self.start_idx + self.current_step
        row = self.df.iloc[row_idx]
        outdoor = float(row["outdoor_temperature"])
        occupancy = int(row["occupancy"])

        a = CONFIG["thermal_a"]
        b = CONFIG["thermal_b"]
        self.indoor_temp += a * env_action - b * (self.indoor_temp - outdoor)

        # Match src.metrics.summarize: combined_score = energy_weight * energy + comfort_weight * violation_minutes.
        energy_cost = abs(env_action)
        out_of_band = (
            self.indoor_temp < CONFIG["comfort_min_c"]
            or self.indoor_temp > CONFIG["comfort_max_c"]
        )
        violation_minutes = CONFIG["freq_minutes"] * int(bool(occupancy) and out_of_band)
        reward = (
            -CONFIG["energy_weight"] * energy_cost
            - CONFIG["comfort_weight"] * violation_minutes
        )

        self.current_step += 1
        terminated = False
        truncated = (
            self.current_step >= self.episode_length
            or self.start_idx + self.current_step >= len(self.df)
        )
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        row_idx = self.start_idx + self.current_step
        if row_idx >= len(self.df):
            row_idx = len(self.df) - 1
        row = self.df.iloc[row_idx]
        ts = pd.to_datetime(row["timestamp"])
        hour = ts.hour + ts.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        predicted = self.forecast.iloc[row_idx] if row_idx < len(self.forecast) else 0.0
        if pd.isna(predicted):
            predicted = 0.0
        return np.array(
            [
                self.indoor_temp,
                float(row["outdoor_temperature"]),
                float(row["occupancy_count"]),
                float(predicted),
                float(hour_sin),
                float(hour_cos),
            ],
            dtype=np.float32,
        )

    def render(self):
        pass


if __name__ == "__main__":
    from src.config import horizon_steps
    from src.data_loader import load_robod
    from src.occupancy_predictor import LSTMOccupancyPredictor

    print("Loading ROBOD Room 2...")
    df = load_robod(rooms=[2])

    print("Loading LSTM forecast...")
    model_path = ROOT / "models" / "lstm_robod_multiroom.pt"
    lstm = LSTMOccupancyPredictor.load(model_path)
    forecast = lstm.predict(df, horizon_steps())

    env = HVACRoomEnv(df=df, forecast=forecast)

    print("Running 10 random episodes...")
    rewards = []
    for ep in range(10):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        print(f"  Episode {ep}: reward = {ep_reward:.1f}")

    print(f"\nMean random reward: {np.mean(rewards):.1f}")
    print(f"Std: {np.std(rewards):.1f}")
    print("(reward sum * -1 should approximate combined_score)")
    print("(For comparison, trained PPO should be significantly higher)")
