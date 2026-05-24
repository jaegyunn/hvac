#!/usr/bin/env python3
"""Train a PPO HVAC policy on ROBOD Room data.

If stable-baselines3 has resolver trouble on Python 3.13, use
`pip install stable-baselines3==2.3.2` as a fallback pin.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import CONFIG, horizon_steps
from src.data_loader import load_robod
from src.hvac_env import HVACRoomEnv
from src.occupancy_predictor import LSTMOccupancyPredictor


def main() -> None:
    started = time.perf_counter()
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "rl").mkdir(parents=True, exist_ok=True)

    df = load_robod(rooms=[args.room])
    lstm = LSTMOccupancyPredictor.load(ROOT / "models" / "lstm_robod_multiroom.pt")
    forecast = lstm.predict(df, horizon_steps())

    def make_env():
        return HVACRoomEnv(df=df, forecast=forecast, train_ratio=0.8)

    env = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(ROOT / "logs" / "rl"),
        n_steps=2048,
        batch_size=64,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(args.output)

    mean_reward = _mean_episode_reward(model, make_env, n_episodes=10)
    print("\nRL training complete")
    print(f"Mean episode reward over 10 eval episodes: {mean_reward:.1f}")
    print(f"Saved PPO policy to: {args.output}")
    print(f"Total time: {time.perf_counter() - started:.2f}s")


def _mean_episode_reward(model: PPO, env_factory, n_episodes: int = 10) -> float:
    rewards = []
    for ep in range(n_episodes):
        env = env_factory()
        obs, _ = env.reset(seed=CONFIG["random_seed"] + ep)
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    return sum(rewards) / len(rewards)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO HVAC controller.")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=CONFIG["random_seed"])
    parser.add_argument("--room", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "ppo_room2.zip")
    return parser.parse_args()


if __name__ == "__main__":
    main()
