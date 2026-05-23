"""Reactive and predictive HVAC controllers with a tiny thermal model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import CONFIG


@dataclass
class BaseController:
    comfort_min: float = CONFIG["comfort_min_c"]
    comfort_max: float = CONFIG["comfort_max_c"]
    thermal_a: float = CONFIG["thermal_a"]
    thermal_b: float = CONFIG["thermal_b"]
    indoor_temperature: float = CONFIG["initial_indoor_temp_c"]

    name: str = "base"

    def choose_action(self, occupancy: int, outdoor_temperature: float, predicted_count: float = 0.0) -> int:
        raise NotImplementedError

    def step(self, row: pd.Series, predicted_count: float = 0.0) -> dict:
        indoor_before = float(self.indoor_temperature)
        outdoor = float(row["outdoor_temperature"])
        occupancy = int(row["occupancy"])
        occupancy_count = int(row.get("occupancy_count", occupancy))
        action = self.choose_action(occupancy, outdoor, float(predicted_count))
        self.indoor_temperature = indoor_before + self.thermal_a * action - self.thermal_b * (indoor_before - outdoor)

        return {
            "timestamp": row["timestamp"],
            "controller": self.name,
            "occupancy": occupancy,
            "occupancy_count": occupancy_count,
            "predicted_count": float(predicted_count),
            "predicted_occupancy": int(float(predicted_count) > 3),
            "outdoor_temperature": outdoor,
            "indoor_before": indoor_before,
            "action": action,
            "indoor_temperature": self.indoor_temperature,
        }


@dataclass
class ReactiveController(BaseController):
    mode_changeover: float = 22.0
    name: str = "reactive"

    def choose_action(self, occupancy: int, outdoor_temperature: float, predicted_count: float = 0.0) -> int:
        if not occupancy:
            return 0
        if outdoor_temperature > self.mode_changeover:
            if self.indoor_temperature > self.comfort_max:
                return -1
            return 0
        if self.indoor_temperature < self.comfort_min:
            return 1
        return 0


@dataclass
class PredictiveController(BaseController):
    precondition_target: float = 23.0
    precondition_count_threshold: float = 2.0
    precondition_deadband: float = 0.5
    mode_changeover: float = 22.0
    name: str = "predictive"

    def choose_action(self, occupancy: int, outdoor_temperature: float, predicted_count: float = 0.0) -> int:
        if occupancy:
            return self._comfort_action(outdoor_temperature)

        if predicted_count > self.precondition_count_threshold:
            return self._precondition_action(outdoor_temperature)
        return 0

    def _comfort_action(self, outdoor_temperature: float) -> int:
        if outdoor_temperature > self.mode_changeover:
            if self.indoor_temperature > self.comfort_max:
                return -1
            return 0
        if self.indoor_temperature < self.comfort_min:
            return 1
        return 0

    def _precondition_action(self, outdoor_temperature: float) -> int:
        if outdoor_temperature > self.mode_changeover:
            if self.indoor_temperature > self.precondition_target + self.precondition_deadband:
                return -1
            return 0
        if self.indoor_temperature < self.precondition_target - self.precondition_deadband:
            return 1
        return 0


def run_controller(controller: BaseController, df: pd.DataFrame, forecast: pd.Series | None = None) -> pd.DataFrame:
    """Run a controller across a scenario and return a step-by-step log."""
    predictions = forecast.fillna(0) if forecast is not None else pd.Series(0.0, index=df.index)
    rows = [
        controller.step(row, float(predictions.loc[idx]))
        for idx, row in df.iterrows()
    ]
    return pd.DataFrame(rows)
