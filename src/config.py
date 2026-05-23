"""Shared defaults for the Phase 1 HVAC simulation."""

CONFIG = {
    "simulation_days": 14,
    "freq_minutes": 5,  # 5-min resolution matches ROBOD; 24 steps = 2 hours horizon
    "comfort_min_c": 20.0,
    "comfort_max_c": 24.0,
    "mode_changeover_c": 22.0,  # outdoor temp threshold for cooling vs heating mode
    "outdoor_min_c": 24.0,
    "outdoor_max_c": 34.0,
    "predictor_horizon_minutes": 120,
    "thermal_a": 1.0,
    "thermal_b": 0.04,
    "initial_indoor_temp_c": 28.0,
    "random_seed": 42,
    "energy_weight": 1.0,
    "comfort_weight": 0.5,
}


def horizon_steps(config: dict = CONFIG) -> int:
    """Return the forecast horizon in simulation steps."""
    return config["predictor_horizon_minutes"] // config["freq_minutes"]
