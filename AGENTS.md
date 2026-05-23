# AGENTS.md — Instructions for Codex

This document tells Codex how to build out the Smart Building HVAC prototype.

## Scope (Phase 1)

Build a **minimal runnable prototype**. No Isaac Sim yet. No real sensor data yet.
Everything runs on synthetic data and produces a clear before/after comparison
between reactive and predictive HVAC control.

## What to Implement

Four small modules under `src/`, plus one entry-point script:

1. **`src/data_loader.py`**
   - Generate synthetic time-series data: occupancy (0/1 or count), outdoor
     temperature, indoor temperature.
   - Provide a `load_synthetic(days: int, freq_minutes: int) -> pd.DataFrame`
     function.
   - Save/load CSVs under `data/`.

2. **`src/occupancy_predictor.py`**
   - Train a simple model (e.g. logistic regression or a small MLP) to predict
     occupancy `k` steps ahead from recent history + time-of-day features.
   - Expose `train(df)` and `predict(df, horizon)`.
   - Start simple — even a "same time yesterday" baseline is fine for v1.

3. **`src/hvac_controller.py`**
   - Two controllers with the same interface:
     - `ReactiveController` — turns HVAC on only when current occupancy is detected
       and indoor temperature is out of comfort range.
     - `PredictiveController` — uses the occupancy forecast to pre-condition the
       room before predicted occupancy.
   - Each step returns an HVAC action (heat / cool / off) and the resulting
     indoor temperature (use a tiny thermal model: `T_next = T + a*action - b*(T - T_outdoor)`).

4. **`src/metrics.py`**
   - Compute: total energy used (sum of |action|), comfort violations (minutes
     outside comfort band while occupied), and a combined score.
   - Provide `compare(reactive_log, predictive_log)` returning a dict / DataFrame.

5. **`scripts/run_demo.py`**
   - Single demo command. End-to-end:
     1. load/generate synthetic data
     2. train the predictor
     3. run both controllers over the same scenario
     4. compute metrics
     5. print a summary table and save plots/CSVs to `results/`
   - Must be runnable as: `python scripts/run_demo.py`

## Constraints

- **Simple Python.** Standard library + NumPy + pandas + scikit-learn + matplotlib.
  No PyTorch, no TensorFlow, no Isaac Sim in Phase 1.
- **No network calls** at runtime.
- **Deterministic** by default — seed all randomness.
- Code should run on a laptop in under a minute.
- Keep modules small and importable; favor pure functions over heavy classes.

## Default Parameters

Use these values across all modules. Centralize them (e.g. a `CONFIG` dict at the
top of `run_demo.py` or a small `src/config.py`) so they're easy to tweak later.

| Parameter | Value | Notes |
|---|---|---|
| Simulation length | **14 days** | Enough to show weekday/weekend patterns, still fast |
| Sampling period | **10 minutes** | HVAC step = one sample |
| Comfort band | **20–24 °C** | Indoor temp outside this counts as a violation *when occupied* |
| Outdoor temp range | sinusoidal, **−2 °C to +12 °C** (winter-ish) | + small Gaussian noise |
| Occupancy schedule | weekdays **09:00–18:00**, occupied = 1 | + small noise / occasional gaps |
| Predictor horizon | **30 minutes** (= 3 steps ahead) | What `PredictiveController` looks at |
| HVAC actions | **{−1: cool, 0: off, +1: heat}** | Discrete, unit magnitude |
| Thermal model | `T_next = T + a·action − b·(T − T_outdoor)` | `a = 0.5`, `b = 0.1` as starting values |
| Initial indoor temp | **22 °C** | Mid-comfort |
| Random seed | **42** | Single global seed |

These are starting values. If predictive ends up looking identical to reactive,
the first knob to turn is `a` / `b` (controller authority vs. thermal leakage) or
the predictor horizon. Don't tune them silently — note any changes in `results/`.

## Definition of Done (Phase 1)

- `python scripts/run_demo.py` runs end-to-end without errors on a fresh checkout.
- `results/` contains a metrics CSV and at least one comparison plot.
- The printed summary clearly shows reactive vs predictive numbers.

## Phase 2 (later, do not start yet)

- Replace synthetic data with logged building data.
- Swap the thermal model for the Isaac Sim digital twin.
- Add a more capable occupancy model (sequence model / gradient boosting).