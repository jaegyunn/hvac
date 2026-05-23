# AGENTS.md — Instructions for Codex

This document tells Codex how to extend the Smart Building HVAC prototype.
Read this fully before any code change.

## Current State

The prototype is well past Phase 1 minimum. Currently implemented:

### Data
- `src/data_loader.py`:
  - `load_synthetic(days, freq_minutes)` — rule-based synthetic generator
    (fallback / baseline)
  - `load_robod(room, start_hour, end_hour, exclude_break)` — ROBOD adapter
    (Tekler et al. 2022, NUS SDE4 building, 5 rooms, 5-min resolution)
- Filters: daytime 06:00–22:00, exclude public holiday (2021-11-04) and
  semester break (2021-12-05 to 2021-12-23)

### Models
- `SameTimeYesterdayPredictor` (baseline) in `src/occupancy_predictor.py`
- `LSTMOccupancyPredictor` (PyTorch, lag-free for true future prediction)
  - Features: hour_sin/cos, dayofweek one-hot, outdoor_temperature
    (no occupancy lag — forces schedule learning)
  - Target: occupancy_count at t+horizon (regression)
  - 80/20 time-based train/test split
  - Save/load via `.save()` / `.load()` to `models/lstm_<scenario>.pt`

### Controllers
- `ReactiveController` — acts only when currently occupied + outside comfort band
- `PredictiveController` — bidirectional pre-conditioning with deadband:
  - `precondition_target=23.0`
  - `precondition_count_threshold` (sensitivity to LSTM forecast)
  - `precondition_deadband` (avoids oscillation around target)
  - Heats or cools toward target when LSTM predicts upcoming occupancy

### Config (`src/config.py`)
- `freq_minutes=5` (matches ROBOD)
- `predictor_horizon_minutes=120` (= 24 steps)
- Summer scenario: outdoor 24~34 °C, indoor initial 28 °C
- Comfort band 20~24 °C
- `thermal_a=1.0`, `thermal_b=0.04` (steady-state covers comfort band)
- Random seed 42

### Entry point
- `python scripts/run_demo.py --data robod --room 2` (default)
- `python scripts/run_demo.py --data synthetic` (fallback)
- Add `--load-model` to skip training and reuse checkpoint

## Roadmap

### Step 2: Multi-room LSTM
- Extend `load_robod` to accept `rooms: list[int]` and concatenate with
  `room_id` column
- Add `room_id` one-hot to LSTM features
- Train on ROBOD rooms 1–5 jointly
- Per-room evaluation in metrics

### Step 3: Evaluation matrix + Sim2Real
- Compare models: baseline (Same-Time-Yesterday), MLP (sklearn),
  LSTM, optionally RF/GBM
- Scenarios: within-room (R2), within-all-rooms, cross-room
  (train on R1+3+4+5, test on R2)
- Sim2Real: compare our predictive simulation against ROBOD's actual
  HVAC operation
  (energy from `chilled_water_energy + fcu_fan_energy` columns)

### Step 4: Isaac Sim closed-loop digital twin
- Building USD on DGX Spark (uses pre-existing room samples; no custom
  modeling)
- NavMesh-based agent simulation generates synthetic occupancy
- Our LSTM + PredictiveController run **inside Isaac Sim's Python runtime**
  (closed loop, not playback)
- Room visualization: color by thermal state, agent visibility by
  occupancy
- Pre-recorded video for the demo; live demo as fallback only

## Constraints

- **Stack**: stdlib + NumPy + pandas + scikit-learn + matplotlib + PyTorch.
  No TensorFlow.
- **Deterministic**: seed all randomness (`random_seed=42` in CONFIG).
- **Mac-friendly**: training runs on Mac (CPU or MPS). DGX Spark is for
  Isaac Sim only.
- **No network calls** at simulation/training time. ROBOD data ships in
  `data/raw/ROBOD/SupplementaryData/`.
- Keep modules small; prefer pure functions over heavy classes.
- Do not silently change defaults in `src/config.py`; if a parameter must
  change, document it in `results/notes.txt` and in the commit message.

## Git Workflow (MANDATORY)

Commit after every meaningful change. Steps:

1. Make the code change.
2. Run `python scripts/run_demo.py --data robod --room 2` and confirm it
   runs end-to-end without error.
3. Stage relevant files: `git add <files>` (avoid `git add -A` to keep
   ignored folders out).
4. Commit with a message in the format:
   `<scope>: <short summary under 60 chars>`

   Examples:
   - `predictor: remove lag features for true future prediction`
   - `controller: add deadband to prevent oscillation`
   - `data_loader: add ROBOD adapter with holiday filters`
   - `metrics: add per-room MAE breakdown`

5. If multiple unrelated changes, split into separate commits.

Never:
- Commit broken code (always verify with demo first)
- Use `--no-verify` or skip hooks
- Bundle unrelated changes
- Commit `data/raw/`, `models/*.pt`, `results/*` (already in `.gitignore`)
- Force-push to `main`

## Definition of Done (per Step)

A step is done when:
- All listed changes are implemented
- `python scripts/run_demo.py --data robod --room 2` runs cleanly
- New metrics make sense (reactive ≠ predictive, predictive comfort
  improvement is visible)
- Changes are committed with clear messages (one logical change per commit)
- Behavior changes documented in `results/notes.txt`