# Smart Building HVAC Control

A prototype for predictive HVAC control in smart buildings. The system uses occupancy
prediction to drive HVAC decisions ahead of time, instead of waiting for sensors to
trigger reactive responses. The long-term target is integration with an Isaac Sim
digital twin; the current focus is a minimal, runnable simulation prototype.

## Project Goal

Demonstrate that predictive HVAC control — driven by an occupancy forecast — reduces
energy use and improves thermal comfort compared to a reactive baseline. The prototype
runs on real building occupancy data (ROBOD) and a synthetic fallback, reporting clear
comparison metrics.

## How to Run

```bash
# 1. (Optional) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run on ROBOD real building data (Room 2, recommended)
python scripts/run_demo.py --data robod --room 2

# 4. (Optional) Run on synthetic data
python scripts/run_demo.py --data synthetic

# 5. (Optional) Skip training, reuse saved checkpoint
python scripts/run_demo.py --data robod --room 2 --load-model
```

Results land in `results/`:
- `metrics.csv` — reactive vs predictive HVAC comparison
- `predictor_metrics.csv` — model MAE/RMSE on test set
- `comparison.png` — time-series visualization
- `predictor_forecasts.csv` — full prediction trace

## Tech Stack

- **Language:** Python 3.10+
- **Core libs:** NumPy, pandas, scikit-learn, matplotlib
- **Deep learning:** PyTorch (LSTM occupancy forecasting)
- **Real data:** ROBOD (Tekler et al. 2022, *Building Simulation*) —
  NUS SDE4 net-zero energy building, 5 rooms, 5-min resolution,
  ground-truth occupancy via surveillance cameras
- **Digital twin (Phase 2):** NVIDIA Isaac Sim on DGX Spark with
  NavMesh agent simulation
- **Structure:** `src/` for library code, `scripts/` for entry points,
  `data/` for inputs, `results/` for outputs, `models/` for checkpoints

## Repository Layout

```
.
├── data/        # synthetic and (later) real input data
├── docs/        # design notes
├── results/     # metrics, plots, logs from runs
├── scripts/     # entry points (run_demo.py, ...)
├── src/         # data loader, predictor, controller, metrics
├── AGENTS.md    # build instructions for Codex
├── CLAUDE.md    # role + context for Claude
└── README.md
```
