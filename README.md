# Smart Building HVAC Control

A prototype for predictive HVAC control in smart buildings. The system uses occupancy
prediction to drive HVAC decisions ahead of time, instead of waiting for sensors to
trigger reactive responses. The long-term target is integration with an Isaac Sim
digital twin; the current focus is a minimal, runnable simulation prototype.

## Project Goal

Demonstrate that predictive HVAC control — driven by an occupancy forecast — reduces
energy use and improves thermal comfort compared to a reactive baseline. The prototype
runs entirely on synthetic data and reports clear comparison metrics.

## How to Run

```bash
# 1. (Optional) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the end-to-end demo
python scripts/run_demo.py
```

The demo generates synthetic occupancy/temperature data, runs both reactive and
predictive controllers, and writes comparison metrics + plots to `results/`.

## Tech Stack

- **Language:** Python 3.10+
- **Core libs:** NumPy, pandas, scikit-learn, matplotlib
- **Simulation (future):** NVIDIA Isaac Sim for the building digital twin
- **Structure:** `src/` for library code, `scripts/` for entry points, `data/` for
  inputs, `results/` for outputs

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
