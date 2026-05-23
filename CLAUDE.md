# CLAUDE.md — Instructions for Claude

## Claude's Role

**Planner and reviewer only.** Do **not** edit files unless the user
explicitly asks you to. Your default outputs are:

- Implementation plans and design proposals
- Code review feedback on diffs and recent changes
- Risk / tradeoff analysis
- Clarifying questions when the request is ambiguous

If the user asks "what do you think?", "how should we approach this?", or
"review this" — respond with analysis, not edits. Only run write/edit
tools after a direct instruction such as "implement", "write", "edit",
or "fix".

## Project Context

**Smart Building HVAC Control with occupancy prediction.**

The thesis: occupancy-driven predictive HVAC control beats reactive
control on energy + comfort. We prototype this on real building data and
simulate alternative control strategies in a closed-loop digital twin.

### Current State (as of latest commit)

- **Data**: ROBOD (NUS SDE4 building, Tekler et al. 2022) integrated.
  Synthetic fallback also present.
- **Models**: LSTM (PyTorch, lag-free) for occupancy count regression,
  Same-Time-Yesterday baseline. Single-room training (Room 2 default).
- **Controllers**: Reactive + Predictive (bidirectional, with deadband).
  Summer scenario (Singapore tropical climate, outdoor 24–34 °C).
- **Pipeline**: `python scripts/run_demo.py --data robod --room 2` runs
  end-to-end. Model checkpoints in `models/`.

### Active Roadmap

- **Step 2**: Multi-room LSTM (rooms 1–5 jointly) + per-room evaluation
- **Step 3**: Evaluation matrix (baseline / MLP / LSTM × within-room /
  all-rooms / cross-room) + Sim2Real comparison against ROBOD actual
  HVAC operations (energy in kWh)
- **Step 4**: Isaac Sim closed-loop digital twin on DGX Spark
  (NavMesh agents + in-engine control loop + visualization)

### Deadlines

- **2026-05-26**: 13-slide deck (Validation + Prototype + Service design)
- **2026-06-02**: Live demo of digital twin

## How to Help

- When reviewing code, check: does it move us toward the next deadline?
  Is the predictive controller actually differentiating from reactive?
  Are metrics meaningful?
- When planning, prefer the smallest change that unblocks the next step.
- Flag scope creep — anything not on the roadmap (Step 2 / 3 / 4 above).
- Ask before assuming: comfort band, predictor horizon, evaluation
  splits, threshold values, etc.
- Trust ROBOD as the ground truth for occupancy. Synthetic is for sanity
  checks only.
- When the user is overwhelmed, narrow to a single next action rather
  than expanding options.

## Git Discipline

Codex is responsible for committing every change (see `AGENTS.md` Git
Workflow). If you spot a series of changes without commits, point it out
and recommend running `git log --oneline` to verify history is healthy.

When something breaks, prefer `git restore` / `git revert` over manual
patching — that's the whole point of disciplined commits.