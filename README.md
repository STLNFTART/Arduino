# PRIMAL LOGIC Framework v1.0.0

Quantum-inspired multi-algorithm control system implemented in DLang with Motor Hand Pro integration.

## Overview

This repository converts the PRIMAL LOGIC LaTeX specification into a production-grade D (Dlang) code base. The framework models:

- Quantum-inspired field evolution and superposition dynamics.
- Plasma field coupling and energy budgeting.
- Temporal coherence assessment across multi-scale signals.
- Adaptive control logic tuned for rapid response (50 μs design target, simulated here at 1 ms).
- Cryogenic revival simulation demonstrating organ-specific tolerances.
- High-level Motor Hand Pro interface that logs deterministic JSON command packets for hardware execution or dry-run testing.

The code emphasizes clarity, reproducibility, and IP traceability via explicit parameter definitions and inline documentation.

## Project Layout

```
./dub.json                      # DUB configuration
./source/app.d                  # CLI entry point orchestrating simulations & hardware commands
./source/primal/config.d        # Canonical parameter definitions and time-scale helpers
./source/primal/quantum.d       # Quantum state integration and superposition tools
./source/primal/plasma.d        # Plasma field diffusion and energy metrics
./source/primal/coherence.d     # Temporal coherence utilities
./source/primal/control.d       # Adaptive alpha controller and exponential memory
./source/primal/hardware/       # Motor Hand Pro client implementation
./source/primal/simulation/     # Cryogenic revival simulator
```

## Build & Run

1. **Build prerequisites**: DMD or LDC compiler with DUB (tested with DMD v2.105+).
2. **Build the executable**:
   ```bash
   dub build
   ```
3. **Run the simulation** (dry-run hardware logging by default):
   ```bash
   ./bin/primal_logic --steps 200 --grid 8 --output-dir ./output/motor_hand_logs
   ```
4. **Enable live hardware writes** (set `--hardware` to forward commands instead of dry run):
   ```bash
   ./bin/primal_logic --hardware --output-dir /var/opt/motor_hand
   ```

Each run emits:
- `simulation_log.csv` capturing per-step alpha, plasma energy, vitality, and collapse probability.
- JSON command payloads for Motor Hand Pro saved in the output directory.

## Testing

Execute unit tests for every module:
```bash
dub test
```

## Data Inspection & Visualization

A minimal Python helper script (`scripts/plot_results.py`) is provided to compute rolling averages and visualize the vitality trajectory for validation.

```bash
python3 scripts/plot_results.py --csv output/motor_hand_logs/simulation_log.csv --column vitality --window 10
```

The script uses pandas to calculate rolling means and matplotlib for plotting; install dependencies via:
```bash
python3 -m pip install pandas matplotlib
```

## Safety & Validation

- Commands enforce grip force bounds (0–120 N) and preserve per-step energy budgets.
- Cryogenic simulation clamps vitality between 0 and 1 while maintaining minimum temperatures ≥ 250 K.
- Hardware interface defaults to dry-run logging, preventing accidental live deployment unless `--hardware` is specified.

## IP Notice

All algorithms and control strategies remain proprietary to Donte Lightfoot with first-inventor filing priority. Redistribution or disclosure requires explicit authorization.
