# Primal Logic Robotic Hand (Python Port)

This repository hosts a self-contained Python implementation of the Primal Logic robotic hand
control framework. It models a quantum-inspired control field, tendon-driven robotic hand, and
analysis utilities for torque data. All numerical helpers are implemented with the Python
standard library so the project runs without external downloads.

## Features

- Modular simulation of a multi-finger hand actuated by adaptive PD controllers.
- Quantum-inspired field model that modulates controller gains.
- Offline-friendly pandas/matplotlib replacements for rolling statistics and plotting.
- Vector sweep tooling to benchmark controller responses across theta values.

## Run Instructions

1. **Demo simulation** – generates a torque log in `artifacts/torques.csv`:
   ```bash
   python3 main.py
   ```

2. **Rolling-average analysis** – produces a pseudo-plot (`.png` text file) using the bundled
   pandas/matplotlib-compatible layers:
   ```bash
   python3 -c "from pathlib import Path; from primal_logic import plot_rolling_average; plot_rolling_average(Path('artifacts/torques.csv'), column='joint_0', window=25)"
   ```

3. **Vector sweep** – explore mean torques for several theta values and write a CSV summary:
   ```bash
   python3 -c "from pathlib import Path; from primal_logic.sweeps import torque_sweep; torque_sweep([0.4, 0.8, 1.2], steps=50, output_path=Path('artifacts/theta_sweep.csv'))"
   ```

## Testing

Basic unit tests and vector sweep regression checks live in `tests/`. Execute them with:

```bash
python3 -m pytest
```

For an additional syntax check run:

```bash
python3 -m compileall primal_logic tests main.py
```

## Plot Interpretation

The generated pseudo-plot stores metadata about each series (raw torque samples and their rolling
means). It is intended for offline inspection or post-processing without requiring the real
matplotlib dependency.
