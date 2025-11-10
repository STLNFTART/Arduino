# Primal Logic Robotic Hand (Python Port)

This repository hosts a Python implementation of the Primal Logic robotic hand control framework.
It models a quantum-inspired control field, tendon-driven robotic hand, and data-analysis
utilities for rolling torque statistics.

## Features

- Modular simulation of a multi-finger hand actuated by adaptive PD controllers.
- Quantum-inspired field model that modulates controller gains.
- Optional serial bridge for streaming torques to embedded microcontrollers.
- Pandas-based analysis that computes rolling averages and generates plots for torque logs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Demo

```bash
python3 main.py
```

The demo saves torques to `artifacts/torques.csv`. Use the analysis helper to compute rolling
statistics:

```bash
python3 -c "from pathlib import Path; from primal_logic import plot_rolling_average; plot_rolling_average(Path('artifacts/torques.csv'), column='joint_0', window=25)"
```

## Testing

Basic unit tests are located in `tests/`. Execute them with:

```bash
python3 -m pytest
```

## Plot Interpretation

The generated plot overlays raw torque signals and their rolling average, providing a
clear view of tendon load trends across the hand's joints.
