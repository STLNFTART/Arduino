# Primal Logic Robotic Hand (Python Port)

This repository hosts a modular Python implementation of the Primal Logic robotic hand control
framework. It models a quantum-inspired control field, tendon-driven robotic hand, and analysis
utilities for torque data. The simulator runs on the Python standard library, while analytics can
leverage real Pandas/Matplotlib or bundled offline fallbacks for air-gapped environments.

## Features

- Multi-finger hand model actuated by adaptive PD controllers with exponential memory kernels.
- Quantum-inspired field module that modulates controller gains based on coherence estimates.
- Vector sweep tooling to benchmark controller responses across θ, α, β, and τ limits.
- Rolling-average analytics and plotting powered by Pandas/Matplotlib (with optional stubs when
  the real libraries are unavailable).
- Git submodule linkage to [`MotorHandPro`](https://github.com/STLNFTART/MotorHandPro) for
  hardware-facing development.
- Recursive Planck operator utilities that expose Donte and Lightfoot constants for advanced
  memory dynamics research.

## Repository Layout

- `primal_logic/` – Core simulation modules (field, hand, trajectory, sweeps, analysis helpers).
- `tests/` – Pytest suite covering the controller, sweeps, and plotting utilities.
- `vendor/` – Lightweight fallbacks for Pandas/Matplotlib used when the real libraries are missing.
- `external/` – Integration hooks and submodules such as `MotorHandPro`.
- `tools/` – Command-line utilities (e.g., `repo_inventory.py`).
- `main.py` – CLI demo that runs a grasp trajectory and optionally logs torques to disk.

## Dependencies

The simulator itself depends only on the Python 3.10+ standard library. For richer analytics,
install the optional extras:

```bash
python3 -m pip install "pandas>=1.5" "matplotlib>=3.6"
```

The `pyproject.toml` also defines the `analysis` extra for packaging workflows:

```bash
python3 -m pip install .[analysis]
```

If external dependencies cannot be installed, the bundled fallbacks provide deterministic text
artifacts so the workflow remains reproducible.

## Run Instructions

1. **Demo simulation** – generates a torque log in `artifacts/torques.csv`:
   ```bash
   python3 main.py
   ```

2. **Rolling-average analysis** – computes a rolling mean for `joint_0` and emits a plot artifact:
   ```bash
   python3 -c "from pathlib import Path; from primal_logic import plot_rolling_average; plot_rolling_average(Path('artifacts/torques.csv'), column='joint_0', window=25)"
   ```

3. **Vector sweeps** – explore controller sensitivities across the key parameters:
   ```bash
   python3 -c "from pathlib import Path; from primal_logic import torque_sweep; torque_sweep([0.4, 0.8, 1.2], steps=50, output_path=Path('artifacts/theta_sweep.csv'))"
   python3 -c "from pathlib import Path; from primal_logic import alpha_sweep; alpha_sweep([0.50, 0.54, 0.58], steps=50, output_path=Path('artifacts/alpha_sweep.csv'))"
   python3 -c "from pathlib import Path; from primal_logic import beta_sweep; beta_sweep([0.4, 0.8, 1.2], steps=50, output_path=Path('artifacts/beta_sweep.csv'))"
   python3 -c "from pathlib import Path; from primal_logic import tau_sweep; tau_sweep([0.5, 0.7, 0.9], steps=50, output_path=Path('artifacts/tau_sweep.csv'))"
   ```

4. **Recursive Planck demo suite** – explore the Donte/Lightfoot formalism:
   ```bash
   python3 demos/demo_primal.py
   python3 demos/demo_cryo.py
   python3 demos/demo_rrt_rif.py
   ```

5. **Repository inventory dataset** – generate CSV/Markdown summaries for reporting:
   ```bash
   python3 tools/repo_inventory.py
   ```

6. **Motor Hand Pro bridge** – fetch the hardware integration submodule:
   ```bash
   git submodule update --init --recursive
   ```

## Testing

Basic unit tests and vector sweep regression checks live in `tests/`. Execute them with:

```bash
python3 -m pytest
```

For an additional syntax check run:

```bash
python3 -m compileall primal_logic tests main.py vendor
```

## Plot Interpretation

Generated plot files are text-based when the stubs are active and PNG images when real
Matplotlib is installed. Each artifact records raw torques and rolling means to support offline
inspection and benchmarking of tendon loads.

## Quantitative Framework Notes

The Lightfoot/Donte constants and the recursive Planck operator are described in detail in
`docs/quantitative_framework.md`. The summary derives the discrete update used by the
`RecursivePlanckOperator` implementation and lists the stability bounds enforced in code.
