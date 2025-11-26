#!/usr/bin/env python3
"""
Wired example: Primal Kernel hand controller sweep using standardized framework.

This demonstrates how to wire an existing simulation (primal_logic)
into the universal experiment results pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Iterable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.framework import ParamGrid, run_parameter_sweep
from primal_logic.hand import RoboticHand
from primal_logic.field import PrimalLogicField
from primal_logic.trajectory import generate_grasp_trajectory
from primal_logic.utils import flatten, mean


def simulate_primal_kernel(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Iterable[Any]]]:
    """
    Run one primal kernel simulation with given config.

    Args:
        config: Parameter dictionary with keys:
            - alpha_base: Controller gain
            - theta: Field coupling strength
            - steps: Simulation steps

    Returns:
        (metrics, time_series) tuple where:
        - metrics: Scalars like mean_torque, mean_coherence, saturation_ratio
        - time_series: Arrays for raw CSV (t, torque, coherence, etc.)
    """
    alpha_base = config["alpha_base"]
    theta = config["theta"]
    steps = config["steps"]

    # Initialize simulation
    hand = RoboticHand(alpha_base=alpha_base)
    field = PrimalLogicField(nx=4, ny=4)
    trajectory = generate_grasp_trajectory(steps, hand.n_fingers, hand.n_joints_per_finger)

    # Run simulation and collect data
    torque_samples: List[float] = []
    coherence_samples: List[float] = []
    time_points: List[float] = []
    saturation_hits = 0
    torque_limit = hand.joint_limits.torque_max

    dt = 0.001  # From constants

    for step in range(steps):
        coherence = field.step(theta)
        coherence_samples.append(coherence)
        time_points.append(step * dt)

        hand.step(trajectory[step], theta=theta, coherence=coherence, step=step)

        latest_torques = flatten(hand.get_torques())
        torque_samples.extend(latest_torques)
        saturation_hits += sum(1 for v in latest_torques if abs(v) >= torque_limit - 1e-9)

    # Compute metrics (summary scalars)
    mean_torque = mean(torque_samples) if torque_samples else 0.0
    mean_coherence = mean(coherence_samples) if coherence_samples else 0.0
    saturation_ratio = (saturation_hits / len(torque_samples)) if torque_samples else 0.0
    stable = saturation_ratio < 0.1  # Stability criterion: <10% saturation

    metrics = {
        "mean_torque": round(mean_torque, 6),
        "mean_coherence": round(mean_coherence, 6),
        "saturation_ratio": round(saturation_ratio, 6),
        "stable": stable,
    }

    # Time series for raw CSV (sample first 200 points to keep files manageable)
    n_samples = min(200, len(time_points))
    time_series = {
        "t": time_points[:n_samples],
        "coherence": coherence_samples[:n_samples],
    }

    return metrics, time_series


def main() -> None:
    """Run full alpha parameter sweep using standardized framework."""

    # Define parameter grid
    grid = ParamGrid(params={
        "alpha_base": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "theta": [0.8, 1.0, 1.2],
        "steps": [200],
    })

    print("Starting primal kernel parameter sweep...")
    print(f"Total configurations: {6 * 3 * 1} = 18")
    print()

    # Run sweep using standardized framework
    output_dir = run_parameter_sweep(
        sim_name="primal_kernel",
        param_grid=grid,
        simulate_fn=simulate_primal_kernel,
        tag="alpha_theta_sweep",
    )

    print()
    print("=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    print(f"Results: {output_dir}")
    print(f"  - summary/summary.csv  (all configs)")
    print(f"  - summary/stats.json   (aggregates)")
    print(f"  - raw/sim_*.csv        (time series)")
    print(f"  - REPORT.md            (summary)")
    print()


if __name__ == "__main__":
    main()
