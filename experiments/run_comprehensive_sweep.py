#!/usr/bin/env python3
"""
COMPREHENSIVE FULL-SPECTRUM PARAMETER SWEEP
Primal Logic Robotic Hand Controller - All Variable Combinations

This sweep covers the complete parameter space:
  - alpha_base: Controller gain (5 values)
  - beta_gain: Memory kernel gain (4 values)
  - theta: Field coupling strength (5 values)
  - torque_max: Joint torque limit (3 values)
  - memory_mode: Kernel type (2 modes)

Total configurations: 5 × 4 × 5 × 3 × 2 = 600

Powered by Primal Tech Invest
www.primaltechinvest.com
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Iterable

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.framework import ParamGrid, run_parameter_sweep
from primal_logic.hand import RoboticHand, JointLimits
from primal_logic.field import PrimalLogicField
from primal_logic.trajectory import generate_grasp_trajectory
from primal_logic.utils import flatten, mean


def simulate_primal_kernel_full(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Iterable[Any]]]:
    """
    Run one comprehensive primal kernel simulation with full parameter set.

    Args:
        config: Parameter dictionary with keys:
            - alpha_base: Controller gain
            - beta_gain: Memory kernel gain
            - theta: Field coupling strength
            - torque_max: Joint torque limit
            - memory_mode: "exponential" or "recursive_planck"
            - steps: Simulation steps

    Returns:
        (metrics, time_series) tuple
    """
    alpha_base = config["alpha_base"]
    beta_gain = config["beta_gain"]
    theta = config["theta"]
    torque_max = config["torque_max"]
    memory_mode = config["memory_mode"]
    steps = config["steps"]

    # Initialize with full parameter set
    limits = JointLimits(torque_max=torque_max)
    hand = RoboticHand(
        alpha_base=alpha_base,
        beta_gain=beta_gain,
        joint_limits=limits,
        memory_mode=memory_mode,
    )
    field = PrimalLogicField(nx=4, ny=4)
    trajectory = generate_grasp_trajectory(steps, hand.n_fingers, hand.n_joints_per_finger)

    # Run simulation
    torque_samples: List[float] = []
    coherence_samples: List[float] = []
    angle_samples: List[float] = []
    velocity_samples: List[float] = []
    time_points: List[float] = []
    saturation_hits = 0

    dt = 0.001

    for step in range(steps):
        coherence = field.step(theta)
        coherence_samples.append(coherence)
        time_points.append(step * dt)

        hand.step(trajectory[step], theta=theta, coherence=coherence, step=step)

        # Collect torques
        latest_torques = flatten(hand.get_torques())
        torque_samples.extend(latest_torques)
        saturation_hits += sum(1 for v in latest_torques if abs(v) >= torque_max - 1e-9)

        # Collect angles and velocities
        angles = flatten(hand.get_angles())
        angle_samples.extend(angles)

        # Compute velocities from states
        velocities = []
        for finger_states in hand.states:
            for joint_state in finger_states:
                velocities.append(joint_state.velocity)
        velocity_samples.extend(velocities)

    # Compute comprehensive metrics
    mean_torque = mean(torque_samples) if torque_samples else 0.0
    max_torque = max(torque_samples) if torque_samples else 0.0
    mean_coherence = mean(coherence_samples) if coherence_samples else 0.0
    mean_angle = mean(angle_samples) if angle_samples else 0.0
    max_velocity = max(abs(v) for v in velocity_samples) if velocity_samples else 0.0
    saturation_ratio = (saturation_hits / len(torque_samples)) if torque_samples else 0.0

    # Stability criteria
    stable = (saturation_ratio < 0.1) and (max_velocity < 8.0)

    # Lipschitz-like smoothness estimate
    torque_changes = [abs(torque_samples[i+1] - torque_samples[i])
                     for i in range(len(torque_samples)-1)] if len(torque_samples) > 1 else [0.0]
    lipschitz_estimate = max(torque_changes) / dt if torque_changes else 0.0

    metrics = {
        "mean_torque": round(mean_torque, 6),
        "max_torque": round(max_torque, 6),
        "mean_coherence": round(mean_coherence, 6),
        "mean_angle": round(mean_angle, 6),
        "max_velocity": round(max_velocity, 6),
        "saturation_ratio": round(saturation_ratio, 6),
        "lipschitz_estimate": round(lipschitz_estimate, 6),
        "stable": stable,
    }

    # Time series (sample first 200 points)
    n_samples = min(200, len(time_points))
    time_series = {
        "t": time_points[:n_samples],
        "coherence": coherence_samples[:n_samples],
    }

    return metrics, time_series


def main() -> None:
    """Run FULL comprehensive parameter sweep across all variables."""

    print("=" * 80)
    print("COMPREHENSIVE FULL-SPECTRUM PARAMETER SWEEP")
    print("Primal Logic Robotic Hand Controller")
    print("=" * 80)
    print()
    print("Powered by Primal Tech Invest")
    print("🔗 www.primaltechinvest.com")
    print()
    print("=" * 80)
    print()

    # Full parameter grid - ALL COMBINATIONS
    grid = ParamGrid(params={
        "alpha_base": [0.3, 0.45, 0.54, 0.6, 0.75],         # 5 values
        "beta_gain": [0.2, 0.5, 0.8, 1.2],                  # 4 values
        "theta": [0.6, 0.8, 1.0, 1.2, 1.5],                 # 5 values
        "torque_max": [0.5, 0.7, 0.9],                      # 3 values
        "memory_mode": ["exponential", "recursive_planck"], # 2 modes
        "steps": [200],                                     # 1 value
    })

    total_configs = 5 * 4 * 5 * 3 * 2
    print(f"Total configurations: {total_configs}")
    print()
    print("Parameter ranges:")
    print("  - alpha_base: [0.3, 0.45, 0.54, 0.6, 0.75]")
    print("  - beta_gain: [0.2, 0.5, 0.8, 1.2]")
    print("  - theta: [0.6, 0.8, 1.0, 1.2, 1.5]")
    print("  - torque_max: [0.5, 0.7, 0.9]")
    print("  - memory_mode: ['exponential', 'recursive_planck']")
    print("  - steps: 200")
    print()
    print("Estimated time: ~10 seconds at 60 configs/sec")
    print()
    print("Running sweep...")
    print()

    # Run comprehensive sweep
    output_dir = run_parameter_sweep(
        sim_name="primal_kernel",
        param_grid=grid,
        simulate_fn=simulate_primal_kernel_full,
        tag="comprehensive_full_sweep",
    )

    print()
    print("=" * 80)
    print("COMPREHENSIVE SWEEP COMPLETE")
    print("=" * 80)
    print(f"Results: {output_dir}")
    print()
    print("Files generated:")
    print(f"  📊 summary/summary.csv  ({total_configs} rows - all configs)")
    print(f"  📈 summary/stats.json   (aggregated statistics)")
    print(f"  📁 raw/sim_*.csv        ({total_configs} time series)")
    print(f"  📝 REPORT.md            (human-readable summary)")
    print()
    print("=" * 80)
    print()
    print("Powered by Primal Tech Invest")
    print("🔗 www.primaltechinvest.com")
    print()


if __name__ == "__main__":
    main()
