#!/usr/bin/env python3
"""
MAXIMUM EXTENDED PARAMETER SWEEP - ULTRA HIGH RESOLUTION
Primal Logic Robotic Hand Controller - ALL Variable Combinations (EXTENDED)

This sweep covers an EXTENDED parameter space with higher resolution:
  - alpha_base: Controller gain (8 values)
  - beta_gain: Memory kernel gain (6 values)
  - theta: Field coupling strength (8 values)
  - torque_max: Joint torque limit (5 values)
  - memory_mode: Kernel type (2 modes)
  - steps: Simulation duration (2 values: 200, 500)

Total configurations: 8 × 6 × 8 × 5 × 2 × 2 = 7,680

This is the MAXIMUM output mode with extended parameter resolution.

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


def simulate_primal_kernel_extended(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Iterable[Any]]]:
    """
    Run one extended primal kernel simulation with full parameter set.

    Args:
        config: Parameter dictionary

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
    saturation_hits = 0

    dt = 0.001

    for step in range(steps):
        coherence = field.step(theta)
        coherence_samples.append(coherence)

        hand.step(trajectory[step], theta=theta, coherence=coherence, step=step)

        # Collect metrics
        latest_torques = flatten(hand.get_torques())
        torque_samples.extend(latest_torques)
        saturation_hits += sum(1 for v in latest_torques if abs(v) >= torque_max - 1e-9)

        angles = flatten(hand.get_angles())
        angle_samples.extend(angles)

        velocities = []
        for finger_states in hand.states:
            for joint_state in finger_states:
                velocities.append(joint_state.velocity)
        velocity_samples.extend(velocities)

    # Compute comprehensive metrics
    mean_torque = mean(torque_samples) if torque_samples else 0.0
    max_torque = max(torque_samples) if torque_samples else 0.0
    min_torque = min(torque_samples) if torque_samples else 0.0
    std_torque = (sum((t - mean_torque)**2 for t in torque_samples) / len(torque_samples))**0.5 if len(torque_samples) > 1 else 0.0

    mean_coherence = mean(coherence_samples) if coherence_samples else 0.0
    min_coherence = min(coherence_samples) if coherence_samples else 0.0

    mean_angle = mean(angle_samples) if angle_samples else 0.0
    max_angle = max(angle_samples) if angle_samples else 0.0

    mean_velocity = mean([abs(v) for v in velocity_samples]) if velocity_samples else 0.0
    max_velocity = max(abs(v) for v in velocity_samples) if velocity_samples else 0.0

    saturation_ratio = (saturation_hits / len(torque_samples)) if torque_samples else 0.0

    # Stability criteria (multiple levels)
    stable_strict = (saturation_ratio < 0.05) and (max_velocity < 6.0)
    stable_moderate = (saturation_ratio < 0.15) and (max_velocity < 7.0)
    stable_relaxed = (saturation_ratio < 0.30) and (max_velocity < 8.0)

    # Lipschitz-like smoothness estimate
    torque_changes = [abs(torque_samples[i+1] - torque_samples[i])
                     for i in range(len(torque_samples)-1)] if len(torque_samples) > 1 else [0.0]
    lipschitz_estimate = max(torque_changes) / dt if torque_changes else 0.0
    mean_lipschitz = mean(torque_changes) / dt if torque_changes else 0.0

    # Energy-like metric
    total_energy = sum(abs(t) for t in torque_samples) * dt

    metrics = {
        "mean_torque": round(mean_torque, 6),
        "max_torque": round(max_torque, 6),
        "min_torque": round(min_torque, 6),
        "std_torque": round(std_torque, 6),
        "mean_coherence": round(mean_coherence, 6),
        "min_coherence": round(min_coherence, 6),
        "mean_angle": round(mean_angle, 6),
        "max_angle": round(max_angle, 6),
        "mean_velocity": round(mean_velocity, 6),
        "max_velocity": round(max_velocity, 6),
        "saturation_ratio": round(saturation_ratio, 6),
        "lipschitz_estimate": round(lipschitz_estimate, 6),
        "mean_lipschitz": round(mean_lipschitz, 6),
        "total_energy": round(total_energy, 6),
        "stable_strict": stable_strict,
        "stable_moderate": stable_moderate,
        "stable_relaxed": stable_relaxed,
    }

    # Time series (sample to keep files manageable)
    n_samples = min(200, len(coherence_samples))
    time_series = {
        "t": [i * dt for i in range(n_samples)],
        "coherence": coherence_samples[:n_samples],
    }

    return metrics, time_series


def main() -> None:
    """Run MAXIMUM extended parameter sweep across all variables with high resolution."""

    print("=" * 80)
    print("MAXIMUM EXTENDED PARAMETER SWEEP - ULTRA HIGH RESOLUTION")
    print("Primal Logic Robotic Hand Controller")
    print("=" * 80)
    print()
    print("Powered by Primal Tech Invest")
    print("🔗 www.primaltechinvest.com")
    print()
    print("=" * 80)
    print()

    # MAXIMUM parameter grid - HIGH RESOLUTION
    grid = ParamGrid(params={
        "alpha_base": [0.25, 0.3, 0.4, 0.45, 0.54, 0.6, 0.7, 0.75],  # 8 values
        "beta_gain": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],                 # 6 values
        "theta": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],          # 8 values
        "torque_max": [0.4, 0.5, 0.7, 0.9, 1.2],                    # 5 values
        "memory_mode": ["exponential", "recursive_planck"],          # 2 modes
        "steps": [200, 500],                                         # 2 durations
    })

    total_configs = 8 * 6 * 8 * 5 * 2 * 2
    print(f"Total configurations: {total_configs}")
    print()
    print("Parameter ranges (EXTENDED):")
    print("  - alpha_base: [0.25, 0.3, 0.4, 0.45, 0.54, 0.6, 0.7, 0.75] (8 values)")
    print("  - beta_gain: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2] (6 values)")
    print("  - theta: [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8] (8 values)")
    print("  - torque_max: [0.4, 0.5, 0.7, 0.9, 1.2] (5 values)")
    print("  - memory_mode: ['exponential', 'recursive_planck'] (2 modes)")
    print("  - steps: [200, 500] (2 durations)")
    print()
    print(f"Estimated time: ~{total_configs / 50:.0f} seconds at 50 configs/sec")
    print()
    print("🚀 LAUNCHING MAXIMUM SWEEP...")
    print()

    # Run maximum extended sweep
    output_dir = run_parameter_sweep(
        sim_name="primal_kernel",
        param_grid=grid,
        simulate_fn=simulate_primal_kernel_extended,
        tag="MAXIMUM_extended_sweep",
    )

    print()
    print("=" * 80)
    print("MAXIMUM EXTENDED SWEEP COMPLETE")
    print("=" * 80)
    print(f"Results: {output_dir}")
    print()
    print("Files generated:")
    print(f"  📊 summary/summary.csv  ({total_configs} rows - ALL CONFIGS)")
    print(f"  📈 summary/stats.json   (aggregated statistics)")
    print(f"  📁 raw/sim_*.csv        ({total_configs} time series)")
    print(f"  📝 REPORT.md            (human-readable summary)")
    print()
    print("Extended metrics computed:")
    print("  - Mean/Max/Min/Std Torque")
    print("  - Mean/Min Coherence")
    print("  - Mean/Max Angle")
    print("  - Mean/Max Velocity")
    print("  - Saturation Ratio")
    print("  - Lipschitz Estimates (Max + Mean)")
    print("  - Total Energy")
    print("  - Stability (3 levels: strict, moderate, relaxed)")
    print()
    print("=" * 80)
    print()
    print("Powered by Primal Tech Invest")
    print("🔗 www.primaltechinvest.com")
    print()


if __name__ == "__main__":
    main()
