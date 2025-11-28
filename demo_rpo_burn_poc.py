"""Demonstrate the RecursiveActuator token burn PoC with simulated actuators.

Run from repo root:
    python demo_rpo_burn_poc.py --mode dry_run
    python demo_rpo_burn_poc.py --mode hedera_testnet  # requires Hedera env vars
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

from billing.rpo_burn_meter import RPOBurnMeter
from primal_logic.hand import RoboticHand
from primal_logic.heart_model import MultiHeartModel


def load_burn_meter(mode: str = "dry_run") -> RPOBurnMeter:
    """Load contract/operator configuration and initialize the burn meter."""

    repo_root = Path(__file__).parent
    operator_config = repo_root / "billing" / "rpo_operator_config.json"
    actuator_map = repo_root / "billing" / "rpo_actuator_addresses.json"
    return RPOBurnMeter.from_config_files(
        operator_config_path=operator_config,
        actuator_map_path=actuator_map,
        mode=mode,
    )


def run_demo(total_sim_time: float = 60.0, dt: float = 0.01, mode: str = "dry_run") -> Dict[str, int]:
    """Run a simple simulation loop and return burned seconds per actuator."""

    burn_meter = load_burn_meter(mode=mode)

    hand = RoboticHand(dt=dt, burn_meter=burn_meter, planck_mode=True)
    heart_model = MultiHeartModel(dt=dt, burn_meter=burn_meter, planck_mode=False)

    desired_angles = [
        [0.2 for _ in range(hand.n_joints_per_finger)] for _ in range(hand.n_fingers)
    ]
    theta = 1.0
    coherence = 0.5
    total_steps = int(total_sim_time / dt)

    for step_idx in range(total_steps):
        phase = step_idx * dt
        hand.step(desired_angles=desired_angles, theta=theta, coherence=coherence, step=step_idx)

        cardiac_input = 0.1 * math.sin(phase)
        brain_setpoint = 0.1 * math.cos(phase)
        heart_model.step(cardiac_input=cardiac_input, brain_setpoint=brain_setpoint, theta=theta)

    burn_report = burn_meter.get_burn_report()

    print("RecursiveActuator Token Burn PoC (Simulated Actuators)")
    print("------------------------------------------------------")
    print(f"Burn mode: {mode}")
    print(f"Total simulated time: {total_sim_time:.2f} s with dt={dt}")
    print("Planck-mode durations:")
    print(f"  primal_logic_hand: {total_sim_time if hand.planck_mode else 0.0:.2f} s")
    print(f"  multi_heart_model: {total_sim_time if heart_model.planck_mode else 0.0:.2f} s")
    print("Burned integer seconds (one RPO per second):")
    for actuator, seconds in burn_report.items():
        print(f"  {actuator}: {seconds} s")

    log_path = Path("rpo_burn_log.csv").resolve()
    print(f"Log file: {log_path}")
    return burn_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        default="dry_run",
        choices=["dry_run", "hedera_testnet"],
        help="Burn mode to use (logs only; hedera_testnet requires Hedera env vars).",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation timestep in seconds.")
    parser.add_argument("--sim_time", type=float, default=60.0, help="Total simulated seconds.")
    args = parser.parse_args()

    run_demo(total_sim_time=args.sim_time, dt=args.dt, mode=args.mode)
