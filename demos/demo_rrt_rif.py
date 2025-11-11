#!/usr/bin/env python3
"""Demonstrate recursive intent field (RIF) behaviour with the RPO memory kernel."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from primal_logic.constants import DT
from primal_logic.field import PrimalLogicField
from primal_logic.hand import RoboticHand
from primal_logic.trajectory import generate_grasp_trajectory
from primal_logic.utils import configure_logging, flatten, mean, write_csv


def main() -> None:
    configure_logging()
    logging.info("Running recursive intent and coherence demo")

    steps = 2000
    hand = RoboticHand(memory_mode="recursive_planck", rpo_alpha=0.35)
    field = PrimalLogicField(nx=6, ny=6)
    trajectory = generate_grasp_trajectory(steps, hand.n_fingers, hand.n_joints_per_finger)

    rows = []
    for step in range(steps):
        theta = 0.9 + 0.1 * math.sin(0.002 * step)
        coherence = field.step(theta=theta)
        hand.step(desired_angles=trajectory[step], theta=theta, coherence=coherence, step=step)
        torques = flatten(hand.get_torques())
        rows.append((step * DT, coherence, mean(torques)))

    output_path = Path("artifacts/rrt_rif_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(str(output_path), ("time_s", "coherence", "mean_torque"), rows)

    avg_coherence = mean([row[1] for row in rows])
    logging.info("Average coherence = %.6f", avg_coherence)
    print(f"Average coherence over simulation = {avg_coherence:.6f}")


if __name__ == "__main__":
    main()
