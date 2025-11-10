"""Demo application for the Primal Logic robotic hand."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import (
    DT,
    SERIAL_BAUD,
    SERIAL_PORT,
    USE_SERIAL,
)
from .field import PrimalLogicField
from .hand import RoboticHand
from .serial_bridge import SerialHandBridge
from .trajectory import generate_grasp_trajectory
from .utils import configure_logging

logger = logging.getLogger(__name__)


def run_demo(steps: int = 3000, log_path: Optional[Path] = None) -> None:
    """Run a simulation demo and optionally log torques to disk."""
    configure_logging()

    bridge = SerialHandBridge(SERIAL_PORT, SERIAL_BAUD) if USE_SERIAL else None
    hand = RoboticHand(bridge=bridge)
    field = PrimalLogicField(nx=8, ny=8)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Logging torques to %s", log_path)

    trajectory = generate_grasp_trajectory(steps, hand.n_fingers, hand.n_joints_per_finger)

    torque_log: list[list[float]] = []
    for step in range(steps):
        theta = 1.0
        coherence = field.step(theta)
        desired = trajectory[step]

        hand.step(desired_angles=desired, theta=theta, coherence=coherence, step=step)

        if step % 500 == 0:
            avg_angle = float(np.mean(hand.get_angles()))
            logger.info("t=%6.3fs | coherence=%5.3f | avg_angle=%5.3f", step * DT, coherence, avg_angle)

        if log_path is not None:
            torque_log.append(hand.get_torques().flatten().tolist())

    if log_path is not None:
        header = ",".join(f"joint_{i}" for i in range(hand.n_fingers * hand.n_joints_per_finger))
        np.savetxt(log_path, np.array(torque_log), delimiter=",", header=header, comments="")
        logger.info("Saved torque log with %d rows", len(torque_log))

    logger.info("Demo complete.")
