"""Module containing configurable constants for the Primal Logic framework."""

from __future__ import annotations

import numpy as np

# Versioning for traceability
VERSION: str = "v1.0.0"

# === Simulation cadence ===
DT: float = 1e-3  # [s] integration step

# === Core parameters ===
ALPHA_DEFAULT: float = 0.54
LAMBDA_DEFAULT: float = 0.115
K_PERF: float = 1.47
K_POWER: float = 0.12
EPSILON: float = 0.1
K_COUPLING: float = 0.5
GAMMA: float = 0.2

# === Temporal and spatial scales ===
TEMPORAL_SCALES: np.ndarray = np.array([0.1e-3, 1.0e-3, 10.0e-3, 100.0e-3])
SPATIAL_SCALES: np.ndarray = np.array([0.1e-3, 1.0e-3, 5.0e-3, 20.0e-3])

# === Energy/phase coupling parameters ===
ENERGY_BUDGET: float = 1.0
PHASE_COUPLING: float = 0.25
SIGMA: float = 0.15

# === Serial configuration ===
USE_SERIAL: bool = False
SERIAL_PORT: str = "/dev/ttyACM0"
SERIAL_BAUD: int = 115200

# === Hand morphology ===
DEFAULT_FINGERS: int = 5
JOINTS_PER_FINGER: int = 3
HAND_MASS: float = 0.05  # [kg] effective tendon mass
HAND_DAMPING: float = 0.02  # [N*s/m] damping constant
