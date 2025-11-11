"""Module containing configurable constants for the Primal Logic framework.

All values are documented with SI units to preserve physical traceability.
The module now exposes the Lightfoot/Donte constants used by the recursive
Planck operator described in the Quantro–Primal formalism.
"""

from __future__ import annotations

from typing import Tuple

# Versioning for traceability
VERSION: str = "v1.0.0"

# === Simulation cadence ===
DT: float = 1e-3  # [s] integration step

# === Core parameters ===
BETA_DEFAULT: float = 0.8  # dimensionless memory kernel gain
ALPHA_DEFAULT: float = 0.54  # Lightfoot's nominal coupling constant
LAMBDA_DEFAULT: float = 0.115  # [1/s] decay rate for field/memory kernels
K_PERF: float = 1.47
K_POWER: float = 0.12
EPSILON: float = 0.1
K_COUPLING: float = 0.5
GAMMA: float = 0.2

# === Temporal and spatial scales ===
TEMPORAL_SCALES: Tuple[float, ...] = (0.1e-3, 1.0e-3, 10.0e-3, 100.0e-3)
SPATIAL_SCALES: Tuple[float, ...] = (0.1e-3, 1.0e-3, 5.0e-3, 20.0e-3)

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

# === Quantitative framework constants ===
# Planck constant in Joule·seconds for reference when computing h_eff.
PLANCK_CONSTANT: float = 6.626_070_15e-34
# Donte's constant couples energetic stability to informational bandwidth.
DONTE_CONSTANT: float = 149.999_231_4
# Lightfoot's constant bounds the damping factor in the Volterra kernel.
LIGHTFOOT_MIN: float = 0.54
LIGHTFOOT_MAX: float = 0.56
LIGHTFOOT_RANGE: Tuple[float, float] = (LIGHTFOOT_MIN, LIGHTFOOT_MAX)
