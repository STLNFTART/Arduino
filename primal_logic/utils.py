"""Utility functions used across the Primal Logic framework."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure logging with a consistent format for reproducibility."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def safe_clip(value: float, lower: float, upper: float) -> float:
    """Clip *value* to [lower, upper] while preserving float semantics."""
    clipped = float(np.clip(value, lower, upper))
    if clipped != value:
        logger.debug("Value %.6f clipped to %.6f", value, clipped)
    return clipped


def laplacian_2d(field: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Compute a discrete 2-D Laplacian with Neumann boundary conditions."""
    lap = np.zeros_like(field)
    lap[1:-1, 1:-1] = (
        -4.0 * field[1:-1, 1:-1]
        + field[2:, 1:-1]
        + field[:-2, 1:-1]
        + field[1:-1, 2:]
        + field[1:-1, :-2]
    ) / (dx * dy)

    # Copy values to the boundary (zero-gradient assumption)
    lap[0, :] = lap[1, :]
    lap[-1, :] = lap[-2, :]
    lap[:, 0] = lap[:, 1]
    lap[:, -1] = lap[:, -2]
    return lap
