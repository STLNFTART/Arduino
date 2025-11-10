"""Utility functions for the Primal Logic framework.

This module avoids third-party numerical dependencies to keep the
repository self-contained. All helpers operate on plain Python lists and
floats so the code runs in minimal offline environments.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence, Tuple

logger = logging.getLogger(__name__)

Number = float
Matrix = List[List[Number]]


def configure_logging(level: int = logging.INFO) -> None:
    """Configure logging with a consistent format for reproducibility."""

    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def safe_clip(value: Number, lower: Number, upper: Number) -> Number:
    """Clip *value* to ``[lower, upper]`` while preserving float semantics."""

    clipped = max(lower, min(upper, float(value)))
    if clipped != value:
        logger.debug("Value %.6f clipped to %.6f", value, clipped)
    return clipped


def mean(values: Sequence[Number]) -> Number:
    """Return the arithmetic mean of *values* with a zero-length guard."""

    if not values:
        return 0.0
    return float(sum(values) / len(values))


def flatten(matrix: Sequence[Sequence[Number]]) -> List[Number]:
    """Flatten a nested sequence into a single list of numbers."""

    return [float(value) for row in matrix for value in row]


def zeros(shape: Tuple[int, ...]) -> Matrix:
    """Create a zero-initialised matrix supporting 2-D or 3-D shapes."""

    if len(shape) == 2:
        rows, cols = shape
        return [[0.0 for _ in range(cols)] for _ in range(rows)]
    if len(shape) == 3:
        dim0, dim1, dim2 = shape
        return [[[0.0 for _ in range(dim2)] for _ in range(dim1)] for _ in range(dim0)]
    raise ValueError(f"Unsupported shape: {shape}")


def zeros_like(matrix: Sequence[Sequence[Number]]) -> Matrix:
    """Return a zero matrix with the same dimensions as *matrix*."""

    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def laplacian_2d(field: Matrix, dx: Number = 1.0, dy: Number = 1.0) -> Matrix:
    """Compute a discrete 2-D Laplacian with Neumann boundary conditions."""

    rows = len(field)
    cols = len(field[0]) if rows else 0
    lap = zeros((rows, cols))

    if rows == 0 or cols == 0:
        return lap

    factor = float(dx * dy) if dx and dy else 1.0

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            lap[i][j] = (
                -4.0 * field[i][j]
                + field[i + 1][j]
                + field[i - 1][j]
                + field[i][j + 1]
                + field[i][j - 1]
            ) / factor

    # Neumann boundary: copy interior neighbours.
    if rows > 1:
        lap[0] = lap[1][:]
        lap[-1] = lap[-2][:]
    if cols > 1:
        for i in range(rows):
            lap[i][0] = lap[i][1]
            lap[i][-1] = lap[i][-2]

    return lap


def write_csv(path: str, header: Sequence[str], rows: Iterable[Sequence[Number]]) -> None:
    """Write rows of numeric data to *path* with *header* columns."""

    import csv

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([f"{float(value):.9f}" for value in row])

