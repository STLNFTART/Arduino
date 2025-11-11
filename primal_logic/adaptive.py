"""Adaptive gain scheduling utilities."""

from __future__ import annotations

import math

from .constants import (
    ALPHA_DEFAULT,
    DT,
    ENERGY_BUDGET,
    PHASE_COUPLING,
    SIGMA,
    SPATIAL_SCALES,
    TEMPORAL_SCALES,
)


def adaptive_alpha(
    step: int,
    avg_energy: float,
    quantum_coherence: float,
    alpha_base: float = ALPHA_DEFAULT,
) -> float:
    """Compute the adaptive alpha gain used in joint controllers."""

    if alpha_base <= 0:
        raise ValueError("alpha_base must be positive for stable control gains")

    base = alpha_base * (1.0 + SIGMA * math.sin(step * 0.001))
    energy_scaling = alpha_base * (avg_energy / (1000.0 * ENERGY_BUDGET))
    coherence_term = alpha_base * PHASE_COUPLING * quantum_coherence

    temporal_influence = 0.0
    for i, scale in enumerate(TEMPORAL_SCALES):
        spatial_ratio = SPATIAL_SCALES[min(i, len(SPATIAL_SCALES) - 1)] / SPATIAL_SCALES[-1]
        temporal_influence += spatial_ratio * math.cos(step * DT / max(scale, 1e-9))

    alpha = base + energy_scaling + coherence_term + 0.1 * temporal_influence
    return float(min(max(alpha, 0.52), 0.56))
