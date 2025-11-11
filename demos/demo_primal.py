#!/usr/bin/env python3
"""Validate recursive Planck operator stability and norm bounds."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from primal_logic.constants import DT, LIGHTFOOT_MAX
from primal_logic.rpo import RecursivePlanckOperator
from primal_logic.utils import configure_logging, write_csv


def _theoretical_bound(alpha: float, lightfoot: float, lam: float, theta: float) -> float:
    """Compute a conservative amplitude bound for the recursive operator."""

    alpha_eff = lightfoot * alpha
    beta_p = lightfoot / (1.0 + lam)
    if not (0 < alpha_eff * DT < 1):  # pragma: no cover - guard for misconfiguration
        raise ValueError("alpha_eff * DT must lie between 0 and 1")
    return (theta / alpha_eff) * (1.0 + abs(beta_p) / (1.0 - alpha_eff * DT))


def main() -> None:
    """Run the Recursive Planck Operator on a sinusoidal input."""

    configure_logging()
    logging.info("Starting Recursive Planck Operator validation demo")

    theta = 1.0
    alpha = 0.4
    operator = RecursivePlanckOperator(alpha=alpha, lightfoot=LIGHTFOOT_MAX)

    samples = []
    for step in range(5000):
        input_value = math.sin(0.01 * step)
        state = operator.step(theta=theta, input_value=input_value, step_index=step)
        samples.append((step * DT, state))

    output_path = Path("artifacts/rpo_primal.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(str(output_path), ("time_s", "state"), samples)

    max_state = max(abs(value) for _, value in samples)
    bound = _theoretical_bound(alpha=alpha, lightfoot=operator.lightfoot, lam=operator.lam, theta=theta)
    logging.info("Maximum |state| = %.6f, theoretical bound = %.6f", max_state, bound)
    print(f"Maximum |state| = {max_state:.6f}; theoretical bound = {bound:.6f}")


if __name__ == "__main__":
    main()
