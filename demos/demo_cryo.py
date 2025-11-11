#!/usr/bin/env python3
"""Compare classical and quantro-inspired thermal noise models."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path

from primal_logic.constants import DT
from primal_logic.rpo import RecursivePlanckOperator
from primal_logic.utils import configure_logging, write_csv


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


def main() -> None:
    configure_logging()
    random.seed(42)
    logging.info("Running cryogenic noise comparison demo")

    steps = 4000
    theta = 0.8
    classical_sigma = 1e-6  # [A] proxy current noise amplitude
    quantro_sigma = 6e-7

    classical_noise = [random.gauss(0.0, classical_sigma) for _ in range(steps)]

    operator = RecursivePlanckOperator(alpha=0.35)
    quantro_noise = []
    for step in range(steps):
        shot_noise = random.gauss(0.0, quantro_sigma)
        filtered = operator.step(theta=theta, input_value=shot_noise, step_index=step)
        quantro_noise.append(filtered)

    output_path = Path("artifacts/cryo_noise.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for step, value in enumerate(classical_noise):
        rows.append(("classical", step * DT, value))
    for step, value in enumerate(quantro_noise):
        rows.append(("quantro", step * DT, value))
    write_csv(str(output_path), ("mode", "time_s", "amplitude"), rows)

    logging.info("Classical RMS = %.3e, Quantro RMS = %.3e", _rms(classical_noise), _rms(quantro_noise))
    print(
        "Classical RMS = %.3e, Quantro RMS = %.3e"
        % (_rms(classical_noise), _rms(quantro_noise))
    )


if __name__ == "__main__":
    main()
