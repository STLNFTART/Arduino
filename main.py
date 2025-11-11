"""Entry point for the Primal Logic robotic hand demo."""

from __future__ import annotations

from pathlib import Path

from primal_logic import run_demo


def main() -> None:
    """Execute the demo and save a torque log for analysis."""
    output = Path("artifacts/torques.csv")
    run_demo(log_path=output)


if __name__ == "__main__":
    main()
