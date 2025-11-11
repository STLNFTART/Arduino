"""CLI for generating repository inventory artifacts.

This script summarises repository contents in both CSV and Markdown table
formats so that empty or placeholder projects can still provide
presentation-ready datasets. The outputs intentionally avoid external
dependencies for maximum portability.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is importable when the script runs without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from primal_logic.inventory import generate_inventory_artifacts
from primal_logic.utils import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the inventory CLI."""

    parser = argparse.ArgumentParser(description="Generate repository inventory datasets.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root to inspect (default: project root)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/repo_inventory.csv"),
        help="Destination CSV path (default: artifacts/repo_inventory.csv)",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/repo_inventory.md"),
        help="Destination Markdown path (default: artifacts/repo_inventory.md)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the repository inventory CLI."""

    args = parse_args()
    configure_logging(getattr(logging, args.log_level))
    logging.getLogger(__name__).info(
        "Generating inventory for root=%s -> csv=%s, markdown=%s", args.root, args.csv, args.markdown
    )
    generate_inventory_artifacts(args.root, args.csv, args.markdown)


if __name__ == "__main__":
    main()

