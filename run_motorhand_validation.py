#!/usr/bin/env python3
"""
Run MotorHandPro validation pipeline including Arduino robotic hand test

This script runs the complete MotorHandPro validation framework which tests
the Primal Logic control law against multiple real-world repositories including
the Arduino robotic hand framework.

Validation targets:
- SpaceX: Rocket landing control
- Tesla: Multi-actuator synchronization
- Firestorm/PX4: Drone stabilization
- CARLA: Autonomous vehicle control
- Tesla Roadster: Motor control
- Arduino: Robotic hand grasp control (15-DOF)

Usage:
    python run_motorhand_validation.py
    python run_motorhand_validation.py --arduino-only

Requirements:
    - numpy
    - MotorHandPro submodule initialized (git submodule update --init)

Patent Pending: U.S. Provisional Patent Application No. 63/842,846
Copyright 2025 Donte Lightfoot - The Phoney Express LLC / Locked In Safety
"""

import sys
import argparse
from pathlib import Path

# Add validation directory to path
VALIDATION_DIR = Path(__file__).parent / "validation"
sys.path.insert(0, str(VALIDATION_DIR.parent))

try:
    from validation.arduino_validation_extension import ArduinoHandValidator
except ImportError as e:
    print(f"ERROR: Failed to import validation framework: {e}")
    print("\nPlease ensure:")
    print("  1. MotorHandPro submodule initialized: git submodule update --init")
    print("  2. NumPy installed: pip install numpy")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run MotorHandPro validation pipeline with Arduino test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all validation tests (SpaceX, Tesla, Firestorm, CARLA, Arduino)
  python run_motorhand_validation.py

  # Run only Arduino robotic hand test
  python run_motorhand_validation.py --arduino-only

  # Export results to JSON
  python run_motorhand_validation.py --export results.json

  # Generate LaTeX report
  python run_motorhand_validation.py --latex report.tex
        """
    )

    parser.add_argument(
        '--arduino-only', action='store_true',
        help='Run only Arduino robotic hand validation test'
    )
    parser.add_argument(
        '--export', type=str, metavar='FILE',
        help='Export results to JSON file (default: integrations/validation_results.json)'
    )
    parser.add_argument(
        '--latex', type=str, metavar='FILE',
        help='Generate LaTeX validation report (default: integrations/validation_report.tex)'
    )

    args = parser.parse_args()

    # Create validator
    print("="*80)
    print("MOTORHANDPRO VALIDATION PIPELINE")
    print("Testing Primal Logic Framework with Arduino Integration")
    print("="*80)
    print()

    validator = ArduinoHandValidator()

    # Run validation tests
    if args.arduino_only:
        print("Running Arduino-only validation...\n")
        result = validator.validate_arduino_robotic_hand()
        results = [result]
    else:
        print("Running complete validation suite...\n")
        results = validator.run_all_validations_with_arduino()

    # Export results if requested
    if args.export:
        validator.export_results_to_json(args.export)
    else:
        # Default export
        default_path = "validation/arduino_results.json"
        validator.export_results_to_json(default_path)

    # Generate LaTeX report if requested
    if args.latex:
        validator.generate_latex_report(args.latex)
    elif not args.arduino_only:
        # Default LaTeX for full run
        default_latex = "validation/arduino_report.tex"
        validator.generate_latex_report(default_latex)

    # Print final status
    print("\n" + "="*80)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"VALIDATION COMPLETE: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All validation tests PASSED")
        print("✓ Primal Logic framework validated across all repositories")
        return 0
    else:
        print(f"✗ {total - passed} test(s) FAILED")
        print("Please review validation results above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
