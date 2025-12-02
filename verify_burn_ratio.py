#!/usr/bin/env python3
"""
Verification Script: Prove 1 RPO Token = 1 Second of Actuation

This script runs comprehensive tests to mathematically prove that the
burn tracking works correctly with a perfect 1:1 ratio.
"""

import sys
from pathlib import Path
import subprocess
import csv
import time

def run_test(duration, mode="hedera_testnet"):
    """Run a burn test and return the results."""
    print(f"\n{'='*80}")
    print(f"Test: {duration}s actuation in {mode} mode")
    print('='*80)

    # Clear old log
    log_file = Path("rpo_burn_log.csv")
    if log_file.exists():
        log_file.unlink()

    # Run demo
    cmd = [
        "python", "demo_primalrwa_integration.py",
        "--demo", "1",
        "--mode", mode,
        "--duration", str(duration)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Count burns
    if log_file.exists():
        with open(log_file) as f:
            reader = csv.DictReader(f)
            burns = list(reader)
            burn_count = len(burns)
    else:
        burn_count = 0

    # Extract result from output
    for line in result.stdout.split('\n'):
        if "Burned seconds:" in line:
            actual_burned = int(line.split(':')[1].strip().split()[0])
            break
    else:
        actual_burned = burn_count

    print(f"\n✓ Test completed")
    print(f"  Expected burns: {int(duration)}")
    print(f"  Actual burns:   {actual_burned}")
    print(f"  Ratio:          {actual_burned / duration:.3f} tokens/second")
    print(f"  Status:         {'PASS ✓' if actual_burned == int(duration) else 'FAIL ✗'}")

    return actual_burned == int(duration)

def main():
    print("="*80)
    print("  VERIFICATION: 1 RPO Token = 1 Second of Actuation")
    print("="*80)

    tests = [
        (3.0, "dry_run"),
        (5.0, "hedera_testnet"),
        (7.5, "hedera_testnet"),
        (10.0, "hedera_testnet"),
    ]

    results = []
    for duration, mode in tests:
        passed = run_test(duration, mode)
        results.append((duration, mode, passed))
        time.sleep(1)  # Small delay between tests

    # Final report
    print("\n" + "="*80)
    print("  FINAL REPORT")
    print("="*80)

    for duration, mode, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {duration:>4.1f}s in {mode:15s} → {status}")

    all_passed = all(p for _, _, p in results)

    print("="*80)
    if all_passed:
        print("  ✓ ALL TESTS PASSED")
        print("  ✓ 1 RPO TOKEN = 1 SECOND OF ACTUATION (VERIFIED)")
    else:
        print("  ✗ SOME TESTS FAILED")
        return 1
    print("="*80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
