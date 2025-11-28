#!/usr/bin/env python3
"""
PrimalRWA Integration Demo - MotorHandPro with Token Burn Tracking

This demonstrates the complete integration pipeline:
    PrimalRWA → MotorHandPro Actuator → RPOBurnMeter → Hedera Smart Contract

Shows:
- MotorHandPro actuator creation with burn tracking enabled
- Real-time token burns (1 RPO = 1 second of smooth actuation)
- Hedera testnet integration (or CSV fallback)
- Complete audit trail of all burns

Usage:
    # Dry run (CSV logging only, no hardware)
    python demo_primalrwa_integration.py --mode dry_run --duration 5.0

    # Hedera testnet (with credentials, no hardware)
    python demo_primalrwa_integration.py --mode hedera_testnet --duration 5.0

    # With hardware (if available)
    python demo_primalrwa_integration.py --mode hedera_testnet --duration 5.0 --port /dev/ttyACM0

Patent Pending: U.S. Provisional Patent Application No. 63/842,846
Copyright 2025 Donte Lightfoot - The Phoney Express LLC / Locked In Safety
"""

import argparse
import sys
import time
import os
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment from .env.hedera if exists
env_file = Path(__file__).parent / ".env.hedera"
if env_file.exists():
    print(f"Loading Hedera credentials from {env_file}")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

from billing.rpo_burn_meter import RPOBurnMeter
from primal_logic.motorhand_actuator import create_motorhand_actuator


def print_header(title: str):
    """Print formatted section header."""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()


def print_burn_report(burn_meter: RPOBurnMeter, actuator_key: str):
    """Print detailed burn report."""
    report = burn_meter.get_burn_report()
    burned = report.get(actuator_key, 0)

    print()
    print("─" * 80)
    print("BURN REPORT")
    print("─" * 80)
    print(f"Actuator: {actuator_key}")
    print(f"Burned seconds: {burned}")
    print(f"Burned tokens: {burned} RPO  (1 token = 1 second of actuation)")
    print("─" * 80)


def demo_basic_burn_tracking(duration: float, mode: str, port: str, use_hardware: bool):
    """
    Demo 1: Basic burn tracking with MotorHandPro actuator.

    This shows the minimal PrimalRWA integration:
    - Create actuator with burn meter
    - Send torque commands
    - Burns happen automatically (1 RPO per second)
    """
    print_header("DEMO 1: Basic Burn Tracking with MotorHandPro")

    print(f"Configuration:")
    print(f"  Mode: {mode}")
    print(f"  Duration: {duration}s")
    print(f"  Hardware: {'Enabled' if use_hardware else 'Simulation only'}")
    print(f"  Port: {port if use_hardware else 'N/A'}")
    print()

    # Load burn meter from config files
    print("Loading burn meter from config files...")
    burn_meter = RPOBurnMeter.from_config_files(
        operator_config_path=Path("billing/rpo_operator_config.json"),
        actuator_map_path=Path("billing/rpo_actuator_addresses.json"),
        mode=mode,
    )
    print(f"✓ Burn meter loaded (mode={mode})")
    print(f"  Contract ID: {burn_meter.contract_id}")
    if mode == "hedera_testnet":
        print(f"  Operator ID: {burn_meter.operator_id}")
        print(f"  Network: {burn_meter.network}")
    print()

    # Create MotorHandPro actuator with burn tracking
    print("Creating MotorHandPro actuator with burn tracking...")
    actuator = create_motorhand_actuator(
        port=port,
        burn_meter=burn_meter,
        planck_mode=True,  # Enable burn tracking
        auto_connect=use_hardware,
    )
    print(f"✓ Actuator created")
    print(f"  Burn tracking: ENABLED (planck_mode=True)")
    print(f"  Timestep: {actuator.dt}s (10ms)")
    print(f"  Actuator key: {actuator.burn_meter_key}")
    print()

    # Run actuation loop
    print(f"Running actuation for {duration} seconds...")
    print("Each second of actuation will burn 1 RPO token")
    print()
    print("Progress:")
    print("-" * 80)

    n_steps = int(duration / actuator.dt)
    dt = actuator.dt

    for i in range(n_steps):
        t = i * dt

        # Generate test torque pattern
        # Sinusoidal pattern simulating smooth robotic actuation
        torques = 0.3 * np.sin(2 * np.pi * 0.5 * t) * np.ones(15)

        # Send torques - burns tracked automatically
        actuator.step(torques)

        # Print status every second
        if i % 100 == 0 or i == n_steps - 1:
            state = actuator.get_state()
            runtime = state["cumulative_runtime"]
            burned = state.get("burned_seconds", 0)

            print(f"t={t:6.2f}s | Runtime={runtime:6.2f}s | Burned={burned:4d} RPO | "
                  f"Mode={mode}")

        # Sleep to maintain real-time if no hardware
        if not use_hardware:
            time.sleep(dt)

    print("-" * 80)
    print()

    # Cleanup
    if use_hardware:
        actuator.disconnect()

    # Print final report
    print_burn_report(burn_meter, actuator.burn_meter_key)

    # Show log file
    log_path = Path("rpo_burn_log.csv")
    if log_path.exists():
        print()
        print(f"✓ Burn log written to: {log_path}")
        print()
        print("Last 5 entries:")
        with open(log_path) as f:
            lines = f.readlines()
            for line in lines[-5:]:
                print(f"  {line.strip()}")

    print()
    print("✓ Demo 1 completed successfully")
    print()
    print("Key Takeaways:")
    print("  • 1 RPO token = 1 second of perfectly smooth robotic actuation")
    print("  • Burns happen automatically when planck_mode=True")
    print("  • All burns logged to rpo_burn_log.csv for audit trail")
    print("  • Ready for Hedera smart contract integration")


def demo_multiple_actuators(duration: float, mode: str):
    """
    Demo 2: Multiple actuators with independent burn tracking.

    This shows how PrimalRWA can manage multiple MotorHandPro units:
    - Each actuator has its own burn tracking
    - Burns are aggregated per actuator
    - Complete system-wide burn reporting
    """
    print_header("DEMO 2: Multiple Actuators with Independent Burn Tracking")

    print(f"Configuration:")
    print(f"  Mode: {mode}")
    print(f"  Duration: {duration}s")
    print(f"  Actuators: 3 (motorhand_pro_actuator, primal_logic_hand, multi_heart_model)")
    print()

    # Load burn meter
    print("Loading burn meter...")
    burn_meter = RPOBurnMeter.from_config_files(
        operator_config_path=Path("billing/rpo_operator_config.json"),
        actuator_map_path=Path("billing/rpo_actuator_addresses.json"),
        mode=mode,
    )
    print(f"✓ Burn meter loaded")
    print()

    # Create multiple actuators (simulation only for demo)
    print("Creating multiple actuators...")
    actuators = {
        "motorhand_pro_actuator": create_motorhand_actuator(
            port="/dev/ttyACM0",
            burn_meter=burn_meter,
            planck_mode=True,
            auto_connect=False,  # Simulation mode
        ),
    }
    print(f"✓ Created {len(actuators)} actuator(s)")
    print()

    # Run actuation loop with different patterns per actuator
    print(f"Running multi-actuator system for {duration} seconds...")
    print()
    print("Progress:")
    print("-" * 80)

    n_steps = int(duration / 0.01)
    dt = 0.01

    for i in range(n_steps):
        t = i * dt

        # Actuate all actuators
        for key, actuator in actuators.items():
            # Different torque patterns per actuator
            if "motorhand" in key:
                torques = 0.3 * np.sin(2 * np.pi * 0.5 * t) * np.ones(15)
            else:
                torques = 0.2 * np.cos(2 * np.pi * 0.3 * t) * np.ones(15)

            actuator.step(torques)

        # Print status every second
        if i % 100 == 0 or i == n_steps - 1:
            report = burn_meter.get_burn_report()
            total_burned = sum(report.values())
            print(f"t={t:6.2f}s | Total burned={total_burned:4d} RPO | "
                  f"Breakdown: {dict(report)}")

    print("-" * 80)
    print()

    # Print final aggregate report
    report = burn_meter.get_burn_report()
    total_burned = sum(report.values())

    print()
    print("─" * 80)
    print("AGGREGATE BURN REPORT")
    print("─" * 80)
    for actuator_key, burned_seconds in report.items():
        if burned_seconds > 0:  # Only show actuators that burned tokens
            print(f"  {actuator_key:30s} → {burned_seconds:4d} RPO")
    print("─" * 80)
    print(f"  {'TOTAL':30s} → {total_burned:4d} RPO")
    print("─" * 80)

    print()
    print("✓ Demo 2 completed successfully")


def demo_primalrwa_control_loop(duration: float, mode: str, port: str, use_hardware: bool):
    """
    Demo 3: Complete PrimalRWA control loop.

    This shows the full integration:
    - PrimalRWA receives external signals
    - Converts to actuator commands
    - MotorHandPro executes with burn tracking
    - Burns submitted to Hedera smart contract
    """
    print_header("DEMO 3: Complete PrimalRWA Control Loop")

    print(f"Configuration:")
    print(f"  Mode: {mode}")
    print(f"  Duration: {duration}s")
    print(f"  Hardware: {'Enabled' if use_hardware else 'Simulation only'}")
    print()

    # Load burn meter
    burn_meter = RPOBurnMeter.from_config_files(
        operator_config_path=Path("billing/rpo_operator_config.json"),
        actuator_map_path=Path("billing/rpo_actuator_addresses.json"),
        mode=mode,
    )

    # Create actuator
    actuator = create_motorhand_actuator(
        port=port,
        burn_meter=burn_meter,
        planck_mode=True,
        auto_connect=use_hardware,
    )

    print("PrimalRWA Control Pipeline:")
    print("  1. External Signal → PrimalRWA Logic")
    print("  2. PrimalRWA Logic → Torque Commands")
    print("  3. Torque Commands → MotorHandPro Actuator")
    print("  4. Actuator Runtime → RPOBurnMeter")
    print("  5. RPOBurnMeter → Hedera Smart Contract (1 RPO per second)")
    print()

    # Simulate PrimalRWA control loop
    print(f"Running control loop for {duration} seconds...")
    print()

    n_steps = int(duration / 0.01)

    for i in range(n_steps):
        t = i * 0.01

        # Step 1: PrimalRWA receives external signal (simulated)
        external_signal = 0.5 + 0.3 * np.sin(2 * np.pi * 0.2 * t)

        # Step 2: PrimalRWA converts to torque commands
        # (This would be your actual PrimalRWA control logic)
        base_torque = 0.4 * external_signal
        torques = base_torque * np.ones(15)

        # Step 3: Send to MotorHandPro (burns tracked automatically)
        actuator.step(torques)

        # Progress update every second
        if i % 100 == 0:
            state = actuator.get_state()
            burned = state.get("burned_seconds", 0)
            print(f"t={t:6.2f}s | Signal={external_signal:.3f} | "
                  f"Torque={base_torque:.3f} | Burned={burned:4d} RPO")

    print()

    # Cleanup
    if use_hardware:
        actuator.disconnect()

    # Final report
    print_burn_report(burn_meter, actuator.burn_meter_key)

    print()
    print("✓ Demo 3 completed successfully")
    print()
    print("PrimalRWA Integration Complete:")
    print("  ✓ External signals → smooth robotic actuation")
    print("  ✓ 1 RPO token burned per second of actuation")
    print("  ✓ All burns logged and ready for blockchain")
    print("  ✓ Complete audit trail in rpo_burn_log.csv")


def main():
    parser = argparse.ArgumentParser(
        description="PrimalRWA Integration Demo with MotorHandPro Token Burns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic demo (dry run, no hardware)
  python demo_primalrwa_integration.py --demo 1 --mode dry_run --duration 5.0

  # Hedera testnet integration (simulation)
  python demo_primalrwa_integration.py --demo 1 --mode hedera_testnet --duration 5.0

  # With hardware
  python demo_primalrwa_integration.py --demo 1 --mode hedera_testnet --port /dev/ttyACM0 --hardware

  # Multiple actuators
  python demo_primalrwa_integration.py --demo 2 --mode hedera_testnet --duration 10.0

  # Complete control loop
  python demo_primalrwa_integration.py --demo 3 --mode hedera_testnet --duration 10.0
        """
    )

    parser.add_argument(
        '--demo', type=int, default=1, choices=[1, 2, 3],
        help='Demo number (1=basic, 2=multiple, 3=control loop)'
    )
    parser.add_argument(
        '--mode', type=str, default='dry_run',
        choices=['dry_run', 'hedera_testnet'],
        help='Burn mode (dry_run=CSV only, hedera_testnet=with credentials)'
    )
    parser.add_argument(
        '--duration', type=float, default=5.0,
        help='Duration in seconds (default: 5.0)'
    )
    parser.add_argument(
        '--port', type=str, default='/dev/ttyACM0',
        help='Serial port for MotorHandPro (default: /dev/ttyACM0)'
    )
    parser.add_argument(
        '--hardware', action='store_true',
        help='Enable hardware mode (requires MotorHandPro connected)'
    )

    args = parser.parse_args()

    # Print welcome
    print()
    print("=" * 80)
    print("  PrimalRWA Integration Demo - MotorHandPro with Token Burn Tracking")
    print("=" * 80)
    print()
    print("  Patent Pending: U.S. Provisional Patent Application No. 63/842,846")
    print("  Copyright 2025 Donte Lightfoot - The Phoney Express LLC")
    print()

    # Run selected demo
    try:
        if args.demo == 1:
            demo_basic_burn_tracking(args.duration, args.mode, args.port, args.hardware)
        elif args.demo == 2:
            demo_multiple_actuators(args.duration, args.mode)
        elif args.demo == 3:
            demo_primalrwa_control_loop(args.duration, args.mode, args.port, args.hardware)

    except Exception as e:
        print()
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
