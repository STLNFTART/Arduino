#!/usr/bin/env python3
"""
MotorHandPro Integration Demo

Demonstrates the complete integration pipeline:
    primal_logic Hand → MotorHandPro Hardware

Shows:
- Connection to MotorHandPro hardware via serial
- Torque streaming from hand simulation
- Primal Logic parameter synchronization (Donte, Lightfoot constants)
- Real-time control energy monitoring
- Optional heart-brain-immune coupling
- Optional Recursive Planck Operator microprocessor

Usage:
    python demos/demo_motorhand_integration.py --port /dev/ttyACM0 --duration 10.0
    python demos/demo_motorhand_integration.py --simulate  # No hardware
    python demos/demo_motorhand_integration.py --full      # All features enabled

Patent Pending: U.S. Provisional Patent Application No. 63/842,846
Copyright 2025 Donte Lightfoot - The Phoney Express LLC / Locked In Safety
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from primal_logic.motorhand_integration import (
    MotorHandProBridge,
    UnifiedPrimalLogicController,
    create_integrated_system,
)
from primal_logic.hand import RoboticHand
from primal_logic.trajectory import GraspTrajectory
from primal_logic.constants import DEFAULT_FINGERS, JOINTS_PER_FINGER


def demo_basic_connection(port: str, duration: float):
    """
    Demo 1: Basic MotorHandPro connection and torque streaming.

    Shows minimal integration for testing hardware connectivity.
    """
    print("=" * 70)
    print("DEMO 1: Basic MotorHandPro Connection")
    print("=" * 70)
    print(f"Port: {port}")
    print(f"Duration: {duration}s")
    print()

    # Create bridge
    bridge = MotorHandProBridge(port=port)

    # Connect to hardware
    print("Connecting to MotorHandPro...")
    if not bridge.connect():
        print("ERROR: Failed to connect to hardware")
        print("Make sure:")
        print("  1. MotorHandPro is connected to USB")
        print("  2. Arduino sketch is uploaded")
        print(f"  3. Port {port} is correct (try ls /dev/ttyACM* or /dev/ttyUSB*)")
        return False

    print("✓ Connected successfully")
    print()

    # Send test torques
    print("Sending test torque pattern...")
    n_steps = int(duration / 0.01)
    n_actuators = DEFAULT_FINGERS * JOINTS_PER_FINGER

    for i in range(n_steps):
        t = i * 0.01

        # Generate sinusoidal test pattern
        torques = 0.3 * np.sin(2 * np.pi * 0.5 * t) * np.ones(n_actuators)

        # Send to hardware
        bridge.send_torques(torques)

        # Print status every second
        if i % 100 == 0:
            state = bridge.get_state()
            print(f"t={t:.2f}s | Ec={state['control_energy']:.4f} | "
                  f"L={state['lipschitz_estimate']:.6f} | "
                  f"Stable={state['stable']}")

        time.sleep(0.01)

    # Disconnect
    bridge.disconnect()
    print("\n✓ Demo completed successfully")
    return True


def demo_hand_simulation(port: str, duration: float, simulate_only: bool = False):
    """
    Demo 2: Hand simulation with MotorHandPro actuation.

    Shows integration of full hand model with hardware control.
    """
    print("=" * 70)
    print("DEMO 2: Hand Simulation with MotorHandPro")
    print("=" * 70)
    print(f"Mode: {'Simulation Only' if simulate_only else 'Hardware Control'}")
    print(f"Duration: {duration}s")
    print()

    # Create hand model
    hand = RoboticHand(
        n_fingers=DEFAULT_FINGERS,
        memory_mode="exponential",
        use_serial=False,  # MotorHandPro handles serial
    )

    # Create MotorHandPro bridge
    bridge = MotorHandProBridge(port=port)

    if not simulate_only:
        print("Connecting to MotorHandPro...")
        if not bridge.connect():
            print("ERROR: Failed to connect. Running in simulation mode only.")
            simulate_only = True
        else:
            print("✓ Connected to hardware")
            print()

    # Create grasp trajectory
    print("Creating grasp trajectory...")
    trajectory = GraspTrajectory(
        n_fingers=DEFAULT_FINGERS,
        grasp_type="power",
        duration=duration
    )

    # Run simulation
    print("Running hand simulation...")
    n_steps = int(duration / 0.01)
    dt = 0.01

    for i in range(n_steps):
        t = i * dt

        # Get target angles from trajectory
        target_angles = trajectory.get_target(t)

        # Step hand simulation
        hand.step(target_angles, dt)

        # Get torques
        torques = hand.get_torques()

        # Send to hardware (if connected)
        if not simulate_only:
            bridge.send_torques(torques)

        # Print status every second
        if i % 100 == 0:
            angles = hand.get_angles()
            if not simulate_only:
                state = bridge.get_state()
                print(f"t={t:.2f}s | "
                      f"Angles=[{angles[0]:.3f}, {angles[1]:.3f}, {angles[2]:.3f}, ...] | "
                      f"Ec={state['control_energy']:.4f} | "
                      f"Stable={state['stable']}")
            else:
                print(f"t={t:.2f}s | "
                      f"Angles=[{angles[0]:.3f}, {angles[1]:.3f}, {angles[2]:.3f}, ...] | "
                      f"Torques=[{torques[0]:.3f}, {torques[1]:.3f}, {torques[2]:.3f}, ...]")

        time.sleep(dt)

    # Cleanup
    if not simulate_only:
        bridge.disconnect()

    print("\n✓ Demo completed successfully")
    return True


def demo_full_integration(port: str, duration: float, simulate_only: bool = False):
    """
    Demo 3: Full integration with all features.

    Shows complete pipeline:
        Field → Hand → RPO → Heart → MotorHandPro
    """
    print("=" * 70)
    print("DEMO 3: Full Primal Logic Integration")
    print("=" * 70)
    print(f"Mode: {'Simulation Only' if simulate_only else 'Hardware Control'}")
    print(f"Duration: {duration}s")
    print()

    print("Creating integrated system...")
    print("  ✓ Hand model (5 fingers, 3 DOF each)")
    print("  ✓ MotorHandPro bridge")
    print("  ✓ Heart-brain-immune model")
    print("  ✓ Recursive Planck Operator (RPO)")
    print()

    # Create complete system
    controller = create_integrated_system(
        port=port,
        use_heart=True,
        use_rpo=True,
        memory_mode="recursive_planck",
    )

    # Connect to hardware
    if not simulate_only:
        print("Connecting to MotorHandPro...")
        if not controller.motorhand.connect():
            print("ERROR: Failed to connect. Running in simulation mode only.")
            simulate_only = True
        else:
            print("✓ Connected to hardware")
            print()

    # Create trajectory
    print("Creating power grasp trajectory...")
    trajectory = GraspTrajectory(
        n_fingers=DEFAULT_FINGERS,
        grasp_type="power",
        duration=duration
    )

    # Run integrated control
    print("\nRunning integrated control system...")
    print("Columns: time | control_energy | lipschitz | heart_rate | brain_activity")
    print("-" * 70)

    n_steps = int(duration / 0.01)

    for i in range(n_steps):
        t = i * 0.01

        # Get target from trajectory
        target = trajectory.get_target(t)

        # Execute control step
        controller.step(target)

        # Print status every second
        if i % 100 == 0:
            state = controller.get_full_state()
            motorhand_state = state['motorhand']

            output = f"t={state['time']:5.2f}s | Ec={motorhand_state['control_energy']:8.4f} | "
            output += f"L={motorhand_state['lipschitz_estimate']:8.6f} | "

            if 'heart' in state:
                output += f"HR={state['heart']['heart_rate']:6.2f} | "
                output += f"Brain={state['heart']['brain_activity']:6.3f}"
            else:
                output += "Heart: N/A"

            print(output)

        time.sleep(0.01)

    # Cleanup
    if not simulate_only:
        controller.motorhand.disconnect()

    print("\n" + "=" * 70)
    print("FINAL STATE SUMMARY")
    print("=" * 70)

    final_state = controller.get_full_state()
    print(f"Total simulation time: {final_state['time']:.2f}s")
    print(f"Total control steps: {final_state['step']}")
    print(f"Final control energy: {final_state['motorhand']['control_energy']:.4f}")
    print(f"Lipschitz constant: {final_state['motorhand']['lipschitz_estimate']:.6f}")
    print(f"System stable: {final_state['motorhand']['stable']}")
    print()

    if 'heart' in final_state:
        print("Physiological State:")
        print(f"  Heart rate: {final_state['heart']['heart_rate']:.2f} BPM")
        print(f"  Brain activity: {final_state['heart']['brain_activity']:.3f}")
        print()

    print("✓ Full integration demo completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="MotorHandPro Integration Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic hardware test
  python demos/demo_motorhand_integration.py --basic

  # Hand simulation with hardware
  python demos/demo_motorhand_integration.py --hand

  # Full integration (all features)
  python demos/demo_motorhand_integration.py --full

  # Simulation only (no hardware required)
  python demos/demo_motorhand_integration.py --simulate --duration 5.0

  # Custom serial port
  python demos/demo_motorhand_integration.py --port /dev/ttyUSB0 --full
        """
    )

    # Demo selection
    parser.add_argument(
        '--basic', action='store_true',
        help='Run basic connection demo (default)'
    )
    parser.add_argument(
        '--hand', action='store_true',
        help='Run hand simulation demo'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Run full integration demo (all features)'
    )

    # Configuration
    parser.add_argument(
        '--port', type=str, default='/dev/ttyACM0',
        help='Serial port for MotorHandPro (default: /dev/ttyACM0)'
    )
    parser.add_argument(
        '--duration', type=float, default=10.0,
        help='Duration in seconds (default: 10.0)'
    )
    parser.add_argument(
        '--simulate', action='store_true',
        help='Simulation mode only (no hardware required)'
    )

    args = parser.parse_args()

    # Determine which demo to run
    if args.full:
        success = demo_full_integration(args.port, args.duration, args.simulate)
    elif args.hand:
        success = demo_hand_simulation(args.port, args.duration, args.simulate)
    else:
        # Default: basic demo
        if args.simulate:
            print("ERROR: Basic demo requires hardware. Use --hand or --full for simulation mode.")
            return 1
        success = demo_basic_connection(args.port, args.duration)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
