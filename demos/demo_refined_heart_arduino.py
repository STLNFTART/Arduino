#!/usr/bin/env python3
"""Demonstration of Refined Heart-Brain Model with Arduino integration.

This demo showcases the physiologically realistic heart-brain coupling model:
1. Van der Pol cardiac oscillator with RSA and baroreflex
2. FitzHugh-Nagumo neural model with slow adaptation
3. Frequency-dependent bidirectional coupling
4. Optional Arduino output streaming via serial

Usage:
    # Simulation only (no Arduino)
    python demos/demo_refined_heart_arduino.py

    # With Arduino connected (requires pyserial)
    python demos/demo_refined_heart_arduino.py --arduino /dev/ttyACM0

    # Custom simulation parameters
    python demos/demo_refined_heart_arduino.py --duration 60.0 --dt 0.001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from primal_logic.refined_heart_brain import RefinedHeartBrainCouplingModel
from primal_logic.heart_arduino_bridge import HeartArduinoBridge


def run_demo(
    duration: float = 60.0,
    dt: float = 0.001,
    arduino_port: str | None = None,
    arduino_baud: int = 115200,
) -> None:
    """Run the refined heart-brain Arduino integration demo.

    Parameters
    ----------
    duration : float
        Simulation duration in seconds.
    dt : float
        Timestep in seconds.
    arduino_port : str | None
        Serial port for Arduino (e.g., '/dev/ttyACM0'). None for simulation only.
    arduino_baud : int
        Baud rate for Arduino communication.
    """
    print("=" * 70)
    print("Refined Heart-Brain Coupling Model - Arduino Integration Demo")
    print("=" * 70)
    print(f"Duration: {duration:.2f}s | Timestep: {dt*1000:.2f}ms")

    # Initialize Refined Heart-Brain Model
    model = RefinedHeartBrainCouplingModel(dt=dt)
    print("\n✓ Refined Heart-Brain Model initialized")
    print("  • Van der Pol cardiac oscillator (RSA + baroreflex)")
    print("  • FitzHugh-Nagumo neural model (3-variable)")
    print("  • Frequency-dependent bidirectional coupling")

    # Initialize Arduino bridge if port specified
    arduino_bridge = None
    if arduino_port:
        try:
            arduino_bridge = HeartArduinoBridge(
                port=arduino_port,
                baud=arduino_baud,
                normalize=True,
            )
            print(f"\n✓ Arduino connected on {arduino_port} at {arduino_baud} baud")
        except Exception as e:
            print(f"\n✗ Arduino connection failed: {e}")
            print("  Continuing in simulation mode...")
            arduino_bridge = None
    else:
        print("\n✓ Running in simulation mode (no Arduino)")

    print("\nSimulation starting...")
    print("-" * 70)
    print("Time  | Heart   | Brain   | HR    | BA    | Coherence | Channels")
    print("-" * 70)

    # Simulation loop
    n_steps = int(duration / dt)
    send_interval = 10  # Send to Arduino every 10 steps

    for step in range(n_steps):
        t = model.time

        # Step the model
        model.step()

        # Send to Arduino at intervals
        if arduino_bridge is not None and step % send_interval == 0:
            # The refined model has get_cardiac_output() which returns [HR, BA, coherence, combined]
            cardiac_output = model.get_cardiac_output()
            arduino_bridge.send_raw_values(cardiac_output)

        # Print status every 5 seconds
        if step % int(5.0 / dt) == 0:
            cardiac_out = model.get_cardiac_output()
            state = model.state

            print(
                f"{t:5.1f}s | "
                f"x={state[0]:6.3f} | "
                f"v={state[2]:6.3f} | "
                f"HR={cardiac_out[0]:5.3f} | "
                f"BA={cardiac_out[1]:5.3f} | "
                f"Coh={cardiac_out[2]:5.3f} | "
                f"[{cardiac_out[0]:.2f},{cardiac_out[1]:.2f},{cardiac_out[2]:.2f},{cardiac_out[3]:.2f}]"
            )

    print("-" * 70)
    print("\n✓ Simulation complete!")

    # Final state analysis
    final_state = model.state
    final_output = model.get_cardiac_output()

    print("\nFinal State:")
    print(f"  Cardiac position (x):       {final_state[0]:.6f}")
    print(f"  Cardiac velocity (v):       {final_state[1]:.6f}")
    print(f"  Neural activation (v_n):    {final_state[2]:.6f}")
    print(f"  Neural recovery (w_n):      {final_state[3]:.6f}")
    print(f"  Neural adaptation (z_n):    {final_state[4]:.6f}")

    print("\nPhysiological Metrics:")
    print(f"  Heart Rate (normalized):    {final_output[0]:.4f}")
    print(f"  Brain Activity (normalized): {final_output[1]:.4f}")
    print(f"  Heart-Brain Coherence:      {final_output[2]:.4f}")
    print(f"  Combined Signal:            {final_output[3]:.4f}")

    print("\nModel Characteristics:")
    print("  • RSA frequency:            0.1 Hz (6 cycles/min)")
    print("  • Baroreflex frequency:     0.04 Hz (2.4 cycles/min)")
    print("  • Neural adaptation time:   ~50 seconds")
    print("  • Integration timestep:     " + f"{dt*1000:.2f} ms")

    print("\nFinal Arduino Output:")
    print(f"  Channel 0 (Heart Rate):     {final_output[0]:.4f}")
    print(f"  Channel 1 (Brain Activity): {final_output[1]:.4f}")
    print(f"  Channel 2 (Coherence):      {final_output[2]:.4f}")
    print(f"  Channel 3 (Combined):       {final_output[3]:.4f}")

    print("\n" + "=" * 70)
    print("Key Features Demonstrated:")
    print("  ✓ Dual-frequency HRV (RSA + baroreflex)")
    print("  ✓ Slow neural entrainment (~30-45 sec)")
    print("  ✓ Physiologically realistic coupling gains")
    print("  ✓ Compatible with existing Arduino bridge")
    print("=" * 70)


def main() -> None:
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="Refined Heart-Brain Model with Arduino integration demo"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Simulation duration in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Timestep in seconds (default: 0.001)",
    )
    parser.add_argument(
        "--arduino",
        type=str,
        default=None,
        help="Arduino serial port (e.g., /dev/ttyACM0 or COM3). Omit for simulation only.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Arduino baud rate (default: 115200)",
    )

    args = parser.parse_args()

    run_demo(
        duration=args.duration,
        dt=args.dt,
        arduino_port=args.arduino,
        arduino_baud=args.baud,
    )


if __name__ == "__main__":
    main()
