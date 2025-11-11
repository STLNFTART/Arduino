#!/usr/bin/env python3
"""Demonstration of Microprocessor-Heart-Arduino integration.

This demo showcases the complete pipeline:
1. Recursive Planck Operator (microprocessor) processes cardiac signals
2. Multi-Heart Model simulates heart-brain coupling with RPO
3. Cardiac output streams to Arduino hardware via serial

Usage:
    # Simulation only (no Arduino)
    python demos/demo_heart_arduino.py

    # With Arduino connected (requires pyserial)
    python demos/demo_heart_arduino.py --arduino /dev/ttyACM0

    # Custom simulation parameters
    python demos/demo_heart_arduino.py --duration 10.0 --dt 0.001
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from primal_logic.constants import DT
from primal_logic.heart_arduino_bridge import (
    HeartArduinoBridge,
    ProcessorHeartArduinoLink,
)
from primal_logic.heart_model import CouplingParameters, MultiHeartModel


def sinusoidal_cardiac_input(t: float, frequency: float = 1.0, amplitude: float = 0.5) -> float:
    """Generate sinusoidal cardiac input simulating rhythmic activity."""
    return amplitude * math.sin(2.0 * math.pi * frequency * t)


def step_brain_setpoint(t: float, step_time: float = 2.0) -> float:
    """Generate step changes in brain setpoint simulating cognitive demands."""
    return 0.5 if t < step_time else 1.0


def run_demo(
    duration: float = 5.0,
    dt: float = DT,
    arduino_port: str | None = None,
    arduino_baud: int = 115200,
    use_frequency_coupling: bool = False,
    use_forcing: bool = False,
) -> None:
    """Run the heart-Arduino integration demo.

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
    use_frequency_coupling : bool
        Use frequency-dependent coupling parameters.
    use_forcing : bool
        Enable dual-frequency forcing (RSA + baroreflex).
    """
    print("=" * 70)
    print("Microprocessor-Heart-Arduino Integration Demo")
    print("=" * 70)
    print(f"Duration: {duration:.2f}s | Timestep: {dt*1000:.2f}ms")

    # Configure coupling parameters
    if use_frequency_coupling:
        print("✓ Using frequency-dependent coupling (RSA + Baroreflex)")
        coupling = CouplingParameters(
            neural_to_cardiac_gain=0.35,  # Mid-range vagal/sympathetic blend
            cardiac_to_neural_gain=0.25,  # Baroreflex feedback strength
            low_freq_weight=0.7,   # Baroreflex dominates slow changes
            high_freq_weight=0.3,  # RSA affects fast dynamics
        )
    else:
        print("✓ Using default coupling parameters")
        coupling = CouplingParameters()

    if use_forcing:
        print(f"✓ Dual-frequency forcing enabled (RSA: {coupling.omega_rsa/(2*math.pi):.3f} Hz, Baro: {coupling.omega_baro/(2*math.pi):.3f} Hz)")

    # Initialize Multi-Heart Model with RPO
    heart_model = MultiHeartModel(
        lambda_heart=0.115,
        lambda_brain=0.092,
        coupling_strength=0.15,  # Deprecated: maintained for backward compatibility
        coupling_params=coupling,
        rpo_alpha=0.4,
        dt=dt,
    )

    # Initialize Arduino bridge if port specified
    arduino_bridge = None
    if arduino_port:
        try:
            arduino_bridge = HeartArduinoBridge(
                port=arduino_port,
                baud=arduino_baud,
                normalize=True,
            )
            print(f"✓ Arduino connected on {arduino_port} at {arduino_baud} baud")
        except Exception as e:
            print(f"✗ Arduino connection failed: {e}")
            print("  Continuing in simulation mode...")
            arduino_bridge = None
    else:
        print("✓ Running in simulation mode (no Arduino)")

    # Create unified processor-heart-Arduino link
    link = ProcessorHeartArduinoLink(
        heart_model=heart_model,
        arduino_bridge=arduino_bridge,
        send_interval=10,  # Send to Arduino every 10 steps (~10ms at 1kHz)
    )

    print("\nSimulation starting...")
    print("-" * 70)

    # Simulation loop
    n_steps = int(duration / dt)
    for step in range(n_steps):
        t = step * dt

        # Generate inputs
        cardiac_input = sinusoidal_cardiac_input(t, frequency=1.2, amplitude=0.6)
        brain_setpoint = step_brain_setpoint(t, step_time=duration / 2.0)
        theta = 1.0  # Command envelope

        # Update the complete pipeline
        link.update(
            cardiac_input=cardiac_input,
            brain_setpoint=brain_setpoint,
            theta=theta,
            use_forcing=use_forcing,
        )

        # Print status every 0.5 seconds
        if step % int(0.5 / dt) == 0:
            state = link.get_state()
            cardiac_out = heart_model.get_cardiac_output()
            print(f"t={t:5.2f}s | ", end="")
            print(f"n_heart={state['n_heart']:6.3f} | ", end="")
            print(f"n_brain={state['n_brain']:6.3f} | ", end="")
            print(f"HR={state['heart_rate']:5.3f} | ", end="")
            print(f"BA={state['brain_activity']:5.3f} | ", end="")
            print(f"Out=[{cardiac_out[0]:.2f},{cardiac_out[1]:.2f},{cardiac_out[2]:.2f},{cardiac_out[3]:.2f}]")

    print("-" * 70)
    print("\n✓ Simulation complete!")
    print("\nFinal State:")
    final_state = link.get_state()
    print(f"  Heart Neural Potential: {final_state['n_heart']:.4f}")
    print(f"  Brain Neural Potential: {final_state['n_brain']:.4f}")
    print(f"  Heart Rate (norm):      {final_state['heart_rate']:.4f}")
    print(f"  Brain Activity (norm):  {final_state['brain_activity']:.4f}")

    # RPO diagnostics
    print("\nRecursive Planck Operator (RPO) Diagnostics:")
    print(f"  Heart RPO state:  {heart_model.rpo_heart.state:.6f}")
    print(f"  Brain RPO state:  {heart_model.rpo_brain.state:.6f}")
    print(f"  h_eff (heart):    {heart_model.rpo_heart.h_eff:.6e}")
    print(f"  β_P (heart):      {heart_model.rpo_heart.beta_p:.6f}")

    final_output = heart_model.get_cardiac_output()
    print("\nFinal Arduino Output:")
    print(f"  Channel 0 (Heart Rate):    {final_output[0]:.4f}")
    print(f"  Channel 1 (Brain Activity): {final_output[1]:.4f}")
    print(f"  Channel 2 (Coherence):     {final_output[2]:.4f}")
    print(f"  Channel 3 (Combined):      {final_output[3]:.4f}")

    print("\n" + "=" * 70)


def main() -> None:
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="Microprocessor-Heart-Arduino integration demo"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Simulation duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DT,
        help=f"Timestep in seconds (default: {DT})",
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
    parser.add_argument(
        "--frequency-coupling",
        action="store_true",
        help="Enable frequency-dependent coupling (RSA + baroreflex)",
    )
    parser.add_argument(
        "--forcing",
        action="store_true",
        help="Enable dual-frequency forcing term in cardiac dynamics",
    )

    args = parser.parse_args()

    run_demo(
        duration=args.duration,
        dt=args.dt,
        arduino_port=args.arduino,
        arduino_baud=args.baud,
        use_frequency_coupling=args.frequency_coupling,
        use_forcing=args.forcing,
    )


if __name__ == "__main__":
    main()
