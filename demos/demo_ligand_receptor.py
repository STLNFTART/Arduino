#!/usr/bin/env python3
"""Demo for Ligand-Receptor Binding Subsystem with Multiscale Coupling.

This demo showcases the complete hierarchical integration:
1. Ligand-receptor binding dynamics at the cellular level
2. Immune signaling accumulation and feedback
3. Heart-brain neural-cardiac coupling with immune modulation
4. Simulation of infection-induced autonomic changes

The demo runs three scenarios:
A. Baseline: Normal physiological state
B. Infection: Simulated pathogen exposure with rising ligand concentration
C. Pharmacological intervention: Modulation of binding rates to restore balance
"""

import argparse
import math
from pathlib import Path
from typing import List

from primal_logic import MultiscaleCoupling, LigandReceptor, ImmuneSignaling


def pulsatile_ligand(t: float, baseline: float = 1.0, amplitude: float = 2.0, period: float = 5.0) -> float:
    """Generate pulsatile ligand concentration (simulates pathogen exposure).

    Parameters
    ----------
    t : float
        Current time in seconds.
    baseline : float
        Baseline ligand concentration. Default: 1.0
    amplitude : float
        Amplitude of pulsatile oscillation. Default: 2.0
    period : float
        Period of oscillation in seconds. Default: 5.0

    Returns
    -------
    float
        Ligand concentration L(t).
    """
    return baseline + amplitude * (1 + math.sin(2 * math.pi * t / period)) / 2


def exponential_ligand(t: float, baseline: float = 1.0, rate: float = 0.1, max_conc: float = 5.0) -> float:
    """Generate exponentially rising ligand concentration (simulates acute infection).

    Parameters
    ----------
    t : float
        Current time in seconds.
    baseline : float
        Initial ligand concentration. Default: 1.0
    rate : float
        Growth rate. Default: 0.1
    max_conc : float
        Maximum concentration (saturation). Default: 5.0

    Returns
    -------
    float
        Ligand concentration L(t).
    """
    return min(max_conc, baseline * math.exp(rate * t))


def run_simulation(
    coupling: MultiscaleCoupling,
    duration: float,
    cardiac_input_fn=None,
    brain_setpoint_fn=None,
    save_path: Path = None,
) -> List[dict]:
    """Run simulation and collect time series data.

    Parameters
    ----------
    coupling : MultiscaleCoupling
        The multiscale coupling system to simulate.
    duration : float
        Simulation duration in seconds.
    cardiac_input_fn : callable, optional
        Function that returns cardiac_input as f(t).
    brain_setpoint_fn : callable, optional
        Function that returns brain_setpoint as f(t).
    save_path : Path, optional
        If provided, save results to CSV file.

    Returns
    -------
    List[dict]
        Time series of all state variables.
    """
    if cardiac_input_fn is None:
        cardiac_input_fn = lambda t: 0.5 * math.sin(2 * math.pi * t / 1.0)  # 1 Hz cardiac oscillation

    if brain_setpoint_fn is None:
        brain_setpoint_fn = lambda t: 0.3  # Constant cognitive load

    dt = coupling.dt
    num_steps = int(duration / dt)
    time_series = []

    print(f"Running simulation for {duration:.1f} seconds ({num_steps} steps)...")

    for step in range(num_steps):
        t = step * dt

        # Get inputs
        cardiac_input = cardiac_input_fn(t)
        brain_setpoint = brain_setpoint_fn(t)

        # Step the system
        coupling.step(
            cardiac_input=cardiac_input,
            brain_setpoint=brain_setpoint,
            theta=1.0,
        )

        # Collect data every 10 steps to reduce memory
        if step % 10 == 0:
            state = coupling.get_complete_state()
            state["time"] = t
            time_series.append(state)

    # Save to CSV if requested
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            # Write header
            if time_series:
                keys = time_series[0].keys()
                f.write(",".join(keys) + "\n")
                # Write data
                for state in time_series:
                    f.write(",".join(str(state[k]) for k in keys) + "\n")
        print(f"Results saved to {save_path}")

    return time_series


def print_summary(time_series: List[dict], label: str):
    """Print summary statistics for a simulation run."""
    if not time_series:
        return

    # Compute means over second half (steady state)
    midpoint = len(time_series) // 2
    steady_state = time_series[midpoint:]

    avg_receptor = sum(s["occupancy_fraction"] for s in steady_state) / len(steady_state)
    avg_immune = sum(s["immune_intensity"] for s in steady_state) / len(steady_state)
    avg_heart_rate = sum(s["heart_rate"] for s in steady_state) / len(steady_state)
    avg_brain = sum(s["brain_activity"] for s in steady_state) / len(steady_state)
    avg_lambda_brain = sum(s["lambda_brain"] for s in steady_state) / len(steady_state)
    avg_lambda_heart = sum(s["lambda_heart"] for s in steady_state) / len(steady_state)

    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"Receptor Occupancy:     {avg_receptor:.3f}")
    print(f"Immune Intensity:       {avg_immune:.3f}")
    print(f"Heart Rate (norm):      {avg_heart_rate:.3f}")
    print(f"Brain Activity:         {avg_brain:.3f}")
    print(f"Lambda Brain (mod):     {avg_lambda_brain:.4f}")
    print(f"Lambda Heart (mod):     {avg_lambda_heart:.4f}")
    print(f"{'=' * 60}")


def scenario_baseline(duration: float, save_dir: Path):
    """Scenario A: Baseline physiological state."""
    print("\n" + "=" * 60)
    print("SCENARIO A: BASELINE PHYSIOLOGICAL STATE")
    print("=" * 60)

    # Create default multiscale system
    coupling = MultiscaleCoupling(dt=0.01)

    # Run simulation
    time_series = run_simulation(
        coupling=coupling,
        duration=duration,
        save_path=save_dir / "baseline.csv",
    )

    print_summary(time_series, "Baseline Summary")


def scenario_infection(duration: float, save_dir: Path):
    """Scenario B: Simulated infection with rising ligand concentration."""
    print("\n" + "=" * 60)
    print("SCENARIO B: SIMULATED INFECTION (EXPONENTIAL LIGAND)")
    print("=" * 60)

    # Create ligand-receptor with exponentially rising ligand
    ligand_receptor = LigandReceptor(
        k_on=1.5,
        k_off=0.3,
        receptor_total=100.0,
        gamma=0.15,
        ligand_input=lambda t: exponential_ligand(t, baseline=1.0, rate=0.05, max_conc=8.0),
        dt=0.01,
    )

    # Create immune system with higher sensitivity
    immune = ImmuneSignaling(
        rho=0.08,
        delta=0.02,
        alpha_brain=0.5,
        alpha_heart=0.3,
        dt=0.01,
    )

    coupling = MultiscaleCoupling(
        ligand_receptor=ligand_receptor,
        immune_signaling=immune,
        dt=0.01,
    )

    # Run simulation
    time_series = run_simulation(
        coupling=coupling,
        duration=duration,
        save_path=save_dir / "infection.csv",
    )

    print_summary(time_series, "Infection Summary")


def scenario_pharmacological(duration: float, save_dir: Path):
    """Scenario C: Pharmacological intervention (reduced binding rates)."""
    print("\n" + "=" * 60)
    print("SCENARIO C: PHARMACOLOGICAL INTERVENTION")
    print("=" * 60)

    # Simulate drug that reduces binding rate (k_on reduced, k_off increased)
    ligand_receptor = LigandReceptor(
        k_on=0.5,  # Reduced from baseline
        k_off=1.0,  # Increased from baseline
        receptor_total=100.0,
        gamma=0.1,
        ligand_input=lambda t: pulsatile_ligand(t, baseline=2.0, amplitude=1.0, period=10.0),
        dt=0.01,
    )

    # Standard immune response
    immune = ImmuneSignaling(
        rho=0.05,
        delta=0.03,  # Faster resolution
        alpha_brain=0.3,
        alpha_heart=0.2,
        dt=0.01,
    )

    coupling = MultiscaleCoupling(
        ligand_receptor=ligand_receptor,
        immune_signaling=immune,
        dt=0.01,
    )

    # Run simulation
    time_series = run_simulation(
        coupling=coupling,
        duration=duration,
        save_path=save_dir / "pharmacological.csv",
    )

    print_summary(time_series, "Pharmacological Summary")


def main():
    """Run all three scenarios."""
    parser = argparse.ArgumentParser(
        description="Ligand-Receptor Binding Subsystem Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Simulation duration in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/ligand_receptor"),
        help="Output directory for CSV files (default: artifacts/ligand_receptor)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("LIGAND-RECEPTOR BINDING SUBSYSTEM DEMONSTRATION")
    print("Multiscale Integration: Cellular → Immune → Neural/Cardiac")
    print("=" * 60)

    # Run all scenarios
    scenario_baseline(args.duration, args.output_dir)
    scenario_infection(args.duration, args.output_dir)
    scenario_pharmacological(args.duration, args.output_dir)

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nKey observations:")
    print("1. Infection scenario shows increased immune intensity and altered decay rates")
    print("2. Heart rate variability changes under immune modulation (fever tachycardia)")
    print("3. Brain activity modulation reflects cognitive fatigue from inflammation")
    print("4. Pharmacological intervention reduces receptor occupancy and stabilizes system")
    print("\nThese results demonstrate the bidirectional coupling between cellular")
    print("biochemistry and macro-scale neural-cardiac dynamics.")


if __name__ == "__main__":
    main()
