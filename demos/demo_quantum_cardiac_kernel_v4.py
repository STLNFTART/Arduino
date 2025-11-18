#!/usr/bin/env python3
"""PRIMAL LOGIC Kernel v4: Quantum-Inspired Cardiac Modeling Demonstration.

This demo showcases the complete Kernel v4 integration:
- Multiple cardiac tissue models (FitzHugh-Nagumo, Hodgkin-Huxley, Windkessel)
- Quantum amplitude states with superposition
- Plasma field collective dynamics
- 15+ PRIMAL LOGIC algorithms
- Multi-scale temporal analysis
- Real-time ECG generation

Usage:
    # Run with FitzHugh-Nagumo model
    python demos/demo_quantum_cardiac_kernel_v4.py

    # Run with Hodgkin-Huxley model
    python demos/demo_quantum_cardiac_kernel_v4.py --model hodgkin_huxley

    # Custom parameters
    python demos/demo_quantum_cardiac_kernel_v4.py --steps 1000 --tissue-size 100
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from primal_logic.cardiac_tissue_models import CardiacModelType
from primal_logic.constants import ALPHA_DEFAULT, DT, LAMBDA_DEFAULT
from primal_logic.kernel_v4 import create_kernel_v4

# Optional: matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not available, visualization disabled")


def generate_cardiac_stimulus(
    tissue_size: tuple[int, int], step: int, frequency: float = 1.0
) -> np.ndarray:
    """Generate spatiotemporal cardiac stimulation pattern.

    Parameters
    ----------
    tissue_size : tuple[int, int]
        Grid dimensions (nx, ny).
    step : int
        Current timestep.
    frequency : float
        Stimulation frequency.

    Returns
    -------
    np.ndarray
        Stimulation pattern.
    """
    nx, ny = tissue_size
    stimulus = np.zeros((nx, ny))

    # Pacemaker region (upper left corner)
    pacemaker_x = nx // 4
    pacemaker_y = ny // 4
    radius = min(nx, ny) // 10

    # Periodic stimulation
    if step % int(100 / frequency) < 10:
        for i in range(max(0, pacemaker_x - radius), min(nx, pacemaker_x + radius)):
            for j in range(max(0, pacemaker_y - radius), min(ny, pacemaker_y + radius)):
                dist = math.sqrt((i - pacemaker_x) ** 2 + (j - pacemaker_y) ** 2)
                if dist < radius:
                    stimulus[i, j] = 1.0 * math.exp(-(dist**2) / (2 * radius**2))

    return stimulus


def run_simulation(
    model_type: CardiacModelType,
    tissue_size: int = 50,
    num_steps: int = 500,
    dt: float = DT,
    alpha: float = ALPHA_DEFAULT,
    lambda_decay: float = LAMBDA_DEFAULT,
    visualize: bool = True,
) -> dict:
    """Run complete Kernel v4 simulation.

    Parameters
    ----------
    model_type : CardiacModelType
        Cardiac model to simulate.
    tissue_size : int
        Grid size (will be tissue_size x tissue_size).
    num_steps : int
        Number of simulation steps.
    dt : float
        Integration timestep.
    alpha : float
        Alpha parameter.
    lambda_decay : float
        Lambda decay parameter.
    visualize : bool
        Whether to generate visualization.

    Returns
    -------
    dict
        Simulation results and metrics.
    """
    print("=" * 80)
    print("PRIMAL LOGIC Kernel v4: Quantum-Cardiac Integration")
    print("=" * 80)
    print(f"Model Type: {model_type.value}")
    print(f"Tissue Size: {tissue_size}x{tissue_size}")
    print(f"Steps: {num_steps}")
    print(f"Timestep: {dt*1000:.2f}ms")
    print(f"Alpha: {alpha:.4f}")
    print(f"Lambda: {lambda_decay:.4f}")
    print("=" * 80)

    # Create Kernel v4 system
    kernel = create_kernel_v4(
        model_type=model_type,
        tissue_size=(tissue_size, tissue_size),
        dt=dt,
        alpha=alpha,
        lambda_decay=lambda_decay,
    )

    # Storage for results
    ecg_signal = []
    energy_trace = []
    coherence_trace = []
    phase_sync_trace = []
    voltage_snapshots = []

    # Simulation loop
    print("\nSimulation running...")
    for step in range(num_steps):
        # Generate cardiac stimulus
        stimulus = generate_cardiac_stimulus(
            (tissue_size, tissue_size), step, frequency=1.2
        )

        # Execute kernel step
        state = kernel.step(cardiac_input=stimulus, theta=1.0)

        # Record metrics
        ecg_signal.append(kernel.get_ecg_signal())
        energy_trace.append(state["total_energy"])
        coherence_trace.append(state["coherence"])
        phase_sync_trace.append(state["phase_sync"])

        # Store voltage snapshots for visualization
        if step % (num_steps // 10) == 0 or step == num_steps - 1:
            voltage_snapshots.append(state["cardiac_voltage"].copy())

        # Progress indicator
        if step % (num_steps // 10) == 0:
            metrics = kernel.get_system_metrics()
            print(f"Step {step:4d} | Energy: {state['total_energy']:6.3f} | "
                  f"Coherence: {state['coherence']:.3f} | "
                  f"Phase Sync: {state['phase_sync']:.3f} | "
                  f"Collapses: {metrics['collapse_count']}")

    print("\n" + "=" * 80)
    print("Simulation complete!")

    # Get final metrics
    final_metrics = kernel.get_system_metrics()
    print("\nFinal System Metrics:")
    print(f"  Total Steps:          {final_metrics['step_count']}")
    print(f"  Quantum Collapses:    {final_metrics['collapse_count']}")
    print(f"  Average Energy:       {final_metrics['avg_energy']:.4f}")
    print(f"  Peak Energy:          {final_metrics['peak_energy']:.4f}")
    print(f"  Average Coherence:    {final_metrics['avg_coherence']:.4f}")
    print(f"  Peak Coherence:       {final_metrics['peak_coherence']:.4f}")
    print(f"  Active Algorithms:    {final_metrics['active_algorithms']}")
    print(f"  Total Executions:     {final_metrics['total_executions']}")

    # Algorithm-specific metrics
    print("\nPRIMAL LOGIC Algorithm Performance:")
    from primal_logic.primal_algorithms import AlgorithmType

    for alg_type in [
        AlgorithmType.QUANTUM_SUPERPOSITION,
        AlgorithmType.PLASMA_FIELD,
        AlgorithmType.TEMPORAL_COHERENCE,
        AlgorithmType.PHASE_SYNCHRONIZATION,
        AlgorithmType.COLLECTIVE_DYNAMICS,
    ]:
        metrics = kernel.algorithm_suite.get_algorithm_metrics(alg_type)
        print(f"  {alg_type.value:25s} | Executions: {metrics.execution_count:4d} | "
              f"Energy: {metrics.total_energy:8.3f}")

    print("=" * 80)

    # Visualization
    if visualize and MATPLOTLIB_AVAILABLE:
        visualize_results(
            ecg_signal,
            energy_trace,
            coherence_trace,
            phase_sync_trace,
            voltage_snapshots,
            model_type,
            alpha,
            lambda_decay,
        )

    return {
        "ecg_signal": ecg_signal,
        "energy_trace": energy_trace,
        "coherence_trace": coherence_trace,
        "phase_sync_trace": phase_sync_trace,
        "voltage_snapshots": voltage_snapshots,
        "final_metrics": final_metrics,
    }


def visualize_results(
    ecg_signal: list,
    energy_trace: list,
    coherence_trace: list,
    phase_sync_trace: list,
    voltage_snapshots: list,
    model_type: CardiacModelType,
    alpha: float,
    lambda_decay: float,
) -> None:
    """Generate comprehensive visualization of results.

    Parameters
    ----------
    ecg_signal : list
        ECG signal trace.
    energy_trace : list
        Energy over time.
    coherence_trace : list
        Coherence over time.
    phase_sync_trace : list
        Phase synchronization over time.
    voltage_snapshots : list
        Voltage field snapshots.
    model_type : CardiacModelType
        Cardiac model type.
    alpha : float
        Alpha parameter.
    lambda_decay : float
        Lambda parameter.
    """
    fig = plt.figure(figsize=(20, 12))

    # ECG Signal
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(ecg_signal, "b-", linewidth=1)
    ax1.set_title(f"Synthetic ECG ({model_type.value})")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    # Energy Trace
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(energy_trace, "r-", linewidth=1)
    ax2.set_title("Quantum Energy Evolution")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Total Energy")
    ax2.grid(True, alpha=0.3)

    # Coherence
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(coherence_trace, "g-", linewidth=1)
    ax3.set_title("Temporal Coherence")
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("Coherence")
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3)

    # Phase Synchronization
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(phase_sync_trace, "purple", linewidth=1)
    ax4.set_title("Phase Synchronization")
    ax4.set_xlabel("Time Steps")
    ax4.set_ylabel("Sync Order")
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3)

    # Voltage Snapshots (4 snapshots)
    num_snapshots = min(4, len(voltage_snapshots))
    for i in range(num_snapshots):
        idx = i * (len(voltage_snapshots) // num_snapshots)
        if idx >= len(voltage_snapshots):
            idx = len(voltage_snapshots) - 1

        ax = plt.subplot(3, 4, 5 + i)
        im = ax.imshow(voltage_snapshots[idx], cmap="RdBu_r", aspect="auto")
        ax.set_title(f"Voltage Field (t={idx * 10})")
        plt.colorbar(im, ax=ax)

    # Energy histogram
    ax9 = plt.subplot(3, 4, 9)
    ax9.hist(energy_trace, bins=30, alpha=0.7, color="orange", edgecolor="black")
    ax9.set_title("Energy Distribution")
    ax9.set_xlabel("Energy")
    ax9.set_ylabel("Frequency")

    # Coherence histogram
    ax10 = plt.subplot(3, 4, 10)
    ax10.hist(coherence_trace, bins=30, alpha=0.7, color="cyan", edgecolor="black")
    ax10.set_title("Coherence Distribution")
    ax10.set_xlabel("Coherence")
    ax10.set_ylabel("Frequency")

    # Phase space (Energy vs Coherence)
    ax11 = plt.subplot(3, 4, 11)
    ax11.scatter(energy_trace, coherence_trace, c=range(len(energy_trace)),
                 cmap="viridis", alpha=0.5, s=10)
    ax11.set_title("Phase Space: Energy vs Coherence")
    ax11.set_xlabel("Energy")
    ax11.set_ylabel("Coherence")
    ax11.grid(True, alpha=0.3)

    # Summary text
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis("off")
    summary_text = f"""
    PRIMAL LOGIC Kernel v4

    Model: {model_type.value}

    Parameters:
      α = {alpha:.4f}
      λ = {lambda_decay:.4f}

    Final Metrics:
      Avg Energy: {np.mean(energy_trace):.3f}
      Peak Energy: {np.max(energy_trace):.3f}
      Avg Coherence: {np.mean(coherence_trace):.3f}
      Avg Phase Sync: {np.mean(phase_sync_trace):.3f}

    Algorithms: 15+ active
    """
    ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes,
              fontfamily="monospace", fontsize=10, verticalalignment="top")

    plt.tight_layout()
    plt.savefig("kernel_v4_results.png", dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved: kernel_v4_results.png")
    plt.show()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PRIMAL LOGIC Kernel v4 Demonstration"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fitzhugh_nagumo",
        choices=["fitzhugh_nagumo", "hodgkin_huxley", "windkessel"],
        help="Cardiac model type",
    )
    parser.add_argument(
        "--tissue-size",
        type=int,
        default=50,
        help="Tissue grid size (default: 50x50)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of simulation steps (default: 500)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DT,
        help=f"Integration timestep (default: {DT})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA_DEFAULT,
        help=f"Alpha parameter (default: {ALPHA_DEFAULT})",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_decay",
        type=float,
        default=LAMBDA_DEFAULT,
        help=f"Lambda decay (default: {LAMBDA_DEFAULT})",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization",
    )

    args = parser.parse_args()

    # Map model name to enum
    model_map = {
        "fitzhugh_nagumo": CardiacModelType.FITZHUGH_NAGUMO,
        "hodgkin_huxley": CardiacModelType.HODGKIN_HUXLEY,
        "windkessel": CardiacModelType.WINDKESSEL,
    }

    model_type = model_map[args.model]

    # Run simulation
    results = run_simulation(
        model_type=model_type,
        tissue_size=args.tissue_size,
        num_steps=args.steps,
        dt=args.dt,
        alpha=args.alpha,
        lambda_decay=args.lambda_decay,
        visualize=not args.no_viz,
    )

    print("\n✓ Demonstration complete!")


if __name__ == "__main__":
    main()
