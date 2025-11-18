"""PRIMAL LOGIC Kernel v4: Quantum-Inspired Cardiac Modeling Integration.

This module integrates quantum-superpositional reasoning with mathematical heart
models using parameters from training: α ≈ 0.52-0.56, λ ≈ 0.11-0.12.

Main orchestrator for the complete PRIMAL LOGIC Kernel v4 system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .cardiac_tissue_models import CardiacModelType, CardiacTissueModel
from .constants import (
    ALPHA_DEFAULT,
    ENERGY_BUDGET,
    EPSILON,
    GAMMA,
    K_COUPLING,
    LAMBDA_DEFAULT,
    PHASE_COUPLING,
    SIGMA,
    TEMPORAL_SCALES,
)
from .primal_algorithms import AlgorithmType, PrimalAlgorithmSuite
from .quantum_plasma_field import PlasmaField, QuantumAmplitudeState


@dataclass
class KernelV4Parameters:
    """Parameters for Kernel v4 quantum-plasma dynamics.

    All parameters derived from training data and PRIMAL LOGIC theory.
    """

    alpha: float = ALPHA_DEFAULT  # From training range 0.52-0.56
    lambda_decay: float = LAMBDA_DEFAULT  # From training range 0.11-0.12
    epsilon: float = EPSILON  # Imaginary component influence
    gamma: float = GAMMA  # Plasma field coupling strength
    sigma: float = SIGMA  # Coherence term scaling
    collapse_threshold: float = 2.0  # Quantum collapse threshold
    energy_budget: float = ENERGY_BUDGET  # Maximum energy per update
    phase_coupling: float = PHASE_COUPLING  # Phase synchronization strength


@dataclass
class PrimalLogicKernelV4:
    """Main Kernel v4 implementation with quantum-plasma-cardiac integration.

    Integrates:
    - Cardiac tissue models (FitzHugh-Nagumo, Hodgkin-Huxley, Windkessel)
    - Quantum amplitude states with superposition
    - Plasma field collective dynamics
    - 15+ PRIMAL LOGIC algorithms

    Parameters
    ----------
    cardiac_model : CardiacTissueModel
        Cardiac tissue model instance.
    params : KernelV4Parameters
        Kernel v4 configuration parameters.
    num_quantum_components : int
        Number of quantum amplitude components.
    """

    cardiac_model: CardiacTissueModel
    params: KernelV4Parameters = field(default_factory=KernelV4Parameters)
    num_quantum_components: int = 4

    # Initialized in __post_init__
    quantum_state: QuantumAmplitudeState = field(init=False)
    plasma_field: PlasmaField = field(init=False)
    algorithm_suite: PrimalAlgorithmSuite = field(init=False)

    # System state tracking
    step_count: int = 0
    collapse_count: int = 0
    energy_history: List[float] = field(default_factory=list)
    coherence_history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize quantum states, plasma fields, and algorithms."""
        tissue_dims = (self.cardiac_model.nx, self.cardiac_model.ny)

        # Initialize quantum amplitude states
        self.quantum_state = QuantumAmplitudeState(
            dimensions=tissue_dims,
            num_components=self.num_quantum_components,
            epsilon=self.params.epsilon,
        )

        # Initialize plasma field
        self.plasma_field = PlasmaField(
            dimensions=tissue_dims,
            coupling_strength=self.params.gamma,
        )

        # Initialize algorithm suite
        self.algorithm_suite = PrimalAlgorithmSuite(
            alpha=self.params.alpha,
            lambda_decay=self.params.lambda_decay,
            energy_budget=self.params.energy_budget,
        )

    def step(
        self,
        cardiac_input: np.ndarray | None = None,
        theta: float = 1.0,
        external_modulation: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray]:
        """Execute single Kernel v4 timestep.

        Performs:
        1. Cardiac tissue dynamics
        2. Quantum state evolution
        3. Plasma field computation
        4. Algorithm execution
        5. Multi-scale integration

        Parameters
        ----------
        cardiac_input : np.ndarray | None
            External cardiac stimulation pattern.
        theta : float
            Command envelope for control.
        external_modulation : np.ndarray | None
            External modulation signal.

        Returns
        -------
        Dict[str, np.ndarray]
            System state including voltage, quantum amplitudes, fields, etc.
        """
        # === 1. CARDIAC TISSUE DYNAMICS ===
        if self.cardiac_model.model_type == CardiacModelType.FITZHUGH_NAGUMO:
            V, W = self.cardiac_model.fitzhugh_nagumo_step(cardiac_input)
            cardiac_voltage = V
        elif self.cardiac_model.model_type == CardiacModelType.HODGKIN_HUXLEY:
            V = self.cardiac_model.hodgkin_huxley_step(cardiac_input)
            cardiac_voltage = V
        elif self.cardiac_model.model_type == CardiacModelType.WINDKESSEL:
            P, Q = self.cardiac_model.windkessel_step(cardiac_input)
            cardiac_voltage = P  # Use pressure as proxy
        else:
            cardiac_voltage = np.zeros((self.cardiac_model.nx, self.cardiac_model.ny))

        # === 2. QUANTUM AMPLITUDE EVOLUTION ===
        # Use cardiac voltage as external field for quantum coupling
        external_field = cardiac_voltage[..., np.newaxis] * self.params.epsilon

        self.quantum_state.evolve(
            alpha=self.params.alpha,
            lambda_decay=self.params.lambda_decay,
            external_field=external_field,
            dt=self.cardiac_model.dt,
        )

        # Apply superposition mixing
        mixing_angle = self.params.phase_coupling * self.step_count * 0.01
        self.quantum_state.apply_superposition(mixing_angle)

        # Get quantum amplitudes
        quantum_amplitudes = self.quantum_state.get_amplitude_magnitude()

        # === 3. PLASMA FIELD COMPUTATION ===
        # Use quantum amplitudes as density for collective field
        plasma_field = self.plasma_field.compute_field(quantum_amplitudes)

        # === 4. ALGORITHM EXECUTION ===

        # Temporal coherence analysis
        coherence = self.algorithm_suite.temporal_coherence(
            self.energy_history, window_size=50
        )
        self.coherence_history.append(coherence)

        # Phase synchronization (using multiple signal channels)
        signals = [
            cardiac_voltage.flatten(),
            quantum_amplitudes.flatten(),
            plasma_field.flatten(),
        ]
        phase_sync = self.algorithm_suite.phase_synchronization(signals)

        # Collective intelligence from quantum amplitudes
        collective_metric = self.algorithm_suite.collective_intelligence(
            quantum_amplitudes.flatten()
        )

        # Energy conservation
        total_energy = np.sum(quantum_amplitudes**2)
        self.energy_history.append(total_energy)

        energy_sources = np.array([total_energy])
        conserved_energy = self.algorithm_suite.energy_conservation(
            energy_sources, max_budget=self.params.energy_budget
        )

        # Quantum collapse detection
        collapse_detected = self.algorithm_suite.quantum_collapse_detection(
            quantum_amplitudes, threshold=self.params.collapse_threshold
        )

        if collapse_detected:
            self.quantum_state.collapse_if_needed(self.params.collapse_threshold)
            self.collapse_count += 1

        # Adaptive gating of cardiac output
        gated_voltage = self.algorithm_suite.adaptive_gating(
            cardiac_voltage, gate_threshold=0.5, step=self.step_count
        )

        # Multi-scale integration
        if len(self.energy_history) > 10:
            integrated_energy = self.algorithm_suite.multi_scale_integration(
                np.array(self.energy_history[-100:])
            )
        else:
            integrated_energy = np.array([total_energy])

        # === 5. SYSTEM STATE UPDATE ===
        self.step_count += 1

        # Return comprehensive state
        return {
            "cardiac_voltage": cardiac_voltage,
            "quantum_amplitudes": quantum_amplitudes,
            "plasma_field": plasma_field,
            "gated_voltage": gated_voltage,
            "total_energy": total_energy,
            "coherence": coherence,
            "phase_sync": phase_sync,
            "collective_metric": collective_metric,
            "collapse_detected": collapse_detected,
            "step": self.step_count,
        }

    def get_ecg_signal(self) -> float:
        """Generate synthetic ECG signal from cardiac tissue.

        Returns
        -------
        float
            ECG signal value.
        """
        return self.cardiac_model.get_ecg_signal()

    def get_system_metrics(self) -> Dict[str, float]:
        """Get comprehensive system performance metrics.

        Returns
        -------
        Dict[str, float]
            System metrics including energy, coherence, algorithms, etc.
        """
        algo_summary = self.algorithm_suite.get_performance_summary()

        return {
            "step_count": self.step_count,
            "collapse_count": self.collapse_count,
            "avg_energy": np.mean(self.energy_history) if self.energy_history else 0.0,
            "peak_energy": np.max(self.energy_history) if self.energy_history else 0.0,
            "avg_coherence": (
                np.mean(self.coherence_history) if self.coherence_history else 0.0
            ),
            "peak_coherence": (
                np.max(self.coherence_history) if self.coherence_history else 0.0
            ),
            "total_algorithm_energy": algo_summary["total_energy"],
            "active_algorithms": algo_summary["active_algorithms"],
            "total_executions": algo_summary["total_executions"],
        }

    def reset(self) -> None:
        """Reset kernel to initial state."""
        # Reinitialize quantum state
        tissue_dims = (self.cardiac_model.nx, self.cardiac_model.ny)
        self.quantum_state = QuantumAmplitudeState(
            dimensions=tissue_dims,
            num_components=self.num_quantum_components,
            epsilon=self.params.epsilon,
        )

        # Reset counters and history
        self.step_count = 0
        self.collapse_count = 0
        self.energy_history.clear()
        self.coherence_history.clear()


def create_kernel_v4(
    model_type: CardiacModelType = CardiacModelType.FITZHUGH_NAGUMO,
    tissue_size: Tuple[int, int] = (50, 50),
    dt: float = 0.1,
    alpha: float = ALPHA_DEFAULT,
    lambda_decay: float = LAMBDA_DEFAULT,
) -> PrimalLogicKernelV4:
    """Factory function to create Kernel v4 system.

    Parameters
    ----------
    model_type : CardiacModelType
        Type of cardiac model to use.
    tissue_size : Tuple[int, int]
        Spatial grid dimensions.
    dt : float
        Integration timestep.
    alpha : float
        Alpha parameter from training range.
    lambda_decay : float
        Lambda decay parameter from training range.

    Returns
    -------
    PrimalLogicKernelV4
        Configured Kernel v4 instance.
    """
    # Create cardiac model
    cardiac_model = CardiacTissueModel(
        model_type=model_type,
        tissue_size=tissue_size,
        dt=dt,
    )

    # Create parameters
    params = KernelV4Parameters(
        alpha=alpha,
        lambda_decay=lambda_decay,
    )

    # Create and return kernel
    kernel = PrimalLogicKernelV4(
        cardiac_model=cardiac_model,
        params=params,
    )

    return kernel


__all__ = [
    "KernelV4Parameters",
    "PrimalLogicKernelV4",
    "create_kernel_v4",
]
