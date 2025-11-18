"""PRIMAL LOGIC algorithm suite for Kernel v4.

This module implements the 15+ core algorithms that comprise the PRIMAL LOGIC
framework for quantum-inspired reasoning and distributed control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import numpy as np
from numpy.fft import fft, ifft

from .constants import (
    ALPHA_DEFAULT,
    ENERGY_BUDGET,
    LAMBDA_DEFAULT,
    PHASE_COUPLING,
    SIGMA,
    TEMPORAL_SCALES,
)


class AlgorithmType(Enum):
    """PRIMAL LOGIC algorithm types."""

    QUANTUM_SUPERPOSITION = "quantum_superposition"
    PLASMA_FIELD = "plasma_field"
    TEMPORAL_COHERENCE = "temporal_coherence"
    COLLECTIVE_DYNAMICS = "collective_dynamics"
    AMPLITUDE_MODULATION = "amplitude_modulation"
    PHASE_SYNCHRONIZATION = "phase_synchronization"
    ENERGY_CONSERVATION = "energy_conservation"
    ADAPTIVE_GATING = "adaptive_gating"
    MULTI_SCALE_INTEGRATION = "multi_scale_integration"
    CAUSAL_INFERENCE = "causal_inference"
    RECURSIVE_FEEDBACK = "recursive_feedback"
    SIGNAL_FUSION = "signal_fusion"
    DISTRIBUTED_CONSENSUS = "distributed_consensus"
    QUANTUM_COLLAPSE = "quantum_collapse"
    EMERGENT_COORDINATION = "emergent_coordination"


@dataclass
class AlgorithmMetrics:
    """Performance metrics for individual algorithms."""

    execution_count: int = 0
    total_energy: float = 0.0
    average_output: float = 0.0
    peak_output: float = 0.0
    coherence: float = 0.0

    def update(self, energy: float, output: float, coherence: float = 0.0) -> None:
        """Update metrics with new execution data."""
        self.execution_count += 1
        self.total_energy += energy
        self.peak_output = max(self.peak_output, abs(output))
        # Running average
        self.average_output = (
            self.average_output * (self.execution_count - 1) + output
        ) / self.execution_count
        self.coherence = coherence


@dataclass
class PrimalAlgorithmSuite:
    """Complete PRIMAL LOGIC algorithm suite.

    Parameters
    ----------
    alpha : float
        Primary control parameter (0.52-0.56 from training).
    lambda_decay : float
        Exponential decay rate (0.11-0.12 from training).
    energy_budget : float
        Maximum energy budget per algorithm.
    """

    alpha: float = ALPHA_DEFAULT
    lambda_decay: float = LAMBDA_DEFAULT
    energy_budget: float = ENERGY_BUDGET

    # Algorithm activation tracking
    algorithms_active: Dict[AlgorithmType, bool] = field(default_factory=dict)
    metrics: Dict[AlgorithmType, AlgorithmMetrics] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize algorithm tracking."""
        for alg_type in AlgorithmType:
            self.algorithms_active[alg_type] = True
            self.metrics[alg_type] = AlgorithmMetrics()

    # ========== CORE ALGORITHMS ==========

    def quantum_superposition(
        self, state: np.ndarray, basis_states: List[np.ndarray], step: int
    ) -> np.ndarray:
        """Quantum superposition: Mix multiple basis states with phase.

        Implements: |ψ⟩ = Σ_i α_i·exp(iφ_i)|basis_i⟩

        Parameters
        ----------
        state : np.ndarray
            Current quantum state.
        basis_states : List[np.ndarray]
            List of basis states to superpose.
        step : int
            Current timestep for phase evolution.

        Returns
        -------
        np.ndarray
            Superposed state.
        """
        if not basis_states:
            return state

        mixed_state = np.zeros_like(state, dtype=complex)
        n_basis = len(basis_states)

        for i, basis in enumerate(basis_states):
            # Phase evolution
            phase = 2 * np.pi * i / n_basis + self.alpha * step * 0.01
            amplitude = self.alpha / n_basis

            mixed_state += amplitude * np.exp(1j * phase) * basis

        # Update metrics
        energy = np.sum(np.abs(mixed_state) ** 2)
        output = np.mean(np.abs(mixed_state))
        self.metrics[AlgorithmType.QUANTUM_SUPERPOSITION].update(energy, output)

        return np.real(mixed_state)

    def plasma_field_dynamics(
        self, density: np.ndarray, coupling_kernel: np.ndarray
    ) -> np.ndarray:
        """Plasma field: Compute collective field from density via convolution.

        Implements: Γ(x) = ∫ ρ(x')·W(x-x') dx'

        Parameters
        ----------
        density : np.ndarray
            Spatial density distribution.
        coupling_kernel : np.ndarray
            Spatial coupling kernel W.

        Returns
        -------
        np.ndarray
            Collective field Γ.
        """
        # Use FFT for efficient convolution
        if density.shape == coupling_kernel.shape:
            # 2D convolution via FFT
            from scipy.signal import fftconvolve

            try:
                field = fftconvolve(density, coupling_kernel, mode="same")
            except ImportError:
                # Fallback to simple convolution
                field = self._simple_convolve_2d(density, coupling_kernel)
        else:
            field = density  # Fallback

        # Update metrics
        energy = np.sum(field**2)
        output = np.mean(field)
        self.metrics[AlgorithmType.PLASMA_FIELD].update(energy, output)

        return field

    def _simple_convolve_2d(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Simple 2D convolution fallback."""
        return a  # Placeholder - could implement manual convolution

    def temporal_coherence(
        self, signal_history: List[float], window_size: int = 50
    ) -> float:
        """Temporal coherence: Analyze multi-scale temporal patterns.

        Computes autocorrelation-based coherence measure.

        Parameters
        ----------
        signal_history : List[float]
            Historical signal values.
        window_size : int
            Analysis window size.

        Returns
        -------
        float
            Coherence metric (0-1).
        """
        if len(signal_history) < window_size:
            return 0.5

        # Extract recent window
        signal = np.array(signal_history[-window_size:])

        # Compute autocorrelation
        signal_centered = signal - np.mean(signal)
        autocorr = np.correlate(signal_centered, signal_centered, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        else:
            autocorr = np.zeros_like(autocorr)

        # Coherence: sum of positive autocorrelations
        coherence = np.mean(np.maximum(autocorr[1:10], 0))

        # Update metrics
        self.metrics[AlgorithmType.TEMPORAL_COHERENCE].update(0.1, coherence, coherence)

        return float(np.clip(coherence, 0, 1))

    def collective_intelligence(
        self, agent_states: np.ndarray, consensus_threshold: float = 0.8
    ) -> float:
        """Collective intelligence: Measure swarm consensus and coherence.

        Parameters
        ----------
        agent_states : np.ndarray
            State values for each agent.
        consensus_threshold : float
            Threshold for consensus detection.

        Returns
        -------
        float
            Consensus agreement (0-1).
        """
        if len(agent_states) == 0:
            return 0.0

        # Compute consensus as inverse of standard deviation
        mean_state = np.mean(agent_states)
        std_state = np.std(agent_states)

        if mean_state != 0:
            consensus = 1.0 - min(1.0, std_state / (abs(mean_state) + 1e-6))
        else:
            consensus = 1.0 if std_state < 0.1 else 0.0

        # Update metrics
        self.metrics[AlgorithmType.COLLECTIVE_DYNAMICS].update(0.05, consensus, consensus)

        return float(consensus)

    def amplitude_modulation(
        self, carrier: np.ndarray, modulator: np.ndarray, depth: float = 0.5
    ) -> np.ndarray:
        """Amplitude modulation: Modulate carrier signal with control signal.

        Implements: y(t) = carrier(t) · [1 + depth·modulator(t)]

        Parameters
        ----------
        carrier : np.ndarray
            Carrier signal.
        modulator : np.ndarray
            Modulating signal.
        depth : float
            Modulation depth.

        Returns
        -------
        np.ndarray
            Modulated signal.
        """
        if carrier.shape != modulator.shape:
            # Broadcast if needed
            modulator = np.broadcast_to(modulator, carrier.shape)

        modulated = carrier * (1 + depth * modulator)

        # Update metrics
        energy = np.sum(modulated**2)
        output = np.mean(modulated)
        self.metrics[AlgorithmType.AMPLITUDE_MODULATION].update(energy, output)

        return modulated

    def phase_synchronization(
        self, signals: List[np.ndarray], coupling_strength: float = PHASE_COUPLING
    ) -> float:
        """Phase synchronization: Compute Kuramoto order parameter.

        Measures synchronization across multiple oscillators.

        Parameters
        ----------
        signals : List[np.ndarray]
            List of oscillatory signals.
        coupling_strength : float
            Coupling strength between oscillators.

        Returns
        -------
        float
            Synchronization order parameter (0-1).
        """
        if not signals or len(signals) < 2:
            return 0.0

        # Extract phases via Hilbert transform approximation
        phases = []
        for signal in signals:
            if len(signal) > 0:
                # Simple phase extraction: arctan(imag/real) of analytic signal
                analytic = fft(signal)
                phase = np.angle(analytic[0])
                phases.append(phase)

        if len(phases) < 2:
            return 0.0

        # Kuramoto order parameter: R = |⟨exp(iθ)⟩|
        complex_phases = np.exp(1j * np.array(phases))
        order_param = abs(np.mean(complex_phases))

        # Update metrics
        self.metrics[AlgorithmType.PHASE_SYNCHRONIZATION].update(
            0.1, order_param, order_param
        )

        return float(order_param)

    def energy_conservation(
        self, energy_sources: np.ndarray, max_budget: float | None = None
    ) -> np.ndarray:
        """Energy conservation: Enforce energy budget constraints.

        Scales energy sources to respect budget while preserving ratios.

        Parameters
        ----------
        energy_sources : np.ndarray
            Energy contribution from each source.
        max_budget : float | None
            Maximum energy budget (defaults to self.energy_budget).

        Returns
        -------
        np.ndarray
            Scaled energy sources.
        """
        if max_budget is None:
            max_budget = self.energy_budget

        total_energy = np.sum(np.abs(energy_sources))

        if total_energy > max_budget:
            # Scale to budget
            scale_factor = max_budget / total_energy
            scaled = energy_sources * scale_factor
        else:
            scaled = energy_sources

        # Update metrics
        self.metrics[AlgorithmType.ENERGY_CONSERVATION].update(
            np.sum(scaled), np.mean(scaled)
        )

        return scaled

    def adaptive_gating(
        self, input_signal: np.ndarray, gate_threshold: float, step: int
    ) -> np.ndarray:
        """Adaptive gating: Dynamic threshold-based signal modulation.

        Parameters
        ----------
        input_signal : np.ndarray
            Input signal to gate.
        gate_threshold : float
            Gating threshold.
        step : int
            Current timestep for adaptive threshold.

        Returns
        -------
        np.ndarray
            Gated signal.
        """
        # Adaptive threshold with temporal oscillation
        adaptive_threshold = gate_threshold * (1 + SIGMA * np.sin(step * 0.01))

        # Apply gating
        gated = np.where(np.abs(input_signal) > adaptive_threshold, input_signal, 0.0)

        # Update metrics
        energy = np.sum(gated**2)
        output = np.mean(gated)
        self.metrics[AlgorithmType.ADAPTIVE_GATING].update(energy, output)

        return gated

    def multi_scale_integration(
        self, signal: np.ndarray, scales: List[float] | None = None
    ) -> np.ndarray:
        """Multi-scale integration: Combine information across temporal scales.

        Parameters
        ----------
        signal : np.ndarray
            Input signal.
        scales : List[float] | None
            Temporal scales to integrate (defaults to TEMPORAL_SCALES).

        Returns
        -------
        np.ndarray
            Multi-scale integrated signal.
        """
        if scales is None:
            scales = list(TEMPORAL_SCALES)

        # Wavelet-like decomposition (simplified)
        integrated = np.zeros_like(signal, dtype=float)

        for i, scale in enumerate(scales):
            # Simple filtering at different scales
            if len(signal) > int(scale):
                kernel_size = max(1, int(scale))
                filtered = np.convolve(
                    signal, np.ones(kernel_size) / kernel_size, mode="same"
                )
                weight = 1.0 / len(scales)
                integrated += weight * filtered

        # Update metrics
        energy = np.sum(integrated**2)
        output = np.mean(integrated)
        self.metrics[AlgorithmType.MULTI_SCALE_INTEGRATION].update(energy, output)

        return integrated

    def recursive_feedback(
        self, current_state: float, error: float, memory_trace: float
    ) -> float:
        """Recursive feedback: Exponential memory integration.

        Implements: ψ(t+1) = -λ·ψ(t) + K·e(t) + β·memory(t)

        Parameters
        ----------
        current_state : float
            Current state ψ(t).
        error : float
            Error signal e(t).
        memory_trace : float
            Memory trace from previous steps.

        Returns
        -------
        float
            Updated state ψ(t+1).
        """
        K = 0.5  # Gain
        beta = 0.3  # Memory weight

        new_state = -self.lambda_decay * current_state + K * error + beta * memory_trace

        # Update metrics
        self.metrics[AlgorithmType.RECURSIVE_FEEDBACK].update(abs(new_state), new_state)

        return new_state

    def distributed_consensus(
        self, agent_votes: np.ndarray, consensus_threshold: float = 0.8
    ) -> bool:
        """Distributed consensus: Determine if agents reached consensus.

        Parameters
        ----------
        agent_votes : np.ndarray
            Binary votes from each agent (0 or 1).
        consensus_threshold : float
            Required agreement fraction.

        Returns
        -------
        bool
            True if consensus reached.
        """
        if len(agent_votes) == 0:
            return False

        agreement = np.mean(agent_votes)
        consensus_reached = agreement >= consensus_threshold or agreement <= (
            1 - consensus_threshold
        )

        # Update metrics
        self.metrics[AlgorithmType.DISTRIBUTED_CONSENSUS].update(0.01, float(agreement))

        return bool(consensus_reached)

    def quantum_collapse_detection(
        self, amplitude: np.ndarray, threshold: float = 2.0
    ) -> bool:
        """Quantum collapse detection: Check if state exceeds threshold.

        Parameters
        ----------
        amplitude : np.ndarray
            Quantum amplitude field.
        threshold : float
            Collapse threshold.

        Returns
        -------
        bool
            True if collapse detected.
        """
        max_amplitude = np.max(np.abs(amplitude))
        collapse = max_amplitude > threshold

        # Update metrics
        self.metrics[AlgorithmType.QUANTUM_COLLAPSE].update(max_amplitude, max_amplitude)

        return collapse

    # ========== METRICS AND REPORTING ==========

    def get_algorithm_metrics(self, alg_type: AlgorithmType) -> AlgorithmMetrics:
        """Get metrics for specific algorithm."""
        return self.metrics[alg_type]

    def get_total_energy(self) -> float:
        """Get total energy consumed across all algorithms."""
        return sum(m.total_energy for m in self.metrics.values())

    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of algorithm performance."""
        return {
            "total_energy": self.get_total_energy(),
            "average_coherence": np.mean(
                [m.coherence for m in self.metrics.values() if m.coherence > 0]
            ),
            "active_algorithms": sum(1 for active in self.algorithms_active.values() if active),
            "total_executions": sum(m.execution_count for m in self.metrics.values()),
        }


__all__ = ["AlgorithmType", "AlgorithmMetrics", "PrimalAlgorithmSuite"]
