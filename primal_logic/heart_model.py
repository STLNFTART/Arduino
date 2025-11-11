"""Multi-Heart Model implementation with heart-brain-immune coupling.

This module implements the Quantro-Primal physiological model described in
docs/quantitative_framework.md. The model couples cardiac actuation (n_h) with
brain neural potential (n_b) through Recursive Planck Operators.

Equations:
    n_h'(t) = −λ_h n_h + f_h(n_b, S_h) + ℛ_P[C(t)]
    n_b'(t) = −λ_b n_b + f_b(n_h, S_b) + ℛ_P[s_set(t)]

Where ℛ_P is the Recursive Planck Operator bridging energetic and informational
domains using Donte's and Lightfoot's constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from .constants import (
    DONTE_CONSTANT,
    DT,
    LAMBDA_DEFAULT,
    LIGHTFOOT_MAX,
    LIGHTFOOT_MIN,
)
from .rpo import RecursivePlanckOperator


@dataclass
class HeartBrainState:
    """State variables for the heart-brain coupled system."""

    n_heart: float = 0.0  # Heart neural potential
    n_brain: float = 0.0  # Brain neural potential
    s_heart: float = 0.0  # Heart sensory feedback
    s_brain: float = 0.0  # Brain sensory feedback


@dataclass
class CouplingParameters:
    """Frequency-dependent coupling parameters for heart-brain interaction.

    Parameters
    ----------
    neural_to_cardiac_gain : float
        Gain for neural (brain) to cardiac (heart) coupling.
        Models vagal/sympathetic drive strength.
    cardiac_to_neural_gain : float
        Gain for cardiac to neural coupling.
        Models baroreflex feedback strength.
    low_freq_weight : float
        Weight for low-frequency coupling (baroreflex dominates slow changes).
        Typically 0.6-0.8 for baroreflex dominance at ~0.04 Hz.
    high_freq_weight : float
        Weight for high-frequency coupling (RSA affects fast dynamics).
        Typically 0.2-0.4 for respiratory sinus arrhythmia at ~0.1-0.15 Hz.
    omega_rsa : float
        Angular frequency for respiratory sinus arrhythmia (radians/s).
        Default: 0.1 * 2π ≈ 0.628 rad/s (~0.1 Hz, typical respiration).
    omega_baro : float
        Angular frequency for baroreflex oscillations (radians/s).
        Default: 0.04 * 2π ≈ 0.251 rad/s (~0.04 Hz, blood pressure regulation).
    amp_rsa : float
        Amplitude of RSA forcing term (default 0.3).
    amp_baro : float
        Amplitude of baroreflex forcing term (default 0.2).
    """

    neural_to_cardiac_gain: float = 0.35  # Mid-range vagal/sympathetic blend
    cardiac_to_neural_gain: float = 0.25  # Baroreflex feedback strength
    low_freq_weight: float = 0.7          # Baroreflex dominates slow changes
    high_freq_weight: float = 0.3         # RSA affects fast dynamics
    omega_rsa: float = 0.1 * 2 * math.pi  # ~0.628 rad/s (respiratory freq)
    omega_baro: float = 0.04 * 2 * math.pi  # ~0.251 rad/s (baroreflex freq)
    amp_rsa: float = 0.3                  # RSA amplitude
    amp_baro: float = 0.2                 # Baroreflex amplitude


@dataclass
class MultiHeartModel:
    """Multi-heart model with heart-brain-immune coupling via RPO.

    Parameters
    ----------
    lambda_heart : float
        Decay rate for heart neural potential (λ_h).
    lambda_brain : float
        Decay rate for brain neural potential (λ_b).
    coupling_strength : float
        Strength of heart-brain coupling (default 0.1).
        DEPRECATED: Use coupling_params for frequency-dependent coupling.
    coupling_params : CouplingParameters
        Frequency-dependent coupling parameters.
    rpo_alpha : float
        RPO damping coefficient (must satisfy 0 < rpo_alpha * dt < 1).
    lightfoot : float
        Lightfoot's constant for neural-mechanical coupling.
    donte : float
        Donte's constant for energy-information scaling.
    dt : float
        Integration timestep in seconds.
    """

    lambda_heart: float = LAMBDA_DEFAULT
    lambda_brain: float = LAMBDA_DEFAULT * 0.8  # Brain decay slightly slower
    coupling_strength: float = 0.1  # Deprecated: use coupling_params instead
    coupling_params: CouplingParameters = field(default_factory=CouplingParameters)
    rpo_alpha: float = 0.4
    lightfoot: float = (LIGHTFOOT_MIN + LIGHTFOOT_MAX) / 2.0
    donte: float = DONTE_CONSTANT
    dt: float = DT

    state: HeartBrainState = field(default_factory=HeartBrainState)
    rpo_heart: RecursivePlanckOperator = field(init=False)
    rpo_brain: RecursivePlanckOperator = field(init=False)
    step_count: int = 0
    time: float = 0.0  # Current simulation time for frequency-dependent forcing

    def __post_init__(self) -> None:
        """Initialize the Recursive Planck Operators for heart and brain."""
        if not (0 < self.rpo_alpha * self.dt < 1):
            raise ValueError("rpo_alpha must satisfy 0 < rpo_alpha * dt < 1 for stability")

        # RPO for heart receives cardiac input C(t)
        self.rpo_heart = RecursivePlanckOperator(
            alpha=self.rpo_alpha,
            lam=self.lambda_heart,
            lightfoot=self.lightfoot,
            donte=self.donte,
            dt=self.dt,
            state=0.0,
        )

        # RPO for brain receives setpoint s_set(t)
        self.rpo_brain = RecursivePlanckOperator(
            alpha=self.rpo_alpha,
            lam=self.lambda_brain,
            lightfoot=self.lightfoot,
            donte=self.donte,
            dt=self.dt,
            state=0.0,
        )

    def forcing_term(self, t: float) -> float:
        """Dual-frequency physiological drive.

        Combines respiratory sinus arrhythmia (RSA) and baroreflex oscillations
        to model realistic heart-brain coupling dynamics.

        Parameters
        ----------
        t : float
            Current simulation time in seconds.

        Returns
        -------
        float
            Combined forcing from RSA and baroreflex components.
        """
        rsa_component = (
            self.coupling_params.amp_rsa * math.sin(self.coupling_params.omega_rsa * t)
        )
        baro_component = (
            self.coupling_params.amp_baro * math.sin(self.coupling_params.omega_baro * t)
        )
        return rsa_component + baro_component

    def frequency_dependent_gain(self, t: float, base_gain: float) -> float:
        """Compute frequency-dependent coupling gain.

        Weights the coupling based on the instantaneous frequency content,
        giving more weight to baroreflex at low frequencies and RSA at high frequencies.

        Parameters
        ----------
        t : float
            Current simulation time in seconds.
        base_gain : float
            Base coupling gain to modulate.

        Returns
        -------
        float
            Frequency-weighted coupling gain.
        """
        # Compute instantaneous phase derivatives to estimate frequency content
        # Low-freq component (baroreflex) varies slowly
        low_freq_phase = math.sin(self.coupling_params.omega_baro * t)
        # High-freq component (RSA) varies quickly
        high_freq_phase = math.sin(self.coupling_params.omega_rsa * t)

        # Weight by frequency components
        weighted_gain = base_gain * (
            self.coupling_params.low_freq_weight * abs(low_freq_phase)
            + self.coupling_params.high_freq_weight * abs(high_freq_phase)
        )
        return weighted_gain

    def f_heart(self, n_brain: float, s_heart: float) -> float:
        """Coupling function from brain to heart with frequency-dependent gain.

        Models afferent vagal and sympathetic drive influenced by brain state,
        weighted by frequency-dependent coupling (RSA + baroreflex).
        """
        gain = self.frequency_dependent_gain(
            self.time, self.coupling_params.neural_to_cardiac_gain
        )
        return gain * math.tanh(n_brain + s_heart)

    def f_brain(self, n_heart: float, s_brain: float) -> float:
        """Coupling function from heart to brain with frequency-dependent gain.

        Models baroreceptor and cardiac feedback influencing brain activity,
        weighted by frequency-dependent coupling (baroreflex dominant at low freq).
        """
        gain = self.frequency_dependent_gain(
            self.time, self.coupling_params.cardiac_to_neural_gain
        )
        return gain * math.tanh(n_heart + s_brain)

    def step(
        self,
        cardiac_input: float,
        brain_setpoint: float,
        theta: float = 1.0,
        use_forcing: bool = False,
    ) -> HeartBrainState:
        """Advance the heart-brain system by one timestep.

        Parameters
        ----------
        cardiac_input : float
            External cardiac input C(t) (e.g., from physical activity or stress).
        brain_setpoint : float
            Brain control setpoint s_set(t) (e.g., cognitive demand or intent).
        theta : float
            Command envelope Θ for the RPO operators (default 1.0).
        use_forcing : bool
            If True, add dual-frequency forcing term to cardiac dynamics (default False).

        Returns
        -------
        HeartBrainState
            Updated state of the heart-brain system.
        """
        # Compute RPO contributions
        rpo_heart_output = self.rpo_heart.step(
            theta=theta,
            input_value=cardiac_input,
            step_index=self.step_count,
        )

        rpo_brain_output = self.rpo_brain.step(
            theta=theta,
            input_value=brain_setpoint,
            step_index=self.step_count,
        )

        # Compute coupling terms with frequency-dependent gains
        f_h = self.f_heart(self.state.n_brain, self.state.s_heart)
        f_b = self.f_brain(self.state.n_heart, self.state.s_brain)

        # Optional dual-frequency forcing (RSA + baroreflex)
        forcing = self.forcing_term(self.time) if use_forcing else 0.0

        # Integrate using forward Euler
        # n_h'(t) = −λ_h n_h + f_h(n_b, S_h) + ℛ_P[C(t)] + forcing(t)
        dn_heart = (
            -self.lambda_heart * self.state.n_heart + f_h + rpo_heart_output + forcing
        )
        self.state.n_heart += dn_heart * self.dt

        # n_b'(t) = −λ_b n_b + f_b(n_h, S_b) + ℛ_P[s_set(t)]
        dn_brain = (
            -self.lambda_brain * self.state.n_brain + f_b + rpo_brain_output
        )
        self.state.n_brain += dn_brain * self.dt

        # Update sensory feedback (simple decay for now)
        self.state.s_heart = self.state.n_heart * 0.5
        self.state.s_brain = self.state.n_brain * 0.5

        # Increment time and step count
        self.time += self.dt
        self.step_count += 1
        return self.state

    def get_heart_rate(self) -> float:
        """Compute synthetic heart rate from heart neural potential.

        Returns a normalized heart rate value based on n_heart state.
        """
        # Map n_heart to a physiological range (e.g., 60-120 bpm normalized to 0-1)
        base_rate = 0.5  # Corresponds to ~90 bpm
        modulation = math.tanh(self.state.n_heart) * 0.3
        return max(0.0, min(1.0, base_rate + modulation))

    def get_brain_activity(self) -> float:
        """Compute synthetic brain activity from brain neural potential.

        Returns a normalized brain activity value based on n_brain state.
        """
        return math.tanh(self.state.n_brain)

    def get_cardiac_output(self) -> List[float]:
        """Get cardiac actuation signals for Arduino output.

        Returns
        -------
        List[float]
            A list of normalized cardiac signals that can be sent to Arduino
            for controlling actuators, LEDs, or other hardware.
        """
        heart_rate = self.get_heart_rate()
        brain_activity = self.get_brain_activity()

        # Generate multi-channel output (e.g., for 4 channels)
        # Channel 0: Heart rate
        # Channel 1: Brain activity
        # Channel 2: Heart-brain coherence
        # Channel 3: Combined signal
        coherence = abs(self.state.n_heart * self.state.n_brain)
        combined = (heart_rate + brain_activity) / 2.0

        return [
            heart_rate,
            brain_activity,
            math.tanh(coherence),
            combined,
        ]

    def reset(self) -> None:
        """Reset the model to initial conditions."""
        self.state = HeartBrainState()
        self.rpo_heart.reset()
        self.rpo_brain.reset()
        self.step_count = 0
        self.time = 0.0


__all__ = ["MultiHeartModel", "HeartBrainState", "CouplingParameters"]
