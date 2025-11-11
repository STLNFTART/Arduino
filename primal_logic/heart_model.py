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
    coupling_strength: float = 0.1
    rpo_alpha: float = 0.4
    lightfoot: float = (LIGHTFOOT_MIN + LIGHTFOOT_MAX) / 2.0
    donte: float = DONTE_CONSTANT
    dt: float = DT

    state: HeartBrainState = field(default_factory=HeartBrainState)
    rpo_heart: RecursivePlanckOperator = field(init=False)
    rpo_brain: RecursivePlanckOperator = field(init=False)
    step_count: int = 0

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

    def f_heart(self, n_brain: float, s_heart: float) -> float:
        """Coupling function from brain to heart.

        Models afferent vagal and sympathetic drive influenced by brain state.
        """
        return self.coupling_strength * math.tanh(n_brain + s_heart)

    def f_brain(self, n_heart: float, s_brain: float) -> float:
        """Coupling function from heart to brain.

        Models baroreceptor and cardiac feedback influencing brain activity.
        """
        return self.coupling_strength * math.tanh(n_heart + s_brain)

    def step(
        self,
        cardiac_input: float,
        brain_setpoint: float,
        theta: float = 1.0,
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

        # Compute coupling terms
        f_h = self.f_heart(self.state.n_brain, self.state.s_heart)
        f_b = self.f_brain(self.state.n_heart, self.state.s_brain)

        # Integrate using forward Euler
        # n_h'(t) = −λ_h n_h + f_h(n_b, S_h) + ℛ_P[C(t)]
        dn_heart = (
            -self.lambda_heart * self.state.n_heart + f_h + rpo_heart_output
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


__all__ = ["MultiHeartModel", "HeartBrainState"]
