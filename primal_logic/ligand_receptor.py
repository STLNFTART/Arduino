"""Ligand-Receptor Binding Subsystem.

This module implements cellular-level biochemical dynamics for ligand-receptor
interactions with feedback from macro-scale neural and cardiac fields.

Equations:
    Ṙ(t) = k_on * L(t) * (R_T - R(t)) - k_off * R(t) + γ * F(t)

Where:
    R(t): Receptor occupancy at time t
    L(t): Ligand concentration at time t
    k_on: Binding rate constant
    k_off: Unbinding rate constant
    R_T: Total receptor density
    F(t): Feedback signal from macro-scale (neural/cardiac) fields
    γ: Coupling coefficient translating physiological state into biochemical modulation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .constants import DT


@dataclass
class LigandReceptorState:
    """State variables for ligand-receptor binding system."""

    receptor_occupancy: float = 0.0  # R(t): current receptor occupancy
    ligand_concentration: float = 0.0  # L(t): current ligand concentration


@dataclass
class LigandReceptor:
    """Ligand-receptor binding dynamics with macro-scale feedback.

    This class models the binding and unbinding of ligands to receptors at the
    cellular level, with feedback from macro-scale physiological states such as
    stress hormones, cytokines, or autonomic signals.

    Parameters
    ----------
    k_on : float
        Binding rate constant (M^-1 s^-1). Default: 1.0
    k_off : float
        Unbinding rate constant (s^-1). Default: 0.5
    receptor_total : float
        Total receptor density (R_T). Default: 100.0
    gamma : float
        Coupling coefficient translating macro-scale feedback into biochemical
        modulation. Default: 0.1
    ligand_input : Optional[Callable[[float], float]]
        Function that provides ligand concentration L(t) as a function of time.
        If None, uses constant concentration. Default: None
    dt : float
        Integration timestep in seconds. Default: DT from constants
    """

    k_on: float = 1.0
    k_off: float = 0.5
    receptor_total: float = 100.0
    gamma: float = 0.1
    ligand_input: Optional[Callable[[float], float]] = None
    dt: float = DT

    state: LigandReceptorState = field(default_factory=LigandReceptorState)
    time: float = 0.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.k_on <= 0:
            raise ValueError("k_on must be positive")
        if self.k_off <= 0:
            raise ValueError("k_off must be positive")
        if self.receptor_total <= 0:
            raise ValueError("receptor_total must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")

    def get_ligand_concentration(self, t: float) -> float:
        """Get ligand concentration at time t.

        Parameters
        ----------
        t : float
            Current time in seconds.

        Returns
        -------
        float
            Ligand concentration L(t).
        """
        if self.ligand_input is not None:
            return self.ligand_input(t)
        # Default: constant baseline concentration
        return 1.0

    def step(self, feedback_signal: float) -> LigandReceptorState:
        """Advance the ligand-receptor system by one timestep.

        Parameters
        ----------
        feedback_signal : float
            Feedback signal F(t) from macro-scale neural or cardiac fields
            (e.g., stress hormones, cytokines, autonomic signals).

        Returns
        -------
        LigandReceptorState
            Updated state of the ligand-receptor system.
        """
        # Get current ligand concentration
        L_t = self.get_ligand_concentration(self.time)
        self.state.ligand_concentration = L_t

        # Current receptor occupancy
        R_t = self.state.receptor_occupancy

        # Compute binding dynamics:
        # Ṙ(t) = k_on * L(t) * (R_T - R(t)) - k_off * R(t) + γ * F(t)
        binding_term = self.k_on * L_t * (self.receptor_total - R_t)
        unbinding_term = self.k_off * R_t
        feedback_term = self.gamma * feedback_signal

        dR_dt = binding_term - unbinding_term + feedback_term

        # Forward Euler integration
        self.state.receptor_occupancy += dR_dt * self.dt

        # Ensure receptor occupancy stays within physical bounds [0, R_T]
        self.state.receptor_occupancy = max(
            0.0, min(self.receptor_total, self.state.receptor_occupancy)
        )

        self.time += self.dt
        return self.state

    def get_occupancy_fraction(self) -> float:
        """Get the fraction of receptors that are occupied.

        Returns
        -------
        float
            Receptor occupancy fraction R(t) / R_T in range [0, 1].
        """
        return self.state.receptor_occupancy / self.receptor_total

    def get_binding_signal(self) -> float:
        """Get a normalized binding signal for upward coupling.

        This signal can be used as input to the immune signaling layer.

        Returns
        -------
        float
            Normalized binding signal in range [0, 1].
        """
        # Use sigmoidal transformation for smooth upward propagation
        occupancy_fraction = self.get_occupancy_fraction()
        return math.tanh(occupancy_fraction * 2.0) * 0.5 + 0.5

    def reset(self) -> None:
        """Reset the model to initial conditions."""
        self.state = LigandReceptorState()
        self.time = 0.0


__all__ = ["LigandReceptor", "LigandReceptorState"]
