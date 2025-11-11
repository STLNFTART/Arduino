"""Multiscale Coupling Integration Layer.

This module integrates the ligand-receptor binding subsystem, immune signaling,
and the existing heart-brain model into a unified multiscale framework:

    Ligand-Receptor → Immune State → Neural + Cardiac Feedback

Each layer passes averaged or filtered variables upward using the hierarchical
coupling defined in the Quantro-Primal framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .constants import DT, LAMBDA_DEFAULT
from .heart_model import HeartBrainState, MultiHeartModel
from .immune_signaling import ImmuneSignaling
from .ligand_receptor import LigandReceptor


@dataclass
class MultiscaleState:
    """Complete state of the multiscale coupled system."""

    ligand_receptor: LigandReceptor
    immune: ImmuneSignaling
    heart_brain: MultiHeartModel

    # Cached base decay rates (stored before modulation)
    lambda_brain_base: float
    lambda_heart_base: float


@dataclass
class MultiscaleCoupling:
    """Integrated multiscale coupling system.

    This class orchestrates the complete hierarchy:
    1. Ligand-receptor binding at cellular level
    2. Immune signaling accumulation
    3. Heart-brain neural-cardiac coupling
    4. Bidirectional feedback between scales

    Parameters
    ----------
    ligand_receptor : Optional[LigandReceptor]
        Ligand-receptor binding subsystem. If None, creates default instance.
    immune_signaling : Optional[ImmuneSignaling]
        Immune signaling subsystem. If None, creates default instance.
    heart_brain_model : Optional[MultiHeartModel]
        Heart-brain coupling model. If None, creates default instance.
    feedback_strength : float
        Strength of downward feedback from macro-scale to cellular level.
        Default: 0.1
    dt : float
        Integration timestep in seconds. Default: DT from constants
    """

    ligand_receptor: Optional[LigandReceptor] = None
    immune_signaling: Optional[ImmuneSignaling] = None
    heart_brain_model: Optional[MultiHeartModel] = None
    feedback_strength: float = 0.1
    dt: float = DT

    state: MultiscaleState = field(init=False)

    def __post_init__(self) -> None:
        """Initialize all subsystems and create the multiscale state."""
        # Create default instances if not provided
        if self.ligand_receptor is None:
            self.ligand_receptor = LigandReceptor(dt=self.dt)

        if self.immune_signaling is None:
            self.immune_signaling = ImmuneSignaling(dt=self.dt)

        if self.heart_brain_model is None:
            self.heart_brain_model = MultiHeartModel(dt=self.dt)

        # Store base decay rates before any modulation
        lambda_brain_base = self.heart_brain_model.lambda_brain
        lambda_heart_base = self.heart_brain_model.lambda_heart

        # Create the complete state
        self.state = MultiscaleState(
            ligand_receptor=self.ligand_receptor,
            immune=self.immune_signaling,
            heart_brain=self.heart_brain_model,
            lambda_brain_base=lambda_brain_base,
            lambda_heart_base=lambda_heart_base,
        )

    def compute_macro_feedback(self) -> float:
        """Compute feedback signal from macro-scale to cellular level.

        This downward feedback F(t) represents stress hormones, cytokines, or
        autonomic signals that modulate ligand-receptor binding.

        Returns
        -------
        float
            Feedback signal F(t) for ligand-receptor subsystem.
        """
        # Combine heart and brain states with immune intensity
        heart_state = self.state.heart_brain.state.n_heart
        brain_state = self.state.heart_brain.state.n_brain
        immune_intensity = self.state.immune.state.intensity

        # Weighted combination representing systemic stress
        macro_signal = (
            0.4 * heart_state +
            0.4 * brain_state +
            0.2 * immune_intensity
        ) * self.feedback_strength

        return macro_signal

    def step(
        self,
        cardiac_input: float = 0.0,
        brain_setpoint: float = 0.0,
        theta: float = 1.0,
    ) -> MultiscaleState:
        """Advance the complete multiscale system by one timestep.

        The integration follows this sequence:
        1. Compute macro-to-micro feedback (downward)
        2. Update ligand-receptor binding
        3. Update immune signaling (upward from cellular)
        4. Modulate heart-brain decay rates
        5. Update heart-brain system with modulated parameters

        Parameters
        ----------
        cardiac_input : float
            External cardiac input C(t). Default: 0.0
        brain_setpoint : float
            Brain control setpoint s_set(t). Default: 0.0
        theta : float
            Command envelope for RPO operators. Default: 1.0

        Returns
        -------
        MultiscaleState
            Updated state of the complete multiscale system.
        """
        # Step 1: Compute downward feedback from macro to cellular level
        macro_feedback = self.compute_macro_feedback()

        # Step 2: Update ligand-receptor binding with macro feedback
        self.state.ligand_receptor.step(feedback_signal=macro_feedback)

        # Step 3: Update immune signaling driven by receptor occupancy
        receptor_signal = self.state.ligand_receptor.get_binding_signal()
        self.state.immune.step(receptor_signal=receptor_signal)

        # Step 4: Modulate neural and cardiac decay rates based on immune state
        lambda_brain_modulated = self.state.immune.modulate_brain_decay(
            self.state.lambda_brain_base
        )
        lambda_heart_modulated = self.state.immune.modulate_heart_decay(
            self.state.lambda_heart_base
        )

        # Apply modulated decay rates to heart-brain model
        self.state.heart_brain.lambda_brain = lambda_brain_modulated
        self.state.heart_brain.lambda_heart = lambda_heart_modulated

        # Step 5: Update heart-brain system
        self.state.heart_brain.step(
            cardiac_input=cardiac_input,
            brain_setpoint=brain_setpoint,
            theta=theta,
        )

        return self.state

    def get_complete_state(self) -> dict:
        """Get a complete snapshot of all state variables.

        Returns
        -------
        dict
            Dictionary containing all state variables across scales:
            - Cellular: receptor_occupancy, ligand_concentration
            - Immune: immune_intensity, inflammation_level
            - Cardiac: n_heart, heart_rate
            - Neural: n_brain, brain_activity
            - Decay rates: lambda_brain, lambda_heart
        """
        return {
            # Cellular level
            "receptor_occupancy": self.state.ligand_receptor.state.receptor_occupancy,
            "ligand_concentration": self.state.ligand_receptor.state.ligand_concentration,
            "occupancy_fraction": self.state.ligand_receptor.get_occupancy_fraction(),
            # Immune level
            "immune_intensity": self.state.immune.state.intensity,
            "inflammation_level": self.state.immune.get_inflammation_level(),
            # Cardiac level
            "n_heart": self.state.heart_brain.state.n_heart,
            "heart_rate": self.state.heart_brain.get_heart_rate(),
            # Neural level
            "n_brain": self.state.heart_brain.state.n_brain,
            "brain_activity": self.state.heart_brain.get_brain_activity(),
            # Modulated parameters
            "lambda_brain": self.state.heart_brain.lambda_brain,
            "lambda_heart": self.state.heart_brain.lambda_heart,
            "lambda_brain_base": self.state.lambda_brain_base,
            "lambda_heart_base": self.state.lambda_heart_base,
        }

    def get_arduino_output(self) -> List[float]:
        """Get signals for Arduino output including immune state.

        Returns
        -------
        List[float]
            Extended cardiac output with immune signals:
            [heart_rate, brain_activity, coherence, combined, immune_intensity]
        """
        cardiac_output = self.state.heart_brain.get_cardiac_output()
        immune_intensity = self.state.immune.get_inflammation_level()
        return cardiac_output + [immune_intensity]

    def reset(self) -> None:
        """Reset all subsystems to initial conditions."""
        self.state.ligand_receptor.reset()
        self.state.immune.reset()
        self.state.heart_brain.reset()

        # Restore base decay rates
        self.state.heart_brain.lambda_brain = self.state.lambda_brain_base
        self.state.heart_brain.lambda_heart = self.state.lambda_heart_base


__all__ = ["MultiscaleCoupling", "MultiscaleState"]
