"""Primal Logic robotic hand control framework (Python port).

This package implements a modular simulation of a quantum-inspired
control field driving a robotic hand model with tendon-like joints,
integrated with a multi-heart physiological model and Arduino hardware.
"""

from .constants import VERSION
from .demo import run_demo
from .analysis import plot_rolling_average
from .rpo import RecursivePlanckOperator
from .sweeps import alpha_sweep, beta_sweep, tau_sweep, torque_sweep
from .heart_model import MultiHeartModel, HeartBrainState
from .heart_arduino_bridge import HeartArduinoBridge, ProcessorHeartArduinoLink

__all__ = [
    "VERSION",
    "run_demo",
    "plot_rolling_average",
    "torque_sweep",
    "alpha_sweep",
    "beta_sweep",
    "tau_sweep",
    "RecursivePlanckOperator",
    "MultiHeartModel",
    "HeartBrainState",
    "HeartArduinoBridge",
    "ProcessorHeartArduinoLink",
]
