"""Primal Logic robotic hand control framework (Python port).

This package implements a modular simulation of a quantum-inspired
control field driving a robotic hand model with tendon-like joints.
"""

from .constants import VERSION
from .demo import run_demo
from .analysis import plot_rolling_average
from .inventory import generate_inventory_artifacts, gather_inventory
from .rpo import RecursivePlanckOperator
from .sweeps import alpha_sweep, beta_sweep, tau_sweep, torque_sweep

__all__ = [
    "VERSION",
    "run_demo",
    "plot_rolling_average",
    "torque_sweep",
    "alpha_sweep",
    "beta_sweep",
    "tau_sweep",
    "gather_inventory",
    "generate_inventory_artifacts",
    "RecursivePlanckOperator",
]
