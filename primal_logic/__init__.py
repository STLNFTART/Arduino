"""Primal Logic robotic hand control framework (Python port).

This package implements a modular simulation of a quantum-inspired
control field driving a robotic hand model with tendon-like joints.
"""

from .constants import VERSION
from .demo import run_demo
from .analysis import plot_rolling_average
from .sweeps import torque_sweep

__all__ = ["VERSION", "run_demo", "plot_rolling_average", "torque_sweep"]
