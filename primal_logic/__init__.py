"""Primal Logic robotic hand control framework (Python port).

This package implements a modular simulation of a quantum-inspired
control field driving a robotic hand model with tendon-like joints,
integrated with a multi-heart physiological model, Arduino hardware,
and MotorHandPro robotic actuator control.
"""

from .constants import VERSION
from .demo import run_demo
from .analysis import plot_rolling_average
from .rpo import RecursivePlanckOperator
from .sweeps import alpha_sweep, beta_sweep, tau_sweep, torque_sweep
from .heart_model import MultiHeartModel, HeartBrainState
from .heart_arduino_bridge import HeartArduinoBridge, ProcessorHeartArduinoLink

# Kernel v4: Quantum-Cardiac Integration (requires numpy)
try:
    from .cardiac_tissue_models import CardiacModelType, CardiacTissueModel
    from .quantum_plasma_field import QuantumAmplitudeState, PlasmaField
    from .primal_algorithms import (
        AlgorithmType,
        AlgorithmMetrics,
        PrimalAlgorithmSuite,
    )
    from .kernel_v4 import (
        KernelV4Parameters,
        PrimalLogicKernelV4,
        create_kernel_v4,
    )
    _KERNEL_V4_AVAILABLE = True
except ImportError:
    _KERNEL_V4_AVAILABLE = False
    # Create placeholder None values
    CardiacModelType = None
    CardiacTissueModel = None
    QuantumAmplitudeState = None
    PlasmaField = None
    AlgorithmType = None
    AlgorithmMetrics = None
    PrimalAlgorithmSuite = None
    KernelV4Parameters = None
    PrimalLogicKernelV4 = None
    create_kernel_v4 = None

# Optional: MotorHandPro integration (requires numpy)
try:
    from .motorhand_integration import (
        MotorHandProBridge,
        UnifiedPrimalLogicController,
        create_integrated_system,
    )
    _MOTORHAND_AVAILABLE = True
except ImportError:
    _MOTORHAND_AVAILABLE = False
    # Create placeholder None values
    MotorHandProBridge = None
    UnifiedPrimalLogicController = None
    create_integrated_system = None

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
    "MotorHandProBridge",
    "UnifiedPrimalLogicController",
    "create_integrated_system",
    # Kernel v4 exports
    "CardiacModelType",
    "CardiacTissueModel",
    "QuantumAmplitudeState",
    "PlasmaField",
    "AlgorithmType",
    "AlgorithmMetrics",
    "PrimalAlgorithmSuite",
    "KernelV4Parameters",
    "PrimalLogicKernelV4",
    "create_kernel_v4",
]
