"""
MotorHandPro Actuator with RPO Burn Meter Integration

This module provides a wrapper around MotorHandProBridge that integrates
with the RPOBurnMeter to track token burns for PrimalRWA integration.

Each second of MotorHandPro actuation triggers 1 RPO token burn.

Patent Pending: U.S. Provisional Patent Application No. 63/842,846
Copyright 2025 Donte Lightfoot - The Phoney Express LLC / Locked In Safety
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

# Ensure billing module is accessible
sys.path.insert(0, str(Path(__file__).parent.parent))

from primal_logic.motorhand_integration import MotorHandProBridge
from billing.rpo_burn_meter import RPOBurnMeter


@dataclass
class MotorHandProActuator:
    """
    MotorHandPro actuator with integrated RPO token burn tracking.

    This class wraps MotorHandProBridge and adds burn meter integration
    for PrimalRWA token burns. Each second of actuation burns 1 RPO token.

    Features:
    - Automatic burn tracking (1 token = 1 second of actuation)
    - Hardware control via MotorHandProBridge
    - Optional planck_mode for enabling/disabling burns
    - Seamless integration with existing burn meter infrastructure

    Attributes:
        port: Serial port for MotorHandPro hardware
        dt: Timestep in seconds (default: 0.01 = 10ms)
        burn_meter: Optional RPOBurnMeter for tracking token burns
        planck_mode: Enable/disable burn tracking (default: False)
        burn_meter_key: Key in actuator address map (default: "motorhand_pro_actuator")
        bridge: MotorHandProBridge instance for hardware communication
    """

    port: str = "/dev/ttyACM0"
    baud: int = 115200
    dt: float = 0.01
    burn_meter: Optional[RPOBurnMeter] = None
    planck_mode: bool = False
    burn_meter_key: str = "motorhand_pro_actuator"
    lambda_value: float = 0.16905
    ke_gain: float = 0.3

    def __post_init__(self):
        """Initialize MotorHandPro bridge after dataclass init."""
        # Create MotorHandPro bridge (simulation mode - bridge will be None)
        # Only create bridge if we'll actually connect to hardware
        self.bridge = None  # Lazy initialization on connect()

        # Store bridge parameters for later
        self._bridge_params = {
            "port": self.port,
            "baud": self.baud,
            "lambda_value": self.lambda_value,
            "ke_gain": self.ke_gain,
        }

        # Track connection state
        self.is_connected = False

        # Cumulative runtime for burn tracking
        self.cumulative_runtime = 0.0

        # Simulated state for when no hardware present
        self._sim_state = {
            "torques": np.zeros(15),
            "control_energy": 0.0,
            "lipschitz_estimate": 0.000130,
            "stable": True,
        }

    def connect(self) -> bool:
        """
        Connect to MotorHandPro hardware.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Create bridge on first connect attempt
            if self.bridge is None:
                self.bridge = MotorHandProBridge(**self._bridge_params)

            self.is_connected = self.bridge.connect()
            return self.is_connected
        except Exception as e:
            print(f"Warning: Failed to connect to hardware: {e}")
            print("Running in simulation mode (burn tracking still active)")
            self.bridge = None
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from MotorHandPro hardware."""
        if self.bridge is not None:
            self.bridge.disconnect()
        self.is_connected = False

    def send_torques(self, torques: np.ndarray) -> bool:
        """
        Send torque commands to MotorHandPro and track burn time.

        This is the primary entry point for actuation. Each call:
        1. Sends torques to hardware via bridge (or simulates if no hardware)
        2. Records runtime in burn meter (if planck_mode=True)
        3. Burns tokens at 1 RPO per second

        Args:
            torques: Array of torque commands (15 values for 5 fingers × 3 joints)

        Returns:
            True if send successful, False otherwise
        """
        # Send to hardware if connected, otherwise simulate
        if self.bridge is not None and self.is_connected:
            success = self.bridge.send_torques(torques)
        else:
            # Simulation mode - just update simulated state
            self._sim_state["torques"] = torques.copy() if isinstance(torques, np.ndarray) else np.array(torques)
            self._sim_state["control_energy"] += np.sum(torques ** 2) * self.dt
            success = True

        if success:
            # Track runtime
            self.cumulative_runtime += self.dt

            # Record burn (if enabled)
            if self.planck_mode and self.burn_meter is not None:
                self.burn_meter.record(self.burn_meter_key, self.dt)

        return success

    def step(self, torques: np.ndarray) -> bool:
        """
        Execute one timestep of actuation.

        Alias for send_torques() to maintain consistency with other actuators.

        Args:
            torques: Array of torque commands

        Returns:
            True if step successful, False otherwise
        """
        return self.send_torques(torques)

    def get_state(self) -> Dict[str, Any]:
        """
        Get complete actuator state including burn tracking.

        Returns:
            Dictionary containing:
                - motorhand: MotorHandPro bridge state (or simulated state)
                - cumulative_runtime: Total actuation time (seconds)
                - planck_mode: Burn tracking enabled/disabled
                - is_connected: Hardware connection status
                - burn_meter_key: Actuator identifier for burns
        """
        # Get motorhand state (from bridge or simulation)
        if self.bridge is not None and self.is_connected:
            motorhand_state = self.bridge.get_state()
        else:
            # Return simulated state
            motorhand_state = self._sim_state.copy()

        state = {
            "motorhand": motorhand_state,
            "cumulative_runtime": self.cumulative_runtime,
            "planck_mode": self.planck_mode,
            "is_connected": self.is_connected,
            "burn_meter_key": self.burn_meter_key,
        }

        # Add burn report if meter available
        if self.burn_meter is not None:
            burn_report = self.burn_meter.get_burn_report()
            state["burned_seconds"] = burn_report.get(self.burn_meter_key, 0)

        return state

    def set_parameters(
        self,
        lambda_value: Optional[float] = None,
        ke_gain: Optional[float] = None,
    ):
        """
        Update Primal Logic control parameters.

        Args:
            lambda_value: Lightfoot constant (0.01 - 1.0 s⁻¹)
            ke_gain: Error gain (0.0 - 1.0)
        """
        self.bridge.set_parameters(lambda_value=lambda_value, ke_gain=ke_gain)

        if lambda_value is not None:
            self.lambda_value = lambda_value
        if ke_gain is not None:
            self.ke_gain = ke_gain


def create_motorhand_actuator(
    port: str = "/dev/ttyACM0",
    burn_meter: Optional[RPOBurnMeter] = None,
    planck_mode: bool = True,
    auto_connect: bool = False,
) -> MotorHandProActuator:
    """
    Factory function to create MotorHandPro actuator with burn tracking.

    Args:
        port: Serial port for MotorHandPro hardware
        burn_meter: RPOBurnMeter instance for token burn tracking
        planck_mode: Enable burn tracking (default: True for PrimalRWA)
        auto_connect: Automatically connect to hardware (default: False)

    Returns:
        MotorHandProActuator instance ready for use

    Example:
        >>> from billing.rpo_burn_meter import RPOBurnMeter
        >>> from primal_logic.motorhand_actuator import create_motorhand_actuator
        >>>
        >>> # Load burn meter
        >>> burn_meter = RPOBurnMeter.from_config_files(
        ...     operator_config_path=Path("billing/rpo_operator_config.json"),
        ...     actuator_map_path=Path("billing/rpo_actuator_addresses.json"),
        ...     mode="hedera_testnet"
        ... )
        >>>
        >>> # Create actuator
        >>> actuator = create_motorhand_actuator(
        ...     port="/dev/ttyACM0",
        ...     burn_meter=burn_meter,
        ...     planck_mode=True,
        ...     auto_connect=True
        ... )
        >>>
        >>> # Use actuator
        >>> torques = np.zeros(15)
        >>> actuator.step(torques)
    """
    actuator = MotorHandProActuator(
        port=port,
        burn_meter=burn_meter,
        planck_mode=planck_mode,
    )

    if auto_connect:
        if not actuator.connect():
            print(f"Warning: Failed to connect to MotorHandPro at {port}")

    return actuator
