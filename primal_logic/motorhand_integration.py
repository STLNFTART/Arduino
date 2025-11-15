"""
MotorHandPro Integration Bridge

Connects the primal_logic robotic hand framework with the MotorHandPro
hardware control system. Provides bidirectional communication for:
- Torque commands from hand model to MotorHandPro actuators
- State feedback from MotorHandPro to control system
- Unified Primal Logic parameter synchronization

Patent Pending: U.S. Provisional Patent Application No. 63/842,846
Copyright 2025 Donte Lightfoot - The Phoney Express LLC / Locked In Safety
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

# Add MotorHandPro to path for integration imports
MOTORHAND_PATH = Path(__file__).parent.parent / "external" / "MotorHandPro"
if MOTORHAND_PATH.exists():
    sys.path.insert(0, str(MOTORHAND_PATH))

from primal_logic.serial_bridge import SerialHandBridge
from primal_logic.constants import (
    DONTE_CONSTANT,
    LIGHTFOOT_MIN,
    LIGHTFOOT_MAX,
    DEFAULT_FINGERS,
    JOINTS_PER_FINGER,
)


class MotorHandProBridge:
    """
    Integration bridge between primal_logic framework and MotorHandPro hardware.

    Features:
    - Torque command streaming to MotorHandPro actuators
    - State feedback from hardware sensors
    - Primal Logic constant synchronization (Donte, Lightfoot)
    - Exponential memory weighting coordination
    - Real-time control panel integration via WebSocket

    Attributes:
        serial_bridge: Low-level serial communication bridge
        n_actuators: Total number of actuators (fingers × joints)
        lambda_value: Lightfoot constant (exponential decay rate)
        donte_value: Donte constant (fixed-point attractor)
        ke_gain: Proportional error gain
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baud: int = 115200,
        n_fingers: int = DEFAULT_FINGERS,
        n_joints_per_finger: int = JOINTS_PER_FINGER,
        lambda_value: float = 0.16905,  # Lightfoot constant
        ke_gain: float = 0.3,
    ):
        """
        Initialize MotorHandPro integration bridge.

        Args:
            port: Serial port for MotorHandPro hardware
            baud: Baud rate (must be 115200 for MotorHandPro)
            n_fingers: Number of fingers on robotic hand
            n_joints_per_finger: Joints per finger (3 DOF)
            lambda_value: Lightfoot constant (0.01 - 1.0 s⁻¹)
            ke_gain: Error gain (0.0 - 1.0)
        """
        self.serial_bridge = SerialHandBridge(port=port, baud=baud)
        self.n_actuators = n_fingers * n_joints_per_finger
        self.lambda_value = lambda_value
        self.donte_value = DONTE_CONSTANT
        self.ke_gain = ke_gain

        # State tracking
        self.current_torques = np.zeros(self.n_actuators)
        self.current_angles = np.zeros(self.n_actuators)
        self.control_energy = 0.0

        # MotorHandPro configuration
        self.motorhand_config = self._load_motorhand_config()

        # WebSocket connection for control panel (optional)
        self.websocket_client = None

    def _load_motorhand_config(self) -> Dict[str, Any]:
        """Load MotorHandPro actuator profile and configuration."""
        config_path = MOTORHAND_PATH / "actuator_profile.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            "actuator_type": "servo",
            "max_torque": 0.7,  # N·m
            "max_velocity": 8.0,  # rad/s
            "response_time": 0.01,  # s
        }

    def connect(self) -> bool:
        """
        Establish connection to MotorHandPro hardware.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.serial_bridge.connect()
            return True
        except Exception as e:
            print(f"Failed to connect to MotorHandPro: {e}")
            return False

    def disconnect(self):
        """Disconnect from MotorHandPro hardware."""
        self.serial_bridge.disconnect()

    def send_torques(self, torques: np.ndarray) -> bool:
        """
        Send torque commands to MotorHandPro actuators.

        Implements Primal Logic control law:
            dψ/dt = -λ·ψ(t) + KE·e(t)

        where torques represent the control command ψ(t).

        Args:
            torques: Array of torque values (N·m) for each actuator
                    Shape: (n_actuators,) or (n_fingers, n_joints_per_finger)

        Returns:
            True if send successful, False otherwise
        """
        if torques.size != self.n_actuators:
            print(f"Warning: Expected {self.n_actuators} torques, got {torques.size}")
            return False

        # Flatten if 2D array
        torques_flat = torques.flatten()

        # Apply exponential memory weighting (Primal Logic)
        # Decay previous torques by λ factor
        decay_factor = np.exp(-self.lambda_value * 0.01)  # Assume 0.01s timestep
        weighted_torques = decay_factor * self.current_torques + (1 - decay_factor) * torques_flat

        # Clip to hardware limits
        max_torque = self.motorhand_config.get("max_torque", 0.7)
        clipped_torques = np.clip(weighted_torques, -max_torque, max_torque)

        # Send via serial bridge
        success = self.serial_bridge.send_torques(clipped_torques)

        if success:
            self.current_torques = clipped_torques
            self._update_control_energy(clipped_torques)

        return success

    def _update_control_energy(self, torques: np.ndarray):
        """
        Update Primal Logic control energy functional.

        Ec(t) = ∫₀^t ψ(τ)·γ(τ) dτ

        This serves as a Lyapunov-like stability metric ensuring bounded convergence.
        """
        # Simplified: assume γ ≈ torque derivative
        gamma = np.sum(np.abs(torques - self.current_torques))
        psi = np.sum(np.abs(torques))
        self.control_energy += psi * gamma * 0.01  # dt = 0.01s

    def get_state(self) -> Dict[str, Any]:
        """
        Get current state of MotorHandPro system.

        Returns:
            Dictionary containing:
                - torques: Current torque commands
                - angles: Joint angles (if feedback available)
                - control_energy: Integrated control energy Ec(t)
                - lambda: Lightfoot constant
                - donte: Donte constant
                - lipschitz_estimate: Estimated Lipschitz constant
        """
        # Calculate Lipschitz estimate
        # F'(D) ≈ c·μ·exp(-μ·D) where c = (150-D)·exp(μ·D)
        c = (150 - self.donte_value) * np.exp(self.lambda_value * self.donte_value)
        lipschitz = c * self.lambda_value * np.exp(-self.lambda_value * self.donte_value)

        return {
            "torques": self.current_torques.tolist(),
            "angles": self.current_angles.tolist(),
            "control_energy": self.control_energy,
            "lambda": self.lambda_value,
            "donte": self.donte_value,
            "ke_gain": self.ke_gain,
            "lipschitz_estimate": lipschitz,
            "stable": lipschitz < 1.0,
        }

    def set_parameters(
        self,
        lambda_value: Optional[float] = None,
        ke_gain: Optional[float] = None
    ):
        """
        Update Primal Logic control parameters.

        Args:
            lambda_value: Lightfoot constant (0.01 - 1.0 s⁻¹)
            ke_gain: Error gain (0.0 - 1.0)
        """
        if lambda_value is not None:
            if 0.01 <= lambda_value <= 1.0:
                self.lambda_value = lambda_value
            else:
                print(f"Warning: lambda must be in [0.01, 1.0], got {lambda_value}")

        if ke_gain is not None:
            if 0.0 <= ke_gain <= 1.0:
                self.ke_gain = ke_gain
            else:
                print(f"Warning: ke_gain must be in [0.0, 1.0], got {ke_gain}")

    def sync_with_control_panel(self, websocket_url: str = "ws://localhost:8765"):
        """
        Connect to MotorHandPro control panel WebSocket for real-time monitoring.

        Args:
            websocket_url: WebSocket URL for control panel
        """
        try:
            import asyncio
            import websockets

            async def connect_websocket():
                async with websockets.connect(websocket_url) as ws:
                    self.websocket_client = ws
                    print(f"Connected to control panel at {websocket_url}")

                    # Send initial state
                    await ws.send(json.dumps({
                        "type": "initialization",
                        "data": self.get_state()
                    }))

            asyncio.run(connect_websocket())
        except ImportError:
            print("websockets library not available. Install with: pip install websockets")
        except Exception as e:
            print(f"Failed to connect to control panel: {e}")

    async def send_state_update(self):
        """Send state update to control panel via WebSocket."""
        if self.websocket_client:
            try:
                await self.websocket_client.send(json.dumps({
                    "type": "visualization_update",
                    "data": {
                        "source": "primal_logic_hand",
                        "primal_logic_analysis": self.get_state()
                    }
                }))
            except Exception as e:
                print(f"Failed to send state update: {e}")


class UnifiedPrimalLogicController:
    """
    Unified controller integrating primal_logic hand model with MotorHandPro hardware.

    Orchestrates:
    - Hand simulation with Recursive Planck Operator (RPO)
    - MotorHandPro hardware actuation
    - Heart-brain-immune physiological coupling
    - Real-time visualization and control

    This provides the complete pipeline:
        Field → Hand → RPO → Heart → MotorHandPro Hardware
    """

    def __init__(
        self,
        hand_model,
        motorhand_bridge: MotorHandProBridge,
        heart_model=None,
        rpo_processor=None,
    ):
        """
        Initialize unified controller.

        Args:
            hand_model: RoboticHand instance from primal_logic.hand
            motorhand_bridge: MotorHandProBridge for hardware communication
            heart_model: Optional MultiHeartModel for physiological coupling
            rpo_processor: Optional RecursivePlanckOperator for microprocessor layer
        """
        self.hand = hand_model
        self.motorhand = motorhand_bridge
        self.heart = heart_model
        self.rpo = rpo_processor

        self.step_count = 0
        self.dt = 0.01  # 10ms timestep

    def step(self, target_angles: Optional[np.ndarray] = None):
        """
        Execute one control timestep through complete pipeline.

        Args:
            target_angles: Optional target joint angles for grasp trajectory
        """
        # 1. Update hand simulation
        if target_angles is not None:
            self.hand.step(target_angles, self.dt)
        else:
            self.hand.step(dt=self.dt)

        # 2. Get torques from hand controllers
        torques = self.hand.get_torques()

        # 3. Process through RPO microprocessor (if available)
        if self.rpo is not None:
            processed_torques = self._apply_rpo(torques)
        else:
            processed_torques = torques

        # 4. Update heart model with control state (if available)
        if self.heart is not None:
            self._update_heart_from_control(processed_torques)

        # 5. Send to MotorHandPro hardware
        self.motorhand.send_torques(processed_torques)

        self.step_count += 1

    def _apply_rpo(self, torques: np.ndarray) -> np.ndarray:
        """
        Process torques through Recursive Planck Operator.

        Bridges energetic and informational domains using Donte's constant.
        """
        # Get RPO processing for control energy
        control_energy = np.sum(torques ** 2)
        processed = self.rpo.process(control_energy, dt=self.dt)

        # Scale torques by RPO modulation
        modulation = processed / (control_energy + 1e-6)
        return torques * np.sqrt(modulation)

    def _update_heart_from_control(self, torques: np.ndarray):
        """
        Update heart model based on control activity.

        Implements heart-brain coupling where motor control influences
        cardiac dynamics through autonomic nervous system.
        """
        # Map control effort to sympathetic activation
        control_effort = np.sqrt(np.sum(torques ** 2))
        sympathetic_drive = np.clip(control_effort / 0.7, 0.0, 1.0)  # Normalize to max torque

        # Update heart with external drive
        if hasattr(self.heart, 'set_external_drive'):
            self.heart.set_external_drive(sympathetic_drive)

    def get_full_state(self) -> Dict[str, Any]:
        """
        Get complete system state across all layers.

        Returns:
            Unified state dictionary with:
                - hand: Hand model state (angles, velocities, torques)
                - motorhand: MotorHandPro hardware state
                - heart: Physiological state (if available)
                - rpo: Microprocessor state (if available)
        """
        state = {
            "step": self.step_count,
            "time": self.step_count * self.dt,
            "hand": {
                "angles": self.hand.get_angles().tolist(),
                "torques": self.hand.get_torques().tolist(),
            },
            "motorhand": self.motorhand.get_state(),
        }

        if self.heart is not None:
            state["heart"] = {
                "heart_rate": self.heart.get_heart_rate(),
                "brain_activity": self.heart.state.n_b if hasattr(self.heart, 'state') else 0.0,
            }

        if self.rpo is not None:
            state["rpo"] = {
                "theta": self.rpo.theta if hasattr(self.rpo, 'theta') else 0.0,
            }

        return state

    def run(self, duration: float, trajectory=None):
        """
        Run unified control system for specified duration.

        Args:
            duration: Simulation duration in seconds
            trajectory: Optional trajectory generator for target angles
        """
        n_steps = int(duration / self.dt)

        for i in range(n_steps):
            if trajectory is not None:
                target = trajectory.get_target(i * self.dt)
            else:
                target = None

            self.step(target)

            # Print status every second
            if i % 100 == 0:
                state = self.get_full_state()
                print(f"t={state['time']:.2f}s | "
                      f"Ec={state['motorhand']['control_energy']:.4f} | "
                      f"L={state['motorhand']['lipschitz_estimate']:.6f} | "
                      f"Stable={state['motorhand']['stable']}")


def create_integrated_system(
    port: str = "/dev/ttyACM0",
    use_heart: bool = True,
    use_rpo: bool = True,
    memory_mode: str = "recursive_planck",
) -> UnifiedPrimalLogicController:
    """
    Factory function to create complete integrated system.

    Args:
        port: Serial port for MotorHandPro hardware
        use_heart: Enable heart-brain physiological coupling
        use_rpo: Enable Recursive Planck Operator microprocessor
        memory_mode: Memory kernel mode ("exponential" or "recursive_planck")

    Returns:
        UnifiedPrimalLogicController ready for execution
    """
    from primal_logic.hand import RoboticHand

    # Create hand model
    hand = RoboticHand(
        n_fingers=DEFAULT_FINGERS,
        memory_mode=memory_mode,
        use_serial=False,  # MotorHandPro bridge handles serial
    )

    # Create MotorHandPro bridge
    motorhand = MotorHandProBridge(port=port)

    # Optional: Create heart model
    heart = None
    if use_heart:
        try:
            from primal_logic.heart_model import MultiHeartModel
            heart = MultiHeartModel()
        except ImportError:
            print("Heart model not available")

    # Optional: Create RPO processor
    rpo = None
    if use_rpo:
        try:
            from primal_logic.rpo import RecursivePlanckOperator
            rpo = RecursivePlanckOperator()
        except ImportError:
            print("RPO processor not available")

    # Create unified controller
    controller = UnifiedPrimalLogicController(
        hand_model=hand,
        motorhand_bridge=motorhand,
        heart_model=heart,
        rpo_processor=rpo,
    )

    return controller
