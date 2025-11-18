#!/usr/bin/env python3
"""
Neurorobotic Hardware Integration Bridge
=========================================

Integrates the closed-loop neurorobotic control system with the existing
MotorHandPro hardware and Primal Logic framework.

Architecture:
    EEG Sensors → Neural Decoder → Sensor Fusion → Closed-Loop Control
                                         ↑                    ↓
    Hand Actuators ← MotorHandPro ← Primal Logic ← Control Commands

This bridge connects:
    - neurorobotic_control.py (brain-to-motion control)
    - motorhand_integration.py (hardware interface)
    - primal_logic framework (bio-hybrid actuation)
"""

import numpy as np
import time
import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import sys

# Import neurorobotic control components
from neurorobotic_control import (
    NeuroroboticControlSystem,
    NeuroSignal,
    SensorData,
    ControlState
)

# Import existing Primal Logic components
from primal_logic.motorhand_integration import (
    MotorHandProBridge,
    UnifiedPrimalLogicController
)
from primal_logic.hand import RoboticHand
from primal_logic.constants import DEFAULT_FINGERS, JOINTS_PER_FINGER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuroHardwareAdapter:
    """
    Adapter between neurorobotic control system and MotorHandPro hardware.

    Provides bidirectional translation:
    - Neural commands → MotorHandPro torques
    - MotorHandPro state → Sensor feedback for closed-loop control
    """

    def __init__(
        self,
        n_dofs: int = 15,
        position_to_torque_gain: float = 5.0,
        max_torque: float = 0.7  # N·m (MotorHandPro limit)
    ):
        self.n_dofs = n_dofs
        self.position_to_torque_gain = position_to_torque_gain
        self.max_torque = max_torque

        # State estimation
        self.previous_positions = np.ones(n_dofs) * 0.5
        self.previous_velocities = np.zeros(n_dofs)
        self.previous_time = time.time()

    def positions_to_torques(
        self,
        commanded_positions: np.ndarray,
        current_positions: np.ndarray
    ) -> np.ndarray:
        """
        Convert position commands to torque commands for MotorHandPro.

        Uses proportional control:
            τ = Kp · (θ_cmd - θ_current)

        Args:
            commanded_positions: Target positions [0, 1] (normalized)
            current_positions: Current positions [0, 1] (normalized)

        Returns:
            Torque commands (N·m) for MotorHandPro
        """
        # Position error
        position_error = commanded_positions - current_positions

        # Proportional torque
        torques = self.position_to_torque_gain * position_error

        # Clip to hardware limits
        torques = np.clip(torques, -self.max_torque, self.max_torque)

        return torques

    def torques_to_positions(
        self,
        torques: np.ndarray,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        Estimate positions from torque commands (forward model).

        Uses simplified actuator dynamics:
            θ̈ = τ / I - b·θ̇

        Args:
            torques: Applied torques (N·m)
            dt: Timestep (seconds)

        Returns:
            Estimated positions [0, 1]
        """
        # Simplified parameters
        inertia = 0.001  # kg·m²
        damping = 0.1  # N·m·s/rad

        # Acceleration
        accelerations = torques / inertia - damping * self.previous_velocities

        # Integrate to velocity
        velocities = self.previous_velocities + accelerations * dt

        # Integrate to position
        positions = self.previous_positions + velocities * dt

        # Clip to valid range
        positions = np.clip(positions, 0, 1)

        # Update state
        self.previous_positions = positions
        self.previous_velocities = velocities
        self.previous_time = time.time()

        return positions

    def create_sensor_data(
        self,
        motorhand_state: Dict[str, Any],
        dt: float = 0.01
    ) -> SensorData:
        """
        Create SensorData from MotorHandPro hardware state.

        Args:
            motorhand_state: State dictionary from MotorHandProBridge.get_state()
            dt: Timestep for numerical differentiation

        Returns:
            SensorData object for neurorobotic controller
        """
        # Get positions from torques (using forward model)
        torques = np.array(motorhand_state['torques'])
        positions = self.torques_to_positions(torques, dt)

        # Estimate velocities (numerical derivative)
        velocities = (positions - self.previous_positions) / dt

        # Estimate accelerations (numerical derivative)
        accelerations = (velocities - self.previous_velocities) / dt

        # Create sensor data
        sensor_data = SensorData(
            timestamp=time.time(),
            joint_positions=positions,
            joint_velocities=velocities,
            joint_accelerations=accelerations,
            tactile_forces=None,  # TODO: Add tactile sensors
            slip_detected=None  # TODO: Add slip detection
        )

        return sensor_data


class NeuroroboticHardwareSystem:
    """
    Complete neurorobotic system with hardware integration.

    Combines:
    - Neurorobotic control (brain signals → motor commands)
    - Primal Logic framework (bio-hybrid control theory)
    - MotorHandPro hardware (15-DOF robotic hand)

    Control pipeline:
        EEG → Neural Decoder → Sensor Fusion → Closed-Loop Controller
                                    ↑                     ↓
        MotorHandPro ← Primal Logic Hand ← Position-to-Torque ← Commands
              ↓
        Proprioceptive Feedback → Sensor Adapter → Sensor Fusion
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        eeg_port: Optional[str] = None,
        n_eeg_channels: int = 64,
        eeg_sampling_rate: float = 256.0,
        control_frequency: float = 100.0,  # Hz
        enable_safety: bool = True,
        use_heart: bool = True,
        use_rpo: bool = True,
        memory_mode: str = "recursive_planck"
    ):
        """
        Initialize complete neurorobotic hardware system.

        Args:
            port: Serial port for MotorHandPro hardware
            eeg_port: Serial port for EEG acquisition (optional)
            n_eeg_channels: Number of EEG channels
            eeg_sampling_rate: EEG sampling rate (Hz)
            control_frequency: Closed-loop control rate (Hz)
            enable_safety: Enable safety monitoring
            use_heart: Enable heart-brain coupling
            use_rpo: Enable Recursive Planck Operator
            memory_mode: Memory kernel mode
        """
        logger.info("Initializing Neurorobotic Hardware System...")

        self.n_dofs = DEFAULT_FINGERS * JOINTS_PER_FINGER
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency

        # 1. Create neurorobotic control system
        self.neuro_system = NeuroroboticControlSystem(
            n_eeg_channels=n_eeg_channels,
            eeg_sampling_rate=eeg_sampling_rate,
            n_dofs=self.n_dofs,
            control_frequency=control_frequency,
            enable_safety=enable_safety
        )

        # 2. Create MotorHandPro bridge
        self.motorhand_bridge = MotorHandProBridge(
            port=port,
            baud=115200,
            n_fingers=DEFAULT_FINGERS,
            n_joints_per_finger=JOINTS_PER_FINGER
        )

        # 3. Create Primal Logic hand model
        self.hand_model = RoboticHand(
            n_fingers=DEFAULT_FINGERS,
            memory_mode=memory_mode,
            use_serial=False  # Bridge handles serial
        )

        # 4. Create unified controller (Primal Logic + MotorHandPro)
        self.primal_controller = UnifiedPrimalLogicController(
            hand_model=self.hand_model,
            motorhand_bridge=self.motorhand_bridge,
            heart_model=None,  # Initialize later if use_heart=True
            rpo_processor=None  # Initialize later if use_rpo=True
        )

        # 5. Create hardware adapter
        self.hardware_adapter = NeuroHardwareAdapter(n_dofs=self.n_dofs)

        # 6. Optional: EEG acquisition
        self.eeg_port = eeg_port
        self.eeg_serial = None
        if eeg_port is not None:
            self._init_eeg_serial(eeg_port, eeg_sampling_rate)

        # 7. Optional: Heart-brain coupling
        if use_heart:
            self._init_heart_model()

        # 8. Optional: RPO processor
        if use_rpo:
            self._init_rpo_processor()

        # State
        self.is_running = False
        self.step_count = 0
        self.telemetry = []

        logger.info("Neurorobotic Hardware System initialized successfully!")

    def _init_eeg_serial(self, port: str, sampling_rate: float) -> None:
        """Initialize serial connection for EEG acquisition."""
        try:
            import serial
            self.eeg_serial = serial.Serial(port, baudrate=115200, timeout=0.001)
            logger.info(f"EEG serial connected on {port}")
        except ImportError:
            logger.warning("pyserial not available, EEG acquisition disabled")
        except Exception as e:
            logger.error(f"Failed to connect EEG serial: {e}")

    def _init_heart_model(self) -> None:
        """Initialize heart-brain coupling model."""
        try:
            from primal_logic.heart_model import MultiHeartModel
            self.primal_controller.heart = MultiHeartModel()
            logger.info("Heart-brain coupling enabled")
        except ImportError:
            logger.warning("Heart model not available")

    def _init_rpo_processor(self) -> None:
        """Initialize Recursive Planck Operator."""
        try:
            from primal_logic.rpo import RecursivePlanckOperator
            self.primal_controller.rpo = RecursivePlanckOperator()
            logger.info("Recursive Planck Operator enabled")
        except ImportError:
            logger.warning("RPO processor not available")

    def calibrate(
        self,
        calibration_duration: float = 10.0,
        resting_state: bool = True
    ) -> None:
        """
        Calibrate the neurorobotic system.

        Args:
            calibration_duration: Duration of calibration (seconds)
            resting_state: If True, calibrate with resting-state EEG
        """
        logger.info(f"Calibrating system (duration={calibration_duration}s)...")

        n_samples = int(calibration_duration * 256)  # Assume 256 Hz EEG
        eeg_data = []

        if self.eeg_serial is not None:
            # Acquire real EEG data
            logger.info("Acquiring EEG calibration data...")
            for _ in range(n_samples):
                eeg_sample = self._read_eeg_sample()
                if eeg_sample is not None:
                    eeg_data.append(eeg_sample)
                time.sleep(1.0 / 256)
        else:
            # Simulate EEG data
            logger.info("Simulating EEG calibration data...")
            eeg_data = np.random.randn(n_samples, 64) * 1e-5

        # Calibrate neural decoder
        eeg_array = np.array(eeg_data)
        self.neuro_system.calibrate(eeg_array)

        logger.info("Calibration complete!")

    def _read_eeg_sample(self) -> Optional[np.ndarray]:
        """Read one EEG sample from serial port."""
        if self.eeg_serial is None:
            return None

        try:
            # Read line from serial (CSV format expected)
            line = self.eeg_serial.readline().decode('ascii').strip()
            if line:
                values = [float(x) for x in line.split(',')]
                return np.array(values)
        except Exception as e:
            logger.warning(f"Failed to read EEG sample: {e}")

        return None

    def run_control_loop(
        self,
        duration: float,
        trajectory_fn: Optional[Callable[[float], np.ndarray]] = None,
        log_frequency: float = 1.0  # Log every N seconds
    ) -> None:
        """
        Run the complete closed-loop neurorobotic control system.

        Args:
            duration: Duration to run (seconds)
            trajectory_fn: Optional function(time) -> target_positions
            log_frequency: How often to log status (seconds)
        """
        logger.info(f"Starting control loop (duration={duration}s)...")

        n_steps = int(duration / self.dt)
        log_interval = int(log_frequency / self.dt)
        self.is_running = True

        for step in range(n_steps):
            if not self.is_running:
                logger.info("Control loop stopped by user")
                break

            current_time = step * self.dt

            # 1. Acquire EEG sample
            if self.eeg_serial is not None:
                eeg_sample = self._read_eeg_sample()
                if eeg_sample is None:
                    eeg_sample = np.random.randn(64) * 1e-5  # Fallback
            else:
                # Simulate EEG with optional trajectory influence
                if trajectory_fn is not None:
                    # Inject trajectory intent into simulated EEG
                    target = trajectory_fn(current_time)
                    eeg_sample = np.random.randn(64) * 1e-5 + target[:64] * 1e-6
                else:
                    eeg_sample = np.random.randn(64) * 1e-5

            # 2. Get current hardware state
            motorhand_state = self.motorhand_bridge.get_state()
            sensor_data = self.hardware_adapter.create_sensor_data(motorhand_state, self.dt)

            # 3. Process through neurorobotic control
            control_state = self.neuro_system.process_step(eeg_sample, sensor_data)

            if control_state is None:
                logger.warning(f"Step {step}: Safety violation, halting")
                break

            # 4. Convert positions to torques
            torques = self.hardware_adapter.positions_to_torques(
                control_state.commanded_positions,
                sensor_data.joint_positions
            )

            # 5. Send through Primal Logic pipeline
            # Update hand model with target positions (converted from commanded positions)
            target_angles = control_state.commanded_positions * np.pi  # [0,1] → [0, π]
            self.primal_controller.hand.step(target_angles, self.dt)

            # Send final torques to hardware
            self.motorhand_bridge.send_torques(torques)

            # 6. Update heart model (if enabled)
            if self.primal_controller.heart is not None:
                self.primal_controller._update_heart_from_control(torques)

            # 7. Log telemetry
            if step % log_interval == 0:
                self._log_status(step, current_time, control_state, motorhand_state)

            # Store telemetry
            self.telemetry.append({
                'step': step,
                'time': current_time,
                'control_state': control_state,
                'motorhand_state': motorhand_state,
                'torques': torques
            })

            self.step_count += 1

        logger.info("Control loop complete!")
        self._print_summary()

    def _log_status(
        self,
        step: int,
        time: float,
        control_state: ControlState,
        motorhand_state: Dict[str, Any]
    ) -> None:
        """Log current system status."""
        neuro_confidence = self.neuro_system.sensor_fusion.get_average_confidence()
        error_norm = np.linalg.norm(control_state.position_error)
        lipschitz = control_state.stability_metric

        logger.info(
            f"t={time:.2f}s | "
            f"Step={step} | "
            f"Error={error_norm:.4f} | "
            f"L_neuro={lipschitz:.4f} | "
            f"L_primal={motorhand_state['lipschitz_estimate']:.4f} | "
            f"Confidence={neuro_confidence:.2f} | "
            f"Stable={motorhand_state['stable']}"
        )

    def _print_summary(self) -> None:
        """Print summary statistics of control session."""
        print("\n" + "=" * 80)
        print("Neurorobotic Control Session Summary")
        print("=" * 80)

        # Neurorobotic stats
        neuro_summary = self.neuro_system.get_telemetry_summary()
        print("\n--- Neural Control ---")
        for key, value in neuro_summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # Hardware stats
        motorhand_state = self.motorhand_bridge.get_state()
        print("\n--- MotorHandPro Hardware ---")
        print(f"  Control Energy (Ec): {motorhand_state['control_energy']:.4f}")
        print(f"  Lipschitz Constant: {motorhand_state['lipschitz_estimate']:.6f}")
        print(f"  Stable: {motorhand_state['stable']}")
        print(f"  Lambda (Lightfoot): {motorhand_state['lambda']:.6f}")
        print(f"  Donte Constant: {motorhand_state['donte']:.4f}")

        # Overall stats
        print("\n--- Overall Performance ---")
        print(f"  Total Steps: {self.step_count}")
        print(f"  Total Time: {self.step_count * self.dt:.2f} s")
        print(f"  Average Cycle Time: {self.dt * 1000:.2f} ms")

        print("=" * 80 + "\n")

    def stop(self) -> None:
        """Stop the control loop."""
        self.is_running = False
        logger.info("Stop signal sent")

    def reset(self) -> None:
        """Reset the system state."""
        self.neuro_system.reset()
        self.primal_controller.step_count = 0
        self.step_count = 0
        self.telemetry = []
        logger.info("System reset")

    def save_telemetry(self, filepath: str) -> None:
        """Save telemetry data to file."""
        import json

        telemetry_serializable = []
        for entry in self.telemetry:
            serializable_entry = {
                'step': entry['step'],
                'time': entry['time'],
                'torques': entry['torques'].tolist(),
                'control_state': {
                    'commanded_positions': entry['control_state'].commanded_positions.tolist(),
                    'position_error': entry['control_state'].position_error.tolist(),
                    'stability_metric': entry['control_state'].stability_metric
                },
                'motorhand_state': entry['motorhand_state']
            }
            telemetry_serializable.append(serializable_entry)

        with open(filepath, 'w') as f:
            json.dump(telemetry_serializable, f, indent=2)

        logger.info(f"Telemetry saved to {filepath}")


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("Closed-Loop Neurorobotic Hardware Control System")
    print("Integrating Brain Signals, Bio-Hybrid Actuators, and Robotic Sensing")
    print("=" * 80)

    # Check if hardware is available
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--simulate':
        port = None  # Simulation mode
        eeg_port = None
        print("\n[SIMULATION MODE - No hardware required]\n")
    else:
        port = "/dev/ttyACM0"
        eeg_port = "/dev/ttyUSB0"
        print("\n[HARDWARE MODE]\n")

    # Create system
    print("Initializing system...")
    try:
        system = NeuroroboticHardwareSystem(
            port=port if port else "/dev/null",
            eeg_port=eeg_port,
            n_eeg_channels=64,
            eeg_sampling_rate=256.0,
            control_frequency=100.0,
            enable_safety=True,
            use_heart=True,
            use_rpo=True,
            memory_mode="recursive_planck"
        )
    except Exception as e:
        print(f"Failed to initialize hardware: {e}")
        print("Falling back to simulation mode...")
        system = NeuroroboticHardwareSystem(
            port="/dev/null",
            eeg_port=None,
            n_eeg_channels=64,
            eeg_sampling_rate=256.0,
            control_frequency=100.0,
            enable_safety=True,
            use_heart=False,
            use_rpo=False
        )

    # Calibrate
    print("\nCalibrating neural decoder...")
    system.calibrate(calibration_duration=5.0)

    # Define trajectory (grasping motion)
    def grasp_trajectory(t: float) -> np.ndarray:
        """Trajectory for closing hand grasp."""
        # Sigmoid function for smooth closing
        progress = 1.0 / (1.0 + np.exp(-2 * (t - 5)))
        return np.ones(15) * progress

    # Run control loop
    print("\nRunning closed-loop control...")
    print("  - Neural signals decoded from EEG")
    print("  - Sensor fusion with proprioceptive feedback")
    print("  - Closed-loop PID control")
    print("  - Bio-hybrid actuation via MotorHandPro")
    print("  - Real-time stability monitoring\n")

    try:
        system.run_control_loop(
            duration=10.0,
            trajectory_fn=grasp_trajectory,
            log_frequency=1.0
        )
    except KeyboardInterrupt:
        print("\nControl interrupted by user")
        system.stop()

    # Save telemetry
    output_path = "artifacts/neurorobotic_telemetry.json"
    Path("artifacts").mkdir(exist_ok=True)
    system.save_telemetry(output_path)

    print(f"\nTelemetry saved to: {output_path}")
    print("\nNeurorobotic hardware control demonstration complete!")
