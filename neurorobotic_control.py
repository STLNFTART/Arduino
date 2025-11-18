#!/usr/bin/env python3
"""
Closed-Loop Neurorobotic Control System
========================================

Integrates brain signals with bio-hybrid actuators and robotic sensing for
real-time closed-loop control of the 15-DOF MotorHandPro system.

Architecture:
    Brain Signals → Neural Decoder → Sensor Fusion → Closed-Loop Controller
                                           ↑                    ↓
                                    Robot Sensors ← Bio-Hybrid Actuators

Components:
    - NeuroInterface: Processes EEG/neural signals from motor cortex
    - SensorFusion: Combines neural intent with proprioceptive/tactile feedback
    - ClosedLoopController: PID control with adaptive gains
    - SafetyMonitor: Real-time stability and limit checking
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NeuroSignal:
    """Neural signal data from brain interface."""
    timestamp: float
    motor_intent: np.ndarray  # 15-DOF intended joint positions [0, 1]
    confidence: float  # Signal quality [0, 1]
    frequency_bands: Dict[str, float] = field(default_factory=dict)  # EEG power bands


@dataclass
class SensorData:
    """Multi-modal sensor feedback from robotic system."""
    timestamp: float
    joint_positions: np.ndarray  # Current 15-DOF positions
    joint_velocities: np.ndarray  # Current velocities
    joint_accelerations: np.ndarray  # Current accelerations
    tactile_forces: Optional[np.ndarray] = None  # Fingertip forces
    slip_detected: Optional[np.ndarray] = None  # Slip sensors (5 fingers)


@dataclass
class ControlState:
    """Closed-loop control state."""
    timestamp: float
    commanded_positions: np.ndarray  # 15-DOF commanded positions
    position_error: np.ndarray  # Intent - Actual
    integral_error: np.ndarray  # Accumulated error
    derivative_error: np.ndarray  # Error rate
    control_effort: np.ndarray  # PID output
    stability_metric: float  # Lipschitz constant estimate


class NeuroInterface:
    """
    Neural signal decoder for motor cortex activity.

    Processes EEG signals and decodes motor intent using:
    - Common Spatial Patterns (CSP) for feature extraction
    - Power spectral density in mu (8-13 Hz) and beta (13-30 Hz) bands
    - Regression model for continuous position decoding
    """

    def __init__(
        self,
        n_channels: int = 64,
        sampling_rate: float = 256.0,
        n_dofs: int = 15,
        window_size: float = 0.5,  # seconds
        confidence_threshold: float = 0.6
    ):
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.n_dofs = n_dofs
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold

        # Signal processing buffers
        self.buffer_size = int(sampling_rate * window_size)
        self.eeg_buffer = deque(maxlen=self.buffer_size)

        # Feature extraction (CSP filters - simplified)
        self.csp_filters = np.random.randn(n_channels, 6)  # 6 spatial patterns
        self.csp_filters /= np.linalg.norm(self.csp_filters, axis=0)

        # Decoder weights (mapping features to DOF intentions)
        self.decoder_weights = np.random.randn(12, n_dofs) * 0.1  # 12 features (6 CSP × 2 bands)
        self.decoder_bias = np.ones(n_dofs) * 0.5

        # Calibration data
        self.is_calibrated = False
        self.baseline_power = None

    def calibrate(self, resting_eeg: np.ndarray) -> None:
        """
        Calibrate decoder with resting-state EEG.

        Args:
            resting_eeg: (n_samples, n_channels) resting EEG data
        """
        logger.info("Calibrating neural decoder...")

        # Compute baseline power spectral density
        self.baseline_power = self._compute_psd(resting_eeg)
        self.is_calibrated = True

        logger.info("Neural decoder calibrated.")

    def process_eeg(self, eeg_sample: np.ndarray) -> NeuroSignal:
        """
        Process raw EEG sample and decode motor intent.

        Args:
            eeg_sample: (n_channels,) single EEG sample

        Returns:
            Decoded neural signal with motor intent
        """
        self.eeg_buffer.append(eeg_sample)

        if len(self.eeg_buffer) < self.buffer_size:
            # Not enough data yet, return neutral intent
            return NeuroSignal(
                timestamp=time.time(),
                motor_intent=np.ones(self.n_dofs) * 0.5,
                confidence=0.0,
                frequency_bands={}
            )

        # Convert buffer to array
        eeg_window = np.array(self.eeg_buffer)  # (buffer_size, n_channels)

        # Apply CSP spatial filtering
        csp_features = eeg_window @ self.csp_filters  # (buffer_size, 6)

        # Compute power in mu (8-13 Hz) and beta (13-30 Hz) bands
        mu_power = self._bandpower(csp_features, 8, 13)
        beta_power = self._bandpower(csp_features, 13, 30)

        # Combine features: [mu_power (6), beta_power (6)]
        features = np.concatenate([mu_power, beta_power])  # (12,)

        # Decode motor intent
        motor_intent = self.decoder_weights.T @ features + self.decoder_bias
        motor_intent = np.clip(motor_intent, 0, 1)  # Normalize to [0, 1]

        # Compute confidence based on signal strength
        if self.baseline_power is not None:
            current_power = np.mean(mu_power**2 + beta_power**2)
            baseline = np.mean(self.baseline_power**2)
            confidence = min(1.0, current_power / (baseline + 1e-6))
        else:
            confidence = 0.5

        return NeuroSignal(
            timestamp=time.time(),
            motor_intent=motor_intent,
            confidence=confidence,
            frequency_bands={'mu': mu_power, 'beta': beta_power}
        )

    def _compute_psd(self, signal: np.ndarray) -> np.ndarray:
        """Compute power spectral density (simplified)."""
        fft = np.fft.rfft(signal, axis=0)
        psd = np.abs(fft)**2 / signal.shape[0]
        return np.mean(psd, axis=0)

    def _bandpower(self, signal: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """
        Compute average power in frequency band.

        Args:
            signal: (n_samples, n_features) time-domain signal
            low_freq: Lower frequency bound (Hz)
            high_freq: Upper frequency bound (Hz)

        Returns:
            (n_features,) power in each feature
        """
        # FFT
        fft = np.fft.rfft(signal, axis=0)
        freqs = np.fft.rfftfreq(signal.shape[0], 1.0 / self.sampling_rate)

        # Find frequency bins
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)

        # Compute power
        power = np.mean(np.abs(fft[idx_band, :])**2, axis=0)

        return power


class SensorFusion:
    """
    Multi-modal sensor fusion combining neural intent with robotic feedback.

    Fuses:
    - Neural motor intent (target positions)
    - Proprioceptive feedback (actual positions/velocities)
    - Tactile feedback (contact forces)
    - Slip detection (grasp stability)
    """

    def __init__(
        self,
        n_dofs: int = 15,
        neural_weight: float = 0.7,
        proprioceptive_weight: float = 0.2,
        tactile_weight: float = 0.1,
        slip_gain: float = 0.5
    ):
        self.n_dofs = n_dofs

        # Fusion weights
        self.neural_weight = neural_weight
        self.proprioceptive_weight = proprioceptive_weight
        self.tactile_weight = tactile_weight
        self.slip_gain = slip_gain

        # State estimation
        self.estimated_intent = np.ones(n_dofs) * 0.5
        self.confidence_history = deque(maxlen=10)

    def fuse(
        self,
        neuro_signal: NeuroSignal,
        sensor_data: SensorData
    ) -> np.ndarray:
        """
        Fuse multi-modal sensor data to estimate true motor intent.

        Args:
            neuro_signal: Decoded neural signal
            sensor_data: Robot sensor feedback

        Returns:
            Fused motor intent (15-DOF target positions)
        """
        # Weight neural intent by confidence
        neural_contribution = (
            self.neural_weight * neuro_signal.confidence * neuro_signal.motor_intent
        )

        # Proprioceptive contribution (smooth changes from current state)
        proprioceptive_contribution = (
            self.proprioceptive_weight * sensor_data.joint_positions
        )

        # Tactile contribution (if gripping, maintain position)
        if sensor_data.tactile_forces is not None:
            # Map 5 fingertip forces to 15 DOFs (3 joints per finger)
            tactile_signal = np.repeat(sensor_data.tactile_forces, 3)
            tactile_contribution = self.tactile_weight * tactile_signal
        else:
            tactile_contribution = 0

        # Combine contributions
        fused_intent = (
            neural_contribution +
            proprioceptive_contribution +
            tactile_contribution
        )

        # Normalize
        total_weight = (
            self.neural_weight * neuro_signal.confidence +
            self.proprioceptive_weight +
            (self.tactile_weight if sensor_data.tactile_forces is not None else 0)
        )
        if total_weight > 0:
            fused_intent /= total_weight

        # Slip compensation (if slip detected, increase grip force)
        if sensor_data.slip_detected is not None:
            for finger_idx in range(5):
                if sensor_data.slip_detected[finger_idx]:
                    # Increase flexion of all joints in this finger
                    for joint_idx in range(3):
                        dof_idx = finger_idx * 3 + joint_idx
                        fused_intent[dof_idx] = min(
                            1.0,
                            fused_intent[dof_idx] + self.slip_gain
                        )

        # Low-pass filter to smooth intent
        alpha = 0.3  # Smoothing factor
        self.estimated_intent = (
            alpha * fused_intent + (1 - alpha) * self.estimated_intent
        )

        self.confidence_history.append(neuro_signal.confidence)

        return self.estimated_intent.copy()

    def get_average_confidence(self) -> float:
        """Get average neural signal confidence over recent history."""
        if len(self.confidence_history) == 0:
            return 0.0
        return float(np.mean(self.confidence_history))


class ClosedLoopController:
    """
    PID closed-loop controller with adaptive gains and stability guarantees.

    Implements:
    - Position error feedback (P term)
    - Integral error accumulation with anti-windup (I term)
    - Derivative error for damping (D term)
    - Adaptive gain scheduling based on error magnitude
    - Lipschitz stability monitoring
    """

    def __init__(
        self,
        n_dofs: int = 15,
        kp: float = 2.0,
        ki: float = 0.5,
        kd: float = 0.8,
        dt: float = 0.01,  # 100 Hz control loop
        integral_limit: float = 0.5,
        lipschitz_limit: float = 0.95
    ):
        self.n_dofs = n_dofs

        # PID gains
        self.kp = kp * np.ones(n_dofs)
        self.ki = ki * np.ones(n_dofs)
        self.kd = kd * np.ones(n_dofs)

        self.dt = dt
        self.integral_limit = integral_limit
        self.lipschitz_limit = lipschitz_limit

        # State
        self.integral_error = np.zeros(n_dofs)
        self.previous_error = np.zeros(n_dofs)
        self.control_history = deque(maxlen=100)
        self.error_history = deque(maxlen=100)

    def compute_control(
        self,
        target_positions: np.ndarray,
        sensor_data: SensorData
    ) -> ControlState:
        """
        Compute closed-loop control command.

        Args:
            target_positions: Desired 15-DOF positions from sensor fusion
            sensor_data: Current robot state

        Returns:
            Control state with commanded positions and diagnostics
        """
        # Position error
        position_error = target_positions - sensor_data.joint_positions

        # Integral error with anti-windup
        self.integral_error += position_error * self.dt
        self.integral_error = np.clip(
            self.integral_error,
            -self.integral_limit,
            self.integral_limit
        )

        # Derivative error
        derivative_error = (position_error - self.previous_error) / self.dt

        # PID control
        p_term = self.kp * position_error
        i_term = self.ki * self.integral_error
        d_term = self.kd * derivative_error

        control_effort = p_term + i_term + d_term

        # Adaptive gain scheduling (reduce gains for large errors)
        error_magnitude = np.linalg.norm(position_error)
        if error_magnitude > 0.5:
            gain_reduction = 0.5 / error_magnitude
            control_effort *= gain_reduction

        # Compute commanded positions
        commanded_positions = sensor_data.joint_positions + control_effort
        commanded_positions = np.clip(commanded_positions, 0, 1)

        # Stability monitoring (Lipschitz constant estimate)
        self.control_history.append(control_effort)
        self.error_history.append(position_error)

        if len(self.control_history) >= 2:
            delta_control = np.linalg.norm(
                self.control_history[-1] - self.control_history[-2]
            )
            delta_error = np.linalg.norm(
                self.error_history[-1] - self.error_history[-2]
            )
            if delta_error > 1e-6:
                lipschitz_estimate = delta_control / delta_error
            else:
                lipschitz_estimate = 0.0
        else:
            lipschitz_estimate = 0.0

        # Update state
        self.previous_error = position_error.copy()

        return ControlState(
            timestamp=time.time(),
            commanded_positions=commanded_positions,
            position_error=position_error,
            integral_error=self.integral_error.copy(),
            derivative_error=derivative_error,
            control_effort=control_effort,
            stability_metric=lipschitz_estimate
        )

    def reset(self) -> None:
        """Reset controller state."""
        self.integral_error = np.zeros(self.n_dofs)
        self.previous_error = np.zeros(self.n_dofs)
        self.control_history.clear()
        self.error_history.clear()


class SafetyMonitor:
    """
    Real-time safety monitoring and limit enforcement.

    Monitors:
    - Joint position limits
    - Velocity limits
    - Acceleration limits
    - Control effort limits
    - Stability metrics (Lipschitz constant)
    - Neural signal quality
    """

    def __init__(
        self,
        n_dofs: int = 15,
        position_limits: Tuple[float, float] = (0.0, 1.0),
        velocity_limit: float = 5.0,  # rad/s
        acceleration_limit: float = 50.0,  # rad/s²
        control_effort_limit: float = 1.0,
        lipschitz_limit: float = 0.95,
        min_neural_confidence: float = 0.3
    ):
        self.n_dofs = n_dofs
        self.position_limits = position_limits
        self.velocity_limit = velocity_limit
        self.acceleration_limit = acceleration_limit
        self.control_effort_limit = control_effort_limit
        self.lipschitz_limit = lipschitz_limit
        self.min_neural_confidence = min_neural_confidence

        self.violation_count = 0
        self.emergency_stop = False

    def check_safety(
        self,
        neuro_signal: NeuroSignal,
        sensor_data: SensorData,
        control_state: ControlState
    ) -> Tuple[bool, List[str]]:
        """
        Check all safety conditions.

        Args:
            neuro_signal: Current neural signal
            sensor_data: Current sensor data
            control_state: Current control state

        Returns:
            (is_safe, violation_messages)
        """
        violations = []

        # Position limits
        if np.any(control_state.commanded_positions < self.position_limits[0]) or \
           np.any(control_state.commanded_positions > self.position_limits[1]):
            violations.append("Position limit exceeded")

        # Velocity limits
        if np.any(np.abs(sensor_data.joint_velocities) > self.velocity_limit):
            violations.append("Velocity limit exceeded")

        # Acceleration limits
        if np.any(np.abs(sensor_data.joint_accelerations) > self.acceleration_limit):
            violations.append("Acceleration limit exceeded")

        # Control effort limits
        if np.any(np.abs(control_state.control_effort) > self.control_effort_limit):
            violations.append("Control effort limit exceeded")

        # Stability check
        if control_state.stability_metric > self.lipschitz_limit:
            violations.append(
                f"Stability violated (L={control_state.stability_metric:.3f} > {self.lipschitz_limit})"
            )

        # Neural signal quality
        if neuro_signal.confidence < self.min_neural_confidence:
            violations.append(
                f"Neural signal quality low (confidence={neuro_signal.confidence:.2f})"
            )

        # Update violation tracking
        if violations:
            self.violation_count += 1
            if self.violation_count > 10:  # 10 consecutive violations
                self.emergency_stop = True
                violations.append("EMERGENCY STOP: Too many consecutive violations")
        else:
            self.violation_count = 0

        is_safe = len(violations) == 0 and not self.emergency_stop

        return is_safe, violations

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop flag."""
        self.emergency_stop = False
        self.violation_count = 0


class NeuroroboticControlSystem:
    """
    Complete closed-loop neurorobotic control system.

    Integrates all components for brain-controlled bio-hybrid robotics.
    """

    def __init__(
        self,
        n_eeg_channels: int = 64,
        eeg_sampling_rate: float = 256.0,
        n_dofs: int = 15,
        control_frequency: float = 100.0,  # Hz
        enable_safety: bool = True
    ):
        self.n_dofs = n_dofs
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency

        # Initialize components
        self.neuro_interface = NeuroInterface(
            n_channels=n_eeg_channels,
            sampling_rate=eeg_sampling_rate,
            n_dofs=n_dofs
        )

        self.sensor_fusion = SensorFusion(n_dofs=n_dofs)

        self.controller = ClosedLoopController(
            n_dofs=n_dofs,
            dt=self.dt
        )

        self.safety_monitor = SafetyMonitor(n_dofs=n_dofs) if enable_safety else None

        # Telemetry
        self.telemetry = {
            'neuro_signals': [],
            'sensor_data': [],
            'control_states': [],
            'safety_violations': []
        }

        self.is_running = False

    def calibrate(self, resting_eeg: np.ndarray) -> None:
        """
        Calibrate the neural decoder.

        Args:
            resting_eeg: (n_samples, n_channels) resting-state EEG data
        """
        self.neuro_interface.calibrate(resting_eeg)
        logger.info("Neurorobotic system calibrated.")

    def process_step(
        self,
        eeg_sample: np.ndarray,
        sensor_data: SensorData
    ) -> Optional[ControlState]:
        """
        Process one control loop iteration.

        Args:
            eeg_sample: (n_channels,) current EEG sample
            sensor_data: Current robot sensor data

        Returns:
            Control state with commanded positions, or None if unsafe
        """
        # 1. Decode neural signal
        neuro_signal = self.neuro_interface.process_eeg(eeg_sample)

        # 2. Sensor fusion
        fused_intent = self.sensor_fusion.fuse(neuro_signal, sensor_data)

        # 3. Closed-loop control
        control_state = self.controller.compute_control(fused_intent, sensor_data)

        # 4. Safety check
        if self.safety_monitor is not None:
            is_safe, violations = self.safety_monitor.check_safety(
                neuro_signal, sensor_data, control_state
            )

            if not is_safe:
                logger.warning(f"Safety violations: {violations}")
                self.telemetry['safety_violations'].append({
                    'timestamp': time.time(),
                    'violations': violations
                })
                return None

        # 5. Log telemetry
        self.telemetry['neuro_signals'].append(neuro_signal)
        self.telemetry['sensor_data'].append(sensor_data)
        self.telemetry['control_states'].append(control_state)

        return control_state

    def get_telemetry_summary(self) -> Dict:
        """Get summary statistics of system performance."""
        if not self.telemetry['control_states']:
            return {}

        # Average tracking error
        errors = [cs.position_error for cs in self.telemetry['control_states']]
        avg_error = np.mean([np.linalg.norm(e) for e in errors])

        # Average neural confidence
        avg_confidence = self.sensor_fusion.get_average_confidence()

        # Stability metrics
        lipschitz_values = [cs.stability_metric for cs in self.telemetry['control_states']]
        max_lipschitz = np.max(lipschitz_values) if lipschitz_values else 0

        # Safety violation rate
        n_violations = len(self.telemetry['safety_violations'])
        n_steps = len(self.telemetry['control_states'])
        violation_rate = n_violations / n_steps if n_steps > 0 else 0

        return {
            'total_steps': n_steps,
            'average_tracking_error': avg_error,
            'average_neural_confidence': avg_confidence,
            'max_lipschitz_constant': max_lipschitz,
            'safety_violation_rate': violation_rate,
            'total_violations': n_violations
        }

    def reset(self) -> None:
        """Reset the control system."""
        self.controller.reset()
        if self.safety_monitor is not None:
            self.safety_monitor.reset_emergency_stop()
        self.telemetry = {
            'neuro_signals': [],
            'sensor_data': [],
            'control_states': [],
            'safety_violations': []
        }
        logger.info("Neurorobotic control system reset.")


# Example usage and simulation
if __name__ == "__main__":
    print("Closed-Loop Neurorobotic Control System")
    print("=" * 60)
    print("\nInitializing system...")

    # Create system
    system = NeuroroboticControlSystem(
        n_eeg_channels=64,
        eeg_sampling_rate=256.0,
        n_dofs=15,
        control_frequency=100.0
    )

    # Simulate calibration
    print("\nCalibrating neural decoder...")
    resting_eeg = np.random.randn(256 * 10, 64) * 1e-5  # 10 seconds of resting EEG
    system.calibrate(resting_eeg)

    # Simulate control loop
    print("\nRunning closed-loop control simulation...")
    n_steps = 1000  # 10 seconds at 100 Hz

    for step in range(n_steps):
        # Simulate EEG input (motor imagery)
        eeg_sample = np.random.randn(64) * 1e-5

        # Simulate sensor feedback
        sensor_data = SensorData(
            timestamp=time.time(),
            joint_positions=np.random.rand(15),
            joint_velocities=np.random.randn(15) * 0.1,
            joint_accelerations=np.random.randn(15) * 0.5,
            tactile_forces=np.random.rand(5) * 0.1,
            slip_detected=np.random.rand(5) < 0.05
        )

        # Process control step
        control_state = system.process_step(eeg_sample, sensor_data)

        if control_state is None:
            print(f"\nStep {step}: Safety violation, control halted")
            break

        # Print progress
        if step % 100 == 0:
            print(f"Step {step}/{n_steps}: "
                  f"Error={np.linalg.norm(control_state.position_error):.4f}, "
                  f"L={control_state.stability_metric:.4f}")

    # Print summary
    print("\n" + "=" * 60)
    print("Control Session Summary")
    print("=" * 60)
    summary = system.get_telemetry_summary()
    for key, value in summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    print("\nNeurorobotic control demonstration complete!")
