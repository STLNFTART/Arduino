"""Arduino integration bridge for the Multi-Heart Model.

This module provides a specialized bridge that connects the Multi-Heart Model
output to Arduino hardware via serial communication. It handles data formatting,
signal conditioning, and real-time streaming of cardiac and brain signals.
"""

from __future__ import annotations

import logging
from typing import Optional

from .heart_model import MultiHeartModel
from .serial_bridge import SerialHandBridge
from .utils import safe_clip

logger = logging.getLogger(__name__)


class HeartArduinoBridge:
    """Bridge between Multi-Heart Model and Arduino hardware.

    This bridge extends the basic serial communication to handle physiological
    signals from the heart-brain model, providing formatted output suitable
    for Arduino-based biofeedback systems, actuators, or displays.

    Parameters
    ----------
    port : str
        Serial port for Arduino (e.g., '/dev/ttyACM0' or 'COM3').
    baud : int
        Baud rate for serial communication (default 115200).
    timeout : float
        Serial read timeout in seconds (default 0.005).
    normalize : bool
        If True, normalize all outputs to [0, 1] range (default True).
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        timeout: float = 0.005,
        normalize: bool = True,
    ) -> None:
        self._serial_bridge = SerialHandBridge(port=port, baud=baud, timeout=timeout)
        self._normalize = normalize
        logger.info("HeartArduinoBridge initialized on %s at %d baud", port, baud)

    def send_heart_signals(self, heart_model: MultiHeartModel) -> None:
        """Send cardiac and brain signals from the heart model to Arduino.

        Parameters
        ----------
        heart_model : MultiHeartModel
            The heart-brain model instance to read signals from.
        """
        cardiac_output = heart_model.get_cardiac_output()

        if self._normalize:
            # Ensure all values are in [0, 1] range
            cardiac_output = [safe_clip(val, 0.0, 1.0) for val in cardiac_output]

        # Format as CSV line for Arduino
        line = ",".join(f"{value:.4f}" for value in cardiac_output) + "\n"
        self._serial_bridge._serial.write(line.encode("ascii"))
        logger.debug("Sent heart signals: %s", line.strip())

    def send_raw_values(self, values: list[float]) -> None:
        """Send raw values directly to Arduino.

        Parameters
        ----------
        values : list[float]
            List of floating-point values to send.
        """
        if self._normalize:
            values = [safe_clip(val, 0.0, 1.0) for val in values]

        line = ",".join(f"{value:.4f}" for value in values) + "\n"
        self._serial_bridge._serial.write(line.encode("ascii"))
        logger.debug("Sent raw values: %s", line.strip())


class ProcessorHeartArduinoLink:
    """Unified link connecting Microprocessor, Multi-Heart Model, and Arduino.

    This class orchestrates the complete pipeline:
    1. Receives processor commands (theta, setpoints)
    2. Updates the heart-brain model with RPO processing
    3. Streams results to Arduino hardware in real-time

    Parameters
    ----------
    heart_model : MultiHeartModel
        The multi-heart model instance.
    arduino_bridge : Optional[HeartArduinoBridge]
        Optional Arduino bridge for hardware output.
    send_interval : int
        Send to Arduino every N steps (default 1 = every step).
    """

    def __init__(
        self,
        heart_model: MultiHeartModel,
        arduino_bridge: Optional[HeartArduinoBridge] = None,
        send_interval: int = 1,
    ) -> None:
        self.heart_model = heart_model
        self.arduino_bridge = arduino_bridge
        self.send_interval = send_interval
        self._step_counter = 0
        logger.info("ProcessorHeartArduinoLink initialized (send_interval=%d)", send_interval)

    def update(
        self,
        cardiac_input: float,
        brain_setpoint: float,
        theta: float = 1.0,
        use_forcing: bool = False,
    ) -> None:
        """Update the system and optionally send to Arduino.

        Parameters
        ----------
        cardiac_input : float
            External cardiac input C(t).
        brain_setpoint : float
            Brain control setpoint s_set(t).
        theta : float
            Command envelope Θ for RPO operators.
        use_forcing : bool
            Enable dual-frequency forcing (RSA + baroreflex).
        """
        # Step the heart-brain model (includes RPO processing)
        self.heart_model.step(
            cardiac_input=cardiac_input,
            brain_setpoint=brain_setpoint,
            theta=theta,
            use_forcing=use_forcing,
        )

        # Send to Arduino if interval reached
        self._step_counter += 1
        if self.arduino_bridge is not None and self._step_counter >= self.send_interval:
            self.arduino_bridge.send_heart_signals(self.heart_model)
            self._step_counter = 0

    def get_state(self) -> dict[str, float]:
        """Get current state of the heart-brain system.

        Returns
        -------
        dict[str, float]
            Dictionary with keys: n_heart, n_brain, heart_rate, brain_activity.
        """
        return {
            "n_heart": self.heart_model.state.n_heart,
            "n_brain": self.heart_model.state.n_brain,
            "heart_rate": self.heart_model.get_heart_rate(),
            "brain_activity": self.heart_model.get_brain_activity(),
        }

    def reset(self) -> None:
        """Reset the heart model and step counter."""
        self.heart_model.reset()
        self._step_counter = 0


__all__ = ["HeartArduinoBridge", "ProcessorHeartArduinoLink"]
