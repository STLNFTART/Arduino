"""Tests for Arduino integration bridge with heart model."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from primal_logic.heart_arduino_bridge import HeartArduinoBridge, ProcessorHeartArduinoLink
from primal_logic.heart_model import MultiHeartModel


@pytest.fixture
def mock_serial():
    """Create a mock serial port."""
    # Mock the serial module that gets imported inside SerialHandBridge.__init__
    with patch("serial.Serial") as mock_serial_class:
        mock_port = Mock()
        mock_serial_class.return_value = mock_port
        yield mock_port


def test_bridge_initialization(mock_serial) -> None:
    """Test that HeartArduinoBridge initializes correctly."""
    bridge = HeartArduinoBridge(port="/dev/ttyACM0", baud=115200, normalize=True)

    assert bridge._normalize is True
    # Verify serial port was opened
    assert mock_serial.write.call_count == 0  # No data sent yet


def test_send_heart_signals_normalization(mock_serial) -> None:
    """Test that heart signals are normalized when sent to Arduino."""
    bridge = HeartArduinoBridge(port="/dev/ttyACM0", normalize=True)
    heart_model = MultiHeartModel()

    # Set extreme values
    heart_model.state.n_heart = 10.0
    heart_model.state.n_brain = -10.0

    bridge.send_heart_signals(heart_model)

    # Verify write was called
    assert mock_serial.write.call_count == 1

    # Get the data that was written
    written_data = mock_serial.write.call_args[0][0].decode("ascii")

    # All values should be between 0 and 1 when normalized
    values = [float(v) for v in written_data.strip().split(",")]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_send_raw_values(mock_serial) -> None:
    """Test sending raw values directly to Arduino."""
    bridge = HeartArduinoBridge(port="/dev/ttyACM0", normalize=True)

    test_values = [0.1, 0.5, 0.9, 1.2]  # Last value exceeds 1.0
    bridge.send_raw_values(test_values)

    assert mock_serial.write.call_count == 1

    written_data = mock_serial.write.call_args[0][0].decode("ascii")
    values = [float(v) for v in written_data.strip().split(",")]

    # Check normalization applied
    assert len(values) == 4
    assert values[-1] == 1.0  # 1.2 should be clipped to 1.0


def test_processor_link_update(mock_serial) -> None:
    """Test ProcessorHeartArduinoLink update method."""
    heart_model = MultiHeartModel()
    bridge = HeartArduinoBridge(port="/dev/ttyACM0")
    link = ProcessorHeartArduinoLink(
        heart_model=heart_model,
        arduino_bridge=bridge,
        send_interval=1,
    )

    # Update should step the model and send to Arduino
    link.update(cardiac_input=0.5, brain_setpoint=0.3, theta=1.0)

    # Verify model was updated
    assert heart_model.step_count == 1

    # Verify data was sent to Arduino
    assert mock_serial.write.call_count == 1


def test_send_interval_throttling(mock_serial) -> None:
    """Test that send_interval controls how often data is sent to Arduino."""
    heart_model = MultiHeartModel()
    bridge = HeartArduinoBridge(port="/dev/ttyACM0")
    link = ProcessorHeartArduinoLink(
        heart_model=heart_model,
        arduino_bridge=bridge,
        send_interval=5,  # Only send every 5 steps
    )

    # Update 3 times (less than send_interval)
    for _ in range(3):
        link.update(cardiac_input=0.5, brain_setpoint=0.3, theta=1.0)

    # Should not have sent yet
    assert mock_serial.write.call_count == 0

    # Update 2 more times to reach send_interval
    for _ in range(2):
        link.update(cardiac_input=0.5, brain_setpoint=0.3, theta=1.0)

    # Now it should have sent once
    assert mock_serial.write.call_count == 1

    # Update 5 more times
    for _ in range(5):
        link.update(cardiac_input=0.5, brain_setpoint=0.3, theta=1.0)

    # Should have sent one more time (total 2)
    assert mock_serial.write.call_count == 2


def test_get_state(mock_serial) -> None:
    """Test getting state from ProcessorHeartArduinoLink."""
    heart_model = MultiHeartModel()
    link = ProcessorHeartArduinoLink(heart_model=heart_model, arduino_bridge=None)

    # Update the model
    link.update(cardiac_input=0.5, brain_setpoint=0.3, theta=1.0)

    # Get state
    state = link.get_state()

    assert "n_heart" in state
    assert "n_brain" in state
    assert "heart_rate" in state
    assert "brain_activity" in state

    assert isinstance(state["n_heart"], float)
    assert isinstance(state["n_brain"], float)
    assert isinstance(state["heart_rate"], float)
    assert isinstance(state["brain_activity"], float)


def test_reset(mock_serial) -> None:
    """Test that reset clears the link state."""
    heart_model = MultiHeartModel()
    bridge = HeartArduinoBridge(port="/dev/ttyACM0")
    link = ProcessorHeartArduinoLink(
        heart_model=heart_model,
        arduino_bridge=bridge,
        send_interval=3,
    )

    # Update several times
    for _ in range(5):
        link.update(cardiac_input=0.5, brain_setpoint=0.3, theta=1.0)

    assert heart_model.step_count == 5

    # Reset
    link.reset()

    assert heart_model.step_count == 0
    assert link._step_counter == 0
    assert heart_model.state.n_heart == 0.0
    assert heart_model.state.n_brain == 0.0


def test_link_without_arduino_bridge() -> None:
    """Test that ProcessorHeartArduinoLink works without Arduino bridge."""
    heart_model = MultiHeartModel()
    link = ProcessorHeartArduinoLink(heart_model=heart_model, arduino_bridge=None)

    # Should work without errors
    link.update(cardiac_input=0.5, brain_setpoint=0.3, theta=1.0)
    assert heart_model.step_count == 1

    state = link.get_state()
    assert state is not None
