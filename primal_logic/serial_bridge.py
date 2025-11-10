"""Serial interface for streaming torques to embedded hardware."""

from __future__ import annotations

import logging
from typing import Iterable

from .utils import safe_clip

logger = logging.getLogger(__name__)


class SerialHandBridge:
    """Optional serial output to a microcontroller expecting CSV torques."""

    def __init__(self, port: str, baud: int, timeout: float = 0.005) -> None:
        try:
            import serial  # type: ignore
        except ImportError as exc:  # pragma: no cover - hardware optional
            raise RuntimeError(
                "pyserial is required for SerialHandBridge; install via `pip install pyserial`."
            ) from exc

        self._serial = serial.Serial(port, baudrate=baud, timeout=timeout)
        logger.info("Serial bridge opened on %s at %d baud", port, baud)

    def send(self, torques: Iterable[Iterable[float]]) -> None:
        flat = [safe_clip(value, -1.0, 1.0) for row in torques for value in row]
        line = ",".join(f"{value:.3f}" for value in flat) + "\n"
        self._serial.write(line.encode("ascii"))
        logger.debug("Sent torques: %s", line.strip())
