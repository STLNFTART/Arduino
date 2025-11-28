"""RPO burn meter for tracking actuator time-in-mode and logging token burns.

This module provides a simple bookkeeping layer that accrues time spent in
Planck/recursive modes for simulated actuators. Each full second of accrued
runtime triggers a placeholder burn entry to ``rpo_burn_log.csv`` so the
Hedera SDK integration can be added later without changing the call sites.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Mapping, Optional

DEFAULT_CONTRACT_ID = "0x35AF4bCa366737d2a433Fe85062Dd7A19F9572d3"
SUPPORTED_MODES = {"dry_run", "hedera_testnet"}

# Environment variable names for the future Hedera SDK integration. These are
# validated in "hedera_testnet" mode even though no network calls are made yet.
ENV_OPERATOR_ID = "HEDERA_OPERATOR_ID"
ENV_OPERATOR_KEY = "HEDERA_OPERATOR_KEY"
ENV_NETWORK = "HEDERA_NETWORK"


class RPOBurnMeter:
    """Accumulate actuator runtime and log simulated burns to CSV.

    Parameters
    ----------
    contract_id : str | None
        Target burn contract identifier. When ``None``, defaults to
        ``DEFAULT_CONTRACT_ID``.
    actuator_address_map : Mapping[str, str]
        Mapping from actuator keys to blockchain addresses.
    mode : str
        Operation mode. ``"dry_run"`` only logs to CSV. ``"hedera_testnet"``
        also logs to CSV with a placeholder transaction ID for future SDK
        integration. In ``hedera_testnet`` mode the environment variables
        ``HEDERA_OPERATOR_ID`` and ``HEDERA_OPERATOR_KEY`` are validated to
        prepare for the eventual live call path.
    """

    def __init__(
        self,
        contract_id: Optional[str],
        actuator_address_map: Mapping[str, str],
        mode: str = "dry_run",
    ) -> None:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported burn mode '{mode}'. Expected one of {SUPPORTED_MODES}.")

        if not actuator_address_map:
            raise ValueError("actuator_address_map must contain at least one actuator entry")

        self.contract_id = contract_id or DEFAULT_CONTRACT_ID
        self.actuator_address_map: Dict[str, str] = dict(actuator_address_map)
        self.mode = mode

        # Optional Hedera configuration loaded from the environment. This is
        # validated in testnet mode so the eventual SDK call path can trust
        # these values without re-checking.
        self.operator_id = os.getenv(ENV_OPERATOR_ID)
        self.operator_key = os.getenv(ENV_OPERATOR_KEY)
        self.network = os.getenv(ENV_NETWORK, "testnet")

        if self.mode == "hedera_testnet":
            missing_vars = [
                name
                for name, value in [
                    (ENV_OPERATOR_ID, self.operator_id),
                    (ENV_OPERATOR_KEY, self.operator_key),
                ]
                if not value
            ]
            if missing_vars:
                raise EnvironmentError(
                    "hedera_testnet mode requires environment variables: "
                    + ", ".join(missing_vars)
                )

        # Track fractional seconds for each actuator that have not yet triggered a burn
        self.unbilled_seconds: Dict[str, float] = {key: 0.0 for key in self.actuator_address_map}
        # Track how many integer seconds have been burned per actuator for reporting
        self.burned_seconds: Dict[str, int] = {key: 0 for key in self.actuator_address_map}

    @classmethod
    def from_config_files(
        cls,
        operator_config_path: Path,
        actuator_map_path: Path,
        mode: str = "dry_run",
    ) -> "RPOBurnMeter":
        """Instantiate a burn meter from JSON config files."""

        operator_config = json.loads(operator_config_path.read_text())
        contract_address = operator_config.get("contract_address") or DEFAULT_CONTRACT_ID

        actuator_map = json.loads(actuator_map_path.read_text())
        if not isinstance(actuator_map, dict):
            raise ValueError("Actuator address map must be a JSON object")

        return cls(contract_id=contract_address, actuator_address_map=actuator_map, mode=mode)

    def record(self, actuator_name: str, dt_seconds: float) -> None:
        """Record elapsed time for an actuator and trigger burns per full second."""

        if actuator_name not in self.actuator_address_map:
            raise KeyError(
                f"Actuator '{actuator_name}' not found in address map. "
                "Update billing/rpo_actuator_addresses.json to include it."
            )

        if dt_seconds <= 0:
            # Ignore non-positive durations to avoid corrupting counters
            return

        self.unbilled_seconds[actuator_name] += dt_seconds
        full_seconds = int(self.unbilled_seconds[actuator_name] // 1.0)

        for _ in range(full_seconds):
            self._burn_one_token(actuator_name, seconds=1)

        self.unbilled_seconds[actuator_name] -= float(full_seconds)

    def _burn_one_token(self, actuator_name: str, seconds: int) -> None:
        """Log a single-token burn for the specified actuator."""

        actuator_address = self.actuator_address_map[actuator_name]
        self._call_burn_contract(actuator_address=actuator_address, seconds=seconds)
        self.burned_seconds[actuator_name] += seconds

    def _call_burn_contract(self, actuator_address: str, seconds: int) -> None:
        """Append a CSV log entry for the would-be burn.

        The Hedera network integration will be added in the future. For now we
        capture the necessary metadata to trace intended burns while validating
        operator credentials in testnet mode. No external network calls are
        issued from this method.
        """

        timestamp = int(time.time())
        log_path = Path("rpo_burn_log.csv")

        if self.mode == "dry_run":
            # Reserve empty fields to keep the CSV schema consistent with the
            # hedera_testnet rows (operator_id, network, tx_id).
            row = (
                f"{timestamp},{self.contract_id},{actuator_address},{seconds},"
                "dry_run,,,")
            row += "\n"
        elif self.mode == "hedera_testnet":
            tx_id = self._build_placeholder_tx_id(timestamp=timestamp, actuator_address=actuator_address)
            row = (
                f"{timestamp},{self.contract_id},{actuator_address},{seconds},"
                f"hedera_testnet,{self.operator_id},{self.network},{tx_id}\n"
            )
        else:
            # The constructor guards mode, but keep a defensive branch for future additions
            raise ValueError(f"Unsupported mode '{self.mode}' in _call_burn_contract")

        self._append_log_row(log_path=log_path, row=row)

    @staticmethod
    def _append_log_row(log_path: Path, row: str) -> None:
        """Append a CSV row, creating the file if needed."""

        if not log_path.exists():
            log_path.write_text(
                "timestamp,contract_id,actuator_address,seconds,mode,operator_id,network,tx_id\n"
            )

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(row)

    @staticmethod
    def _build_placeholder_tx_id(timestamp: int, actuator_address: str) -> str:
        """Return a deterministic placeholder transaction id for logging.

        This keeps the CSV schema aligned with a future Hedera SDK integration
        without performing any real signing or submission.
        """

        suffix = actuator_address[-6:]
        return f"SIMULATED_TX_{timestamp}_{suffix}"

    def get_burn_report(self) -> Dict[str, int]:
        """Return how many integer seconds have been burned per actuator."""

        return dict(self.burned_seconds)
