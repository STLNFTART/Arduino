"""Lightweight pandas replacement used for offline testing.

The implementation here supports the minimal subset of pandas required by
this repository: ``DataFrame`` construction from dictionaries, CSV I/O,
column selection producing ``Series`` objects, and a ``rolling().mean()``
operation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, MutableMapping, Sequence


class Series:
    """Simple numeric series that mimics pandas' Series API."""

    def __init__(self, data: Iterable[float]):
        self._data = [float(value) for value in data]

    def __iter__(self) -> Iterator[float]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> float:
        return self._data[index]

    def to_list(self) -> List[float]:
        return list(self._data)

    def rolling(self, window: int, min_periods: int = 1) -> "Rolling":
        return Rolling(self._data, window, min_periods)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Series({self._data!r})"


class Rolling:
    """Rolling window helper supporting mean aggregation."""

    def __init__(self, data: Sequence[float], window: int, min_periods: int) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if min_periods <= 0:
            raise ValueError("min_periods must be positive")
        self._data = list(float(value) for value in data)
        self._window = window
        self._min_periods = min_periods

    def mean(self) -> Series:
        results: List[float] = []
        for idx in range(len(self._data)):
            start = max(0, idx - self._window + 1)
            window_values = self._data[start : idx + 1]
            if len(window_values) < self._min_periods:
                results.append(float("nan"))
            else:
                results.append(sum(window_values) / len(window_values))
        return Series(results)


class DataFrame:
    """Minimal numeric DataFrame supporting CSV export and column selection."""

    def __init__(self, data: MutableMapping[str, Iterable[float]]):
        converted: Dict[str, List[float]] = {
            key: [float(value) for value in values] for key, values in data.items()
        }
        lengths = {len(values) for values in converted.values()}
        if lengths and len(lengths) != 1:
            raise ValueError("All columns must have the same length")
        self._data = converted

    @property
    def columns(self) -> List[str]:
        return list(self._data.keys())

    @property
    def index(self) -> range:
        return range(len(self))

    def __len__(self) -> int:
        if not self._data:
            return 0
        first_column = next(iter(self._data.values()))
        return len(first_column)

    def __getitem__(self, key: str) -> Series:
        return Series(self._data[key])

    def __setitem__(self, key: str, values: Iterable[float]) -> None:
        values_list = list(values) if not isinstance(values, Series) else values.to_list()
        if len(values_list) != len(self):
            raise ValueError("Assigned column must match existing length")
        self._data[key] = [float(value) for value in values_list]

    def to_csv(self, path: Path, index: bool = False) -> None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            headers = list(self._data.keys())
            if index:
                writer.writerow(["index", *headers])
            else:
                writer.writerow(headers)
            for row_idx in range(len(self)):
                row = [self._data[column][row_idx] for column in headers]
                if index:
                    writer.writerow([row_idx, *row])
                else:
                    writer.writerow(row)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"DataFrame(columns={self.columns!r}, rows={len(self)})"


def read_csv(path: Path | str) -> DataFrame:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            headers = next(reader)
        except StopIteration as exc:  # pragma: no cover - invalid input
            raise ValueError("CSV file is empty") from exc

        columns: Dict[str, List[float]] = {header: [] for header in headers}
        for row in reader:
            for header, value in zip(headers, row):
                columns[header].append(float(value))
        return DataFrame(columns)


__all__ = ["DataFrame", "Series", "read_csv"]
