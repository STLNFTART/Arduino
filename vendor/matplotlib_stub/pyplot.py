"""Minimal pyplot replacement writing textual plot summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass
class SeriesRecord:
    x: List[float]
    y: List[float]
    label: str
    alpha: float
    linewidth: float


@dataclass
class Axes:
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    grid_enabled: bool = False
    grid_style: Tuple[str, float] = ("--", 0.3)
    series: List[SeriesRecord] = field(default_factory=list)

    def plot(
        self,
        x: Iterable[float],
        y: Iterable[float],
        label: str = "",
        alpha: float = 1.0,
        linewidth: float = 1.0,
    ) -> SeriesRecord:
        record = SeriesRecord(
            list(float(v) for v in x),
            list(float(v) for v in y),
            label,
            float(alpha),
            float(linewidth),
        )
        self.series.append(record)
        return record

    def set_xlabel(self, text: str) -> None:
        self.xlabel = text

    def set_ylabel(self, text: str) -> None:
        self.ylabel = text

    def set_title(self, text: str) -> None:
        self.title = text

    def legend(self) -> None:  # pragma: no cover - metadata only
        return None

    def grid(self, enabled: bool, linestyle: str = "--", alpha: float = 0.3) -> None:
        self.grid_enabled = bool(enabled)
        self.grid_style = (linestyle, float(alpha))


@dataclass
class Figure:
    axes: Axes
    size: Tuple[float, float]

    def tight_layout(self) -> None:  # pragma: no cover - no-op
        return None

    def savefig(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as handle:
            handle.write("Primal Logic pseudo-plot\n")
            handle.write(f"size={self.size}\n")
            handle.write(f"title={self.axes.title}\n")
            handle.write(f"xlabel={self.axes.xlabel}\n")
            handle.write(f"ylabel={self.axes.ylabel}\n")
            handle.write(f"grid={self.axes.grid_enabled},{self.axes.grid_style}\n")
            for idx, record in enumerate(self.axes.series):
                handle.write(
                    "series {} label={} alpha={} linewidth={} points={}\n".format(
                        idx, record.label, record.alpha, record.linewidth, len(record.x)
                    )
                )
                preview = list(zip(record.x[:5], record.y[:5]))
                handle.write(f"preview={preview}\n")


def subplots(figsize: Tuple[float, float] = (6.4, 4.8)) -> Tuple[Figure, Axes]:
    axes = Axes()
    fig = Figure(axes=axes, size=figsize)
    return fig, axes


def close(_figure: Figure) -> None:  # pragma: no cover - compatibility
    return None


__all__ = ["Figure", "Axes", "subplots", "close"]
