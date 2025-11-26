import json
import csv
import time
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Iterable, Callable, List, Tuple
from itertools import product
from datetime import datetime
import math

# ---------- 1. CONFIG & UTILS ----------

@dataclass
class ParamGrid:
    """Simple parameter grid."""
    params: Dict[str, Iterable[Any]]

    def iter_configs(self) -> Iterable[Dict[str, Any]]:
        keys = list(self.params.keys())
        values = [list(v) for v in self.params.values()]
        for combo in product(*values):
            yield dict(zip(keys, combo))


def timestamp_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------- 2. RESULT LOGGER ----------

class RunLogger:
    def __init__(self, sim_name: str, tag: str = "sweep"):
        self.sim_name = sim_name
        self.tag = tag
        self.run_id = f"{timestamp_tag()}_{tag}"
        self.base_dir = os.path.join("experiments", "runs", sim_name, self.run_id)

        self.raw_dir = os.path.join(self.base_dir, "raw")
        self.summary_dir = os.path.join(self.base_dir, "summary")
        self.plots_dir = os.path.join(self.base_dir, "plots")

        ensure_dir(self.raw_dir)
        ensure_dir(self.summary_dir)
        ensure_dir(self.plots_dir)

        self.summary_rows: List[Dict[str, Any]] = []

    def save_raw_csv(self, index: int, series: Dict[str, Iterable[Any]]) -> str:
        """
        series: e.g. {"t": [...], "x": [...], "y": [...]} (all same length)
        """
        fname = os.path.join(self.raw_dir, f"sim_{index:04d}.csv")
        keys = list(series.keys())
        length = len(next(iter(series.values())))

        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(length):
                writer.writerow([series[k][i] for k in keys])
        return fname

    def add_summary_row(self, row: Dict[str, Any]) -> None:
        self.summary_rows.append(row)

    def finalize_summary(self) -> None:
        if not self.summary_rows:
            return
        keys = sorted(self.summary_rows[0].keys())
        # summary.csv
        csv_path = os.path.join(self.summary_dir, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in self.summary_rows:
                writer.writerow(r)

        # stats.json (simple aggregate)
        stats = self._compute_stats(keys)
        json_path = os.path.join(self.summary_dir, "stats.json")
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)

    def _compute_stats(self, keys: List[str]) -> Dict[str, Any]:
        numeric: Dict[str, List[float]] = {}
        for r in self.summary_rows:
            for k in keys:
                v = r.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numeric.setdefault(k, []).append(float(v))
        stats = {}
        for k, arr in numeric.items():
            if not arr:
                continue
            arr_sorted = sorted(arr)
            n = len(arr_sorted)
            mean = sum(arr_sorted) / n
            stats[k] = {
                "count": n,
                "min": arr_sorted[0],
                "max": arr_sorted[-1],
                "mean": mean,
                "median": arr_sorted[n // 2],
            }
        return stats

    def write_report(self, title: str, meta: Dict[str, Any]) -> None:
        path = os.path.join(self.base_dir, "REPORT.md")
        lines = [
            f"# {title}",
            "",
            f"**Simulation:** `{self.sim_name}`",
            f"**Run ID:** `{self.run_id}`",
            "",
            "## Meta",
            "",
        ]
        for k, v in meta.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")
        lines.append("## Summary Stats")
        lines.append("")
        stats_path = os.path.join(self.summary_dir, "stats.json")
        if os.path.exists(stats_path):
            lines.append(f"See `summary/stats.json` and `summary/summary.csv`.")
        else:
            lines.append("No stats available.")
        with open(path, "w") as f:
            f.write("\n".join(lines))


# ---------- 3. GENERIC SWEEP RUNNER ----------

def run_parameter_sweep(
    sim_name: str,
    param_grid: ParamGrid,
    simulate_fn: Callable[[Dict[str, Any]], Tuple[Dict[str, Any], Dict[str, Iterable[Any]]]],
    tag: str = "sweep",
) -> str:
    """
    simulate_fn(config) -> (metrics, time_series)
      metrics: dict of scalars (e.g., {"lipschitz": 0.63, "stable": True})
      time_series: dict of arrays for CSV logging (e.g., {"t": [...], "x": [...], "u": [...]})
    """
    logger = RunLogger(sim_name=sim_name, tag=tag)
    start = time.time()
    total = 0
    stable_count = 0

    for idx, cfg in enumerate(param_grid.iter_configs()):
        metrics, series = simulate_fn(cfg)
        metrics_with_params = {**cfg, **metrics}
        logger.add_summary_row(metrics_with_params)
        if series:
            logger.save_raw_csv(index=idx, series=series)

        total += 1
        if metrics_with_params.get("stable", False):
            stable_count += 1

    elapsed = time.time() - start
    logger.finalize_summary()
    meta = {
        "total_configs": total,
        "stable_count": stable_count,
        "stability_rate": f"{100.0 * stable_count / max(total, 1):.2f}%",
        "elapsed_sec": round(elapsed, 3),
        "configs_per_sec": round(total / max(elapsed, 1e-9), 2),
    }
    logger.write_report(title="Comprehensive Parameter Sweep", meta=meta)
    return logger.base_dir
