# Universal Experiment Results Pattern

**Standardized framework for parameter sweeps across all simulation repositories.**

This directory implements the universal experiment/results pattern used across all repos (Multi-Heart-Model, UAV, Van Allen, Optimus, stealth, Mars, Arduino, etc.).

---

## 1. Directory Structure

```
experiments/
  configs/
    <sim_name>_sweep_*.json      # parameter grids
  runs/
    <sim_name>/
      YYYYMMDD_HHMMSS_<tag>/
        raw/
          sim_<index>.csv        # raw time-series / trajectories
        summary/
          summary.csv            # one row per config
          stats.json             # aggregates, min/max/mean, etc.
        plots/
          *.png                  # heatmaps, convergence, etc.
        REPORT.md                # human-readable run summary
  framework.py                   # core results framework (reusable)
  run_<sim_name>_sweep.py        # wired runner for specific sim
```

**Every sim = one folder per run with the same structure.**

---

## 2. Core Framework

The `framework.py` module provides:

- **ParamGrid**: Simple parameter grid generator using itertools.product
- **RunLogger**: Automatic directory creation, CSV logging, summary stats, REPORT.md generation
- **run_parameter_sweep()**: Generic sweep runner that takes a simulation function

### Usage Pattern

```python
from experiments.framework import ParamGrid, run_parameter_sweep

def simulate_my_system(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Iterable[Any]]]:
    """
    Run one simulation with given config.

    Returns:
        (metrics, time_series) where:
        - metrics: Dict of scalars (e.g., {"lipschitz": 0.63, "stable": True})
        - time_series: Dict of arrays (e.g., {"t": [...], "x": [...], "u": [...]})
    """
    # Your simulation code here
    ...
    return metrics, time_series

# Define parameter grid
grid = ParamGrid(params={
    "param1": [0.1, 0.2, 0.3],
    "param2": [10, 20, 30],
})

# Run sweep
output_dir = run_parameter_sweep(
    sim_name="my_sim",
    param_grid=grid,
    simulate_fn=simulate_my_system,
    tag="full_sweep",
)
```

---

## 3. Wiring a New Sim

### Step 1: Create a wrapper function

Your simulation function must:
1. Accept a config dict with parameters
2. Return `(metrics, time_series)` tuple
3. metrics = dict of scalars (floats, ints, bools)
4. time_series = dict of arrays (all same length)

### Step 2: Define parameter grid

Create a JSON config in `configs/` (optional) or define in code:

```python
grid = ParamGrid(params={
    "alpha": [0.3, 0.5, 0.7],
    "beta": [0.1, 0.2],
    "steps": [1000],
})
```

### Step 3: Run the sweep

```python
output_dir = run_parameter_sweep(
    sim_name="your_sim_name",
    param_grid=grid,
    simulate_fn=your_wrapper_function,
    tag="descriptive_tag",
)
```

The framework handles:
- Directory creation
- CSV logging (raw + summary)
- Stats aggregation (min/max/mean/median)
- REPORT.md generation
- Timestamped run IDs

---

## 4. Example: Primal Kernel Sweep

See `run_primal_kernel_sweep.py` for a complete example.

**What it does:**
- Sweeps `alpha_base` (controller gain) and `theta` (field coupling)
- Runs RoboticHand + PrimalLogicField simulation for each config
- Collects metrics: mean_torque, mean_coherence, saturation_ratio, stable
- Saves time series: t, coherence

**Run it:**
```bash
python experiments/run_primal_kernel_sweep.py
```

**Output:**
```
experiments/runs/primal_kernel/20251126_074045_alpha_theta_sweep/
  ├── raw/
  │   ├── sim_0000.csv  (time series for config 0)
  │   ├── sim_0001.csv
  │   └── ...
  ├── summary/
  │   ├── summary.csv   (all configs, one row each)
  │   └── stats.json    (aggregates)
  ├── plots/
  │   └── (empty, for future heatmaps)
  └── REPORT.md         (human-readable summary)
```

---

## 5. Rollout Plan

### Phase 1: Standardize existing repos
- [x] Arduino (primal_kernel)
- [ ] Multi-Heart-Model
- [ ] UAV swarm
- [ ] Van Allen radiation
- [ ] Optimus
- [ ] Stealth
- [ ] Mars

### Phase 2: Add plotting utilities
- Heatmap generation for 2D parameter sweeps
- Convergence plots
- Stability region visualization

### Phase 3: Shared submodule (optional)
- Create `experiment-framework` git submodule
- Reuse across all repos
- Single source of truth for framework.py

---

## 6. Key Benefits

1. **Consistency**: Every repo uses the same structure
2. **Discoverability**: Results are always in `experiments/runs/<sim>/YYYYMMDD_HHMMSS_<tag>/`
3. **Automation**: Framework handles all logging, stats, and reporting
4. **Reproducibility**: Configs + timestamps + REPORT.md make runs traceable
5. **Scalability**: Same pattern works for 10 configs or 10,000 configs

---

## 7. Advanced Usage

### Custom stability criteria

```python
def simulate_fn(config):
    # ... run simulation ...
    stable = (lipschitz < 1.0) and (saturation_ratio < 0.05)
    metrics = {"lipschitz": lipschitz, "stable": stable, ...}
    return metrics, time_series
```

The framework counts `stable=True` configs and reports stability rate.

### Time series downsampling

If your sim runs for millions of steps, downsample before returning:

```python
time_series = {
    "t": t_array[::100],  # every 100th point
    "x": x_array[::100],
}
```

### Empty time series

If you don't need raw time series, return empty dict:

```python
return metrics, {}
```

---

## Notes

- No fluff. Just directories, CSVs, and a reusable Python module.
- Drop `framework.py` into every repo, wire your sim, done.
- Plots dir is reserved for future heatmap/visualization scripts.

---

**Next steps:**
1. Run `run_primal_kernel_sweep.py` to see the pattern in action
2. Check `experiments/runs/primal_kernel/<latest>/` for output
3. Wire your own simulations using the same pattern
