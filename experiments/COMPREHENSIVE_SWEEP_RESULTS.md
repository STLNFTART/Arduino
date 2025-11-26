# Comprehensive Full-Spectrum Parameter Sweep Results

**Primal Logic Robotic Hand Controller**
**Date:** 2025-11-26
**Run ID:** `20251126_205409_comprehensive_full_sweep`

---

## Executive Summary

✅ **Successfully completed comprehensive parameter sweep**
📊 **600 unique configurations tested**
⚡ **Performance:** 53.42 configs/sec (11.23 seconds total)
🔗 **Powered by [Primal Tech Invest](https://www.primaltechinvest.com)**

---

## Parameter Space Coverage

This sweep explored **all possible combinations** of the following parameters:

| Parameter | Values Tested | Count |
|-----------|--------------|-------|
| `alpha_base` | [0.3, 0.45, 0.54, 0.6, 0.75] | 5 |
| `beta_gain` | [0.2, 0.5, 0.8, 1.2] | 4 |
| `theta` | [0.6, 0.8, 1.0, 1.2, 1.5] | 5 |
| `torque_max` | [0.5, 0.7, 0.9] | 3 |
| `memory_mode` | ["exponential", "recursive_planck"] | 2 |
| `steps` | [200] | 1 |

**Total combinations:** 5 × 4 × 5 × 3 × 2 = **600 configurations**

---

## Key Metrics Aggregated

### Controller Gain (alpha_base)
- **Range tested:** 0.30 - 0.75
- **Mean:** 0.528
- **Median:** 0.54 (Lightfoot nominal)

### Memory Kernel Gain (beta_gain)
- **Range tested:** 0.20 - 1.20
- **Mean:** 0.675
- **Median:** 0.80 (default)

### Field Coupling (theta)
- **Range tested:** 0.60 - 1.50
- **Mean:** 1.02
- **Median:** 1.00

### Torque Limits (torque_max)
- **Range tested:** 0.50 - 0.90 N·m
- **Mean:** 0.70 N·m
- **Median:** 0.70 N·m

---

## Performance Metrics

### Mean Torque Output
- **Min:** 0.450 N·m
- **Max:** 0.780 N·m
- **Mean:** 0.616 N·m
- **Median:** 0.618 N·m

### Saturation Ratio
- **Min:** 79%
- **Max:** 85%
- **Mean:** 82.1%
- **Median:** 82%
- ⚠️ High saturation across all configs (controller pushing limits)

### Maximum Velocity
- **Min:** 1.74 rad/s
- **Max:** 3.01 rad/s
- **Mean:** 2.38 rad/s
- **Median:** 2.39 rad/s
- ✅ All configs well below 8.0 rad/s limit

### Coherence (Field Stability)
- **Min:** 0.999995
- **Max:** 0.999999
- **Mean:** 0.999998
- **Median:** 0.999998
- ✅ Exceptional field coherence across all configs

### Lipschitz Estimate (Smoothness)
- **Min:** 30.17
- **Max:** 39.11
- **Mean:** 35.18
- **Median:** 35.80

---

## Stability Analysis

**Stability Criterion:**
`stable = (saturation_ratio < 0.1) AND (max_velocity < 8.0)`

**Results:**
- **Stable configurations:** 0 / 600 (0.00%)
- **Primary failure mode:** Saturation ratio too high (>79% in all cases)

**Interpretation:**
The controller is operating near torque limits across the entire parameter space. This suggests:
1. Trajectory demands are aggressive relative to torque limits
2. Joint inertia/damping may need tuning
3. Alternative: Relax stability criterion to `saturation_ratio < 0.85`

---

## Memory Mode Comparison

The sweep tested both memory kernel implementations:

1. **Exponential Memory Kernel** (300 configs)
   - Standard exponential decay with gain modulation
   - More predictable, lower computational cost

2. **Recursive Planck Memory Kernel** (300 configs)
   - Quantum-inspired recursive formulation
   - Leverages Lightfoot/Donte constants
   - Potentially better long-term coherence

**Note:** Results show minimal performance difference between modes in this parameter range. Further analysis of time-series data recommended.

---

## Files Generated

```
experiments/runs/primal_kernel/20251126_205409_comprehensive_full_sweep/
├── raw/
│   ├── sim_0000.csv
│   ├── sim_0001.csv
│   └── ... (600 time series files)
├── summary/
│   ├── summary.csv   (600 rows, 14 columns)
│   └── stats.json    (aggregated statistics)
├── plots/
│   └── (reserved for heatmaps)
└── REPORT.md
```

---

## Recommendations

### 1. Torque Limit Exploration
High saturation suggests we're hitting physical limits. Consider:
- Testing higher `torque_max` values: [0.9, 1.2, 1.5]
- Analyzing which joint angles correlate with saturation
- Investigating alternative trajectories

### 2. Stability Region Mapping
Generate heatmaps for:
- `alpha_base` vs `beta_gain` (controller/memory interaction)
- `theta` vs `torque_max` (coupling vs limits)
- `saturation_ratio` across 2D parameter slices

### 3. Memory Mode Deep Dive
- Compare exponential vs recursive_planck on long simulations (1000+ steps)
- Analyze coherence retention over extended runs
- Identify parameter regimes where recursive_planck outperforms

### 4. Adaptive Trajectory Tuning
- Current trajectory may be too aggressive
- Consider torque-aware trajectory generation
- Test smooth vs step-like desired angle profiles

---

## Next Steps

1. ✅ **Completed:** Full parameter sweep (600 configs)
2. 🔄 **In Progress:** Deploy framework to other repos
3. 📊 **Pending:** Generate heatmaps for 2D parameter slices
4. 🔬 **Pending:** Extended run analysis (1000+ steps)
5. 🤖 **Pending:** Deploy to Multi-Heart-Model, UAV, Van Allen, etc.

---

## Technical Details

**Simulation Parameters:**
- Time step: `dt = 1e-3 s`
- Steps per config: 200
- Hand morphology: 5 fingers × 3 joints = 15 DOF
- Field grid: 4×4
- Integration: Euler forward

**Metrics Computed:**
- Mean/max torque
- Mean coherence
- Mean angle
- Max velocity
- Saturation ratio
- Lipschitz smoothness estimate
- Stability flag

**Framework Version:** Universal Experiment Results Pattern v1.0

---

## Reproducibility

To reproduce this sweep:

```bash
cd /home/user/Arduino
python experiments/run_comprehensive_sweep.py
```

**Dependencies:**
- Python 3.8+
- primal_logic module (hand, field, trajectory, utils)
- experiments/framework.py

**Expected runtime:** ~10-15 seconds on modern hardware

---

## Data Access

**Summary CSV:**
`experiments/runs/primal_kernel/20251126_205409_comprehensive_full_sweep/summary/summary.csv`

**Time Series:**
`experiments/runs/primal_kernel/20251126_205409_comprehensive_full_sweep/raw/sim_*.csv`

**Stats JSON:**
`experiments/runs/primal_kernel/20251126_205409_comprehensive_full_sweep/summary/stats.json`

---

**Powered by Primal Tech Invest**
🔗 [www.primaltechinvest.com](https://www.primaltechinvest.com)

*Universal Experiment Results Framework - Standardized across all simulation repositories*

---

**Document Version:** 1.0
**Last Updated:** 2025-11-26
**Contact:** [www.primaltechinvest.com](https://www.primaltechinvest.com)
