#!/usr/bin/env python3
"""
RUN ALL EXPERIMENT SWEEPS - MAXIMUM OUTPUT MODE

Executes all individual parameter sweeps (alpha, beta, theta, torque)
from the original primal_logic.sweeps module.

Powered by Primal Tech Invest
www.primaltechinvest.com
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from primal_logic.sweeps import alpha_sweep, beta_sweep, tau_sweep, torque_sweep

def main() -> None:
    print("=" * 80)
    print("RUNNING ALL INDIVIDUAL PARAMETER SWEEPS")
    print("Primal Logic Robotic Hand Controller")
    print("=" * 80)
    print()
    print("Powered by Primal Tech Invest")
    print("🔗 www.primaltechinvest.com")
    print()
    print("=" * 80)
    print()

    output_dir = Path("experiments/runs/individual_sweeps")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Torque/Theta Sweep
    print("1. Running TORQUE/THETA SWEEP...")
    print("   Parameters: theta = [0.5, 0.8, 1.0, 1.2, 1.5]")
    print("   Steps: 200")
    torque_path = output_dir / "torque_sweep.csv"
    torque_results = torque_sweep([0.5, 0.8, 1.0, 1.2, 1.5], steps=200, output_path=torque_path)
    print(f"   ✓ Complete: {len(torque_results)} configs")
    print(f"   Output: {torque_path}")
    print()

    # Alpha Sweep
    print("2. Running ALPHA SWEEP...")
    print("   Parameters: alpha_base = [0.3, 0.4, 0.5, 0.54, 0.6, 0.7, 0.8]")
    print("   Steps: 200")
    alpha_path = output_dir / "alpha_sweep.csv"
    alpha_results = alpha_sweep([0.3, 0.4, 0.5, 0.54, 0.6, 0.7, 0.8], steps=200, output_path=alpha_path)
    print(f"   ✓ Complete: {len(alpha_results)} configs")
    print(f"   Output: {alpha_path}")
    print()

    # Beta Sweep
    print("3. Running BETA SWEEP...")
    print("   Parameters: beta_gain = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5]")
    print("   Steps: 200")
    beta_path = output_dir / "beta_sweep.csv"
    beta_results = beta_sweep([0.2, 0.5, 0.8, 1.0, 1.2, 1.5], steps=200, output_path=beta_path)
    print(f"   ✓ Complete: {len(beta_results)} configs")
    print(f"   Output: {beta_path}")
    print()

    # Tau (Torque Limit) Sweep
    print("4. Running TAU (TORQUE LIMIT) SWEEP...")
    print("   Parameters: torque_max = [0.3, 0.5, 0.7, 0.9, 1.2, 1.5]")
    print("   Steps: 200")
    tau_path = output_dir / "tau_sweep.csv"
    tau_results = tau_sweep([0.3, 0.5, 0.7, 0.9, 1.2, 1.5], steps=200, output_path=tau_path)
    print(f"   ✓ Complete: {len(tau_results)} configs")
    print(f"   Output: {tau_path}")
    print()

    print("=" * 80)
    print("ALL INDIVIDUAL SWEEPS COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Torque sweep: {len(torque_results)} configs")
    print(f"  - Alpha sweep: {len(alpha_results)} configs")
    print(f"  - Beta sweep: {len(beta_results)} configs")
    print(f"  - Tau sweep: {len(tau_results)} configs")
    print(f"  - Total: {len(torque_results) + len(alpha_results) + len(beta_results) + len(tau_results)} configs")
    print()
    print(f"All outputs saved to: {output_dir}")
    print()
    print("Powered by Primal Tech Invest")
    print("🔗 www.primaltechinvest.com")
    print()


if __name__ == "__main__":
    main()
