"""Vector sweep utilities for parameter exploration."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from .field import PrimalLogicField
from .hand import JointLimits, RoboticHand
from .trajectory import generate_grasp_trajectory
from .utils import flatten, mean, write_csv


def _simulate_statistics(
    hand: RoboticHand,
    field: PrimalLogicField,
    trajectory: Sequence[Sequence[Sequence[float]]],
    theta: float,
    steps: int,
) -> Tuple[float, float, float]:
    """Run the coupled hand/field simulation and collect summary statistics."""

    torques: List[float] = []
    coherence_samples: List[float] = []
    saturation_hits = 0
    torque_limit = hand.joint_limits.torque_max

    for step in range(steps):
        coherence = field.step(theta)
        coherence_samples.append(coherence)
        hand.step(trajectory[step], theta=theta, coherence=coherence, step=step)

        latest_torques = flatten(hand.get_torques())
        torques.extend(latest_torques)
        saturation_hits += sum(1 for value in latest_torques if abs(value) >= torque_limit - 1e-9)

    mean_torque = mean(torques)
    mean_coherence = mean(coherence_samples)
    saturation_ratio = (saturation_hits / len(torques)) if torques else 0.0
    return mean_torque, mean_coherence, saturation_ratio


def torque_sweep(
    theta_values: Sequence[float],
    steps: int = 200,
    output_path: Path | None = None,
) -> List[Tuple[float, float, float, float]]:
    """Sweep over constant ``theta`` values and record torque statistics."""

    results: List[Tuple[float, float, float, float]] = []
    for theta in theta_values:
        hand = RoboticHand()
        field = PrimalLogicField(nx=4, ny=4)
        trajectory = generate_grasp_trajectory(steps, hand.n_fingers, hand.n_joints_per_finger)
        mean_torque, mean_coherence, saturation_ratio = _simulate_statistics(hand, field, trajectory, theta, steps)
        results.append((theta, mean_torque, mean_coherence, saturation_ratio))

    if output_path is not None:
        header = ["theta", "mean_torque", "mean_coherence", "saturation_ratio"]
        write_csv(str(output_path), header, results)

    return results


def alpha_sweep(
    alpha_values: Sequence[float],
    theta: float = 1.0,
    steps: int = 200,
    output_path: Path | None = None,
) -> List[Tuple[float, float, float, float]]:
    """Evaluate controller behaviour across a vector of base alpha gains."""

    results: List[Tuple[float, float, float, float]] = []
    for alpha in alpha_values:
        hand = RoboticHand(alpha_base=alpha)
        field = PrimalLogicField(nx=4, ny=4)
        trajectory = generate_grasp_trajectory(steps, hand.n_fingers, hand.n_joints_per_finger)
        stats = _simulate_statistics(hand, field, trajectory, theta, steps)
        results.append((alpha, *stats))

    if output_path is not None:
        header = ["alpha_base", "mean_torque", "mean_coherence", "saturation_ratio"]
        write_csv(str(output_path), header, results)

    return results


def beta_sweep(
    beta_values: Sequence[float],
    theta: float = 1.0,
    steps: int = 200,
    output_path: Path | None = None,
) -> List[Tuple[float, float, float, float]]:
    """Evaluate memory kernel gains (β) across a parameter vector."""

    results: List[Tuple[float, float, float, float]] = []
    for beta in beta_values:
        if beta <= 0:
            raise ValueError("beta gains must be positive to maintain causal memory dynamics")
        hand = RoboticHand(beta_gain=beta)
        field = PrimalLogicField(nx=4, ny=4)
        trajectory = generate_grasp_trajectory(steps, hand.n_fingers, hand.n_joints_per_finger)
        stats = _simulate_statistics(hand, field, trajectory, theta, steps)
        results.append((beta, *stats))

    if output_path is not None:
        header = ["beta_gain", "mean_torque", "mean_coherence", "saturation_ratio"]
        write_csv(str(output_path), header, results)

    return results


def tau_sweep(
    torque_limits: Sequence[float],
    theta: float = 1.0,
    steps: int = 200,
    output_path: Path | None = None,
) -> List[Tuple[float, float, float, float]]:
    """Sweep torque limits (τ_max) to study saturation behaviour."""

    results: List[Tuple[float, float, float, float]] = []
    for tau_max in torque_limits:
        if tau_max <= 0:
            raise ValueError("tau_max must be positive")
        limits = JointLimits(torque_max=tau_max)
        hand = RoboticHand(joint_limits=limits)
        field = PrimalLogicField(nx=4, ny=4)
        trajectory = generate_grasp_trajectory(steps, hand.n_fingers, hand.n_joints_per_finger)
        stats = _simulate_statistics(hand, field, trajectory, theta, steps)
        results.append((tau_max, *stats))

    if output_path is not None:
        header = ["tau_max", "mean_torque", "mean_coherence", "saturation_ratio"]
        write_csv(str(output_path), header, results)

    return results


__all__ = [
    "torque_sweep",
    "alpha_sweep",
    "beta_sweep",
    "tau_sweep",
]
