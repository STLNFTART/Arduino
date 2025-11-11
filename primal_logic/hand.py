"""Robotic hand model composed of tendons and joints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .adaptive import adaptive_alpha
from .constants import (
    ALPHA_DEFAULT,
    BETA_DEFAULT,
    DEFAULT_FINGERS,
    DT,
    HAND_DAMPING,
    HAND_MASS,
    JOINTS_PER_FINGER,
    LAMBDA_DEFAULT,
)
from .memory import ExponentialMemoryKernel, RecursivePlanckMemoryKernel
from .serial_bridge import SerialHandBridge
from .utils import safe_clip


@dataclass
class JointLimits:
    """Physical limits for a joint."""

    angle_min: float = 0.0  # [rad]
    angle_max: float = 1.2  # [rad]
    vel_max: float = 8.0  # [rad/s]
    torque_max: float = 0.7  # [N*m]


@dataclass
class JointState:
    """State of a joint, tracked in simulation units."""

    angle: float = 0.0  # [rad]
    velocity: float = 0.0  # [rad/s]


@dataclass
class HandJointController:
    """PD controller with memory augmentation."""

    limits: JointLimits
    mem_kernel: ExponentialMemoryKernel
    alpha_base: float = ALPHA_DEFAULT
    kp: float = 12.0
    kd: float = 0.6
    last_tau: float = 0.0

    def compute_torque(
        self,
        desired_angle: float,
        state: JointState,
        theta: float,
        coherence: float,
        step: int,
    ) -> float:
        """Compute torque using PD and adaptive memory contributions."""

        avg_energy = abs(state.angle) + abs(state.velocity)
        alpha = adaptive_alpha(
            step,
            avg_energy=avg_energy,
            quantum_coherence=coherence,
            alpha_base=self.alpha_base,
        )

        error = desired_angle - state.angle
        d_error = -state.velocity

        u_mem = self.mem_kernel.update(theta=theta, error=error, step_index=step)
        u_pd = alpha * (self.kp * error + self.kd * d_error)

        tau = safe_clip(u_pd + u_mem, -self.limits.torque_max, self.limits.torque_max)
        self.last_tau = tau
        return tau


@dataclass
class RoboticHand:
    """Robotic hand composed of multiple fingers and tendon-driven joints."""

    n_fingers: int = DEFAULT_FINGERS
    n_joints_per_finger: int = JOINTS_PER_FINGER
    dt: float = DT
    joint_limits: JointLimits = field(default_factory=JointLimits)
    mass: float = HAND_MASS
    damping: float = HAND_DAMPING
    alpha_base: float = ALPHA_DEFAULT
    beta_gain: float = BETA_DEFAULT
    bridge: Optional[SerialHandBridge] = None
    memory_mode: str = "exponential"
    rpo_alpha: float = 0.4  # ensures alpha * dt < 1.0 for dt = 1e-3

    states: List[List[JointState]] = field(init=False)
    controllers: List[List[HandJointController]] = field(init=False)

    def __post_init__(self) -> None:
        if self.alpha_base <= 0:
            raise ValueError("alpha_base must be positive for controller stability")
        if self.beta_gain <= 0:
            raise ValueError("beta_gain must be positive for memory dynamics")

        if self.memory_mode not in {"exponential", "recursive_planck"}:
            raise ValueError("memory_mode must be 'exponential' or 'recursive_planck'")

        if self.memory_mode == "recursive_planck" and not (0 < self.rpo_alpha * self.dt < 1):
            raise ValueError("rpo_alpha must satisfy 0 < rpo_alpha * dt < 1 for stability")

        self.states = [
            [JointState() for _ in range(self.n_joints_per_finger)] for _ in range(self.n_fingers)
        ]
        self.controllers = []
        for _finger in range(self.n_fingers):
            finger_ctrls: List[HandJointController] = []
            for _ in range(self.n_joints_per_finger):
                if self.memory_mode == "exponential":
                    kernel = ExponentialMemoryKernel(lam=LAMBDA_DEFAULT, gain=self.beta_gain)
                else:
                    kernel = RecursivePlanckMemoryKernel(
                        alpha=self.rpo_alpha,
                        lam=LAMBDA_DEFAULT,
                        lightfoot=self.alpha_base,
                    )
                finger_ctrls.append(
                    HandJointController(
                        limits=self.joint_limits,
                        mem_kernel=kernel,
                        alpha_base=self.alpha_base,
                    )
                )
            self.controllers.append(finger_ctrls)

    def step(self, desired_angles: List[List[float]], theta: float, coherence: float, step: int) -> None:
        """Advance the hand dynamics by one time step."""

        for finger in range(self.n_fingers):
            for joint in range(self.n_joints_per_finger):
                state: JointState = self.states[finger][joint]
                controller = self.controllers[finger][joint]
                target_angle = float(desired_angles[finger][joint])
                torque = controller.compute_torque(target_angle, state, theta, coherence, step)

                acceleration = (torque - self.damping * state.velocity) / self.mass
                state.velocity += acceleration * self.dt
                state.angle += state.velocity * self.dt

                state.velocity = safe_clip(state.velocity, -self.joint_limits.vel_max, self.joint_limits.vel_max)
                state.angle = safe_clip(state.angle, self.joint_limits.angle_min, self.joint_limits.angle_max)

        if self.bridge is not None:
            self.apply_torques()

    def get_angles(self) -> List[List[float]]:
        """Return a copy of all joint angles for downstream analysis."""

        return [[state.angle for state in finger] for finger in self.states]

    def get_torques(self) -> List[List[float]]:
        """Return the latest torques from each controller."""

        return [[ctrl.last_tau for ctrl in finger] for finger in self.controllers]

    def apply_torques(self) -> None:
        """Send torques to the hardware bridge if available."""

        if self.bridge is not None:
            self.bridge.send(self.get_torques())
