"""Robotic hand model composed of tendons and joints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .adaptive import adaptive_alpha
from .constants import (
    DEFAULT_FINGERS,
    DT,
    HAND_DAMPING,
    HAND_MASS,
    JOINTS_PER_FINGER,
    LAMBDA_DEFAULT,
)
from .memory import ExponentialMemoryKernel
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
        alpha = adaptive_alpha(step, avg_energy=avg_energy, quantum_coherence=coherence)

        error = desired_angle - state.angle
        d_error = -state.velocity

        u_mem = self.mem_kernel.update(theta=theta, error=error)
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
    bridge: Optional[SerialHandBridge] = None

    states: np.ndarray = field(init=False)
    controllers: List[List[HandJointController]] = field(init=False)

    def __post_init__(self) -> None:
        self.states = np.empty((self.n_fingers, self.n_joints_per_finger), dtype=object)
        self.controllers = []
        for finger in range(self.n_fingers):
            finger_ctrls: List[HandJointController] = []
            for _ in range(self.n_joints_per_finger):
                finger_ctrls.append(
                    HandJointController(
                        limits=self.joint_limits,
                        mem_kernel=ExponentialMemoryKernel(lam=LAMBDA_DEFAULT, gain=0.8),
                    )
                )
            self.controllers.append(finger_ctrls)

        for finger in range(self.n_fingers):
            for joint in range(self.n_joints_per_finger):
                self.states[finger, joint] = JointState()

    def step(self, desired_angles: np.ndarray, theta: float, coherence: float, step: int) -> None:
        """Advance the hand dynamics by one time step."""
        for finger in range(self.n_fingers):
            for joint in range(self.n_joints_per_finger):
                state: JointState = self.states[finger, joint]
                controller = self.controllers[finger][joint]
                target_angle = float(desired_angles[finger, joint])
                torque = controller.compute_torque(target_angle, state, theta, coherence, step)

                acceleration = (torque - self.damping * state.velocity) / self.mass
                state.velocity += acceleration * self.dt
                state.angle += state.velocity * self.dt

                state.velocity = safe_clip(state.velocity, -self.joint_limits.vel_max, self.joint_limits.vel_max)
                state.angle = safe_clip(state.angle, self.joint_limits.angle_min, self.joint_limits.angle_max)

        if self.bridge is not None:
            self.apply_torques()

    def get_angles(self) -> np.ndarray:
        """Return a copy of all joint angles for downstream analysis."""
        angles = np.zeros((self.n_fingers, self.n_joints_per_finger))
        for finger in range(self.n_fingers):
            for joint in range(self.n_joints_per_finger):
                state: JointState = self.states[finger, joint]
                angles[finger, joint] = state.angle
        return angles

    def get_torques(self) -> np.ndarray:
        """Return the latest torques from each controller."""
        torques = np.zeros((self.n_fingers, self.n_joints_per_finger))
        for finger in range(self.n_fingers):
            for joint in range(self.n_joints_per_finger):
                torques[finger, joint] = self.controllers[finger][joint].last_tau
        return torques

    def apply_torques(self) -> None:
        """Send torques to the hardware bridge if available."""
        if self.bridge is not None:
            self.bridge.send(self.get_torques())
