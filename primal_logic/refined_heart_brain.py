"""
Refined Heart-Brain Coupling Models
Implements physiologically realistic HRV and neural entrainment dynamics

Uses only Python standard library (math, random) for compatibility.
"""

import math
import random
from typing import Tuple, Optional, List


class RefinedVanDerPolOscillator:
    """
    Van der Pol cardiac oscillator with dual-frequency physiological drive.

    Includes:
    - Respiratory Sinus Arrhythmia (RSA): ~0.1 Hz
    - Baroreflex oscillations: ~0.04 Hz
    """

    def __init__(
        self,
        mu: float = 1.2,
        omega_base: float = 1.0,
        omega_rsa: float = 0.628,      # 0.1 Hz * 2π
        omega_baro: float = 0.251,     # 0.04 Hz * 2π
        amp_rsa: float = 0.35,         # Increased for visibility
        amp_baro: float = 0.25,        # Increased for visibility
        noise_level: float = 0.02
    ):
        """
        Args:
            mu: Nonlinearity parameter (damping)
            omega_base: Base oscillation frequency
            omega_rsa: Respiratory sinus arrhythmia frequency (rad/s)
            omega_baro: Baroreflex frequency (rad/s)
            amp_rsa: RSA amplitude
            amp_baro: Baroreflex amplitude
            noise_level: Physiological noise amplitude
        """
        self.mu = mu
        self.omega_base = omega_base
        self.omega_rsa = omega_rsa
        self.omega_baro = omega_baro
        self.amp_rsa = amp_rsa
        self.amp_baro = amp_baro
        self.noise_level = noise_level

    def forcing_term(self, t: float) -> float:
        """Dual-frequency physiological drive with noise"""
        rsa = self.amp_rsa * math.sin(self.omega_rsa * t)
        baro = self.amp_baro * math.sin(self.omega_baro * t)
        noise = self.noise_level * random.gauss(0, 1) if self.noise_level > 0 else 0
        return rsa + baro + noise

    def derivatives(
        self,
        state: List[float],
        t: float,
        coupling_input: float = 0.0
    ) -> List[float]:
        """
        Compute derivatives for Van der Pol oscillator.

        Args:
            state: [position, velocity]
            t: Current time
            coupling_input: External coupling from neural system

        Returns:
            [dx/dt, dv/dt]
        """
        x, v = state
        drive = self.forcing_term(t)

        dx = v
        dv = (self.mu * (1 - x**2) * v
              - self.omega_base**2 * x
              + drive
              + coupling_input)

        return [dx, dv]

    def get_instantaneous_frequency(self, velocity: float) -> float:
        """Estimate instantaneous heart rate from velocity"""
        # Maps velocity to normalized heart rate
        return 0.5 + 0.3 * math.tanh(velocity)


class RefinedFitzHughNagumo:
    """
    FitzHugh-Nagumo neural model with slow adaptation variable.

    Three-variable system:
    - v: Fast neural activation (spikes)
    - w: Recovery variable (refractoriness)
    - z: Slow adaptation (long-term entrainment)
    """

    def __init__(
        self,
        a: float = 0.7,
        b: float = 0.8,
        tau: float = 15.0,           # Increased from ~3 to slow down
        tau_adapt: float = 50.0,     # Very slow adaptation timescale
        stimulus_amplitude: float = 0.15,
        stimulus_frequency: float = 0.3,
        adaptation_strength: float = 0.05
    ):
        """
        Args:
            a, b: Standard FHN parameters
            tau: Recovery timescale (higher = slower dynamics)
            tau_adapt: Adaptation timescale (controls entrainment speed)
            stimulus_amplitude: External drive amplitude
            stimulus_frequency: External drive frequency
            adaptation_strength: How strongly adaptation affects dynamics
        """
        self.a = a
        self.b = b
        self.tau = tau
        self.tau_adapt = tau_adapt
        self.stimulus_amplitude = stimulus_amplitude
        self.stimulus_frequency = stimulus_frequency
        self.adaptation_strength = adaptation_strength

    def stimulus(self, t: float) -> float:
        """External stimulus with slow modulation"""
        return self.stimulus_amplitude * math.sin(self.stimulus_frequency * t)

    def derivatives(
        self,
        state: List[float],
        t: float,
        coupling_input: float = 0.0
    ) -> List[float]:
        """
        Compute derivatives for three-variable FHN system.

        Args:
            state: [v, w, z] - activation, recovery, adaptation
            t: Current time
            coupling_input: External coupling from cardiac system

        Returns:
            [dv/dt, dw/dt, dz/dt]
        """
        v, w, z = state
        stim = self.stimulus(t)

        # Fast activation with adaptation feedback
        dv = v - v**3/3 - w + stim + coupling_input - self.adaptation_strength * z

        # Medium-speed recovery
        dw = (v + self.a - self.b * w) / self.tau

        # Very slow adaptation (tracks mean activation)
        dz = (v - z) / self.tau_adapt

        return [dv, dw, dz]

    def get_activity_level(self, v: float, z: float) -> float:
        """Compute normalized brain activity level"""
        # Sigmoid with adaptation-adjusted threshold
        return 1.0 / (1.0 + math.exp(-2.0 * (v - z)))


class RefinedCouplingParameters:
    """
    Frequency-dependent bidirectional coupling between heart and brain.
    """

    def __init__(
        self,
        neural_to_cardiac_gain: float = 0.35,    # Vagal/sympathetic blend
        cardiac_to_neural_gain: float = 0.25,    # Baroreflex strength
        low_freq_weight: float = 0.7,            # Slow baroreflex dominance
        high_freq_weight: float = 0.3,           # Fast RSA contribution
        delay_n2c: float = 0.15,                 # Neural→cardiac delay (sec)
        delay_c2n: float = 0.25                  # Cardiac→neural delay (sec)
    ):
        """
        Args:
            neural_to_cardiac_gain: Brain→heart coupling strength
            cardiac_to_neural_gain: Heart→brain coupling strength
            low_freq_weight: Weight for slow oscillations
            high_freq_weight: Weight for fast oscillations
            delay_n2c: Vagal transmission delay
            delay_c2n: Baroreceptor delay
        """
        self.n2c_gain = neural_to_cardiac_gain
        self.c2n_gain = cardiac_to_neural_gain
        self.low_freq_weight = low_freq_weight
        self.high_freq_weight = high_freq_weight
        self.delay_n2c = delay_n2c
        self.delay_c2n = delay_c2n

    def neural_to_cardiac_coupling(
        self,
        neural_state: float,
        frequency_content: float = 0.5
    ) -> float:
        """
        Compute neural influence on cardiac system.

        Args:
            neural_state: Current neural activation
            frequency_content: 0=low freq, 1=high freq

        Returns:
            Coupling force
        """
        freq_weight = (self.low_freq_weight * (1 - frequency_content) +
                      self.high_freq_weight * frequency_content)
        return self.n2c_gain * freq_weight * neural_state

    def cardiac_to_neural_coupling(
        self,
        cardiac_state: float,
        cardiac_velocity: float
    ) -> float:
        """
        Compute cardiac influence on neural system (baroreflex).

        Args:
            cardiac_state: Current heart position
            cardiac_velocity: Current heart rate change

        Returns:
            Coupling force
        """
        # Baroreflex responds to both position and rate of change
        return self.c2n_gain * (0.6 * cardiac_state + 0.4 * cardiac_velocity)


class RefinedHeartBrainCouplingModel:
    """
    Complete heart-brain coupling system with refined dynamics.

    This model provides physiologically realistic HRV patterns including:
    - Respiratory Sinus Arrhythmia (RSA) at ~0.1 Hz
    - Baroreflex oscillations at ~0.04 Hz
    - Neural entrainment with slow adaptation (~30-45 sec)
    - Frequency-dependent bidirectional coupling
    """

    def __init__(
        self,
        cardiac_model: Optional[RefinedVanDerPolOscillator] = None,
        neural_model: Optional[RefinedFitzHughNagumo] = None,
        coupling: Optional[RefinedCouplingParameters] = None,
        dt: float = 0.001
    ):
        """
        Args:
            cardiac_model: Van der Pol cardiac oscillator instance
            neural_model: FitzHugh-Nagumo neural model instance
            coupling: Coupling parameters instance
            dt: Integration timestep in seconds
        """
        self.cardiac = cardiac_model or RefinedVanDerPolOscillator()
        self.neural = neural_model or RefinedFitzHughNagumo()
        self.coupling = coupling or RefinedCouplingParameters()
        self.dt = dt

        # State: [x_cardiac, v_cardiac, v_neural, w_neural, z_neural]
        self.state = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.time = 0.0
        self.step_count = 0

        # History for delay implementation and coherence computation
        self.cardiac_history: List[float] = []
        self.neural_history: List[float] = []

    def coupled_derivatives(
        self,
        state: List[float],
        t: float
    ) -> List[float]:
        """
        Compute coupled system derivatives.

        Args:
            state: [x_c, v_c, v_n, w_n, z_n] - cardiac (2) + neural (3)
            t: Current time

        Returns:
            Full state derivative vector
        """
        # Unpack state
        x_cardiac, v_cardiac = state[0:2]
        v_neural, w_neural, z_neural = state[2:5]

        # Compute frequency content (simplified - could use FFT for real implementation)
        freq_content = 0.5  # Placeholder - in production, analyze recent history

        # Compute coupling forces
        n2c = self.coupling.neural_to_cardiac_coupling(v_neural, freq_content)
        c2n = self.coupling.cardiac_to_neural_coupling(x_cardiac, v_cardiac)

        # Get derivatives from each subsystem
        cardiac_derivs = self.cardiac.derivatives(
            [x_cardiac, v_cardiac],
            t,
            coupling_input=n2c
        )

        neural_derivs = self.neural.derivatives(
            [v_neural, w_neural, z_neural],
            t,
            coupling_input=c2n
        )

        return cardiac_derivs + neural_derivs

    def step(self) -> List[float]:
        """
        Advance the coupled system by one timestep using RK4 integration.

        Returns:
            Updated state vector
        """
        t = self.time
        y = self.state
        dt = self.dt

        # RK4 integration
        k1 = self.coupled_derivatives(y, t)

        y_k2 = [y[i] + 0.5*dt*k1[i] for i in range(len(y))]
        k2 = self.coupled_derivatives(y_k2, t + 0.5*dt)

        y_k3 = [y[i] + 0.5*dt*k2[i] for i in range(len(y))]
        k3 = self.coupled_derivatives(y_k3, t + 0.5*dt)

        y_k4 = [y[i] + dt*k3[i] for i in range(len(y))]
        k4 = self.coupled_derivatives(y_k4, t + dt)

        # Update state
        self.state = [y[i] + (dt/6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(len(y))]
        self.time += dt
        self.step_count += 1

        # Store history for coherence computation
        self.cardiac_history.append(self.state[0])
        self.neural_history.append(self.state[2])

        # Keep history bounded (last 10000 samples)
        if len(self.cardiac_history) > 10000:
            self.cardiac_history.pop(0)
            self.neural_history.pop(0)

        return self.state

    def simulate(
        self,
        t_span: Tuple[float, float],
        initial_state: Optional[List[float]] = None
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Run coupled simulation over a time span.

        Args:
            t_span: (t_start, t_end)
            initial_state: Optional initial state [x_c, v_c, v_n, w_n, z_n]

        Returns:
            (times, states) as lists
        """
        if initial_state is not None:
            self.state = list(initial_state)

        t_start, t_end = t_span
        self.time = t_start

        n_steps = int((t_end - t_start) / self.dt)
        times = [t_start + i * self.dt for i in range(n_steps)]
        states = [[0.0] * len(self.state) for _ in range(n_steps)]
        states[0] = list(self.state)

        for i in range(1, n_steps):
            states[i] = self.step()

        return times, states

    def get_heart_rate(self) -> float:
        """Get normalized heart rate from current state"""
        velocity = self.state[1]
        return self.cardiac.get_instantaneous_frequency(velocity)

    def get_brain_activity(self) -> float:
        """Get normalized brain activity from current state"""
        v_neural = self.state[2]
        z_neural = self.state[4]
        return self.neural.get_activity_level(v_neural, z_neural)

    def get_cardiac_output(self) -> List[float]:
        """
        Get cardiac actuation signals compatible with Arduino output.

        Returns:
            List[float]: 4-channel output [heart_rate, brain_activity, coherence, combined]
        """
        heart_rate = self.get_heart_rate()
        brain_activity = self.get_brain_activity()

        # Compute coherence from state correlation
        x_cardiac = self.state[0]
        v_neural = self.state[2]
        coherence = abs(math.tanh(x_cardiac * v_neural))

        # Combined signal
        combined = (heart_rate + brain_activity) / 2.0

        return [
            heart_rate,
            brain_activity,
            coherence,
            combined,
        ]

    def compute_coherence(
        self,
        cardiac_state: List[float],
        neural_state: List[float],
        window_size: int = 1000
    ) -> float:
        """
        Compute heart-brain coherence using phase synchronization.

        Args:
            cardiac_state: Recent cardiac position history
            neural_state: Recent neural activation history
            window_size: Number of samples to analyze

        Returns:
            Coherence measure [0, 1]
        """
        if len(cardiac_state) < window_size:
            return 0.0

        # Get recent window
        c_window = cardiac_state[-window_size:]
        n_window = neural_state[-window_size:]

        # Compute means
        c_mean = sum(c_window) / len(c_window)
        n_mean = sum(n_window) / len(n_window)

        # Compute standard deviations
        c_var = sum((x - c_mean)**2 for x in c_window) / len(c_window)
        n_var = sum((x - n_mean)**2 for x in n_window) / len(n_window)
        c_std = math.sqrt(c_var) if c_var > 0 else 1e-8
        n_std = math.sqrt(n_var) if n_var > 0 else 1e-8

        # Compute correlation
        covariance = sum((c_window[i] - c_mean) * (n_window[i] - n_mean) for i in range(len(c_window))) / len(c_window)
        correlation = covariance / (c_std * n_std)

        return abs(correlation)

    def reset(self) -> None:
        """Reset the model to initial conditions"""
        self.state = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.time = 0.0
        self.step_count = 0
        self.cardiac_history = []
        self.neural_history = []


__all__ = [
    "RefinedVanDerPolOscillator",
    "RefinedFitzHughNagumo",
    "RefinedCouplingParameters",
    "RefinedHeartBrainCouplingModel",
]
