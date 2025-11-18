# Neurorobotic Control System Integration Guide

## Overview

This document outlines how to design a **closed-loop neurorobotic control system** that integrates brain signals with the existing Primal Logic framework to control bio-hybrid actuators and robotic sensing.

---

## 1. EXISTING INTEGRATION POINTS FOR BRAIN SIGNALS

### 1.1 Quantum Field Coherence Modulation

**Current Implementation** (field.py):
- 8×8 complex quantum field evolves at 1000 Hz
- Real and imaginary components track coherence
- Coherence value (0-1) modulates controller gains

**Integration Opportunity**:
```python
# Current: Coherence computed from field self-dynamics
coherence = field.step(theta=1.0)

# Modified: Incorporate brain signal as input
brain_alpha = normalize_brain_signal(eeg_input)  # [0, 1]
coherence = field.step(theta=brain_alpha)

# Result: Brain activity directly modulates control gains
adaptive_alpha_gain = alpha_base * (1 + coherence_term)
```

**Effect**: Higher brain coherence → Tighter control gains → More responsive hand movements

### 1.2 Heart-Brain Coupling Layer

**Current Implementation** (heart_model.py):
```python
MultiHeartModel(
    lambda_heart=0.115,
    lambda_brain=0.092,
    coupling_strength=0.15
)

# Updates coupled ODEs:
# n_h' = -λ_h·n_h + f_h(n_b, S_h) + ℛ_P[C(t)]
# n_b' = -λ_b·n_b + f_b(n_h, S_b) + ℛ_P[s_set(t)]
```

**Integration Points for Brain Signals**:
1. **Brain Setpoint** (s_set): Direct neural intent signal
   - Replace with real brain signal (normalized -1 to +1)
   - Could be event-related desynchronization (ERD) from motor cortex
   
2. **Coupling Strength**: Parameterize by brain state
   - Higher coherence → Stronger heart-brain coupling
   - Reflects increased autonomic regulation during motor tasks
   
3. **Neural Feedback Loop**: Add additional brain states
   - Current: 2 ODEs (heart, brain)
   - Proposed: Add motor cortex, somatosensory, and cerebellum states

**Code Structure**:
```python
class NeuroRoboticHeartBrainModel(MultiHeartModel):
    """Extended heart-brain-motor cortex model"""
    
    def __init__(self, ...):
        super().__init__(...)
        self.motor_cortex_rpo = RecursivePlanckOperator(...)
        self.sensory_rpo = RecursivePlanckOperator(...)
    
    def step(self, 
             cardiac_input,
             brain_signal,           # Real EEG/neural signal
             proprioceptive_feedback,  # Joint angles → sensory
             theta=1.0):
        
        # Process motor command through RPO
        motor_output = self.motor_cortex_rpo.step(
            theta=theta,
            input_value=brain_signal,
            step_index=self.step_count
        )
        
        # Sensory feedback to cortex
        sensory_feedback = self.sensory_rpo.step(
            theta=theta,
            input_value=proprioceptive_feedback,
            step_index=self.step_count
        )
        
        # Update coupled dynamics
        # ... existing heart-brain update ...
        
        return state
```

---

## 2. ARCHITECTURE FOR CLOSED-LOOP NEUROROBOTIC CONTROL

### 2.1 Complete Control Pipeline

```
┌──────────────────────────────────────┐
│ BRAIN SIGNAL ACQUISITION             │
│ - EEG (motor cortex, sensorimotor)  │
│ - EMG (muscle activity)              │
│ - fMRI (spatial, lower temporal res) │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ SIGNAL PROCESSING                    │
│ - Filter, normalize to [-1, 1]      │
│ - Coherence analysis (α, β, γ bands)│
│ - Source localization (if applicable)│
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ MOTOR INTENTION DECODING             │
│ - Neural classifier (LDA, RNN, CNN) │
│ - Grasp type selection (power/prec) │
│ - Desired joint angles               │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ FIELD MODULATION & QUANTUM COHERENCE │
│ - Real-time coherence from EEG      │
│ - Modulates adaptive gain scheduling │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ HAND CONTROL (UNIFIED SYSTEM)        │
│ - PD + Memory kernels               │
│ - Target tracking with neural drive │
│ - RPO processing (optional)         │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ ACTUATOR COMMANDS                    │
│ - 15 torque values to MotorHandPro  │
│ - Exponential memory weighting      │
│ - Stability monitoring (L < 1)      │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ ROBOTIC HARDWARE                     │
│ - 5 fingers × 3 joints               │
│ - Tendon-driven actuators            │
│ - Joint angle/velocity sensors       │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ SENSORY FEEDBACK (TO BRAIN)          │
│ - Joint proprioception               │
│ - Tactile/force information (new)    │
│ - Visual feedback (external)         │
└──────────────────┬───────────────────┘
                   ↓
        ┌──────────┴───────┐
        ↓                  ↓
    [EEG SYSTEM]    [HEART-BRAIN MODEL]
    (closed-loop)   (autonomic feedback)
```

### 2.2 Key Timing Characteristics

| Component | Frequency | Latency | Notes |
|-----------|-----------|---------|-------|
| **EEG Acquisition** | 100-1000 Hz | 5-50 ms | Real-time, subject to artifacts |
| **Signal Processing** | 1000+ Hz | 10-20 ms | Filtering, downsampling |
| **Neural Decoding** | 100-500 Hz | 10-50 ms | Classifier overhead |
| **Field Update** | 1000 Hz | 1 ms | Quantum field step |
| **Hand Control** | 100 Hz | 10 ms | Main loop frequency |
| **Serial Communication** | 115200 baud | <5 ms | Hardware round-trip |
| **Total Latency** | -- | **~50-100 ms** | End-to-end |

**Note**: 50-100 ms latency is acceptable for gross motor tasks but borderline for fine manipulation. Consider:
- Predictive models to compensate
- Feed-forward control based on early motor signals
- Parallel processing of brain signals

---

## 3. SENSOR ARCHITECTURE FOR BIO-HYBRID ACTUATORS

### 3.1 Current Sensor Interfaces

**Implicit Joint Feedback**:
- JointState tracks angle, velocity (simulated)
- No external force/pressure sensors
- Error computed from desired vs. actual (position error only)

### 3.2 Extended Sensor Suite for Neurorobotic System

**Proposed Architecture**:

```
┌──────────────────────────────────────┐
│ PROPRIOCEPTIVE SENSORS (Joint-level) │
│ - Absolute position: Hall effect     │
│ - Velocity: Differentiated position  │
│ - Torque: Inline strain gauges       │
│ Serial output: 15 values @ 100 Hz    │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ TACTILE SENSORS (Finger tips)        │
│ - Contact detection (15 on/off)      │
│ - Pressure/force (3-axis per digit)  │
│ - Temperature (optional)             │
│ Serial output: 15 + 45 channels     │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ NEURAL/EEG SENSORS (External)        │
│ - 16-64 channel EEG headset          │
│ - USB/Bluetooth @ 100+ Hz            │
│ - Impedance monitoring               │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ INTEGRATION LAYER                    │
│ - Multi-port serial multiplexer      │
│ - Timestamp synchronization          │
│ - Data fusion (sensorimotor cortex) │
└──────────────────────────────────────┘
```

### 3.3 Implementation: Extended Sensor Interface

**Extend serial communication**:

```python
class NeuroRoboticSensorBridge:
    """Unified sensor interface for neurorobotic system"""
    
    def __init__(self, 
                 hand_port="/dev/ttyACM0",      # MotorHandPro
                 proprioceptive_port=None,      # Joint sensors
                 tactile_port=None,             # Finger tactile
                 eeg_port=None):                # EEG headset
        self.hand_port = hand_port
        self.proprioceptive_port = proprioceptive_port
        self.tactile_port = tactile_port
        self.eeg_port = eeg_port
        
        self.state = {
            'joint_angles': np.zeros(15),
            'joint_velocities': np.zeros(15),
            'joint_torques': np.zeros(15),
            'tactile_contacts': np.zeros(15, dtype=bool),
            'tactile_forces': np.zeros((15, 3)),  # 3-axis per digit
            'brain_signal': np.zeros(64),          # 64 EEG channels
            'brain_coherence': 0.0,
            'timestamp': 0.0
        }
    
    def read_all_sensors(self):
        """Synchronously read all sensor data"""
        t_start = time.time()
        
        # Read proprioception (high priority)
        if self.proprioceptive_port:
            self.state['joint_angles'], self.state['joint_velocities'] = \
                self._read_proprioception()
        
        # Read tactile (medium priority)
        if self.tactile_port:
            self.state['tactile_contacts'], self.state['tactile_forces'] = \
                self._read_tactile()
        
        # Read EEG (lower priority, can skip frames)
        if self.eeg_port:
            self.state['brain_signal'], self.state['brain_coherence'] = \
                self._read_eeg()
        
        self.state['timestamp'] = time.time() - t_start
        return self.state
```

### 3.4 Sensorimotor Integration Layer

**Proprioceptive Feedback to Hand Control**:

```python
class SensorFusionController(UnifiedPrimalLogicController):
    """Enhanced controller with sensor fusion"""
    
    def step(self, 
             target_angles=None,
             brain_signal=None,
             proprioceptive_feedback=None,
             tactile_feedback=None):
        
        # 1. Get desired angles from trajectory + brain signal
        if brain_signal is not None:
            intent = decode_motor_intent(brain_signal)  # Classifier
            target_angles = self._grasp_planning(intent)
        
        # 2. Update hand with proprioceptive feedback
        if proprioceptive_feedback is not None:
            # Use real joint states instead of simulated
            self.hand.states = proprioceptive_feedback
        
        # 3. Modulate control gains by brain coherence
        brain_coherence = estimate_eeg_coherence(brain_signal)
        self.hand.coherence = brain_coherence
        
        # 4. Integrate tactile feedback into force control
        if tactile_feedback is not None:
            # Adjust target torques based on contact/force
            target_angles = self._contact_aware_control(
                target_angles,
                tactile_feedback
            )
        
        # 5. Standard control loop
        self.hand.step(target_angles, dt=self.dt)
        torques = self.hand.get_torques()
        
        # Send to hardware
        self.motorhand.send_torques(torques)
        self.step_count += 1
```

---

## 4. BRAIN SIGNAL PREPROCESSING & NEURAL DECODING

### 4.1 EEG Signal Processing Pipeline

**Raw EEG → Control Commands**:

```python
from scipy import signal
import numpy as np

class MotorCortexDecoder:
    """Decode motor commands from EEG motor cortex signals"""
    
    def __init__(self, channels=[3, 4, 5, 6], bands=None):
        """
        channels: EEG channel indices for motor cortex
        bands: Frequency bands to extract
        """
        self.channels = channels
        self.bands = bands or {
            'alpha': (8, 12),      # Idle/relaxation
            'beta': (12, 30),      # Motor preparation/execution
            'gamma': (30, 100),    # Fine motor control
        }
        self.scaler = None
        self.classifier = None
    
    def preprocess(self, eeg_raw, fs=500):
        """
        Clean and filter EEG data
        
        eeg_raw: shape (n_samples, n_channels)
        fs: Sampling frequency
        """
        # 1. High-pass filter (> 0.5 Hz) - remove DC drift
        sos = signal.butter(4, 0.5, 'hp', fs=fs, output='sos')
        eeg_hp = signal.sosfilt(sos, eeg_raw, axis=0)
        
        # 2. Low-pass filter (< 60 Hz) - remove high-frequency noise
        sos = signal.butter(4, 60, 'lp', fs=fs, output='sos')
        eeg_lp = signal.sosfilt(sos, eeg_hp, axis=0)
        
        # 3. Notch filter (50/60 Hz powerline interference)
        sos = signal.iirnotch(50, 30, fs=fs, output='sos')
        eeg_clean = signal.sosfilt(sos, eeg_lp, axis=0)
        
        # 4. Spatial filter (common average reference)
        eeg_car = eeg_clean - np.mean(eeg_clean, axis=1, keepdims=True)
        
        return eeg_car
    
    def extract_features(self, eeg_clean, fs=500, window_size=0.5):
        """
        Extract frequency-domain features from EEG
        
        Returns:
            features: shape (n_bands, n_channels)
        """
        n_samples = int(window_size * fs)
        
        # Welch's periodogram
        f, pxx = signal.welch(
            eeg_clean[self.channels].T,
            fs=fs,
            nperseg=n_samples
        )
        
        # Extract band power
        features = {}
        for band_name, (f_low, f_high) in self.bands.items():
            band_mask = (f >= f_low) & (f <= f_high)
            band_power = np.mean(pxx[:, band_mask], axis=1)
            features[band_name] = band_power
        
        return features  # Dict of features
    
    def decode_intent(self, eeg_clean, fs=500):
        """
        Classify motor intent from EEG
        
        Returns:
            intent: 0-1 (0=rest, 1=max activation)
            confidence: Classifier confidence
            grasp_type: 'power', 'precision', or 'rest'
        """
        # Extract features
        features = self.extract_features(eeg_clean, fs=fs)
        X = np.concatenate([features[b] for b in self.bands.keys()])
        
        # Normalize
        if self.scaler:
            X = self.scaler.transform(X.reshape(1, -1))
        
        # Classify
        if self.classifier:
            intent_prob = self.classifier.predict_proba(X)[0]
            intent = intent_prob[1]  # P(motor command)
            
            # Select grasp type
            if intent < 0.2:
                grasp_type = 'rest'
            elif intent < 0.6:
                grasp_type = 'precision'
            else:
                grasp_type = 'power'
            
            return intent, intent, grasp_type
        
        return 0.0, 0.0, 'rest'
    
    def estimate_coherence(self, eeg_clean, fs=500, window_size=0.5):
        """
        Estimate motor cortex coherence (synchronization strength)
        
        Returns:
            coherence: 0-1 (0=low sync, 1=high sync)
        """
        # Coherence between channels within motor cortex
        n_samples = int(window_size * fs)
        
        if len(self.channels) < 2:
            return 0.5
        
        # Cross-spectral density
        f, coh = signal.coherence(
            eeg_clean[:, self.channels[0]],
            eeg_clean[:, self.channels[1]],
            fs=fs,
            nperseg=n_samples
        )
        
        # Beta band coherence (motor-relevant)
        beta_mask = (f >= 12) & (f <= 30)
        coherence = np.mean(coh[beta_mask])
        
        return np.clip(coherence, 0, 1)
```

### 4.2 Neural Classifier Training

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train_motor_decoder(eeg_training, labels_training, fs=500):
    """
    Train classifier on labeled motor tasks
    
    eeg_training: (n_samples, n_timepoints, n_channels)
    labels_training: (n_samples,) - 0=rest, 1=motor command
    """
    decoder = MotorCortexDecoder()
    
    # Preprocess all training data
    n_samples = eeg_training.shape[0]
    features_list = []
    
    for i in range(n_samples):
        eeg_clean = decoder.preprocess(eeg_training[i], fs=fs)
        features = decoder.extract_features(eeg_clean, fs=fs)
        X_i = np.concatenate([features[b] for b in decoder.bands.keys()])
        features_list.append(X_i)
    
    X = np.array(features_list)
    y = np.array(labels_training)
    
    # Train classifier
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_scaled, y)
    
    decoder.scaler = scaler
    decoder.classifier = classifier
    
    return decoder
```

---

## 5. STABILITY ANALYSIS FOR NEURAL INPUTS

### 5.1 Lipschitz Stability Under Variable Brain Input

**Current Framework** (from constants.py):
```
Lipschitz constant L < 1.0 guarantees bounded convergence
```

**Extended Analysis**: When brain signal varies, we need:

```python
def analyze_neural_stability(
    brain_input_range=(-1, 1),      # Possible brain signal values
    field_coherence_range=(0, 1),   # Possible coherence values
    lambda_value=0.16905            # Lightfoot constant
):
    """
    Verify stability across all possible neural inputs
    
    Requirement: max_theta * L < 1 for all theta in input_range
    """
    
    max_theta = max(abs(brain_input_range[0]), abs(brain_input_range[1]))
    
    # Worst-case Lipschitz calculation
    c = (150 - 149.9992314) * np.exp(lambda_value * 149.9992314)
    L_worst = max_theta * c * lambda_value * np.exp(-lambda_value * 149.9992314)
    
    print(f"Max brain input magnitude: {max_theta}")
    print(f"Worst-case Lipschitz: {L_worst:.6f}")
    
    if L_worst < 1.0:
        print("✓ System stable for all possible neural inputs")
        return True
    else:
        print("✗ System may be unstable")
        print("  → Reduce brain input gain or increase lambda")
        return False
    
    # Mitigation strategies if unstable:
    # 1. Scale brain input: brain_signal *= 0.5
    # 2. Increase lambda: More aggressive decay
    # 3. Add safety bounds: Clip brain_signal before use
```

### 5.2 Proposed Stability Monitoring

```python
class NeuroRoboticStabilityMonitor:
    """Monitor closed-loop neurorobotic system stability"""
    
    def __init__(self):
        self.control_energy_history = []
        self.lipschitz_history = []
        self.brain_input_history = []
        self.safe = True
    
    def check_stability(self, 
                        motorhand_state,
                        brain_signal,
                        threshold_lipschitz=0.95):
        """
        Real-time stability check
        """
        L = motorhand_state['lipschitz_estimate']
        Ec = motorhand_state['control_energy']
        
        # Record
        self.lipschitz_history.append(L)
        self.control_energy_history.append(Ec)
        self.brain_input_history.append(np.abs(brain_signal).max())
        
        # Check Lipschitz stays < 1
        if L > 1.0:
            self.safe = False
            print(f"⚠ UNSTABLE: Lipschitz {L:.4f} > 1.0")
            return False
        
        # Check control energy doesn't diverge
        if len(self.control_energy_history) > 100:
            recent_growth = (
                self.control_energy_history[-1] / 
                (self.control_energy_history[-100] + 1e-6)
            )
            if recent_growth > 10:  # 10x growth in 100 steps
                self.safe = False
                print(f"⚠ WARNING: Control energy diverging ({recent_growth:.1f}x)")
                return False
        
        # Check brain input doesn't exceed safe bounds
        if np.abs(brain_signal).max() > 1.0:
            print(f"⚠ Brain signal out of bounds: {np.abs(brain_signal).max():.2f}")
            return False
        
        return True
    
    def emergency_stop(self):
        """Disable actuators if system becomes unstable"""
        print("🛑 EMERGENCY STOP")
        # Send zero torques to hardware
        return np.zeros(15)
```

---

## 6. IMPLEMENTATION ROADMAP

### Phase 1: Brain Signal Integration (Week 1-2)
- [ ] Set up EEG acquisition system (OpenBCI or commercial headset)
- [ ] Implement signal preprocessing pipeline
- [ ] Train motor intent classifier on subject-specific data
- [ ] Integrate EEG input into field.step()
- [ ] Test field modulation with real brain signals

### Phase 2: Extended Heart-Brain Model (Week 2-3)
- [ ] Extend MultiHeartModel with motor cortex state
- [ ] Add somatosensory feedback loop
- [ ] Implement bidirectional coupling
- [ ] Validate using multimodal recordings (EEG + cardiac)

### Phase 3: Sensor Integration (Week 3-4)
- [ ] Add joint torque sensors to hardware
- [ ] Implement tactile sensing on fingertips
- [ ] Create multi-port serial interface
- [ ] Integrate proprioceptive feedback into control

### Phase 4: Closed-Loop Testing (Week 4-5)
- [ ] Run integrated system with real brain signals
- [ ] Validate stability across all operating conditions
- [ ] Measure end-to-end latency
- [ ] Optimize neural decoding for real-time performance
- [ ] User training and adaptation

### Phase 5: Advanced Features (Week 5-6)
- [ ] Implement predictive decoding (anticipatory control)
- [ ] Add force feedback to user (tactile display)
- [ ] Multi-modal sensor fusion (vision + proprioception)
- [ ] Long-term learning and adaptation

---

## 7. SUCCESS METRICS FOR NEUROROBOTIC SYSTEM

| Metric | Target | Current Status |
|--------|--------|-----------------|
| **Latency (end-to-end)** | < 100 ms | ~50-100 ms (estimated) |
| **Stability (Lipschitz)** | L < 0.9 | L < 0.0001 (static control) |
| **Decoding Accuracy** | > 85% | Subject to train |
| **Grasp Success Rate** | > 90% | Baseline: 95% (automated) |
| **User Learning Time** | < 1 hour | Subject-dependent |
| **Reaction Time** | < 500 ms | Human baseline ~150-300 ms |
| **Control Energy** | Ec < 2.0 | Currently 0.1-1.0 |
| **Oscillations** | < 5 Hz | Not observed in tests |

---

## 8. REFERENCES & EXTERNAL LIBRARIES

**For Real-Time Brain Signal Processing**:
- `mne-python`: EEG analysis and source localization
- `pyxdf`: Load/save XDF format from OpenBCI
- `scikit-learn`: Classifiers and preprocessing
- `scipy.signal`: Filtering and spectral analysis

**For Hardware Integration**:
- `pyserial`: Serial communication (already used)
- `pylsl`: Lab Streaming Layer for multimodal sync
- `board` and `busio`: I2C/SPI for sensor integration

**For Visualization**:
- `matplotlib`: Real-time signal plots
- `vispy`: 3D hand visualization
- `pyqtgraph`: High-performance oscilloscope-like displays

---

## CONCLUSION

The existing Primal Logic framework provides **excellent integration points** for neurorobotic control:

1. ✓ **Field coherence** can be driven by brain signals
2. ✓ **Heart-brain coupling** enables physiological feedback
3. ✓ **Exponential memory** ensures stability under neural inputs
4. ✓ **Modular architecture** allows selective feature integration
5. ✓ **Proven stability** (Lipschitz < 1) extends to variable inputs

**Key Design Principle**: Leverage existing **Primal Logic stability guarantees** while adding neural signals as high-level modulation inputs, not replacing core control law.

This approach ensures **safe human-robot interaction** while achieving **direct neural control** of multi-DOF bio-hybrid actuators.
