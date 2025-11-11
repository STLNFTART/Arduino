# Microprocessor-Heart-Arduino Integration

This document describes the complete integration pipeline linking the microprocessor (Recursive Planck Operator), multi-heart physiological model, and Arduino hardware.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. MICROPROCESSOR (RPO)                                        │
│     ├─ RecursivePlanckOperator (primal_logic/rpo.py)           │
│     ├─ Processes cardiac and brain signals                     │
│     ├─ Uses Donte's constant (𝓓 ≈ 149.999)                    │
│     └─ Uses Lightfoot's constant (𝓛 ∈ [0.54, 0.56])          │
│                    ↓                                            │
│  2. MULTI-HEART MODEL (primal_logic/heart_model.py)             │
│     ├─ Heart-brain-immune coupling equations:                  │
│     │   • n_h'(t) = −λ_h n_h + f_h(n_b, S_h) + ℛ_P[C(t)]      │
│     │   • n_b'(t) = −λ_b n_b + f_b(n_h, S_b) + ℛ_P[s_set(t)]  │
│     ├─ Integrates two RPO instances (heart + brain)            │
│     └─ Generates physiological signals (HR, activity, etc.)    │
│                    ↓                                            │
│  3. ARDUINO BRIDGE (primal_logic/heart_arduino_bridge.py)       │
│     ├─ HeartArduinoBridge: Serial communication                │
│     ├─ ProcessorHeartArduinoLink: Unified orchestration        │
│     ├─ Formats signals as CSV for Arduino                      │
│     └─ Streams at configurable intervals                       │
│                    ↓                                            │
│  4. ARDUINO HARDWARE                                            │
│     ├─ USB Serial (115200 baud, /dev/ttyACM0)                  │
│     ├─ Receives 4-channel cardiac output:                      │
│     │   • Channel 0: Heart rate (normalized)                   │
│     │   • Channel 1: Brain activity (normalized)               │
│     │   • Channel 2: Heart-brain coherence                     │
│     │   • Channel 3: Combined signal                           │
│     └─ Drives actuators, LEDs, or biofeedback systems          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Microprocessor: Recursive Planck Operator (RPO)

**File**: `primal_logic/rpo.py`

The RPO is the computational "microprocessor" that bridges energetic and informational domains:

```python
from primal_logic import RecursivePlanckOperator

# Initialize RPO with Donte and Lightfoot constants
rpo = RecursivePlanckOperator(
    alpha=0.4,           # Damping coefficient
    lightfoot=0.55,      # Neural-mechanical coupling
    donte=149.9992314,   # Energy-information scaling
    dt=0.001,            # 1ms timestep
)

# Process signal at each timestep
output = rpo.step(theta=1.0, input_value=cardiac_signal, step_index=k)
```

**Key Parameters**:
- **α (alpha)**: Damping coefficient from Volterra kernel. Must satisfy `0 < α·Δt < 1` for stability.
- **𝓛 (Lightfoot)**: Blends neural and mechanical domains, clipped to `[0.54, 0.56]`.
- **𝓓 (Donte)**: Scales effective Planck constant: `h_eff = h / 𝓓`.
- **β_P**: Recursive coupling coefficient: `β_P = 𝓛 / (1 + λ)`.

### 2. Multi-Heart Model

**File**: `primal_logic/heart_model.py`

Implements the Quantro-Primal physiological equations with two coupled RPO instances:

```python
from primal_logic import MultiHeartModel

# Initialize heart-brain model
heart_model = MultiHeartModel(
    lambda_heart=0.115,      # Heart decay rate
    lambda_brain=0.092,      # Brain decay rate
    coupling_strength=0.15,  # Heart-brain coupling
    rpo_alpha=0.4,           # RPO damping
)

# Step the model
state = heart_model.step(
    cardiac_input=0.5,       # External cardiac input C(t)
    brain_setpoint=1.0,      # Brain setpoint s_set(t)
    theta=1.0,               # Command envelope
)

# Get physiological outputs
heart_rate = heart_model.get_heart_rate()           # 0-1 normalized
brain_activity = heart_model.get_brain_activity()   # -1 to 1
cardiac_output = heart_model.get_cardiac_output()   # 4-channel array
```

**State Variables**:
- **n_heart**: Heart neural potential (affected by brain feedback + RPO)
- **n_brain**: Brain neural potential (affected by heart feedback + RPO)
- **s_heart**: Heart sensory feedback
- **s_brain**: Brain sensory feedback

**Coupling Functions**:
- **f_heart(n_brain, s_heart)**: Vagal/sympathetic drive from brain to heart
- **f_brain(n_heart, s_brain)**: Baroreceptor feedback from heart to brain

### 3. Arduino Integration Bridge

**File**: `primal_logic/heart_arduino_bridge.py`

Two classes handle Arduino communication:

#### HeartArduinoBridge
Basic serial interface for heart signals:

```python
from primal_logic import HeartArduinoBridge, MultiHeartModel

bridge = HeartArduinoBridge(
    port="/dev/ttyACM0",  # Arduino USB port
    baud=115200,          # Communication speed
    normalize=True,       # Normalize to [0,1]
)

# Send signals to Arduino
bridge.send_heart_signals(heart_model)
```

#### ProcessorHeartArduinoLink
Unified orchestration layer:

```python
from primal_logic import ProcessorHeartArduinoLink

link = ProcessorHeartArduinoLink(
    heart_model=heart_model,
    arduino_bridge=bridge,
    send_interval=10,  # Send every 10 steps (~10ms)
)

# Update entire pipeline
link.update(
    cardiac_input=0.6,
    brain_setpoint=1.0,
    theta=1.0,
)

# Query state
state = link.get_state()
# Returns: {'n_heart': ..., 'n_brain': ..., 'heart_rate': ..., 'brain_activity': ...}
```

## Arduino Output Format

The bridge sends CSV-formatted data to Arduino via serial:

```
0.7234,0.4512,0.3267,0.5873\n
```

**Channel assignments**:
- **Channel 0**: Normalized heart rate (0-1 range, ~60-120 bpm)
- **Channel 1**: Brain activity level (-1 to 1, tanh-normalized)
- **Channel 2**: Heart-brain coherence (0-1, based on n_h × n_b)
- **Channel 3**: Combined signal (average of heart rate + brain activity)

### Example Arduino Sketch

```cpp
// Arduino code to receive cardiac signals
void setup() {
  Serial.begin(115200);
  pinMode(9, OUTPUT);   // Heart rate LED
  pinMode(10, OUTPUT);  // Brain activity LED
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');

    // Parse CSV
    float heart_rate = line.substring(0, line.indexOf(',')).toFloat();
    line = line.substring(line.indexOf(',') + 1);
    float brain_activity = line.substring(0, line.indexOf(',')).toFloat();

    // Map to PWM (0-255)
    int hr_pwm = (int)(heart_rate * 255);
    int ba_pwm = (int)((brain_activity + 1.0) / 2.0 * 255);

    analogWrite(9, hr_pwm);
    analogWrite(10, ba_pwm);
  }
}
```

## Running the Demo

### Simulation Mode (No Arduino)

```bash
python demos/demo_heart_arduino.py --duration 10.0
```

### With Arduino Connected

```bash
# Linux/macOS
python demos/demo_heart_arduino.py --arduino /dev/ttyACM0 --duration 10.0

# Windows
python demos/demo_heart_arduino.py --arduino COM3 --duration 10.0
```

### Custom Parameters

```bash
python demos/demo_heart_arduino.py \
  --duration 20.0 \
  --dt 0.0005 \
  --arduino /dev/ttyACM0 \
  --baud 115200
```

## Python API Usage

### Basic Integration

```python
from primal_logic import (
    MultiHeartModel,
    HeartArduinoBridge,
    ProcessorHeartArduinoLink,
)

# 1. Create heart model with RPO
heart = MultiHeartModel(
    lambda_heart=0.115,
    lambda_brain=0.092,
    rpo_alpha=0.4,
)

# 2. Create Arduino bridge (optional)
arduino = HeartArduinoBridge(port="/dev/ttyACM0")

# 3. Create unified link
link = ProcessorHeartArduinoLink(
    heart_model=heart,
    arduino_bridge=arduino,
    send_interval=10,
)

# 4. Run simulation loop
for step in range(1000):
    link.update(
        cardiac_input=0.5,
        brain_setpoint=1.0,
        theta=1.0,
    )

    if step % 100 == 0:
        state = link.get_state()
        print(f"Step {step}: HR={state['heart_rate']:.3f}")
```

### Advanced: Direct RPO Access

```python
from primal_logic import MultiHeartModel

heart = MultiHeartModel()

# Access underlying RPO instances
rpo_heart = heart.rpo_heart
rpo_brain = heart.rpo_brain

# Check RPO diagnostics
print(f"Heart RPO h_eff: {rpo_heart.h_eff:.6e}")
print(f"Heart RPO β_P: {rpo_heart.beta_p:.4f}")
print(f"Heart RPO state: {rpo_heart.state:.4f}")
```

## Hardware Requirements

### Arduino Setup

1. **Board**: Arduino Uno, Mega, or compatible (ATmega328P/2560)
2. **Connection**: USB cable (Type A to Type B)
3. **Drivers**: CH340/FTDI drivers if needed
4. **Power**: USB power sufficient for most applications

### Serial Port Configuration

- **Linux**: `/dev/ttyACM0` or `/dev/ttyUSB0`
- **macOS**: `/dev/cu.usbmodem*` or `/dev/cu.usbserial*`
- **Windows**: `COM3`, `COM4`, etc.

Check port with:
```bash
# Linux/macOS
ls /dev/tty*

# Or use Python
python -m serial.tools.list_ports
```

### Required Python Package

```bash
pip install pyserial
```

## Theory: RPO in Heart-Brain Coupling

The Recursive Planck Operator provides bounded, resonant feedback in the heart-brain equations:

### Heart Equation
```
n_h'(t) = −λ_h n_h + f_h(n_b, S_h) + ℛ_P[C(t)]
          └─decay─┘   └─coupling──┘   └─RPO────┘
```

The RPO term `ℛ_P[C(t)]` processes cardiac input through:
1. **Exponential decay**: Governed by λ_h
2. **Recursive resonance**: Via `sin(2π k Δt / h_eff) · y_k`
3. **Lightfoot damping**: Ensures bounded response

### Brain Equation
```
n_b'(t) = −λ_b n_b + f_b(n_h, S_b) + ℛ_P[s_set(t)]
```

Similar structure with brain-specific decay rate λ_b.

### Stability Guarantee

For any inputs, the system satisfies:
```
‖y‖_∞ ≤ (M · Θ̄ / α_eff) · [1 + |β_P| / (1 − α_eff Δt)]
```

This prevents runaway oscillations even under recursive feedback.

## Testing

Run tests for the integration:

```bash
# Test heart model
pytest tests/test_heart_model.py

# Test Arduino bridge (requires mock serial)
pytest tests/test_heart_arduino_bridge.py

# Run all tests
pytest tests/
```

## Troubleshooting

### Serial Port Access Denied

**Linux**:
```bash
sudo usermod -a -G dialout $USER
# Then log out and back in
```

**Permission error**:
```bash
sudo chmod 666 /dev/ttyACM0
```

### Arduino Not Detected

1. Check USB connection
2. Upload blink sketch to verify board works
3. Check drivers (CH340/FTDI)
4. Try different USB port/cable

### Import Errors

Ensure pyserial is installed:
```bash
pip install pyserial
```

Not `serial` package (different library).

### RPO Stability Issues

If experiencing unbounded growth:
- Ensure `0 < alpha * dt < 1` (default: `0.4 * 0.001 = 0.0004`)
- Check Lightfoot constant is in `[0.54, 0.56]`
- Verify input signals are bounded

## References

- **Quantitative Framework**: `docs/quantitative_framework.md`
- **RPO Theory**: See `primal_logic/rpo.py` docstrings
- **Heart Model**: `primal_logic/heart_model.py`
- **Arduino Bridge**: `primal_logic/heart_arduino_bridge.py`
- **Demo**: `demos/demo_heart_arduino.py`

## See Also

- `demos/demo_primal.py` - RPO stability validation
- `demos/demo_cryo.py` - Quantum vs. classical noise comparison
- `demos/demo_rrt_rif.py` - Recursive intent demonstration
- `primal_logic/hand.py` - Robotic hand with RPO integration
