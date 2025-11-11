# Ligand-Receptor Binding Subsystem

## Overview

The Ligand-Receptor Binding Subsystem introduces cellular-level biochemical dynamics to the Primal Logic framework, creating a complete multiscale integration from molecular binding events to macro-scale neural-cardiac physiology.

This subsystem enables simulation of:
- Infection-induced autonomic changes (fever tachycardia, cognitive fatigue)
- Pharmacological modulation and therapeutic interventions
- Biofeedback-assisted therapy for immune dysregulation
- Systemic inflammation effects on heart rate variability and brain function

## Architecture

The subsystem follows a hierarchical, bidirectional coupling architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    MACRO SCALE                              │
│  ┌───────────────────────────────────────────────────┐     │
│  │   Heart-Brain Coupling (MultiHeartModel)          │     │
│  │   • Neural potential (n_b)                        │     │
│  │   • Cardiac potential (n_h)                       │     │
│  │   • Recursive Planck Operators                    │     │
│  └───────────────────────────────────────────────────┘     │
│                           ↕                                 │
│             Modulated Decay Rates (λ_b, λ_h)                │
│                           ↕                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │   Immune Signaling (ImmuneSignaling)              │     │
│  │   • Immune intensity I(t)                         │     │
│  │   • Metabolic feedback                            │     │
│  └───────────────────────────────────────────────────┘     │
│                           ↕                                 │
│             Receptor Binding Signal                         │
│                           ↕                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │   Ligand-Receptor Binding (LigandReceptor)        │     │
│  │   • Receptor occupancy R(t)                       │     │
│  │   • Ligand concentration L(t)                     │     │
│  └───────────────────────────────────────────────────┘     │
│                    CELLULAR SCALE                           │
└─────────────────────────────────────────────────────────────┘
         ↕ Feedback Signal F(t) (stress, cytokines)
```

## Mathematical Formulation

### 1. Ligand-Receptor Binding Dynamics

At the cellular level, receptor occupancy `R(t)` and ligand concentration `L(t)` are governed by:

```
Ṙ(t) = k_on · L(t) · (R_T - R(t)) - k_off · R(t) + γ · F(t)
```

**Parameters:**
- `k_on`: Binding rate constant (M⁻¹ s⁻¹)
- `k_off`: Unbinding rate constant (s⁻¹)
- `R_T`: Total receptor density
- `F(t)`: Feedback signal from macro-scale (neural/cardiac fields)
- `γ`: Coupling coefficient (physiological → biochemical modulation)

**Physical Interpretation:**
- First term: New binding events (proportional to available receptors)
- Second term: Spontaneous unbinding
- Third term: Modulation by systemic stress, hormones, cytokines

### 2. Immune Signaling Accumulation

Immune intensity `I(t)` represents cumulative inflammation/cytokine levels:

```
İ(t) = ρ · R(t) - δ · I(t)
```

**Parameters:**
- `ρ`: Production rate (receptor binding → immune response)
- `δ`: Decay rate (immune signal resolution)

This creates a **low-pass filter** that accumulates rapid cellular events into a slower systemic immune state.

### 3. Metabolic Feedback to Neural/Cardiac Systems

The immune state modulates decay rates in the heart-brain model:

```
λ_b(t) = λ_b0 · (1 + α_b · I(t))
λ_h(t) = λ_h0 · (1 + α_h · I(t))
```

**Parameters:**
- `λ_b0, λ_h0`: Baseline decay rates
- `α_b, α_h`: Modulation coefficients (brain, heart)

**Biological Basis:**
- Increased `λ_b`: Faster neural decay → cognitive fatigue, reduced attention span
- Increased `λ_h`: Altered cardiac autonomic balance → tachycardia, reduced HRV

### 4. Macro-to-Cellular Feedback

Downward feedback `F(t)` from macro-scale to cellular level:

```
F(t) = β · (w_h · n_h(t) + w_b · n_b(t) + w_i · I(t))
```

**Default weights:**
- `w_h = 0.4`: Heart contribution (sympathetic/vagal tone)
- `w_b = 0.4`: Brain contribution (stress response)
- `w_i = 0.2`: Immune feed-forward
- `β`: Overall feedback strength

This implements **stress-induced receptor modulation** (e.g., cortisol affecting immune cell receptor expression).

## Implementation

### Core Classes

#### `LigandReceptor`
Implements cellular-level binding dynamics with macro feedback.

```python
from primal_logic import LigandReceptor

# Create receptor system
receptor = LigandReceptor(
    k_on=1.0,           # Binding rate
    k_off=0.5,          # Unbinding rate
    receptor_total=100, # Total receptors
    gamma=0.1,          # Feedback coupling
    dt=0.01             # Timestep
)

# Step with feedback from macro scale
state = receptor.step(feedback_signal=macro_feedback)
occupancy = receptor.get_occupancy_fraction()
```

#### `ImmuneSignaling`
Implements immune accumulation and decay rate modulation.

```python
from primal_logic import ImmuneSignaling

# Create immune system
immune = ImmuneSignaling(
    rho=0.05,        # Production rate
    delta=0.02,      # Decay rate
    alpha_brain=0.3, # Brain modulation
    alpha_heart=0.2, # Heart modulation
    dt=0.01
)

# Step with receptor signal
state = immune.step(receptor_signal=binding_signal)

# Get modulated decay rates
lambda_b = immune.modulate_brain_decay(lambda_b0)
lambda_h = immune.modulate_heart_decay(lambda_h0)
```

#### `MultiscaleCoupling`
Orchestrates the complete hierarchical integration.

```python
from primal_logic import MultiscaleCoupling

# Create integrated system
coupling = MultiscaleCoupling(dt=0.01)

# Step the complete hierarchy
state = coupling.step(
    cardiac_input=0.5,
    brain_setpoint=0.3,
    theta=1.0
)

# Get complete state across all scales
full_state = coupling.get_complete_state()
# Returns: receptor_occupancy, immune_intensity, n_heart, n_brain, etc.
```

## Demo Scenarios

The `demos/demo_ligand_receptor.py` script demonstrates three key scenarios:

### Scenario A: Baseline Physiological State
Normal homeostatic conditions with constant ligand concentration.

**Expected Results:**
- Steady receptor occupancy ~66%
- Low immune intensity ~0.15
- Stable heart rate and brain activity
- Minimal decay rate modulation

### Scenario B: Simulated Infection
Exponentially rising ligand concentration (acute pathogen exposure).

**Expected Results:**
- Increased receptor occupancy ~85%
- Rising immune intensity ~0.27
- Elevated decay rates (λ_b ↑, λ_h ↑)
- Simulates: fever tachycardia, cognitive fatigue, reduced HRV

**Biological Correlates:**
- Cytokine storm
- Systemic inflammatory response
- Autonomic dysregulation

### Scenario C: Pharmacological Intervention
Reduced binding rate (k_on ↓) and increased unbinding (k_off ↑).

**Expected Results:**
- Decreased receptor occupancy ~59%
- Normalized immune intensity ~0.15
- Stabilized decay rates
- Restored autonomic balance

**Therapeutic Implications:**
- Receptor antagonists
- Anti-inflammatory drugs
- Immune checkpoint modulation

## Running the Demo

```bash
# Basic demo (30 seconds)
python3 demos/demo_ligand_receptor.py

# Short test run
python3 demos/demo_ligand_receptor.py --duration 10.0

# Custom output directory
python3 demos/demo_ligand_receptor.py --output-dir results/my_experiment
```

**Output Files:**
- `baseline.csv`: Time series for Scenario A
- `infection.csv`: Time series for Scenario B
- `pharmacological.csv`: Time series for Scenario C

Each CSV contains:
- Time
- Receptor occupancy, ligand concentration
- Immune intensity, inflammation level
- Heart rate, brain activity
- Modulated decay rates (λ_b, λ_h)

## Applications

### 1. Infection Modeling
Simulate how pathogen load affects autonomic function:
```python
# Exponential growth (acute infection)
ligand_fn = lambda t: min(5.0, 1.0 * math.exp(0.1 * t))

receptor = LigandReceptor(
    ligand_input=ligand_fn,
    k_on=1.5,
    k_off=0.3
)
```

### 2. Pharmacological Testing
Test drug effects by varying binding parameters:
```python
# Receptor antagonist
receptor = LigandReceptor(
    k_on=0.5,   # Reduced binding
    k_off=1.0   # Enhanced unbinding
)
```

### 3. Biofeedback Therapy
Implement neural/cardiac training to offset immune dysregulation:
```python
# Increase heart-brain coherence to reduce macro feedback
brain_setpoint = lambda t: 0.5 * math.sin(2 * math.pi * t / 10)
cardiac_input = lambda t: 0.5 * math.sin(2 * math.pi * t / 10)
```

### 4. Stress Response
Model chronic stress effects on immune function:
```python
# High macro-to-cellular feedback
coupling = MultiscaleCoupling(feedback_strength=0.3)
```

## Integration with Existing Systems

### Heart-Arduino Bridge
Extend cardiac output to include immune signals:
```python
from primal_logic import MultiscaleCoupling, HeartArduinoBridge

coupling = MultiscaleCoupling()
bridge = HeartArduinoBridge(port="/dev/ttyACM0")

# Get extended output (includes immune intensity)
arduino_signals = coupling.get_arduino_output()
# [heart_rate, brain_activity, coherence, combined, immune_intensity]

bridge.send_cardiac_signal(arduino_signals)
```

### Recursive Planck Operator
The subsystem inherits RPO stability guarantees from the heart-brain layer:
- Bounded signal propagation
- Resonant coupling between scales
- Energy-information domain bridging

## Parameter Guidelines

### Typical Physiological Ranges

**Ligand-Receptor:**
- `k_on`: 0.5 - 2.0 (higher = stronger binding)
- `k_off`: 0.2 - 1.0 (higher = faster unbinding)
- `R_T`: 50 - 200 (receptor density)
- `γ`: 0.05 - 0.2 (feedback coupling)

**Immune Signaling:**
- `ρ`: 0.03 - 0.1 (production rate)
- `δ`: 0.01 - 0.05 (decay rate)
- `α_b`: 0.2 - 0.5 (brain modulation)
- `α_h`: 0.1 - 0.3 (heart modulation)

**Multiscale Coupling:**
- `feedback_strength`: 0.05 - 0.3 (macro → cellular)

### Stability Considerations

1. **Receptor binding must converge:**
   - Ensure `k_off > 0` (always decay)
   - Limit `γ · F(t)` to prevent unbounded growth

2. **Immune intensity should be bounded:**
   - `δ > 0` (immune signals resolve)
   - Typical steady-state: `I_ss = (ρ/δ) · R_avg`

3. **Decay rate modulation must preserve stability:**
   - Keep `α_b · I(t) < 5` (max 5x increase)
   - Monitor heart-brain system for oscillations

## Validation and Testing

### Unit Tests
Test individual components:
```python
# Test receptor binding equilibrium
receptor = LigandReceptor(k_on=1.0, k_off=0.5, receptor_total=100)
for _ in range(1000):
    receptor.step(feedback_signal=0.0)
# Should converge to ~67% occupancy

# Test immune accumulation
immune = ImmuneSignaling(rho=0.05, delta=0.02)
for _ in range(1000):
    immune.step(receptor_signal=1.0)
# Should converge to I_ss = 0.05/0.02 = 2.5
```

### Integration Tests
Verify multiscale coupling stability:
```python
coupling = MultiscaleCoupling()
for _ in range(10000):
    coupling.step(cardiac_input=0.5, brain_setpoint=0.3)
# All state variables should remain bounded
```

## Future Extensions

### 1. Multiple Receptor Types
Add receptor subtypes with different kinetics:
```python
receptor_alpha = LigandReceptor(k_on=1.0, k_off=0.5)
receptor_beta = LigandReceptor(k_on=2.0, k_off=0.2)
```

### 2. Spatial Heterogeneity
Implement receptor density gradients across tissues.

### 3. Nonlinear Feedback
Replace linear modulation with Hill equation cooperativity:
```python
λ_b(t) = λ_b0 · (1 + α_b · I^n / (K^n + I^n))
```

### 4. Temporal Delays
Add delay differential equations for hormone transport:
```python
F(t) = γ · neural_state(t - τ_delay)
```

## References

1. **Ligand-Receptor Kinetics**: Classical biochemical binding theory
2. **Immune Modulation**: Cytokine-mediated autonomic regulation
3. **Heart Rate Variability**: Task Force guidelines (1996)
4. **Recursive Planck Operator**: See `docs/quantitative_framework.md`

## See Also

- `docs/quantitative_framework.md` - Mathematical foundations
- `docs/processor_heart_arduino_integration.md` - Hardware integration
- `primal_logic/heart_model.py` - Heart-brain coupling implementation
- `demos/demo_heart_arduino.py` - Arduino integration demo
