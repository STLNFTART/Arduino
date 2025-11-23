# CLAUDE.md - AI Assistant Guide for Primal Logic Robotic Hand Framework

## Repository Overview

This repository implements a **Primal Logic robotic hand control framework** - a sophisticated quantum-inspired control system for a 15-DOF (5 fingers × 3 joints) tendon-driven robotic hand with integrated physiological coupling and Arduino hardware support.

**Core Technologies:**
- Python 3.10+ (standard library only for core simulation)
- Optional: pandas, matplotlib for analytics
- Arduino hardware integration via serial communication
- Git submodule: MotorHandPro hardware control

**Project Type:** Research/Production hybrid with validated control framework
**License:** Not specified
**Current Version:** v1.0.0

---

## Critical Conventions & Standards

### Code Quality Standards

1. **Python Version:** Requires Python 3.10+ (uses modern type hints, dataclasses)
2. **Import Style:** Absolute imports preferred; `from __future__ import annotations` for forward references
3. **Type Hints:** Required for all public APIs; use `typing` module extensively
4. **Dataclasses:** Preferred for state containers (see `JointState`, `HeartBrainState`)
5. **Docstrings:** Triple-quoted docstrings for modules and classes
6. **Line Length:** Not strictly enforced but generally reasonable (<120 chars)

### Testing Requirements

```bash
# Run test suite (REQUIRED before commits)
python3 -m pytest

# Syntax validation (recommended)
python3 -m compileall primal_logic tests main.py vendor

# Validation suite (for major changes)
python3 run_motorhand_validation.py
```

**Test Coverage Requirements:**
- All new control algorithms MUST have unit tests
- Hardware integration code MUST have simulation mode tests
- RPO/memory kernels MUST validate stability guarantees (Lipschitz < 1)
- Critical files with existing tests: `test_hand.py`, `test_rpo.py`, `test_heart_model.py`

### Git Workflow

**Branch Naming Convention:**
- Feature branches: `claude/feature-name-<session-id>`
- Current branch: `claude/claude-md-mibdc4u1oidbtubf-019HUbxszZyd6P8JmN13rxCM`

**Commit Guidelines:**
1. Descriptive commit messages focusing on "why" not "what"
2. Reference issue/PR numbers when applicable
3. Run tests before committing
4. Use `git submodule update --init --recursive` when cloning

**Submodules:**
- `external/MotorHandPro` - Arduino hardware firmware (115200 baud)

---

## Architecture & Module Organization

### Directory Structure

```
/home/user/Arduino/
├── primal_logic/              # Core framework modules
│   ├── hand.py                # 15-DOF hand model with PD controllers
│   ├── field.py               # Quantum-inspired coherence field (8×8)
│   ├── rpo.py                 # Recursive Planck Operator (microprocessor)
│   ├── heart_model.py         # Heart-brain-immune coupling
│   ├── memory.py              # Exponential & RPO memory kernels
│   ├── motorhand_integration.py  # Hardware bridge & unified controller
│   ├── heart_arduino_bridge.py   # Cardiac signal Arduino bridge
│   ├── serial_bridge.py       # Low-level serial communication
│   ├── trajectory.py          # Grasp trajectory planning
│   ├── adaptive.py            # Adaptive gain scheduling
│   ├── constants.py           # All physical/control constants
│   ├── sweeps.py              # Parameter sweep utilities
│   ├── analysis.py            # Rolling average analytics
│   ├── utils.py               # Helper utilities
│   └── __init__.py            # Package exports
├── tests/                     # Pytest test suite
├── demos/                     # Demonstration scripts
├── docs/                      # Comprehensive documentation
├── external/                  # Git submodules (MotorHandPro)
├── vendor/                    # Offline fallbacks (pandas/matplotlib stubs)
├── validation/                # Validation framework extensions
├── main.py                    # CLI demo entry point
└── run_motorhand_validation.py  # Validation pipeline
```

### Control Architecture (Hierarchical)

```
┌─────────────────────────────────────┐
│ APPLICATION LAYER                   │
│ - Grasp planning (trajectory.py)   │
│ - Power/precision/tripod grasps    │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ FIELD LAYER                         │
│ - Quantum coherence (field.py)     │
│ - Modulates controller gains       │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ CONTROL LAYER                       │
│ - PD + exponential memory (hand.py)│
│ - 15 joint controllers             │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ MICROPROCESSOR LAYER                │
│ - Recursive Planck Operator (rpo.py)│
│ - Donte: 149.9992314               │
│ - Lightfoot: 0.54-0.56             │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ PHYSIOLOGICAL LAYER                 │
│ - Heart-brain coupling (heart_model)│
│ - Sympathetic/vagal feedback       │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ HARDWARE LAYER                      │
│ - MotorHandPro bridge              │
│ - Serial 115200 baud               │
│ - 15 servo actuators               │
└─────────────────────────────────────┘
```

---

## Key Components Deep Dive

### 1. Constants & Configuration (`primal_logic/constants.py`)

**Universal Physical Constants:**
```python
PLANCK_CONSTANT = 6.626070e-34  # J·s
DONTE_CONSTANT = 149.9992314    # Fixed-point attractor
LIGHTFOOT_MIN = 0.54            # Damping bound (lower)
LIGHTFOOT_MAX = 0.56            # Damping bound (upper)
```

**Simulation Parameters:**
```python
DT = 1e-3                       # 1ms timestep (1000 Hz)
BETA_DEFAULT = 0.8              # Memory kernel gain
ALPHA_DEFAULT = 0.54            # Coupling constant
LAMBDA_DEFAULT = 0.115          # 1/s decay rate
```

**Hardware Configuration:**
```python
DEFAULT_FINGERS = 5
JOINTS_PER_FINGER = 3           # Total: 15 DOF
SERIAL_BAUD = 115200            # MUST match Arduino firmware
SERIAL_PORT = "/dev/ttyACM0"    # Linux default
```

**IMPORTANT:** Never modify `DONTE_CONSTANT` or `PLANCK_CONSTANT` - these are derived from theoretical framework.

### 2. Robotic Hand Model (`primal_logic/hand.py`)

**Key Classes:**
- `JointState`: Dataclass for angle/velocity state
- `JointLimits`: Physical constraints (angle: 0-1.2 rad, torque: 0.7 N·m max)
- `HandJointController`: PD controller with memory kernel
- `RoboticHand`: Main hand simulation (15 joints)

**Control Law:**
```python
# PD component
u_pd = alpha * (Kp * error + Kd * d_error)

# Memory component (exponential or RPO)
u_mem = controller.mem_kernel.update(theta, error, step_index)

# Combined torque
tau = clip(u_pd + u_mem, -torque_max, torque_max)
```

**Usage Pattern:**
```python
from primal_logic import RoboticHand

hand = RoboticHand(
    n_fingers=5,
    n_joints_per_finger=3,
    alpha_base=0.54,
    memory_type="exponential"  # or "recursive_planck"
)

# Each timestep
hand.step(
    desired_angles=target,  # Shape: (5, 3)
    theta=1.0,
    coherence=0.8,
    step=iteration
)

torques = hand.get_torques()  # Shape: (15,)
```

### 3. Recursive Planck Operator (`primal_logic/rpo.py`)

**Purpose:** Microprocessor layer ensuring bounded, resonant control with Lipschitz stability guarantees.

**Key Equations:**
```python
# Discrete update (1ms timestep)
sin_arg = 2π * step_index * dt / h_eff
resonance = sin(sin_arg) * state

state_new = (1 - alpha*dt) * state + theta*dt * (input + beta_p * resonance)
```

**Critical Parameters:**
- `alpha`: From `LIGHTFOOT_MIN-MAX` range (0.54-0.56)
- `beta_p`: Resonance coupling strength
- `h_eff`: Effective Planck constant (NOT the fundamental constant)

**Stability Check:**
```python
c = (150 - DONTE_CONSTANT) * exp(lambda * DONTE_CONSTANT)
L = c * lambda * exp(-lambda * DONTE_CONSTANT)
# MUST have L < 1.0 for stability
```

### 4. MotorHandPro Hardware Integration (`primal_logic/motorhand_integration.py`)

**Two Main Classes:**

**MotorHandProBridge** - Low-level hardware communication:
```python
bridge = MotorHandProBridge(
    port="/dev/ttyACM0",
    baud=115200,              # CRITICAL: Must be 115200
    n_fingers=5,
    n_joints_per_finger=3,
    lambda_value=0.16905,     # Lightfoot constant
    ke_gain=0.3
)

bridge.connect()
bridge.send_torques(torques_array)  # np.ndarray shape (15,)
state = bridge.get_state()          # Returns control_energy, lipschitz, etc.
bridge.disconnect()
```

**UnifiedPrimalLogicController** - Complete integration:
```python
controller = UnifiedPrimalLogicController(
    hand_model=hand,
    motorhand_bridge=bridge,
    heart_model=None,         # Optional MultiHeartModel
    rpo_processor=None        # Optional RecursivePlanckOperator
)

# Control loop
for i in range(num_steps):
    controller.step(target_angles)
    full_state = controller.get_full_state()
```

**Serial Protocol:**
- Format: CSV with newline termination
- Example: `0.123,0.234,0.156,...,0.089\n` (15 comma-separated floats)
- Precision: 3 decimal places
- Range: Normalized to [-1.0, 1.0]

### 5. Heart-Brain Physiological Model (`primal_logic/heart_model.py`)

**Purpose:** Models heart-brain-immune coupling via RPO for bio-hybrid actuation research.

**Core State Equations:**
```python
# Heart dynamics
n_h' = -λ_h * n_h + f_h(n_b, S_h) + RPO_heart[C(t)]

# Brain dynamics
n_b' = -λ_b * n_b + f_b(n_h, S_b) + RPO_brain[s_set(t)]

# Coupling functions
f_heart = coupling_strength * tanh(n_brain + s_heart)
f_brain = coupling_strength * tanh(n_heart + s_brain)
```

**Usage:**
```python
from primal_logic import MultiHeartModel

heart = MultiHeartModel(
    lambda_heart=0.115,
    lambda_brain=0.092,
    coupling_strength=0.15
)

state = heart.step(
    cardiac_input=0.7,
    brain_setpoint=0.5,
    theta=1.0
)

# Get Arduino-ready outputs
cardiac_output = heart.get_cardiac_output()  # 4 channels
```

### 6. Memory Kernels (`primal_logic/memory.py`)

**Two Types:**

**ExponentialMemoryKernel** - Standard exponential decay:
```python
decay = exp(-λ * dt)
memory_new = decay * memory_old + theta * error * dt
output = -gain * memory_new
```

**RecursivePlanckMemoryKernel** - Quantum-inspired resonance:
```python
# Uses RPO internally
output = rpo.step(theta, error, step_index)
```

**When to Use:**
- Exponential: Fast, guaranteed stability, simpler
- RPO: Enhanced resonant coupling, better for physiological integration

---

## Development Workflows

### Adding a New Control Feature

1. **Define constants** in `primal_logic/constants.py`
2. **Implement algorithm** in appropriate module
3. **Add unit tests** in `tests/test_<module>.py`
4. **Validate stability** - ensure Lipschitz < 1.0
5. **Add demo** in `demos/demo_<feature>.py` (optional)
6. **Update documentation** in `docs/` if significant
7. **Run full test suite** before committing

**Example - Adding New Memory Kernel:**
```python
# 1. In primal_logic/memory.py
class NewMemoryKernel:
    def __init__(self, param1, param2):
        self._state = 0.0
        # ...

    def update(self, theta, error, step_index=None):
        # Implement update logic
        # MUST ensure bounded output
        return output

# 2. In tests/test_memory.py
def test_new_kernel_stability():
    kernel = NewMemoryKernel(...)
    for i in range(1000):
        out = kernel.update(1.0, 0.1, i)
        assert abs(out) < 10.0  # Bounded check
```

### Working with Hardware

**Safety Checklist:**
1. ✓ Verify serial port permissions (`ls -l /dev/ttyACM0`)
2. ✓ Confirm baud rate is 115200 (MUST match firmware)
3. ✓ Test in simulation mode first (`--simulate` flag in demos)
4. ✓ Check torque limits (< 0.7 N·m per joint)
5. ✓ Monitor Lipschitz stability estimate (< 1.0)
6. ✓ Implement emergency stop in control loop

**Port Detection:**
```bash
# Linux/Mac
ls /dev/ttyACM* /dev/ttyUSB*

# Add user to dialout group (Linux)
sudo usermod -a -G dialout $USER
```

**Testing Pipeline:**
```bash
# 1. Simulation only (no hardware)
python3 demos/demo_motorhand_integration.py --full --simulate --duration 5.0

# 2. Basic connection test
python3 demos/demo_motorhand_integration.py --basic --port /dev/ttyACM0

# 3. Full hardware integration
python3 demos/demo_motorhand_integration.py --full --duration 10.0
```

### Running Validation Suite

```bash
# Complete validation (requires numpy)
python3 run_motorhand_validation.py

# Arduino robotic hand test only
python3 run_motorhand_validation.py --arduino-only

# Export results
python3 run_motorhand_validation.py --export results.json --latex report.tex
```

**Validation Criteria (ALL MUST PASS):**
- Lipschitz contractivity: L < 1.0
- Bounded control energy
- Finite-time convergence
- Max torque < 0.7 N·m (allows 1.5 N·m transient overshoot)
- Final oscillations < 5 zero crossings

---

## Common Patterns & Idioms

### Pattern 1: Safe Numerical Clipping

**Always use safe clipping to prevent numerical issues:**
```python
from primal_logic.utils import safe_clip

# Good
value = safe_clip(computation, min_val, max_val)

# Avoid
value = max(min_val, min(max_val, computation))  # Doesn't handle NaN
```

### Pattern 2: Optional Hardware Graceful Degradation

**Hardware components should gracefully degrade to simulation:**
```python
# In primal_logic/__init__.py
try:
    from .motorhand_integration import MotorHandProBridge
    _MOTORHAND_AVAILABLE = True
except ImportError:
    _MOTORHAND_AVAILABLE = False
    MotorHandProBridge = None

# In user code
if bridge is not None:
    bridge.send_torques(torques)
else:
    # Simulation mode - just log
    print(f"Simulated torques: {torques}")
```

### Pattern 3: Offline Analytics Fallbacks

**Use vendor stubs when pandas/matplotlib unavailable:**
```python
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    from vendor.pandas_stub import pd
    from vendor.matplotlib_stub import pyplot as plt

# Code works identically - stubs generate text artifacts
```

### Pattern 4: Timestep-Aware Algorithms

**Many algorithms require explicit step indexing for resonance terms:**
```python
# Good - provides step_index
for step in range(num_steps):
    rpo_output = rpo.step(theta=1.0, input_value=signal, step_index=step)

# Bad - missing step_index (will fail for RPO kernels)
rpo_output = rpo.step(theta=1.0, input_value=signal)
```

### Pattern 5: State Container Immutability

**Use dataclasses for state, but create new instances for updates:**
```python
from dataclasses import dataclass

@dataclass
class JointState:
    angle: float = 0.0
    velocity: float = 0.0

# Good - modify in place for performance (simulation loop)
state.angle += velocity * dt

# Also acceptable - functional style
new_state = JointState(
    angle=state.angle + velocity * dt,
    velocity=state.velocity + acceleration * dt
)
```

---

## Documentation Structure

**Existing Documentation Files:**

1. **README.md** - Quick start, features, run instructions
2. **CODEBASE_ARCHITECTURE_ANALYSIS.md** - Comprehensive architecture analysis (THIS IS THE MAIN REFERENCE)
3. **NEUROROBOTIC_INTEGRATION_GUIDE.md** - Brain-signal integration guide
4. **NEUROROBOTIC_CONTROL.md** - Neurorobotic control theory
5. **ANALYSIS_SUMMARY.md** - Analysis tooling summary
6. **QUICK_REFERENCE.md** - Quick command reference
7. **docs/processor_heart_arduino_integration.md** - Heart-Arduino bridge docs
8. **docs/motorhand_pro_integration.md** - MotorHandPro hardware docs
9. **docs/quantitative_framework.md** - Mathematical framework (Donte/Lightfoot)

**When to Update Documentation:**
- New module → Add section to CODEBASE_ARCHITECTURE_ANALYSIS.md
- New hardware interface → Add to relevant hardware integration doc
- New control algorithm → Update quantitative_framework.md if theory changes
- Breaking API change → Update README.md and relevant docs

---

## Debugging & Troubleshooting

### Common Issues

**1. Serial Port Permission Denied**
```bash
# Error: Permission denied: '/dev/ttyACM0'
# Fix:
sudo usermod -a -G dialout $USER
# Log out and log back in
```

**2. Import Errors for Optional Dependencies**
```python
# Error: ModuleNotFoundError: No module named 'numpy'
# This is expected - motorhand_integration requires numpy
# Either:
pip install numpy
# Or use simulation mode without hardware
```

**3. Lipschitz Stability Violation**
```python
# Error: Lipschitz estimate > 1.0 (unstable system)
# Causes:
# - Lambda too high
# - Donte constant modified
# - Memory kernel misconfigured
# Fix: Reduce lambda_value or ke_gain in MotorHandProBridge
```

**4. Arduino Not Responding**
```bash
# Check connection
ls /dev/ttyACM*

# Check if another process is using port
lsof /dev/ttyACM0

# Verify baud rate in Arduino firmware (MUST be 115200)
```

### Debug Mode Patterns

**Add verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# In modules, add
logger = logging.getLogger(__name__)
logger.debug(f"State: {state}, Torque: {torque}")
```

**Monitor stability in real-time:**
```python
state = bridge.get_state()
print(f"Lipschitz: {state['lipschitz_estimate']:.6f}, "
      f"Stable: {state['stable']}, "
      f"Energy: {state['control_energy']:.4f}")
```

---

## Performance Considerations

### Simulation Performance

**Typical Benchmarks:**
- Hand simulation loop: 1000 Hz (1ms timestep)
- Serial communication latency: < 5ms
- Torque computation: < 0.1ms per joint
- Full system (hand + RPO + heart): ~100 Hz achievable

**Optimization Tips:**
1. **Use exponential kernel** instead of RPO for faster simulation
2. **Reduce field grid size** (8×8 is default, can go to 4×4)
3. **Disable heart model** if not needed for bio-hybrid research
4. **Batch serial writes** - send all 15 torques at once, not individually

### Memory Usage

**Typical Memory Footprint:**
- Hand model: ~10 KB (15 joints × state)
- RPO instance: ~1 KB
- Heart model: ~2 KB
- Total simulation: < 100 KB

**Large-Scale Simulations:**
- Use NumPy arrays instead of lists where possible
- Clear torque logs periodically if running > 1M steps
- Consider memory-mapped files for long-running data collection

---

## Testing Best Practices

### Unit Test Structure

**Follow existing test patterns:**
```python
# In tests/test_<module>.py
import pytest
from primal_logic.<module> import ClassName

def test_basic_functionality():
    """Test basic operation"""
    obj = ClassName(param1, param2)
    result = obj.method()
    assert result == expected

def test_edge_case_bounds():
    """Test boundary conditions"""
    obj = ClassName()
    # Test limits
    assert obj.compute(0.0) >= 0.0
    assert obj.compute(1.0) <= 1.0

@pytest.mark.parametrize("input,expected", [
    (0.0, 0.0),
    (0.5, 0.25),
    (1.0, 1.0),
])
def test_parametrized(input, expected):
    """Test multiple cases"""
    assert abs(function(input) - expected) < 1e-6
```

### Integration Testing

**Hardware integration tests should have simulation fallback:**
```python
@pytest.mark.skipif(not hardware_available(),
                    reason="Hardware not connected")
def test_hardware_integration():
    bridge = MotorHandProBridge(port="/dev/ttyACM0")
    # Test with real hardware
```

### Validation Framework

**Custom validators in `validation/arduino_validation_extension.py`:**
- Extends base validation for robotic hand specifics
- Checks 15-DOF grasp phases (approach → contact → stabilize)
- Validates torque saturation, oscillations, convergence time

---

## Integration Points for AI Assistants

### What AI Assistants Should Know

**1. This is a Research + Production Codebase:**
- Theoretical framework (Quantro-Primal) is well-established - don't modify core constants
- Production quality: requires tests, validation, stability proofs
- Research quality: encourages experimentation with new kernels, coupling schemes

**2. Stability is Non-Negotiable:**
- ALWAYS verify Lipschitz < 1.0 for new control algorithms
- ALWAYS test memory kernels for bounded output
- ALWAYS clip torques to physical limits (0.7 N·m)

**3. Hardware Integration is Safety-Critical:**
- Test in simulation first (`--simulate` mode)
- Never push untested code that controls physical actuators
- Verify serial protocol matches firmware (115200 baud, CSV format)

**4. Mathematical Rigor:**
- Donte constant (149.9992314) is derived - don't round
- Lightfoot range (0.54-0.56) is theoretically bounded
- Planck constant (6.626070e-34) is fundamental physics constant

**5. Optional Dependencies:**
- Core simulation: Python stdlib only
- Analytics: pandas, matplotlib (has vendor stubs)
- Hardware: pyserial (required for actual hardware)
- MotorHandPro: numpy (required for hardware bridge)

### When to Ask for Clarification

**Ask the user if:**
- They want to modify Donte or Lightfoot constants (usually NO)
- They want hardware integration (requires physical setup)
- They want to add new theoretical framework components
- Breaking changes to public API are needed
- Performance requirements change (current: 100-1000 Hz)

**Proceed without asking if:**
- Adding new demo scripts
- Fixing bugs that don't change API
- Adding tests for existing functionality
- Improving documentation
- Adding type hints to existing code

---

## Quick Command Reference

```bash
# === Setup ===
git clone <repo-url>
git submodule update --init --recursive
python3 -m pip install -e .
python3 -m pip install -e .[analysis]  # Optional analytics

# === Testing ===
python3 -m pytest                        # Run all tests
python3 -m pytest tests/test_hand.py     # Run specific test
python3 -m compileall primal_logic       # Syntax check

# === Validation ===
python3 run_motorhand_validation.py                    # Full suite
python3 run_motorhand_validation.py --arduino-only     # Hand only

# === Demos ===
python3 main.py                                        # Basic simulation
python3 demos/demo_primal.py                          # RPO demo
python3 demos/demo_heart_arduino.py --duration 10     # Heart-Arduino
python3 demos/demo_motorhand_integration.py --full --simulate  # Full sim

# === Hardware ===
ls /dev/ttyACM*                         # Find Arduino port
sudo usermod -a -G dialout $USER        # Grant serial access
python3 demos/demo_motorhand_integration.py --basic --port /dev/ttyACM0

# === Analysis ===
python3 -c "from primal_logic import plot_rolling_average; plot_rolling_average('artifacts/torques.csv', 'joint_0', 25)"
```

---

## File Modification Guidelines

### Files You Can Freely Modify

- `demos/*.py` - Add new demonstrations
- `tests/*.py` - Add new tests
- `docs/*.md` - Update documentation
- `main.py` - Modify CLI demo
- `primal_logic/adaptive.py` - Experiment with gain scheduling
- `primal_logic/trajectory.py` - Add new grasp patterns

### Files Requiring Careful Modification

- `primal_logic/hand.py` - Core dynamics (validate stability)
- `primal_logic/rpo.py` - RPO implementation (verify theory)
- `primal_logic/memory.py` - Memory kernels (ensure boundedness)
- `primal_logic/motorhand_integration.py` - Hardware safety-critical
- `primal_logic/constants.py` - Review implications before changing

### Files You Should Rarely Modify

- `primal_logic/utils.py` - Stable utility functions
- `vendor/*` - Offline stubs (maintain compatibility)
- `.gitmodules` - Submodule configuration

### Files You Should NEVER Modify Without Theoretical Justification

- Core constants in `constants.py`:
  - `DONTE_CONSTANT`
  - `PLANCK_CONSTANT`
  - `LIGHTFOOT_MIN/MAX` range

---

## Common AI Assistant Tasks

### Task: Add a new grasp trajectory

1. Open `primal_logic/trajectory.py`
2. Add new function following existing pattern
3. Add test in `tests/test_trajectory.py`
4. Optionally add demo in `demos/`

### Task: Debug serial communication issue

1. Check port: `ls /dev/ttyACM*`
2. Verify permissions: `ls -l /dev/ttyACM0`
3. Test with basic demo: `--basic` flag
4. Check baud rate (MUST be 115200)
5. Monitor logs with `logging.DEBUG`

### Task: Add new control parameter

1. Define in `primal_logic/constants.py` with SI units
2. Add to relevant class `__init__` method
3. Update docstrings
4. Add parameter sweep in `primal_logic/sweeps.py`
5. Validate stability with new parameter range

### Task: Optimize simulation performance

1. Profile with `cProfile` to find bottleneck
2. Consider switching RPO → exponential kernel
3. Reduce field grid size (8×8 → 4×4)
4. Use NumPy arrays instead of lists
5. Batch serial writes if hardware-connected

---

## Version History & Migration Notes

**Current Version: v1.0.0**

**Major Milestones:**
- Initial release: Core hand simulation + field dynamics
- v0.9: Added RPO microprocessor layer
- v0.95: Heart-brain physiological coupling
- v1.0: MotorHandPro hardware integration

**No Breaking Changes Yet** - API is stable

**Future Compatibility:**
- Planning: Multi-modal sensor integration
- Considering: Real-time brain signal inputs (EEG/BCI)
- Research: Bio-hybrid actuator support

---

## Additional Resources

**Key Papers & Theory:**
- `docs/quantitative_framework.md` - Mathematical derivations
- Quantro-Primal formalism: Donte/Lightfoot constants derivation

**External Links:**
- MotorHandPro firmware: https://github.com/STLNFTART/MotorHandPro
- Arduino serial reference: https://www.arduino.cc/reference/en/language/functions/communication/serial/

**Contact & Support:**
- Issue tracker: GitHub issues
- Author: Donte Lightfoot (see pyproject.toml)

---

## Summary for AI Assistants

**DO:**
- ✓ Run tests before committing (`pytest`)
- ✓ Verify Lipschitz < 1.0 for new control algorithms
- ✓ Use simulation mode (`--simulate`) before hardware testing
- ✓ Follow existing code patterns (dataclasses, type hints)
- ✓ Add unit tests for new functionality
- ✓ Update documentation for significant changes
- ✓ Clip torques to physical limits (0.7 N·m)
- ✓ Check serial port permissions on Linux

**DON'T:**
- ✗ Modify Donte or Planck constants without theoretical justification
- ✗ Push untested hardware control code
- ✗ Change serial baud rate (MUST be 115200)
- ✗ Skip stability validation for new algorithms
- ✗ Assume hardware is connected (check gracefully)
- ✗ Round physical constants (preserve full precision)
- ✗ Modify submodule files directly (use upstream PRs)

**Remember:**
This is a **safety-critical research framework** with real hardware integration. Prioritize stability, testing, and documentation. When in doubt, simulate first, validate thoroughly, then deploy to hardware.
