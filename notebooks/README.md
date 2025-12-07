# Jupyter Notebooks - Primal Logic Framework

**Comprehensive interactive tutorials and demonstrations for the Arduino Primal Logic robotic hand control framework.**

---

## 📚 Notebook Collection

This directory contains **11 comprehensive Jupyter notebooks** covering all aspects of the Primal Logic framework:

### 🎯 Getting Started

| # | Notebook | Description | Topics |
|---|----------|-------------|--------|
| **00** | [Index and Overview](00_Index_and_Overview.ipynb) | Master index and quick start | Setup, imports, first demo |
| **01** | [Introduction to Primal Logic](01_Introduction_to_Primal_Logic.ipynb) | Theoretical foundations | Donte constant, Lightfoot constant, RPO, Lipschitz stability |
| **02** | [Robotic Hand Basics](02_Robotic_Hand_Basics.ipynb) | 15-DOF hand control | Joint dynamics, grasp simulation, trajectory tracking |

### 🔬 Core Framework

| # | Notebook | Description | Topics |
|---|----------|-------------|--------|
| **03** | [Parameter Exploration](03_Parameter_Exploration.ipynb) | Interactive parameter tuning | Theta, Alpha, Beta, Tau sweeps |
| **04** | [Heart-Brain Coupling](04_Heart_Brain_Coupling.ipynb) | Physiological integration | Multi-heart model, cardiac-neural coupling |

### 🤖 Hardware & Integration

| # | Notebook | Description | Topics |
|---|----------|-------------|--------|
| **05** | [MotorHandPro Integration](05_MotorHandPro_Integration.ipynb) | Hardware bridge | Serial communication, real-time control |

### 🛰️ Space Applications

| # | Notebook | Description | Topics |
|---|----------|-------------|--------|
| **06** | [Satellite Systems](06_Satellite_Systems.ipynb) | 50,000-satellite constellation | Orbital mechanics, coverage analysis |
| **07** | [Space Environment](07_Space_Environment.ipynb) | Radiation & EMP simulation | Radiation belts, magnetic fields, satellite health |

### ✅ Validation & Testing

| # | Notebook | Description | Topics |
|---|----------|-------------|--------|
| **08** | [Validation Framework](08_Validation_Framework.ipynb) | Cross-repository validation | SpaceX, Tesla, PX4, CARLA, Arduino |

### 🎮 Interactive & Advanced

| # | Notebook | Description | Topics |
|---|----------|-------------|--------|
| **09** | [Interactive Demos](09_Interactive_Demos.ipynb) | Live parameter tuning | ipywidgets, real-time visualization |
| **10** | [Advanced Topics](10_Advanced_Topics.ipynb) | Deep dives | Quantum fields, field coupling, anti-gravity |

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Navigate to notebooks directory
cd notebooks/

# Install dependencies
pip install jupyter numpy matplotlib pandas pytest pyserial ipywidgets
```

### 2. Launch Jupyter

```bash
# Start Jupyter server
jupyter notebook

# Or Jupyter Lab
jupyter lab
```

### 3. Start Learning

1. Open **[00_Index_and_Overview.ipynb](00_Index_and_Overview.ipynb)** for quick start
2. Follow the numbered sequence for structured learning
3. Jump to specific topics as needed

---

## 📋 Notebook Features

All notebooks include:

✅ **Theory & Background** - Mathematical foundations
✅ **Working Code Examples** - Copy-paste ready
✅ **Visualizations** - Matplotlib plots and animations
✅ **Interactive Widgets** - Real-time parameter tuning (where applicable)
✅ **Exercises** - Hands-on learning opportunities
✅ **Next Steps** - Clear progression path

---

## 🎓 Learning Paths

### Path 1: Beginners (Start Here!)
1. **00 Index** → **01 Introduction** → **02 Robotic Hand** → **03 Parameter Exploration**

### Path 2: Researchers
1. **01 Introduction** → **10 Advanced Topics** → **08 Validation Framework**

### Path 3: Hardware Engineers
1. **02 Robotic Hand** → **05 MotorHandPro** → **04 Heart-Brain Coupling**

### Path 4: Space Applications
1. **01 Introduction** → **06 Satellite Systems** → **07 Space Environment**

### Path 5: Full Coverage
Follow notebooks **00 → 10** in numerical order

---

## 🔧 Technical Requirements

### Minimum Requirements
- Python 3.10+
- Jupyter Notebook or JupyterLab
- 4GB RAM
- Standard libraries (no GPU required)

### Recommended Setup
- Python 3.11+
- JupyterLab 4.0+
- 8GB RAM
- Modern web browser (Chrome, Firefox, Edge)

### Optional Hardware
- **Arduino**: For serial communication demos (Notebook 04, 05)
- **MotorHandPro**: For hardware integration (Notebook 05)
- **EEG Device**: For neurorobotic demos (advanced use)

---

## 📦 Dependencies

### Core (Required)
```bash
numpy>=1.20          # Numerical operations
matplotlib>=3.5      # Visualization
```

### Framework (Auto-installed)
```bash
# Primal logic modules (from ../primal_logic/)
primal_logic.hand
primal_logic.field
primal_logic.rpo
primal_logic.heart_model
```

### Optional (Enhanced Features)
```bash
pandas>=1.5          # Data analysis (Notebook 03)
ipywidgets>=8.0      # Interactive controls (Notebook 09)
pyserial>=3.5        # Arduino communication (Notebook 04, 05)
pytest>=7.0          # Testing (Notebook 08)
```

---

## 💡 Usage Examples

### Example 1: Run Single Notebook Cell
```python
# In any notebook
import sys, os
sys.path.insert(0, os.path.abspath('..'))

from primal_logic.hand import RoboticHand
hand = RoboticHand()
hand.step(desired_angles)
```

### Example 2: Execute Complete Notebook
```bash
# From command line
jupyter nbconvert --execute --to notebook \
  01_Introduction_to_Primal_Logic.ipynb
```

### Example 3: Export to HTML
```bash
jupyter nbconvert --to html 02_Robotic_Hand_Basics.ipynb
```

---

## 🐛 Troubleshooting

### Issue: Module not found
**Solution:** Ensure repository root is in Python path
```python
import sys, os
sys.path.insert(0, os.path.abspath('..'))
```

### Issue: Serial port error (Notebook 04, 05)
**Solution:** Check Arduino connection and port
```bash
# Linux
ls /dev/ttyACM*
ls /dev/ttyUSB*

# macOS
ls /dev/cu.usb*

# Windows
# Check Device Manager
```

### Issue: Widgets not displaying (Notebook 09)
**Solution:** Enable Jupyter widgets extension
```bash
jupyter nbextension enable --py widgetsnbextension
```

### Issue: Matplotlib not showing plots
**Solution:** Use inline backend
```python
%matplotlib inline
import matplotlib.pyplot as plt
```

---

## 📊 Notebook Content Summary

### Code Statistics
- **Total notebooks:** 11
- **Total code cells:** ~150+
- **Total markdown cells:** ~100+
- **Estimated completion time:** 10-15 hours (all notebooks)

### Topics Covered
- Primal Logic theory ✅
- Robotic hand control ✅
- Parameter optimization ✅
- Physiological coupling ✅
- Hardware integration ✅
- Satellite systems ✅
- Space environment ✅
- Cross-repo validation ✅
- Interactive demos ✅
- Advanced quantum fields ✅

---

## 🎯 Learning Outcomes

After completing these notebooks, you will be able to:

1. ✅ Understand Primal Logic theoretical foundations
2. ✅ Implement robotic hand control systems
3. ✅ Tune control parameters for optimal performance
4. ✅ Integrate physiological signals (heart-brain coupling)
5. ✅ Connect to hardware (Arduino, MotorHandPro)
6. ✅ Simulate satellite constellations (50,000+ satellites)
7. ✅ Model space environment effects (radiation, EMP)
8. ✅ Validate control algorithms across multiple platforms
9. ✅ Create interactive demonstrations
10. ✅ Apply quantum-inspired coherence fields

---

## 📚 Related Documentation

### Repository Documentation
- **[Main README](../README.md)** - Repository overview
- **[Quick Reference](../QUICK_REFERENCE.md)** - API quick reference
- **[Architecture](../CODEBASE_ARCHITECTURE_ANALYSIS.md)** - System architecture
- **[Neurorobotic Guide](../NEUROROBOTIC_INTEGRATION_GUIDE.md)** - Brain-computer interface

### Technical Documents
- **[Primal Logic Framework](../external/MotorHandPro/PRIMAL_LOGIC_FRAMEWORK.md)** - Mathematical framework
- **[Quantitative Framework](../docs/quantitative_framework.md)** - Constants and equations
- **[MotorHandPro Integration](../docs/motorhand_pro_integration.md)** - Hardware integration
- **[Heart-Arduino Bridge](../docs/processor_heart_arduino_integration.md)** - Serial protocol

### Test Suite
- **[Unit Tests](../tests/)** - 86 test files
- **[Test Results](../TEST_RUN_RESULTS.md)** - Latest test run
- **[Validation](../validation/)** - Cross-repo validation

---

## 🔬 Research & Citation

If you use these notebooks in your research, please cite:

```bibtex
@software{primal_logic_notebooks_2025,
  title={Primal Logic Robotic Hand Control Framework - Jupyter Notebooks},
  author={Lightfoot, Donte},
  year={2025},
  publisher={The Phoney Express LLC / Locked In Safety},
  note={U.S. Provisional Patent Application No. 63/842,846}
}
```

---

## 🤝 Contributing

To add new notebooks:

1. Follow naming convention: `NN_Topic_Name.ipynb`
2. Include proper metadata (kernelspec, language_info)
3. Add markdown explanations before code cells
4. Test all code cells before committing
5. Update this README with new notebook info

---

## 📧 Support

For questions or issues:

1. Review the notebook's markdown cells for explanations
2. Check the **[Main README](../README.md)** troubleshooting section
3. Examine working examples in **[demos/](../demos/)** directory
4. Inspect **[test files](../tests/)** for additional examples

---

## 🎉 Get Started Now!

**Open [00_Index_and_Overview.ipynb](00_Index_and_Overview.ipynb) to begin your journey!**

---

**Copyright 2025** Donte Lightfoot - The Phoney Express LLC / Locked In Safety
**Patent Pending:** U.S. Provisional Patent Application No. 63/842,846
