# Local Setup Guide - PrimalRWA Integration

Complete guide to clone and run the PrimalRWA integration on your local machine.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/STLNFTART/Arduino.git
cd Arduino

# Checkout the integration branch
git checkout claude/actuator-entry-points-01PMwk9JxxZBjj2A65winAwT

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install numpy pandas matplotlib pyserial

# Configure Hedera credentials
cp .env.hedera.example .env.hedera
# Edit .env.hedera with your credentials

# Run the demo!
python demo_primalrwa_integration.py --demo 1 --mode hedera_testnet --duration 5.0
```

---

## 📋 Detailed Setup Instructions

### Step 1: Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/STLNFTART/Arduino.git
cd Arduino

# Checkout the PrimalRWA integration branch
git checkout claude/actuator-entry-points-01PMwk9JxxZBjj2A65winAwT

# Initialize MotorHandPro submodule
git submodule update --init --recursive
```

### Step 2: Set Up Python Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Verify activation:**
```bash
which python  # Should show path in .venv
# On Windows: where python
```

### Step 3: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Additional packages for integration
pip install numpy pandas matplotlib pyserial

# Optional: Hedera SDK (for live burns)
pip install hedera-sdk-python
```

**Verify installation:**
```bash
python -c "import numpy, pandas, matplotlib; print('✓ All packages installed')"
```

### Step 4: Configure Hedera Credentials

**Option A: Use Template (Recommended)**

```bash
# Copy the example file
cp .env.hedera.example .env.hedera

# Edit with your credentials
nano .env.hedera  # or vim, code, etc.
```

**Fill in your credentials:**
```bash
# Get these from https://portal.hedera.com/
HEDERA_OPERATOR_ID=0.0.YOUR_ACCOUNT_ID
HEDERA_OPERATOR_KEY=0xYOUR_PRIVATE_KEY
HEDERA_EVM_ADDRESS=0xYOUR_EVM_ADDRESS
HEDERA_NETWORK=testnet
HEDERA_CONTRACT_ID=0x35AF4bCa366737d2a433Fe85062Dd7A19F9572d3
```

**Option B: Export Environment Variables**

```bash
export HEDERA_OPERATOR_ID="0.0.7344342"
export HEDERA_OPERATOR_KEY="0x084e30a..."
export HEDERA_NETWORK="testnet"
export HEDERA_CONTRACT_ID="0x35AF4bCa366737d2a433Fe85062Dd7A19F9572d3"
```

**On Windows:**
```cmd
set HEDERA_OPERATOR_ID=0.0.7344342
set HEDERA_OPERATOR_KEY=0x084e30a...
set HEDERA_NETWORK=testnet
set HEDERA_CONTRACT_ID=0x35AF4bCa366737d2a433Fe85062Dd7A19F9572d3
```

### Step 5: Verify Setup

```bash
# Test basic imports
python -c "
from billing.rpo_burn_meter import RPOBurnMeter
from primal_logic.motorhand_actuator import create_motorhand_actuator
print('✓ Imports successful')
"

# Check environment
python -c "
import os
print('✓ Operator ID:', os.getenv('HEDERA_OPERATOR_ID', 'NOT SET'))
print('✓ Network:', os.getenv('HEDERA_NETWORK', 'NOT SET'))
"
```

---

## 🎮 Running the Demos

### Demo 1: Basic Burn Tracking (5 seconds)

```bash
python demo_primalrwa_integration.py --demo 1 --mode hedera_testnet --duration 5.0
```

**Expected output:**
```
Running actuation for 5.0 seconds...
t=  0.00s | Runtime=  0.01s | Burned=   0 RPO
t=  1.00s | Runtime=  1.01s | Burned=   1 RPO ✓
t=  2.00s | Runtime=  2.01s | Burned=   2 RPO ✓
t=  3.00s | Runtime=  3.01s | Burned=   3 RPO ✓
t=  4.00s | Runtime=  4.01s | Burned=   4 RPO ✓
t=  5.00s | Runtime=  5.00s | Burned=   5 RPO ✓

BURN REPORT
Burned seconds: 5
Burned tokens: 5 RPO
```

### Demo 2: Multiple Actuators (10 seconds)

```bash
python demo_primalrwa_integration.py --demo 2 --mode hedera_testnet --duration 10.0
```

### Demo 3: Complete PrimalRWA Control Loop (10 seconds)

```bash
python demo_primalrwa_integration.py --demo 3 --mode hedera_testnet --duration 10.0
```

### Run All Verification Tests

```bash
python verify_burn_ratio.py
```

**Expected output:**
```
================================================================================
  VERIFICATION: 1 RPO Token = 1 Second of Actuation
================================================================================

Test: 3.0s actuation in dry_run mode
✓ Test completed
  Expected burns: 3
  Actual burns:   3
  Ratio:          1.000 tokens/second
  Status:         PASS ✓

... (more tests)

================================================================================
  ✓ ALL TESTS PASSED
  ✓ 1 RPO TOKEN = 1 SECOND OF ACTUATION (VERIFIED)
================================================================================
```

---

## 🔧 Advanced Usage

### Custom Duration

```bash
# Run for any duration
python demo_primalrwa_integration.py --demo 1 --mode hedera_testnet --duration 15.0
```

### Dry Run Mode (No Credentials Needed)

```bash
# Test without Hedera credentials
python demo_primalrwa_integration.py --demo 1 --mode dry_run --duration 3.0
```

### With Hardware (MotorHandPro Connected)

```bash
# Connect to real hardware
python demo_primalrwa_integration.py \
  --demo 1 \
  --mode hedera_testnet \
  --duration 5.0 \
  --port /dev/ttyACM0 \
  --hardware
```

**On Windows:** Use `--port COM3` (or your COM port)

---

## 📊 Check Results

### View Burn Log

```bash
# View all burns
cat rpo_burn_log.csv

# View last 10 burns
tail -10 rpo_burn_log.csv

# Count total burns
tail -n +2 rpo_burn_log.csv | wc -l
```

### Analyze Burns with Python

```python
import pandas as pd

# Load burn log
df = pd.read_csv('rpo_burn_log.csv')

# Summary
print(f"Total burns: {len(df)}")
print(f"Total seconds: {df['seconds'].sum()}")
print(f"Unique actuators: {df['actuator_address'].nunique()}")

# Per actuator
print("\nBurns per actuator:")
print(df.groupby('actuator_address')['seconds'].sum())
```

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Reinstall packages
pip install -r requirements.txt
pip install numpy pandas matplotlib pyserial
```

### Issue: "Hedera credentials not found"

```bash
# Check if .env.hedera exists
ls -la .env.hedera

# Check if variables are set
python -c "import os; print(os.getenv('HEDERA_OPERATOR_ID'))"

# Make sure you're in the Arduino directory
pwd  # Should show /path/to/Arduino
```

### Issue: "No such file or directory: rpo_operator_config.json"

```bash
# Make sure you're in the repo root
cd /path/to/Arduino

# Verify files exist
ls billing/rpo_operator_config.json
ls billing/rpo_actuator_addresses.json
```

### Issue: Serial port not found (when using hardware)

**Linux:**
```bash
# List available ports
ls /dev/ttyACM* /dev/ttyUSB*

# Add user to dialout group
sudo usermod -a -G dialout $USER
# Log out and log back in
```

**Windows:**
```cmd
# Check Device Manager → Ports (COM & LPT)
# Use the COM port shown (e.g., COM3)
```

**macOS:**
```bash
# List ports
ls /dev/cu.usbmodem*
ls /dev/tty.usbmodem*
```

### Issue: Permission denied on Python

```bash
# Make sure virtual environment is activated
which python  # Should show .venv/bin/python

# If not, activate it
source .venv/bin/activate
```

---

## 🔐 Security Best Practices

### Protect Your Credentials

**DO NOT commit `.env.hedera` to git!** (Already in `.gitignore`)

```bash
# Verify it's ignored
git status  # Should NOT show .env.hedera

# If it shows up, add to .gitignore
echo ".env.hedera" >> .gitignore
```

### Use Environment Variables in Production

```bash
# Instead of .env.hedera file, use system environment variables
export HEDERA_OPERATOR_ID="0.0.7344342"
export HEDERA_OPERATOR_KEY="0x..."

# Or use a secrets manager (AWS Secrets Manager, Vault, etc.)
```

### Separate Testnet and Mainnet

```bash
# Testnet credentials
cp .env.hedera.example .env.hedera.testnet

# Mainnet credentials
cp .env.hedera.example .env.hedera.mainnet

# Use different files for different environments
python demo_primalrwa_integration.py --demo 1 --mode dry_run
```

---

## 📦 Project Structure

```
Arduino/
├── billing/
│   ├── rpo_burn_meter.py              # Core burn tracking
│   ├── hedera_integration.py          # Hedera SDK integration
│   ├── rpo_operator_config.json       # Contract config
│   └── rpo_actuator_addresses.json    # Actuator addresses
├── primal_logic/
│   ├── motorhand_actuator.py          # Actuator wrapper
│   ├── motorhand_integration.py       # MotorHandPro bridge
│   ├── hand.py                        # Hand simulation
│   └── heart_model.py                 # Heart model
├── contracts/
│   └── RPOBurnContract.sol            # Smart contract
├── docs/
│   ├── PRIMALRWA_INTEGRATION.md       # Integration guide
│   └── SMART_CONTRACT_DEPLOYMENT.md   # Deployment guide
├── demo_primalrwa_integration.py      # Main demo script
├── verify_burn_ratio.py               # Test verification
├── deploy_contract.py                 # Contract deployment
├── .env.hedera.example                # Credentials template
├── requirements.txt                   # Python dependencies
└── README.md                          # Main README
```

---

## 🚀 Next Steps

### 1. Test Locally
```bash
python demo_primalrwa_integration.py --demo 1 --mode dry_run --duration 5.0
```

### 2. Configure Hedera
```bash
# Get credentials from portal.hedera.com
# Update .env.hedera
```

### 3. Run with Hedera Testnet
```bash
python demo_primalrwa_integration.py --demo 1 --mode hedera_testnet --duration 5.0
```

### 4. Deploy Smart Contract (Optional)
```bash
# See docs/SMART_CONTRACT_DEPLOYMENT.md
python deploy_contract.py --network testnet --rpo-token 0x...
```

### 5. Integrate with Your Application
```python
from billing.rpo_burn_meter import RPOBurnMeter
from primal_logic.motorhand_actuator import create_motorhand_actuator

# Your code here
```

---

## 📞 Support

**Documentation:**
- Integration Guide: `docs/PRIMALRWA_INTEGRATION.md`
- Deployment Guide: `docs/SMART_CONTRACT_DEPLOYMENT.md`
- Test Results: `TEST_RESULTS.md`

**Get Help:**
- GitHub Issues: https://github.com/STLNFTART/Arduino/issues
- Hedera Discord: https://hedera.com/discord
- Hedera Docs: https://docs.hedera.com/

---

## ✅ Verification Checklist

Before running, verify:

- [ ] Repository cloned
- [ ] Branch checked out: `claude/actuator-entry-points-01PMwk9JxxZBjj2A65winAwT`
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip list` shows numpy, pandas, etc.)
- [ ] `.env.hedera` configured with credentials
- [ ] In Arduino directory (`pwd` shows path to Arduino)
- [ ] Can import modules: `python -c "from billing.rpo_burn_meter import RPOBurnMeter"`

**Ready to run:**
```bash
python demo_primalrwa_integration.py --demo 1 --mode hedera_testnet --duration 5.0
```

---

**Built with cutting-edge robotics and blockchain technology**
**Ready for real-world deployment** 🚀
