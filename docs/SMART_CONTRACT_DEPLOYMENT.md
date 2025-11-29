# Smart Contract Deployment Guide

## Prerequisites

### 1. Install Solidity Compiler

```bash
# Install solc
npm install -g solc

# Or use Docker
docker pull ethereum/solc:stable
```

### 2. Install Hedera SDK

```bash
pip install hedera-sdk-python
```

### 3. Get Hedera Credentials

1. Go to https://portal.hedera.com/
2. Create account (testnet or mainnet)
3. Copy your account ID and private key
4. Update `.env.hedera`

---

## Deployment Steps

### Step 1: Compile the Smart Contract

```bash
# Compile Solidity contract
solc --bin --abi contracts/RPOBurnContract.sol -o contracts/build/

# This creates:
# - contracts/build/RPOBurnContract.bin (bytecode)
# - contracts/build/RPOBurnContract.abi (interface)
```

**Note:** You'll need an RPO token contract first. If you don't have one yet, deploy an ERC20 token first.

### Step 2: Deploy RPO Token (if needed)

If you don't have an RPO token contract yet:

```bash
# Deploy basic ERC20 token first
python deploy_rpo_token.py --network testnet --name "Recursive Planck Operator" --symbol "RPO"
```

### Step 3: Deploy Burn Contract

```bash
# Deploy to testnet
python deploy_contract.py \
  --network testnet \
  --bytecode contracts/build/RPOBurnContract.bin \
  --rpo-token 0xYOUR_RPO_TOKEN_ADDRESS

# Deploy to mainnet (when ready)
python deploy_contract.py \
  --network mainnet \
  --bytecode contracts/build/RPOBurnContract.bin \
  --rpo-token 0xYOUR_RPO_TOKEN_ADDRESS
```

### Step 4: Authorize Actuators

After deployment, authorize your actuators:

```python
from hedera import Client, AccountId, PrivateKey, ContractExecuteTransaction

# Connect to Hedera
client = Client.forTestnet()
client.setOperator(account_id, private_key)

# Authorize motorhand_pro_actuator
tx = (
    ContractExecuteTransaction()
    .setContractId(contract_id)
    .setGas(100000)
    .setFunction(
        "authorizeActuator",
        ["0x0000000000000000000000000000000000000003"]  # motorhand_pro_actuator
    )
)

response = tx.execute(client)
receipt = response.getReceipt(client)
print(f"Actuator authorized: {receipt.status}")
```

### Step 5: Update Configuration

Update `billing/rpo_operator_config.json`:

```json
{
  "owner_address": "0x536f51e53111755f9d1327d41fe6b21a9b2b2ba1",
  "contract_address": "0xYOUR_DEPLOYED_CONTRACT_ADDRESS",
  "network": "testnet",
  "deployed": true
}
```

Update `.env.hedera`:

```bash
HEDERA_CONTRACT_ID=0xYOUR_DEPLOYED_CONTRACT_ADDRESS
```

---

## Quick Start (Without Real Deployment)

For testing without deploying, the current stub mode works perfectly:

```bash
# Just run the demos - they use CSV logging
python demo_primalrwa_integration.py --demo 1 --mode hedera_testnet --duration 5.0

# The CSV log (rpo_burn_log.csv) is ready for batch submission later
```

---

## Going Live

### Option 1: Use Existing Contract (Recommended)

If you already have a deployed contract at `0x35AF4bCa366737d2a433Fe85062Dd7A19F9572d3`:

1. Just update `billing/hedera_integration.py` to use real SDK calls:

```python
# In HederaContractBurner.burn_tokens()
tx = (
    ContractExecuteTransaction()
    .setContractId(self.contract_id)
    .setGas(100000)
    .setFunction(
        "burnTokens",
        [actuator_address, seconds]
    )
)

tx_response = tx.execute(self.client)
receipt = tx_response.getReceipt(self.client)
```

2. Run the demo:

```bash
python demo_primalrwa_integration.py --demo 1 --mode hedera_testnet --duration 5.0
```

### Option 2: Deploy New Contract

Follow the full deployment steps above.

---

## Verify Deployment

After deployment, verify it works:

```bash
# Test burn
python -c "
from billing.hedera_integration import load_hedera_burner_from_env

burner = load_hedera_burner_from_env()
result = burner.burn_tokens(
    actuator_address='0x0000000000000000000000000000000000000003',
    seconds=1
)
print(f'Success: {result.success}')
print(f'TX ID: {result.transaction_id}')
burner.close()
"
```

---

## Troubleshooting

### Error: "hedera-sdk-python not installed"

```bash
pip install hedera-sdk-python
```

### Error: "Insufficient balance"

- Make sure your Hedera account has HBAR for gas fees
- Testnet: Get free HBAR from portal.hedera.com
- Mainnet: Purchase HBAR

### Error: "Actuator not authorized"

```bash
# Call authorizeActuator on the contract
python authorize_actuator.py --actuator 0x0000000000000000000000000000000000000003
```

---

## Cost Estimates

### Testnet (FREE)
- Get free HBAR from portal.hedera.com
- Unlimited testing

### Mainnet
- Contract deployment: ~$1-5 in HBAR
- Per burn transaction: ~$0.0001-0.001 in HBAR
- Monthly cost (1000 burns/day): ~$3-30 in HBAR

---

## Next Steps

1. ✅ Compile contract (`solc`)
2. ✅ Deploy RPO token (if needed)
3. ✅ Deploy burn contract (`python deploy_contract.py`)
4. ✅ Authorize actuators
5. ✅ Update configuration
6. ✅ Test with demo (`python demo_primalrwa_integration.py`)
7. ✅ Monitor burns in real-time

---

## Support

For help with deployment:
- Hedera Docs: https://docs.hedera.com/
- Hedera Discord: https://hedera.com/discord
- GitHub Issues: https://github.com/STLNFTART/Arduino/issues
