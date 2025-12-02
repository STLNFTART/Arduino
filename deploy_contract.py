#!/usr/bin/env python3
"""
Deploy RPO Burn Contract to Hedera Network

This script deploys the RPO token burn contract to Hedera testnet or mainnet.

Usage:
    python deploy_contract.py --network testnet
    python deploy_contract.py --network mainnet

Requires:
    - Hedera account with HBAR for gas
    - Environment variables set (HEDERA_OPERATOR_ID, HEDERA_OPERATOR_KEY)
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Load Hedera SDK
try:
    from hedera import (
        Client,
        AccountId,
        PrivateKey,
        FileCreateTransaction,
        FileAppendTransaction,
        ContractCreateTransaction,
        ContractFunctionParameters,
        Hbar,
    )
    HEDERA_AVAILABLE = True
except ImportError:
    HEDERA_AVAILABLE = False
    print("ERROR: hedera-sdk-python not installed")
    print("Install with: pip install hedera-sdk-python")
    sys.exit(1)


def load_contract_bytecode(contract_path: Path) -> bytes:
    """Load compiled contract bytecode."""
    if not contract_path.exists():
        raise FileNotFoundError(f"Contract bytecode not found: {contract_path}")

    with open(contract_path, 'rb') as f:
        return f.read()


def create_hedera_client(network: str) -> Client:
    """Create Hedera client for specified network."""
    operator_id = os.getenv("HEDERA_OPERATOR_ID")
    operator_key = os.getenv("HEDERA_OPERATOR_KEY")

    if not operator_id or not operator_key:
        raise EnvironmentError(
            "Missing Hedera credentials. Set HEDERA_OPERATOR_ID and HEDERA_OPERATOR_KEY"
        )

    # Create client
    if network == "testnet":
        client = Client.forTestnet()
    elif network == "mainnet":
        client = Client.forMainnet()
    else:
        raise ValueError(f"Invalid network: {network}")

    # Set operator
    account_id = AccountId.fromString(operator_id)
    private_key = PrivateKey.fromString(operator_key.removeprefix("0x"))
    client.setOperator(account_id, private_key)

    return client


def deploy_contract(
    client: Client,
    bytecode: bytes,
    rpo_token_address: str,
) -> str:
    """
    Deploy RPO burn contract to Hedera network.

    Args:
        client: Hedera client
        bytecode: Compiled contract bytecode
        rpo_token_address: Address of RPO token contract

    Returns:
        Contract address (hex format)
    """
    print("\n" + "="*80)
    print("DEPLOYING RPO BURN CONTRACT TO HEDERA")
    print("="*80)

    # Step 1: Upload bytecode to Hedera File Service
    print("\nStep 1: Uploading bytecode to Hedera File Service...")

    file_tx = FileCreateTransaction()
    file_tx.setKeys([client.getOperatorKey()])
    file_tx.setContents(bytecode[:4096])  # First chunk

    file_response = file_tx.execute(client)
    file_receipt = file_response.getReceipt(client)
    file_id = file_receipt.fileId

    print(f"✓ File created: {file_id}")

    # Append remaining bytecode if needed
    if len(bytecode) > 4096:
        print("  Appending remaining bytecode chunks...")
        for i in range(4096, len(bytecode), 4096):
            chunk = bytecode[i:i+4096]
            append_tx = FileAppendTransaction()
            append_tx.setFileId(file_id)
            append_tx.setContents(chunk)
            append_tx.execute(client).getReceipt(client)
        print(f"✓ Complete bytecode uploaded ({len(bytecode)} bytes)")

    # Step 2: Create contract
    print("\nStep 2: Creating contract on Hedera network...")

    # Constructor parameters (RPO token address)
    constructor_params = ContractFunctionParameters()
    constructor_params.addAddress(rpo_token_address)

    contract_tx = ContractCreateTransaction()
    contract_tx.setBytecodeFileId(file_id)
    contract_tx.setGas(100000)
    contract_tx.setConstructorParameters(constructor_params)

    contract_response = contract_tx.execute(client)
    contract_receipt = contract_response.getReceipt(client)
    contract_id = contract_receipt.contractId

    print(f"✓ Contract deployed: {contract_id}")

    # Convert to EVM address format
    contract_address = f"0x{contract_id.toSolidityAddress()}"

    print("\n" + "="*80)
    print("DEPLOYMENT COMPLETE")
    print("="*80)
    print(f"Contract ID: {contract_id}")
    print(f"Contract Address: {contract_address}")
    print("="*80)

    return contract_address


def save_deployment_info(contract_address: str, network: str):
    """Save deployment information to config file."""
    config_path = Path("billing/rpo_operator_config.json")

    with open(config_path) as f:
        config = json.load(f)

    config["contract_address"] = contract_address
    config["network"] = network
    config["deployed"] = True

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Deployment info saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Deploy RPO Burn Contract to Hedera")
    parser.add_argument(
        '--network',
        type=str,
        default='testnet',
        choices=['testnet', 'mainnet'],
        help='Hedera network to deploy to'
    )
    parser.add_argument(
        '--bytecode',
        type=str,
        default='contracts/RPOBurnContract.bin',
        help='Path to compiled contract bytecode'
    )
    parser.add_argument(
        '--rpo-token',
        type=str,
        required=True,
        help='Address of RPO token contract (EVM format)'
    )

    args = parser.parse_args()

    # Load environment
    env_file = Path(".env.hedera")
    if env_file.exists():
        print(f"Loading credentials from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value

    try:
        # Load bytecode
        print("Loading contract bytecode...")
        bytecode = load_contract_bytecode(Path(args.bytecode))
        print(f"✓ Loaded bytecode ({len(bytecode)} bytes)")

        # Create client
        print(f"\nConnecting to Hedera {args.network}...")
        client = create_hedera_client(args.network)
        print("✓ Connected")

        # Deploy contract
        contract_address = deploy_contract(
            client=client,
            bytecode=bytecode,
            rpo_token_address=args.rpo_token,
        )

        # Save deployment info
        save_deployment_info(contract_address, args.network)

        print("\n✓ Deployment successful!")
        print(f"\nNext steps:")
        print(f"  1. Authorize actuators: Use authorizeActuator({actuator_address})")
        print(f"  2. Update .env.hedera with new contract address")
        print(f"  3. Run integration: python demo_primalrwa_integration.py")

        client.close()
        return 0

    except Exception as e:
        print(f"\n✗ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
