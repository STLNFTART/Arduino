# RecursiveActuator Token Burn PoC (Simulated Actuators)

This proof-of-concept wires the simulated actuators in `primal_logic` to an `RPOBurnMeter` that tracks how long each actuator spends in **planck_mode**. Every full second accumulated in that mode triggers a would-be burn of 1 RPO on contract `0x35AF4bCa366737d2a433Fe85062Dd7A19F9572d3`. In the current build, burns are only logged to `rpo_burn_log.csv`; there are **no real Hedera network calls** even when using `hedera_testnet` mode.

- `owner_address`: `0x536f51e53111755F9D1327D41fE6b21a9b2B2BA1` (reserved for future Hedera operator credentials provided via environment variables, never committed).
- Actuator addresses: `billing/rpo_actuator_addresses.json` maps human-readable actuator IDs to placeholder blockchain addresses.
- Contract address: `billing/rpo_operator_config.json` carries the burn contract address.

## How the PoC works
1. `billing/rpo_burn_meter.RPOBurnMeter` loads the operator and actuator address maps.
2. Actuators (`RoboticHand` and `MultiHeartModel`) accept an optional `burn_meter`, a `planck_mode` flag, and report their timestep `dt` to the burn meter after advancing dynamics.
3. For each full second accumulated per actuator while `planck_mode=True`, the burn meter appends a CSV row to `rpo_burn_log.csv` recording the timestamp, contract id, actuator address, seconds burned, mode, operator id (if provided), network, and a simulated transaction ID when using `hedera_testnet` mode.

CSV schema (written with header on first write):

```text
timestamp,contract_id,actuator_address,seconds,mode,operator_id,network,tx_id
```

## Running the demo
From the repository root:

```bash
# Dry-run logging only
python demo_rpo_burn_poc.py --mode dry_run

# Testnet-prep logging (still offline). Requires Hedera env vars.
HEDERA_OPERATOR_ID=testnet-id \
HEDERA_OPERATOR_KEY=testnet-key \
python demo_rpo_burn_poc.py --mode hedera_testnet
```

Environment variables for `hedera_testnet` (validated even though no network calls are made):

- `HEDERA_OPERATOR_ID`
- `HEDERA_OPERATOR_KEY`
- `HEDERA_NETWORK` (optional, defaults to `testnet`)

The script simulates 60 seconds with 10 ms steps, wiring the same `RPOBurnMeter` into two actuators (hand and heart model). After the run it prints:

- Total simulated seconds spent in `planck_mode` per actuator
- Total integer seconds burned per actuator (one token per second in planck_mode)
- The filesystem path to `rpo_burn_log.csv`

The log file resides in the repository root and accumulates all simulated burns across runs.
