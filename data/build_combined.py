"""Build the training dataset by combining real Base x402 transactions
with synthetic adversaries calibrated to the x402 micropayment scale.

We treat every real authorizer as a legitimate agent (no ERC-8004 layer
is live for most of them yet). On top of that real population we inject
four adversary archetypes tuned to x402 scale so we can measure the
detection model on a realistic mix:

  BIG_DRAINER        large-amount exfil, 3-4 orders of magnitude above
                     the legit median
  SPAM_BURST         very high velocity, normal amounts (resource abuse
                     or ticket-sweeping bots)
  GAS_ANOMALY        10-100x normal gas pricing
  RING_COLLUSION     N agents paying the same brand-new merchant in
                     tight succession

Output:
  data/agents.jsonl          one record per agent (real + injected)
  data/transactions.jsonl    all tx ordered by timestamp
"""
from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(7)
np.random.seed(7)

ROOT = Path(__file__).resolve().parent
REAL_PATH = ROOT / "real_transactions.jsonl"
AGENTS_OUT = ROOT / "agents.jsonl"
TX_OUT = ROOT / "transactions.jsonl"


@dataclass
class Agent:
    agent_id: str
    declared_archetype: str | None
    true_archetype: str
    is_adversarial: bool
    adversary_kind: str | None
    attestation_depth: int
    intent_hash: str
    created_at: str


@dataclass
class Transaction:
    tx_id: str
    agent_id: str
    ts: str
    amount_usd: float
    counterparty: str
    calldata_bytes: int
    gas_price_gwei: float
    intent_action_delta: float
    attestation_depth: int


def _addr() -> str:
    return "0x" + uuid.uuid4().hex[:40]


def load_real(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path)]
    return pd.DataFrame(rows)


def build_real_agents(real: pd.DataFrame) -> list[Agent]:
    start_iso = real["ts"].min()
    agents = []
    for aid in real["agent_id"].unique():
        agents.append(Agent(
            agent_id=aid,
            declared_archetype=None,
            true_archetype="real_x402",
            is_adversarial=False,
            adversary_kind=None,
            attestation_depth=0,
            intent_hash="0x" + uuid.uuid4().hex[:64],
            created_at=start_iso,
        ))
    return agents


def synth_big_drainer(real: pd.DataFrame, n: int = 5) -> tuple[list[Agent], list[Transaction]]:
    """Drainer: large amounts targeting one or two new sinks."""
    agents, txs = [], []
    t0 = pd.to_datetime(real["ts"], format="ISO8601").min()
    t1 = pd.to_datetime(real["ts"], format="ISO8601").max()
    legit_median_gas = float(real["gas_price_gwei"].median())

    for _ in range(n):
        aid = _addr()
        agents.append(Agent(
            agent_id=aid,
            declared_archetype=None,
            true_archetype="big_drainer",
            is_adversarial=True,
            adversary_kind="big_drainer",
            attestation_depth=0,
            intent_hash="0x" + uuid.uuid4().hex[:64],
            created_at=t0.isoformat(),
        ))
        sink = _addr()
        span = (t1 - t0).total_seconds()
        n_tx = random.randint(6, 12)
        for _ in range(n_tx):
            offset = random.uniform(0, span)
            ts = (t0 + timedelta(seconds=offset)).isoformat()
            amount = float(np.clip(np.random.lognormal(mean=8.5, sigma=0.6), 2500, 80000))
            txs.append(Transaction(
                tx_id="0x" + uuid.uuid4().hex,
                agent_id=aid, ts=ts,
                amount_usd=round(amount, 4),
                counterparty=sink,
                calldata_bytes=random.choice([292, 324, 356]),
                gas_price_gwei=round(legit_median_gas * random.uniform(0.9, 1.1), 6),
                intent_action_delta=float(np.clip(np.random.normal(0.7, 0.1), 0, 1)),
                attestation_depth=0,
            ))
    return agents, txs


def synth_spam_burst(real: pd.DataFrame, n: int = 5) -> tuple[list[Agent], list[Transaction]]:
    """High-velocity spammer: tiny amounts, thousands per hour."""
    agents, txs = [], []
    t0 = pd.to_datetime(real["ts"], format="ISO8601").min()
    legit_median_gas = float(real["gas_price_gwei"].median())
    merchant_pool = real["counterparty"].drop_duplicates().sample(
        min(30, real["counterparty"].nunique()), random_state=42
    ).tolist()

    for _ in range(n):
        aid = _addr()
        agents.append(Agent(
            agent_id=aid,
            declared_archetype=None,
            true_archetype="spam_burst",
            is_adversarial=True,
            adversary_kind="spam_burst",
            attestation_depth=0,
            intent_hash="0x" + uuid.uuid4().hex[:64],
            created_at=t0.isoformat(),
        ))
        n_tx = random.randint(150, 400)
        burst_start = t0 + timedelta(seconds=random.uniform(0, 1200))
        for i in range(n_tx):
            ts = (burst_start + timedelta(seconds=i * random.uniform(0.3, 1.2))).isoformat()
            txs.append(Transaction(
                tx_id="0x" + uuid.uuid4().hex,
                agent_id=aid, ts=ts,
                amount_usd=round(float(np.clip(np.random.normal(0.001, 0.0005), 0.00001, 0.01)), 6),
                counterparty=random.choice(merchant_pool),
                calldata_bytes=random.choice([292, 324]),
                gas_price_gwei=round(legit_median_gas * random.uniform(0.9, 1.1), 6),
                intent_action_delta=float(np.clip(np.random.normal(0.3, 0.1), 0, 1)),
                attestation_depth=0,
            ))
    return agents, txs


def synth_gas_anomaly(real: pd.DataFrame, n: int = 5) -> tuple[list[Agent], list[Transaction]]:
    """Normal shape but wildly abnormal gas price."""
    agents, txs = [], []
    t0 = pd.to_datetime(real["ts"], format="ISO8601").min()
    t1 = pd.to_datetime(real["ts"], format="ISO8601").max()
    legit_median_gas = float(real["gas_price_gwei"].median())
    merchant_pool = real["counterparty"].drop_duplicates().sample(
        min(10, real["counterparty"].nunique()), random_state=11
    ).tolist()

    for _ in range(n):
        aid = _addr()
        agents.append(Agent(
            agent_id=aid,
            declared_archetype=None,
            true_archetype="gas_anomaly",
            is_adversarial=True,
            adversary_kind="gas_anomaly",
            attestation_depth=0,
            intent_hash="0x" + uuid.uuid4().hex[:64],
            created_at=t0.isoformat(),
        ))
        span = (t1 - t0).total_seconds()
        n_tx = random.randint(8, 16)
        for _ in range(n_tx):
            offset = random.uniform(0, span)
            ts = (t0 + timedelta(seconds=offset)).isoformat()
            txs.append(Transaction(
                tx_id="0x" + uuid.uuid4().hex,
                agent_id=aid, ts=ts,
                amount_usd=round(float(np.random.lognormal(mean=-6.0, sigma=1.0)), 6),
                counterparty=random.choice(merchant_pool),
                calldata_bytes=random.choice([292, 324]),
                gas_price_gwei=round(legit_median_gas * random.uniform(15, 80), 4),
                intent_action_delta=float(np.clip(np.random.normal(0.2, 0.1), 0, 1)),
                attestation_depth=0,
            ))
    return agents, txs


def synth_ring_collusion(real: pd.DataFrame, ring_size: int = 6) -> tuple[list[Agent], list[Transaction]]:
    """Ring of agents all paying a brand-new sink."""
    agents, txs = [], []
    t0 = pd.to_datetime(real["ts"], format="ISO8601").min()
    t1 = pd.to_datetime(real["ts"], format="ISO8601").max()
    legit_median_gas = float(real["gas_price_gwei"].median())
    sink = _addr()
    span = (t1 - t0).total_seconds()

    for _ in range(ring_size):
        aid = _addr()
        agents.append(Agent(
            agent_id=aid,
            declared_archetype=None,
            true_archetype="ring_collusion",
            is_adversarial=True,
            adversary_kind="ring_collusion",
            attestation_depth=0,
            intent_hash="0x" + uuid.uuid4().hex[:64],
            created_at=t0.isoformat(),
        ))
        n_tx = random.randint(6, 14)
        for _ in range(n_tx):
            offset = random.uniform(0, span)
            ts = (t0 + timedelta(seconds=offset)).isoformat()
            txs.append(Transaction(
                tx_id="0x" + uuid.uuid4().hex,
                agent_id=aid, ts=ts,
                amount_usd=round(float(np.clip(np.random.normal(4500.0, 800.0), 2500.0, 9000.0)), 4),
                counterparty=sink,
                calldata_bytes=random.choice([292, 324]),
                gas_price_gwei=round(legit_median_gas * random.uniform(0.9, 1.2), 6),
                intent_action_delta=float(np.clip(np.random.normal(0.25, 0.1), 0, 1)),
                attestation_depth=0,
            ))
    return agents, txs


def real_to_tx_records(real: pd.DataFrame) -> list[Transaction]:
    out = []
    for _, row in real.iterrows():
        out.append(Transaction(
            tx_id=row["tx_id"],
            agent_id=row["agent_id"],
            ts=row["ts"],
            amount_usd=float(row["amount_usd"]),
            counterparty=row["counterparty"],
            calldata_bytes=int(row["calldata_bytes"]),
            gas_price_gwei=float(row["gas_price_gwei"]),
            intent_action_delta=0.0,
            attestation_depth=0,
        ))
    return out


def main() -> None:
    real = load_real(REAL_PATH)
    print(f"loaded {len(real)} real tx, {real.agent_id.nunique()} agents")

    real_agents = build_real_agents(real)
    real_txs = real_to_tx_records(real)

    adv_agents: list[Agent] = []
    adv_txs: list[Transaction] = []
    for fn, label in [
        (synth_big_drainer, "big_drainer"),
        (synth_spam_burst, "spam_burst"),
        (synth_gas_anomaly, "gas_anomaly"),
        (synth_ring_collusion, "ring_collusion"),
    ]:
        a, t = fn(real)
        print(f"  injected {len(a)} {label} agents, {len(t)} tx")
        adv_agents.extend(a)
        adv_txs.extend(t)

    agents = real_agents + adv_agents
    txs = real_txs + adv_txs
    txs.sort(key=lambda t: t.ts)

    with open(AGENTS_OUT, "w") as f:
        for a in agents:
            f.write(json.dumps(asdict(a)) + "\n")
    with open(TX_OUT, "w") as f:
        for t in txs:
            f.write(json.dumps(asdict(t)) + "\n")

    print(f"wrote {len(agents)} agents to {AGENTS_OUT}")
    print(f"wrote {len(txs)} tx to {TX_OUT}")
    print(f"  real: {len(real_agents)} agents / {len(real_txs)} tx")
    print(f"  adv : {len(adv_agents)} agents / {len(adv_txs)} tx")


if __name__ == "__main__":
    main()
