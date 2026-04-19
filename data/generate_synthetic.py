"""Synthetic x402 transaction generator.

Emulates the behavioral signatures of realistic agent archetypes plus
adversarial patterns. Output mirrors what a Base x402 facilitator would
expose: signed intent, payer/payee, amount, timing, calldata shape.

Each agent carries an ERC-8004 style identity record that declares its
intended archetype. Adversaries either have no attestation or declare an
archetype that does not match their behavior, which is the signal our
risk layer exploits.
"""
from __future__ import annotations

import json
import os
import random
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

random.seed(42)
np.random.seed(42)

OUT_PATH = Path(__file__).resolve().parent / "transactions.jsonl"
AGENTS_PATH = Path(__file__).resolve().parent / "agents.jsonl"


# ---------------------------------------------------------------------------
# Archetypes. Each defines a behavioral signature that clustering should
# discover without labels.
# ---------------------------------------------------------------------------

ARCHETYPES = {
    "subscription_payer": dict(
        amount_mu=29.0, amount_sigma=0.8,
        interval_hours_mu=24 * 30, interval_hours_sigma=6,
        counterparty_pool=3, calldata_bytes=68,
        active_hours=(0, 24), gas_multiplier=1.0,
    ),
    "trading_bot": dict(
        amount_mu=850.0, amount_sigma=600.0,
        interval_hours_mu=0.05, interval_hours_sigma=0.03,
        counterparty_pool=12, calldata_bytes=260,
        active_hours=(0, 24), gas_multiplier=1.6,
    ),
    "data_buyer": dict(
        amount_mu=0.50, amount_sigma=0.15,
        interval_hours_mu=0.002, interval_hours_sigma=0.001,
        counterparty_pool=4, calldata_bytes=180,
        active_hours=(0, 24), gas_multiplier=1.0,
    ),
    "oracle_publisher": dict(
        amount_mu=0.08, amount_sigma=0.02,
        interval_hours_mu=0.25, interval_hours_sigma=0.02,
        counterparty_pool=1, calldata_bytes=420,
        active_hours=(0, 24), gas_multiplier=1.2,
    ),
    "treasury_rebalancer": dict(
        amount_mu=12000.0, amount_sigma=4000.0,
        interval_hours_mu=24 * 7, interval_hours_sigma=24,
        counterparty_pool=5, calldata_bytes=140,
        active_hours=(9, 17), gas_multiplier=1.0,
    ),
    "arbitrage_bot": dict(
        amount_mu=2200.0, amount_sigma=800.0,
        interval_hours_mu=0.1, interval_hours_sigma=0.08,
        counterparty_pool=20, calldata_bytes=310,
        active_hours=(0, 24), gas_multiplier=2.0,
    ),
}

ADVERSARIES = {
    # Agent registered as subscription payer, behaves like a drainer.
    "identity_mismatch_drainer": dict(
        declared="subscription_payer",
        amount_mu=9500.0, amount_sigma=2000.0,
        interval_hours_mu=0.01, interval_hours_sigma=0.005,
        counterparty_pool=1, calldata_bytes=68,
        active_hours=(2, 5), gas_multiplier=3.0,
    ),
    # Compromised trading bot siphoning funds to a single address.
    "compromised_trading_bot": dict(
        declared="trading_bot",
        amount_mu=4500.0, amount_sigma=200.0,
        interval_hours_mu=0.008, interval_hours_sigma=0.002,
        counterparty_pool=1, calldata_bytes=140,
        active_hours=(0, 24), gas_multiplier=2.5,
    ),
    # Agent with no attestation. Behavior resembles nothing in training.
    "unattested_exfil": dict(
        declared=None,
        amount_mu=1800.0, amount_sigma=900.0,
        interval_hours_mu=0.03, interval_hours_sigma=0.02,
        counterparty_pool=2, calldata_bytes=48,
        active_hours=(3, 6), gas_multiplier=4.0,
    ),
    # Swarm collusion. Many agents all sending to the same sink.
    "collusion_ring_member": dict(
        declared="data_buyer",
        amount_mu=450.0, amount_sigma=50.0,
        interval_hours_mu=0.5, interval_hours_sigma=0.1,
        counterparty_pool=1, calldata_bytes=180,
        active_hours=(0, 24), gas_multiplier=1.1,
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    agent_id: str
    declared_archetype: str | None
    true_archetype: str
    is_adversarial: bool
    adversary_kind: str | None
    attestation_depth: int  # 0 if unattested, 1-4 for valid chains
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
    intent_action_delta: float  # 0 normal, up to 1 for intent drift
    attestation_depth: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _addr() -> str:
    return "0x" + uuid.uuid4().hex[:40]


def _in_active_hours(ts: datetime, window: tuple[int, int]) -> bool:
    lo, hi = window
    h = ts.hour
    if lo <= hi:
        return lo <= h < hi
    return h >= lo or h < hi


def _sample_interval(mu: float, sigma: float) -> float:
    val = np.random.normal(mu, sigma)
    return max(val, 0.0005)


def _sample_amount(mu: float, sigma: float) -> float:
    val = np.random.normal(mu, sigma)
    return round(max(val, 0.01), 4)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_agents(n_legit: int = 240, n_adversarial: int = 30) -> list[Agent]:
    agents: list[Agent] = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Balance archetypes so clustering has enough density per class.
    archetype_cycle = list(ARCHETYPES.keys()) * ((n_legit // len(ARCHETYPES)) + 1)
    random.shuffle(archetype_cycle)

    for i in range(n_legit):
        archetype = archetype_cycle[i]
        depth = random.choice([2, 3, 3, 4])  # valid chain
        agents.append(Agent(
            agent_id=_addr(),
            declared_archetype=archetype,
            true_archetype=archetype,
            is_adversarial=False,
            adversary_kind=None,
            attestation_depth=depth,
            intent_hash="0x" + uuid.uuid4().hex[:64],
            created_at=(start + timedelta(hours=random.randint(0, 24 * 90))).isoformat(),
        ))

    for _ in range(n_adversarial):
        kind = random.choice(list(ADVERSARIES))
        cfg = ADVERSARIES[kind]
        declared = cfg["declared"]
        depth = 0 if declared is None else random.choice([1, 2])
        agents.append(Agent(
            agent_id=_addr(),
            declared_archetype=declared,
            true_archetype=kind,
            is_adversarial=True,
            adversary_kind=kind,
            attestation_depth=depth,
            intent_hash="0x" + uuid.uuid4().hex[:64],
            created_at=(start + timedelta(hours=random.randint(0, 24 * 90))).isoformat(),
        ))

    return agents


def generate_transactions(
    agents: list[Agent],
    horizon_days: int = 30,
    collusion_sink: str | None = None,
) -> list[Transaction]:
    start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=horizon_days)
    txs: list[Transaction] = []

    sink = collusion_sink or _addr()

    for agent in agents:
        cfg = ADVERSARIES[agent.adversary_kind] if agent.is_adversarial else ARCHETYPES[agent.true_archetype]
        pool = [_addr() for _ in range(cfg["counterparty_pool"])]
        if agent.is_adversarial and agent.adversary_kind == "collusion_ring_member":
            pool = [sink]

        ts = start + timedelta(hours=random.uniform(0, cfg["interval_hours_mu"]))
        per_agent_cap = 1500  # cap so high-frequency archetypes do not dominate
        emitted = 0
        while ts < end and emitted < per_agent_cap:
            if not _in_active_hours(ts, cfg["active_hours"]):
                ts += timedelta(hours=1)
                continue

            amount = _sample_amount(cfg["amount_mu"], cfg["amount_sigma"])
            counterparty = random.choice(pool)
            calldata = int(np.random.normal(cfg["calldata_bytes"], 10))
            gas = round(np.random.normal(0.05 * cfg["gas_multiplier"], 0.01), 4)

            # intent_action_delta: humans never produce this signal.
            # Honest agents stay near 0. Compromised/injected agents drift.
            if agent.is_adversarial and agent.adversary_kind in {
                "identity_mismatch_drainer",
                "compromised_trading_bot",
                "unattested_exfil",
            }:
                delta = float(np.clip(np.random.normal(0.7, 0.15), 0, 1))
            else:
                delta = float(np.clip(np.random.normal(0.05, 0.03), 0, 1))

            txs.append(Transaction(
                tx_id="0x" + uuid.uuid4().hex,
                agent_id=agent.agent_id,
                ts=ts.isoformat(),
                amount_usd=amount,
                counterparty=counterparty,
                calldata_bytes=max(calldata, 4),
                gas_price_gwei=max(gas, 0.001),
                intent_action_delta=delta,
                attestation_depth=agent.attestation_depth,
            ))
            emitted += 1
            ts += timedelta(hours=_sample_interval(cfg["interval_hours_mu"], cfg["interval_hours_sigma"]))

    return txs


def main() -> None:
    os.makedirs(OUT_PATH.parent, exist_ok=True)

    agents = generate_agents(n_legit=200, n_adversarial=30)
    txs = generate_transactions(agents, horizon_days=30)

    with open(AGENTS_PATH, "w") as f:
        for a in agents:
            f.write(json.dumps(asdict(a)) + "\n")

    with open(OUT_PATH, "w") as f:
        for t in txs:
            f.write(json.dumps(asdict(t)) + "\n")

    legit = sum(1 for a in agents if not a.is_adversarial)
    adv = sum(1 for a in agents if a.is_adversarial)
    print(f"agents: {len(agents)} ({legit} legitimate, {adv} adversarial)")
    print(f"transactions: {len(txs)}")
    print(f"wrote: {AGENTS_PATH}")
    print(f"wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
