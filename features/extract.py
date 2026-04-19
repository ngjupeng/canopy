"""Feature extraction.

Two extraction modes:
  1. Per-agent aggregate features for training the behavioral clusters.
  2. Rolling window features for scoring a live transaction stream.

Key features (designed so honest agents fall on a low-dim manifold and
adversaries drift off it):
  amount_log_mu, amount_log_sigma, amount_cv
  interval_log_mu, interval_log_sigma
  counterparty_entropy, counterparty_unique_ratio
  calldata_mu, calldata_sigma
  gas_mu
  night_ratio (fraction of tx between 0-6 UTC)
  intent_action_delta_mu     <- attestation-derived signal
  attestation_depth          <- ERC-8004 identity feature
"""
from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TX_PATH = ROOT / "data" / "transactions.jsonl"
AGENT_PATH = ROOT / "data" / "agents.jsonl"
OUT_PATH = ROOT / "features" / "agent_features.csv"

FEATURE_COLS = [
    "amount_log_mu", "amount_log_sigma", "amount_cv",
    "interval_log_mu", "interval_log_sigma",
    "counterparty_entropy", "counterparty_unique_ratio",
    "calldata_mu", "calldata_sigma",
    "gas_mu",
    "night_ratio",
    "intent_action_delta_mu",
    "attestation_depth",
    "top_sink_shared_fanin",  # graph feature: how many OTHER agents
                               # send to this agent's top counterparty
    "top_sink_volume_share",   # fraction of own volume to top counterparty
]


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _counterparty_fanin(tx_df: pd.DataFrame) -> dict[str, int]:
    """For each counterparty address, count how many distinct agents send to it."""
    return (
        tx_df.groupby("counterparty")["agent_id"]
        .nunique()
        .to_dict()
    )


def extract_agent_features(tx_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-agent features over the full history."""
    fanin = _counterparty_fanin(tx_df)
    rows = []
    for agent_id, sub in tx_df.groupby("agent_id"):
        sub = sub.sort_values("ts")
        amounts = sub["amount_usd"].to_numpy()
        log_amt = np.log1p(amounts)

        ts = pd.to_datetime(sub["ts"], format="ISO8601").to_numpy()
        intervals_s = np.diff(ts).astype("timedelta64[s]").astype(float)
        intervals_h = intervals_s / 3600.0
        intervals_h = intervals_h[intervals_h > 0]
        log_int = np.log1p(intervals_h) if len(intervals_h) else np.array([0.0])

        cp_volume = sub.groupby("counterparty")["amount_usd"].sum()
        cp_counts = cp_volume.to_numpy()
        cp_entropy = _entropy(cp_counts) if len(cp_counts) else 0.0
        cp_unique_ratio = len(cp_counts) / max(len(sub), 1)

        top_sink = cp_volume.idxmax() if len(cp_volume) else None
        top_sink_volume_share = float(cp_volume.max() / cp_volume.sum()) if len(cp_volume) else 0.0
        top_sink_shared_fanin = float(fanin.get(top_sink, 1)) if top_sink else 1.0

        calldata = sub["calldata_bytes"].to_numpy()
        gas = sub["gas_price_gwei"].to_numpy()
        hours = pd.to_datetime(sub["ts"], format="ISO8601").dt.hour.to_numpy()
        night_ratio = float(((hours >= 0) & (hours < 6)).mean())

        rows.append({
            "agent_id": agent_id,
            "tx_count": len(sub),
            "amount_log_mu": float(log_amt.mean()),
            "amount_log_sigma": float(log_amt.std() + 1e-9),
            "amount_cv": float(amounts.std() / (amounts.mean() + 1e-9)),
            "interval_log_mu": float(log_int.mean()),
            "interval_log_sigma": float(log_int.std() + 1e-9),
            "counterparty_entropy": cp_entropy,
            "counterparty_unique_ratio": cp_unique_ratio,
            "calldata_mu": float(calldata.mean()),
            "calldata_sigma": float(calldata.std() + 1e-9),
            "gas_mu": float(gas.mean()),
            "night_ratio": night_ratio,
            "intent_action_delta_mu": float(sub["intent_action_delta"].mean()),
            "attestation_depth": int(sub["attestation_depth"].iloc[0]),
            "top_sink_shared_fanin": top_sink_shared_fanin,
            "top_sink_volume_share": top_sink_volume_share,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rolling window features for live scoring (used by API)
# ---------------------------------------------------------------------------

@dataclass
class AgentWindow:
    agent_id: str
    amounts: deque
    timestamps: deque
    counterparties: deque
    calldata: deque
    gas: deque
    deltas: deque
    attestation_depth: int

    @classmethod
    def new(cls, agent_id: str, attestation_depth: int, size: int = 64) -> "AgentWindow":
        return cls(
            agent_id=agent_id,
            amounts=deque(maxlen=size),
            timestamps=deque(maxlen=size),
            counterparties=deque(maxlen=size),
            calldata=deque(maxlen=size),
            gas=deque(maxlen=size),
            deltas=deque(maxlen=size),
            attestation_depth=attestation_depth,
        )

    def add(self, tx: dict) -> None:
        self.amounts.append(float(tx["amount_usd"]))
        self.timestamps.append(pd.to_datetime(tx["ts"]))
        self.counterparties.append(tx["counterparty"])
        self.calldata.append(int(tx["calldata_bytes"]))
        self.gas.append(float(tx["gas_price_gwei"]))
        self.deltas.append(float(tx["intent_action_delta"]))

    def features(self, fanin_lookup: dict[str, int] | None = None) -> np.ndarray | None:
        if len(self.amounts) < 5:
            return None
        amounts = np.array(self.amounts)
        log_amt = np.log1p(amounts)
        ts = np.array([t.to_datetime64() for t in self.timestamps])
        intervals_s = np.diff(ts).astype("timedelta64[s]").astype(float)
        intervals_h = intervals_s / 3600.0
        intervals_h = intervals_h[intervals_h > 0]
        log_int = np.log1p(intervals_h) if len(intervals_h) else np.array([0.0])

        cp_vol: dict[str, float] = {}
        for cp, amt in zip(self.counterparties, amounts):
            cp_vol[cp] = cp_vol.get(cp, 0.0) + float(amt)
        cp_counts = np.array(list(cp_vol.values()), dtype=float)
        cp_entropy = _entropy(cp_counts) if len(cp_counts) else 0.0
        cp_unique_ratio = len(cp_counts) / max(len(self.amounts), 1)

        top_sink = max(cp_vol, key=cp_vol.get) if cp_vol else None
        top_share = float(max(cp_vol.values()) / sum(cp_vol.values())) if cp_vol else 0.0
        top_fanin = float((fanin_lookup or {}).get(top_sink, 1)) if top_sink else 1.0

        calldata = np.array(self.calldata)
        gas = np.array(self.gas)
        hours = np.array([t.hour for t in self.timestamps])
        night_ratio = float(((hours >= 0) & (hours < 6)).mean())
        return np.array([
            float(log_amt.mean()),
            float(log_amt.std() + 1e-9),
            float(amounts.std() / (amounts.mean() + 1e-9)),
            float(log_int.mean()),
            float(log_int.std() + 1e-9),
            cp_entropy,
            cp_unique_ratio,
            float(calldata.mean()),
            float(calldata.std() + 1e-9),
            float(gas.mean()),
            night_ratio,
            float(np.mean(self.deltas)),
            float(self.attestation_depth),
            top_fanin,
            top_share,
        ])


def main() -> None:
    txs = [json.loads(line) for line in open(TX_PATH)]
    tx_df = pd.DataFrame(txs)
    agent_df = pd.DataFrame([json.loads(l) for l in open(AGENT_PATH)])

    feats = extract_agent_features(tx_df)
    feats = feats.merge(
        agent_df[["agent_id", "declared_archetype", "true_archetype",
                  "is_adversarial", "adversary_kind"]],
        on="agent_id", how="left",
    )

    feats.to_csv(OUT_PATH, index=False)
    print(f"extracted features for {len(feats)} agents")
    print(f"feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"wrote: {OUT_PATH}")
    print(feats[["agent_id", "tx_count", "is_adversarial", "true_archetype"]].head())


if __name__ == "__main__":
    main()
