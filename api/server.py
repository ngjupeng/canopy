"""Canopy risk scoring API.

POST /score  accepts a transaction from an agent. We keep a rolling window
per agent, extract features once the window has >=5 tx, and return a risk
score with component breakdown.

Also accepts an ERC-8004 style declared_archetype so we can detect
identity-behavior mismatch in real time.

Run:
    .venv/bin/uvicorn api.server:app --reload --port 8000
"""
from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features.extract import AgentWindow  # noqa: E402

MODEL_PATH = ROOT / "model" / "canopy_model.pkl"

with open(MODEL_PATH, "rb") as f:
    ARTIFACT = pickle.load(f)

SCALER = ARTIFACT["scaler"]
CLUSTERER = ARTIFACT["clusterer"]
GMM = ARTIFACT["gmm"]
CLUSTER_ARCHETYPE = ARTIFACT["cluster_archetype_map"]
B_LO = ARTIFACT["b_lo"]
B_HI = ARTIFACT["b_hi"]
FEATURE_COLS = ARTIFACT["feature_cols"]
FANIN_P95 = ARTIFACT.get("fanin_p95", 10.0)

import hdbscan  # noqa: E402

app = FastAPI(title="Canopy Risk API", version="0.1.0")

WINDOWS: dict[str, AgentWindow] = {}
FANIN: dict[str, int] = {}
AGENTS_SENDING_TO: dict[str, set[str]] = {}


class Transaction(BaseModel):
    tx_id: str
    agent_id: str
    ts: str
    amount_usd: float
    counterparty: str
    calldata_bytes: int
    gas_price_gwei: float
    intent_action_delta: float = Field(0.0, ge=0.0, le=1.0)
    attestation_depth: int = 0
    declared_archetype: str | None = None


class ScoreResponse(BaseModel):
    tx_id: str
    agent_id: str
    status: Literal["ok", "warmup"]
    risk_score: float
    behavior_score: float
    identity_score: float
    intent_score: float
    graph_score: float
    assigned_cluster: int | None
    declared_archetype: str | None
    expected_archetype: str | None
    decision: Literal["accept", "review", "block"]
    reasons: list[str]


def _score_features(feats: np.ndarray, declared: str | None) -> dict:
    z = SCALER.transform(feats.reshape(1, -1))
    ll = float(GMM.score_samples(z)[0])
    behavior = -ll
    b_norm = float(np.clip((behavior - B_LO) / max(B_HI - B_LO, 1e-9), 0, 1))

    cluster_ids, _ = hdbscan.approximate_predict(CLUSTERER, z)
    cluster_id = int(cluster_ids[0])

    if declared is None:
        # ERC-8004 not yet universal; missing attestation is neutral.
        identity = 0.0
        expected = None
    elif cluster_id == -1:
        identity = 0.6
        expected = None
    else:
        expected = CLUSTER_ARCHETYPE.get(cluster_id)
        if expected is None:
            identity = 0.4
        elif expected != declared:
            identity = 0.9
        else:
            identity = 0.0

    fanin = float(feats[-2])
    share = float(feats[-1])
    fanin_excess = max((fanin - FANIN_P95) / max(FANIN_P95, 1.0), 0.0)
    fanin_excess = min(fanin_excess / 3.0, 1.0)
    graph = fanin_excess if share >= 0.9 else 0.0

    intent = float(feats[FEATURE_COLS.index("intent_action_delta_mu")])

    risk = float(np.clip(
        0.55 * b_norm + 0.15 * identity + 0.10 * intent + 0.20 * graph, 0, 1
    ))

    reasons = []
    if b_norm >= 0.5:
        # Identify WHICH features are most anomalous by z-score so the
        # reason text is specific rather than a generic "off-manifold".
        z_flat = z[0]
        feature_labels = {
            "amount_log_mu": "transaction amount",
            "amount_log_sigma": "amount variance",
            "amount_cv": "amount volatility",
            "interval_log_mu": "transaction velocity",
            "interval_log_sigma": "timing variance",
            "counterparty_entropy": "counterparty diversity",
            "counterparty_unique_ratio": "counterparty spread",
            "calldata_mu": "calldata size",
            "calldata_sigma": "calldata variance",
            "gas_mu": "gas price",
            "night_ratio": "night-hours activity",
            "intent_action_delta_mu": "intent drift",
            "attestation_depth": "attestation chain",
            "top_sink_shared_fanin": "sink fan-in",
            "top_sink_volume_share": "sink concentration",
        }
        anomalies = []
        for i, col in enumerate(FEATURE_COLS):
            if i < len(z_flat) and abs(z_flat[i]) > 2.0:
                label = feature_labels.get(col, col)
                direction = "high" if z_flat[i] > 0 else "low"
                anomalies.append((abs(z_flat[i]), f"{label} is abnormally {direction} ({abs(z_flat[i]):.1f}x std dev)"))
        anomalies.sort(key=lambda x: -x[0])
        if anomalies:
            top = anomalies[:2]
            reasons.extend(detail for _, detail in top)
        else:
            reasons.append(f"off-manifold behavior (score={b_norm:.2f})")
    if identity >= 0.6:
        if expected and declared and expected != declared:
            reasons.append(
                f"identity mismatch: declared {declared!r}, behavior matches {expected!r}"
            )
        elif declared is None:
            reasons.append("agent has no ERC-8004 attestation")
        else:
            reasons.append("agent sits off any known cluster")
    if intent >= 0.3:
        reasons.append(f"intent-action drift (delta={intent:.2f})")
    if graph >= 0.5:
        reasons.append(
            f"collusion signal: top sink shared with {int(fanin)} agents, {share:.0%} volume"
        )

    if risk >= 0.60:
        decision = "block"
    elif risk >= 0.40:
        decision = "review"
    else:
        decision = "accept"

    return {
        "risk_score": risk,
        "behavior_score": b_norm,
        "identity_score": identity,
        "intent_score": intent,
        "graph_score": graph,
        "assigned_cluster": None if cluster_id == -1 else cluster_id,
        "expected_archetype": expected,
        "decision": decision,
        "reasons": reasons,
    }


@app.get("/")
def root() -> dict:
    return {
        "service": "Canopy Risk API",
        "version": "0.1.0",
        "model": {
            "feature_count": len(FEATURE_COLS),
            "clusters": len(CLUSTER_ARCHETYPE),
            "archetypes": sorted(set(CLUSTER_ARCHETYPE.values())),
        },
    }


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction) -> ScoreResponse:
    AGENTS_SENDING_TO.setdefault(tx.counterparty, set()).add(tx.agent_id)
    FANIN[tx.counterparty] = len(AGENTS_SENDING_TO[tx.counterparty])

    win = WINDOWS.get(tx.agent_id)
    if win is None:
        win = AgentWindow.new(tx.agent_id, tx.attestation_depth)
        WINDOWS[tx.agent_id] = win
    win.add(tx.model_dump())

    feats = win.features(FANIN)
    if feats is None:
        return ScoreResponse(
            tx_id=tx.tx_id,
            agent_id=tx.agent_id,
            status="warmup",
            risk_score=0.0,
            behavior_score=0.0,
            identity_score=0.0,
            intent_score=0.0,
            graph_score=0.0,
            assigned_cluster=None,
            declared_archetype=tx.declared_archetype,
            expected_archetype=None,
            decision="accept",
            reasons=["warmup: need 5 tx before scoring"],
        )

    result = _score_features(feats, tx.declared_archetype)
    return ScoreResponse(
        tx_id=tx.tx_id,
        agent_id=tx.agent_id,
        status="ok",
        declared_archetype=tx.declared_archetype,
        **result,
    )


@app.get("/health")
def health() -> dict:
    return {"ok": True, "agents_tracked": len(WINDOWS)}


# ---------------------------------------------------------------------------
# Demo UI assets and feed
# ---------------------------------------------------------------------------

UI_DIR = ROOT / "ui"
DATA_TX = ROOT / "data" / "transactions.jsonl"
DATA_AGENTS = ROOT / "data" / "agents.jsonl"

if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/ui")
def ui_index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


# Pre-curated demo feed: balanced mix of real legit + adversaries, shuffled
# so the demo has dramatic variety without being predictable.
_DEMO_FEED: list[dict] | None = None


def _load_demo_feed() -> list[dict]:
    """Curate a demo feed so the UI sees real scoring, not just warmup.

    Pairs with prewarm_windows() (called on startup): we pick a small set
    of high-activity agents and adversaries, push their first 5 tx into
    the API state silently, then expose the rest of their tx for the live
    demo. Result: every card on screen is a meaningful scored tx.
    """
    global _DEMO_FEED
    if _DEMO_FEED is not None:
        return _DEMO_FEED

    agents_meta: dict[str, dict] = {}
    if DATA_AGENTS.exists():
        for line in open(DATA_AGENTS):
            a = json.loads(line)
            agents_meta[a["agent_id"]] = a

    txs: list[dict] = []
    if DATA_TX.exists():
        for line in open(DATA_TX):
            txs.append(json.loads(line))

    by_agent: dict[str, list[dict]] = {}
    for t in txs:
        by_agent.setdefault(t["agent_id"], []).append(t)
    for v in by_agent.values():
        v.sort(key=lambda t: t["ts"])

    # Top legit agents by tx count (active subscribers and bots that
    # produce enough volume to actually be scored).
    legit_agents = [
        (aid, agent_txs) for aid, agent_txs in by_agent.items()
        if not agents_meta.get(aid, {}).get("is_adversarial", False)
    ]
    legit_agents.sort(key=lambda kv: -len(kv[1]))
    top_legit = legit_agents[:35]

    adv_agents = [
        (aid, agent_txs) for aid, agent_txs in by_agent.items()
        if agents_meta.get(aid, {}).get("is_adversarial", False)
    ]

    def _mark(agent_id: str, t: dict) -> dict:
        meta = agents_meta.get(agent_id, {})
        return {
            **t,
            "_is_adversarial": meta.get("is_adversarial", False),
            "_kind": meta.get("adversary_kind") or meta.get("true_archetype", "real_x402"),
            "_declared_archetype": meta.get("declared_archetype"),
        }

    # Each agent: first 5 tx are reserved for pre-warm; rest are the
    # streamable demo feed.
    PREWARM = 5
    cap_per_agent = 14
    legit_streams = [list(map(lambda t: _mark(aid, t), txs[PREWARM:cap_per_agent]))
                     for aid, txs in top_legit if len(txs) > PREWARM]
    adv_streams = [list(map(lambda t: _mark(aid, t), txs[PREWARM:cap_per_agent]))
                   for aid, txs in adv_agents if len(txs) > PREWARM]

    random.seed(0)

    # Flatten and shuffle each pool independently.
    legit_flat: list[dict] = []
    for s in legit_streams:
        legit_flat.extend(s)
    random.shuffle(legit_flat)

    adv_flat: list[dict] = []
    for s in adv_streams:
        adv_flat.extend(s)
    random.shuffle(adv_flat)

    # Interleave: 3 legit then 1 adversary = ~25% adversary density.
    feed: list[dict] = []
    li = ai = 0
    while li < len(legit_flat) or ai < len(adv_flat):
        for _ in range(3):
            if li < len(legit_flat):
                feed.append(legit_flat[li]); li += 1
        if ai < len(adv_flat):
            feed.append(adv_flat[ai]); ai += 1

    _DEMO_FEED = feed
    return feed


@app.get("/demo/feed")
def demo_feed(cursor: int = 0, limit: int = 20) -> dict:
    """Return the next slice of the curated demo feed.

    The frontend pulls slices, scores each tx via /score, and animates
    the result. cursor wraps around so the demo loops indefinitely.
    """
    feed = _load_demo_feed()
    if not feed:
        return {"items": [], "next_cursor": 0, "total": 0}
    n = len(feed)
    start = cursor % n
    items = []
    for i in range(limit):
        idx = (start + i) % n
        items.append(feed[idx])
    return {"items": items, "next_cursor": (start + limit) % n, "total": n}


@app.on_event("startup")
def prewarm_windows() -> None:
    """Pre-feed the first 5 tx of every demo agent so the live UI sees
    real scores from the first card on screen instead of warmup placeholders.
    """
    if not DATA_TX.exists() or not DATA_AGENTS.exists():
        return

    agents_meta = {json.loads(l)["agent_id"]: json.loads(l) for l in open(DATA_AGENTS)}
    txs = [json.loads(l) for l in open(DATA_TX)]
    by_agent: dict[str, list[dict]] = {}
    for t in txs:
        by_agent.setdefault(t["agent_id"], []).append(t)
    for v in by_agent.values():
        v.sort(key=lambda t: t["ts"])

    # Same agent set as the demo feed
    legit = sorted(
        [(aid, ts) for aid, ts in by_agent.items()
         if not agents_meta.get(aid, {}).get("is_adversarial", False)],
        key=lambda kv: -len(kv[1])
    )[:35]
    adv = [(aid, ts) for aid, ts in by_agent.items()
           if agents_meta.get(aid, {}).get("is_adversarial", False)]

    warmed = 0
    for agent_id, agent_txs in legit + adv:
        for tx in agent_txs[:5]:
            AGENTS_SENDING_TO.setdefault(tx["counterparty"], set()).add(agent_id)
            FANIN[tx["counterparty"]] = len(AGENTS_SENDING_TO[tx["counterparty"]])
            win = WINDOWS.get(agent_id)
            if win is None:
                win = AgentWindow.new(agent_id, tx.get("attestation_depth", 0))
                WINDOWS[agent_id] = win
            win.add(tx)
        warmed += 1
    print(f"[canopy] pre-warmed {warmed} agents")
