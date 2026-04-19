"""Live scoring demo against the running API.

Streams real transactions from our synthetic dataset for one legitimate
agent and one attacking agent, printing the risk trajectory as each tx
lands.

Usage:
    .venv/bin/python demo/live_demo.py [--host http://127.0.0.1:8765]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TX_PATH = ROOT / "data" / "transactions.jsonl"
AGENT_PATH = ROOT / "data" / "agents.jsonl"


def load_agents() -> dict:
    out = {}
    for line in open(AGENT_PATH):
        a = json.loads(line)
        out[a["agent_id"]] = a
    return out


def post(host: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        host + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


COLORS = {
    "accept": "\033[92m",
    "review": "\033[93m",
    "block": "\033[91m",
    "end": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
}


def pretty(result: dict) -> str:
    d = result["decision"]
    color = COLORS[d]
    return (
        f"{color}{d.upper():<6}{COLORS['end']} "
        f"risk={result['risk_score']:.2f} "
        f"b={result['behavior_score']:.2f} "
        f"id={result['identity_score']:.2f} "
        f"intent={result['intent_score']:.2f} "
        f"graph={result['graph_score']:.2f}"
    )


def run_agent(host: str, agent: dict, txs: list[dict], max_tx: int = 40) -> None:
    label = "ADVERSARY" if agent["is_adversarial"] else "LEGIT   "
    kind = agent.get("adversary_kind") or agent["true_archetype"]
    declared = agent.get("declared_archetype") or "<unattested>"
    print()
    print(f"{COLORS['bold']}{label} {kind}{COLORS['end']}  declared={declared}  agent={agent['agent_id'][:12]}")
    print("-" * 90)
    blocked_once = False
    for i, tx in enumerate(txs[:max_tx], 1):
        payload = {
            "tx_id": tx["tx_id"],
            "agent_id": tx["agent_id"],
            "ts": tx["ts"],
            "amount_usd": tx["amount_usd"],
            "counterparty": tx["counterparty"],
            "calldata_bytes": tx["calldata_bytes"],
            "gas_price_gwei": tx["gas_price_gwei"],
            "intent_action_delta": tx["intent_action_delta"],
            "attestation_depth": tx["attestation_depth"],
            "declared_archetype": agent.get("declared_archetype"),
        }
        result = post(host, "/score", payload)
        if result["status"] == "warmup":
            print(f"  tx#{i:<3} {COLORS['dim']}warmup{COLORS['end']}")
            continue
        print(f"  tx#{i:<3} {pretty(result)}")
        if result["reasons"] and not blocked_once:
            for r in result["reasons"]:
                print(f"         -> {r}")
        if result["decision"] == "block":
            blocked_once = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8765")
    args = ap.parse_args()

    agents = load_agents()
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for line in open(TX_PATH):
        t = json.loads(line)
        by_agent[t["agent_id"]].append(t)
    for txs in by_agent.values():
        txs.sort(key=lambda t: t["ts"])

    # Pick one of each kind we have enough tx for.
    picks = []
    seen_kinds = set()
    for agent_id, agent in agents.items():
        k = agent.get("adversary_kind") or agent["true_archetype"]
        if k in seen_kinds:
            continue
        if len(by_agent[agent_id]) < 10:
            continue
        picks.append((agent, by_agent[agent_id]))
        seen_kinds.add(k)
        if len(picks) >= 10:
            break

    print(f"{COLORS['bold']}=== CANOPY LIVE DEMO ==={COLORS['end']}")
    print(f"Streaming to {args.host}/score")
    for agent, txs in picks:
        run_agent(args.host, agent, txs)


if __name__ == "__main__":
    main()
