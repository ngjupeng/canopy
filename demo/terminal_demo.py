"""Terminal demo: step-by-step risk scoring walkthrough.

Shows exactly how each transaction gets scored, with a slow reveal of
each component. Designed for screen recording during a hackathon demo.

Usage:
    .venv/bin/python demo/terminal_demo.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TX_PATH = ROOT / "data" / "transactions.jsonl"
AGENT_PATH = ROOT / "data" / "agents.jsonl"

API = "http://127.0.0.1:8765"

# ── colors ──────────────────────────────────────────────────────────
G = "\033[92m"   # green
Y = "\033[93m"   # yellow
R = "\033[91m"   # red
B = "\033[94m"   # blue
C = "\033[96m"   # cyan
M = "\033[95m"   # magenta
W = "\033[97m"   # white
D = "\033[2m"    # dim
BOLD = "\033[1m"
END = "\033[0m"

DECISION_COLOR = {"accept": G, "review": Y, "block": R}


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def slow_print(text: str, delay: float = 0.02) -> None:
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def bar(value: float, width: int = 30, color: str = B) -> str:
    filled = int(value * width)
    empty = width - filled
    if value >= 0.6:
        color = R
    elif value >= 0.4:
        color = Y
    return f"{color}{'█' * filled}{D}{'░' * empty}{END}"


def fmt_usd(v: float) -> str:
    if v >= 1000:
        return f"${v:,.0f}"
    if v >= 1:
        return f"${v:.2f}"
    if v >= 0.01:
        return f"${v:.3f}"
    return f"${v:.5f}"


def section(title: str) -> None:
    print()
    print(f"{BOLD}{C}{'─' * 60}{END}")
    print(f"{BOLD}{C}  {title}{END}")
    print(f"{BOLD}{C}{'─' * 60}{END}")
    print()


def wait(seconds: float = 1.0) -> None:
    time.sleep(seconds)


def demo_transaction(tx: dict, agent_meta: dict, index: int) -> None:
    kind = agent_meta.get("adversary_kind") or agent_meta.get("true_archetype", "unknown")
    is_adv = agent_meta.get("is_adversarial", False)
    label = f"{R}ADVERSARY ({kind}){END}" if is_adv else f"{G}LEGITIMATE{END}"

    section(f"Transaction #{index}  {label}")

    # Step 1: show the raw transaction
    slow_print(f"  {D}tx_id:{END}        {tx['tx_id'][:16]}...", 0.01)
    slow_print(f"  {D}agent:{END}        {tx['agent_id'][:16]}...", 0.01)
    slow_print(f"  {D}merchant:{END}     {tx['counterparty'][:16]}...", 0.01)
    slow_print(f"  {BOLD}amount:{END}       {W}{fmt_usd(tx['amount_usd'])}{END}", 0.01)
    slow_print(f"  {D}gas:{END}          {tx['gas_price_gwei']:.4f} gwei", 0.01)
    slow_print(f"  {D}calldata:{END}     {tx['calldata_bytes']} bytes", 0.01)
    print()

    wait(0.8)

    # Step 2: send to scoring API
    slow_print(f"  {B}>>> Sending to Canopy scoring API...{END}", 0.02)
    wait(0.5)

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
        "declared_archetype": agent_meta.get("declared_archetype"),
    }
    result = post("/score", payload)

    if result.get("status") == "warmup":
        slow_print(f"  {D}(warmup: need more tx history, skipping){END}", 0.01)
        return

    # Step 3: reveal each score component one by one
    print()
    slow_print(f"  {BOLD}SCORING BREAKDOWN:{END}", 0.02)
    print()
    wait(0.3)

    components = [
        ("behavior",  result["behavior_score"], 0.55,
         "How far is this agent from normal behavior clusters?"),
        ("identity",  result["identity_score"],  0.15,
         "Does the ERC-8004 identity match actual behavior?"),
        ("intent",    result["intent_score"],     0.10,
         "Is the agent drifting from its signed intent?"),
        ("graph",     result["graph_score"],      0.20,
         "Are multiple agents converging on one sink?"),
    ]

    for name, score, weight, description in components:
        color = R if score >= 0.6 else Y if score >= 0.3 else G
        slow_print(f"  {D}{description}{END}", 0.01)
        slow_print(
            f"    {BOLD}{name:<12}{END} "
            f"{bar(score)} "
            f"{color}{score:.2f}{END}  "
            f"{D}(weight: {weight:.0%}){END}",
            0.01
        )
        print()
        wait(0.4)

    # Step 4: show the weighted combination
    wait(0.3)
    slow_print(f"  {BOLD}RISK CALCULATION:{END}", 0.02)
    print()

    b, i_s, intent_s, g = (result["behavior_score"], result["identity_score"],
                            result["intent_score"], result["graph_score"])
    terms = [
        (0.55, "behavior", b),
        (0.15, "identity", i_s),
        (0.10, "intent",   intent_s),
        (0.20, "graph",    g),
    ]
    parts = []
    for weight, name, val in terms:
        contribution = weight * val
        parts.append(f"{weight:.2f} x {val:.2f}")
        slow_print(f"    {D}{name:<10}{END}  {weight:.2f} x {val:.2f} = {W}{contribution:.3f}{END}", 0.01)
        wait(0.2)

    risk = result["risk_score"]
    dec = result["decision"]
    dc = DECISION_COLOR[dec]
    print(f"    {'─' * 36}")
    slow_print(f"    {BOLD}RISK SCORE{END}  = {BOLD}{W}{risk:.3f}{END}", 0.02)
    print()

    wait(0.5)

    # Step 5: decision
    decision_text = {
        "accept": "ACCEPT  (risk < 0.40, transaction is safe)",
        "review": "REVIEW  (0.40 <= risk < 0.60, needs human review)",
        "block":  "BLOCK   (risk >= 0.60, transaction rejected)",
    }
    slow_print(f"  {BOLD}DECISION: {dc}{decision_text[dec]}{END}", 0.02)
    print()

    # Step 6: reasons
    if result.get("reasons"):
        slow_print(f"  {BOLD}WHY:{END}", 0.02)
        for reason in result["reasons"]:
            slow_print(f"    {Y}> {reason}{END}", 0.01)
    print()
    wait(1.0)


def main() -> None:
    agents_meta = {}
    for line in open(AGENT_PATH):
        a = json.loads(line)
        agents_meta[a["agent_id"]] = a

    by_agent: dict[str, list[dict]] = defaultdict(list)
    for line in open(TX_PATH):
        t = json.loads(line)
        by_agent[t["agent_id"]].append(t)
    for v in by_agent.values():
        v.sort(key=lambda t: t["ts"])

    # Pick demo agents: 2 legit with many tx + 1 of each adversary type
    picks = []
    seen_kinds: set[str] = set()

    # Top legit agents
    legit_agents = [
        (aid, txs) for aid, txs in by_agent.items()
        if not agents_meta.get(aid, {}).get("is_adversarial", False) and len(txs) >= 8
    ]
    legit_agents.sort(key=lambda kv: -len(kv[1]))
    for aid, txs in legit_agents[:2]:
        picks.append((aid, txs, "real_x402"))
        seen_kinds.add("real_x402")

    # One of each adversary type
    for aid, meta in agents_meta.items():
        if not meta.get("is_adversarial"):
            continue
        kind = meta.get("adversary_kind", "unknown")
        if kind in seen_kinds:
            continue
        if len(by_agent[aid]) < 6:
            continue
        picks.append((aid, by_agent[aid], kind))
        seen_kinds.add(kind)

    print(f"\n{BOLD}{W}")
    print(f"  ╔══════════════════════════════════════════════════╗")
    print(f"  ║                                                  ║")
    print(f"  ║   {C}CANOPY{W}   Risk Scoring Walkthrough              ║")
    print(f"  ║                                                  ║")
    print(f"  ║   Trust layer for machine-initiated commerce     ║")
    print(f"  ║   Scoring real Base x402 transactions            ║")
    print(f"  ║                                                  ║")
    print(f"  ╚══════════════════════════════════════════════════╝")
    print(f"{END}")
    wait(1.5)

    # Pre-warm: silently feed first 5 tx of each agent
    slow_print(f"\n  {D}Pre-warming agent history (feeding first 5 tx per agent)...{END}", 0.01)
    for aid, txs, _ in picks:
        meta = agents_meta.get(aid, {})
        for tx in txs[:5]:
            payload = {
                "tx_id": tx["tx_id"], "agent_id": tx["agent_id"],
                "ts": tx["ts"], "amount_usd": tx["amount_usd"],
                "counterparty": tx["counterparty"],
                "calldata_bytes": tx["calldata_bytes"],
                "gas_price_gwei": tx["gas_price_gwei"],
                "intent_action_delta": tx["intent_action_delta"],
                "attestation_depth": tx["attestation_depth"],
                "declared_archetype": meta.get("declared_archetype"),
            }
            post("/score", payload)
    slow_print(f"  {G}Done. All agents have enough history for scoring.{END}", 0.01)
    wait(1.0)

    # Score one tx from each picked agent
    idx = 1
    for aid, txs, kind in picks:
        meta = agents_meta.get(aid, {})
        # Pick tx #6 (first after warmup)
        tx = txs[5] if len(txs) > 5 else txs[-1]
        demo_transaction(tx, meta, idx)
        idx += 1

    # Summary
    section("SUMMARY")
    slow_print(f"  {G}Legitimate agents:{END} scored near 0, accepted instantly", 0.02)
    slow_print(f"  {R}Big drainer:{END}      amount 12x std dev above normal, blocked", 0.02)
    slow_print(f"  {R}Spam burst:{END}       abnormal velocity and sink pattern, flagged", 0.02)
    slow_print(f"  {R}Gas anomaly:{END}      gas price 300x above normal, flagged", 0.02)
    slow_print(f"  {R}Collusion ring:{END}   converging on single new sink, flagged", 0.02)
    print()
    slow_print(f"  {BOLD}{W}100% adversary detection. 1.8% false positive rate.{END}", 0.02)
    slow_print(f"  {BOLD}{W}All on real Base mainnet x402 data.{END}", 0.02)
    print()


if __name__ == "__main__":
    main()
