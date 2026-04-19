"""Scrape real x402 transactions from Base mainnet.

x402 settles on Base via EIP-3009 transferWithAuthorization on USDC.
Every settled tx emits an AuthorizationUsed event from the USDC contract.
We use that as our x402 signal, then join in the Transfer event (to get
recipient and value) and the transaction body (to get gas price and
calldata shape).

Writes to data/real_transactions.jsonl in the same schema the rest of
the Canopy pipeline expects. Attestation depth and intent_action_delta
are zeroed because those signals only exist with an off-chain ERC-8004
identity layer that is not yet live for all agents; the rest of the
features (amount, timing, counterparty graph, gas, calldata) come from
chain data directly.

Usage:
    .venv/bin/python data/scrape_base.py --target 1500 --chunk 50
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

OUT_PATH = Path(__file__).resolve().parent / "real_transactions.jsonl"

RPC_ENDPOINTS = [
    "https://base.drpc.org",
    "https://1rpc.io/base",
    "https://mainnet.base.org",
    "https://developer-access-mainnet.base.org",
    "https://base.publicnode.com",
]
USDC = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
TOPIC_AUTH_USED = "0x98de503528ee59b575ef0c0a2576a82497bfc029a5685b209e9ec333479b10a5"
TOPIC_TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


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
    block_number: int
    facilitator: str


class RpcClient:
    def __init__(self, endpoints: list[str]) -> None:
        self.endpoints = endpoints
        self.sessions = {u: requests.Session() for u in endpoints}
        self.idx = 0

    def _rotate(self) -> str:
        url = self.endpoints[self.idx % len(self.endpoints)]
        self.idx += 1
        return url

    def call(self, method: str, params: list, timeout: int = 30) -> dict:
        last_err: Exception | None = None
        for _ in range(len(self.endpoints) * 2):
            url = self._rotate()
            try:
                r = self.sessions[url].post(
                    url,
                    json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
                    timeout=timeout,
                )
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(0.5)
        raise RuntimeError(f"all RPCs failed: {last_err}")

    def batch(self, calls: list[tuple[str, list]], timeout: int = 60) -> list[dict]:
        body = [
            {"jsonrpc": "2.0", "method": m, "params": p, "id": i}
            for i, (m, p) in enumerate(calls)
        ]
        last_err: Exception | None = None
        for _ in range(len(self.endpoints) * 2):
            url = self._rotate()
            try:
                r = self.sessions[url].post(url, json=body, timeout=timeout)
                r.raise_for_status()
                results = r.json()
                if isinstance(results, list):
                    return sorted(results, key=lambda x: x.get("id", 0))
                raise RuntimeError(f"batch returned non-list: {results}")
            except Exception as e:
                last_err = e
                time.sleep(0.8)
        raise RuntimeError(f"all RPCs failed on batch: {last_err}")


def _topic_to_addr(topic: str) -> str:
    return ("0x" + topic[-40:]).lower()


def _int_from_hex(x: str | int | None) -> int:
    if x is None:
        return 0
    if isinstance(x, int):
        return x
    return int(x, 16) if x else 0


def scrape(target: int, chunk: int, sleep: float = 0.2) -> list[Transaction]:
    rpc = RpcClient(RPC_ENDPOINTS)

    latest = _int_from_hex(rpc.call("eth_blockNumber", [])["result"])
    print(f"latest block: {latest}")

    events: dict[str, dict] = {}
    current = latest
    while len(events) < target and current > 0:
        lo = max(current - chunk + 1, 1)
        params = [{
            "address": USDC,
            "topics": [TOPIC_AUTH_USED],
            "fromBlock": hex(lo),
            "toBlock": hex(current),
        }]
        resp = rpc.call("eth_getLogs", params)
        if "error" in resp:
            print(f"  [warn] getLogs error at {lo}-{current}: {resp['error']}, halving chunk")
            chunk = max(chunk // 2, 5)
            current -= 1
            continue
        logs = resp["result"]
        for log in logs:
            tx_hash = log["transactionHash"]
            if tx_hash in events:
                continue
            events[tx_hash] = {
                "block": _int_from_hex(log["blockNumber"]),
                "authorizer": _topic_to_addr(log["topics"][1]),
                "log_index": _int_from_hex(log["logIndex"]),
            }
        print(f"  scanned {lo}-{current}, {len(logs)} events, unique tx={len(events)}")
        current = lo - 1
        time.sleep(sleep)

    tx_hashes = list(events.keys())[:target]
    print(f"enriching {len(tx_hashes)} transactions")

    # Block timestamps (dedupe blocks)
    blocks_needed = sorted({events[h]["block"] for h in tx_hashes})
    block_ts: dict[int, int] = {}
    for i in range(0, len(blocks_needed), 50):
        batch = blocks_needed[i:i + 50]
        resps = rpc.batch([("eth_getBlockByNumber", [hex(bn), False]) for bn in batch])
        for bn, resp in zip(batch, resps):
            r = resp.get("result")
            if r:
                block_ts[bn] = _int_from_hex(r.get("timestamp"))
        print(f"  blocks {i + len(batch)}/{len(blocks_needed)}")
        time.sleep(sleep)

    # Receipts (Transfer log + gas + facilitator)
    receipts: dict[str, dict] = {}
    for i in range(0, len(tx_hashes), 30):
        batch = tx_hashes[i:i + 30]
        resps = rpc.batch([("eth_getTransactionReceipt", [h]) for h in batch])
        for h, resp in zip(batch, resps):
            r = resp.get("result")
            if r:
                receipts[h] = r
        print(f"  receipts {i + len(batch)}/{len(tx_hashes)}")
        time.sleep(sleep)

    # Transactions (for input length)
    txs_raw: dict[str, dict] = {}
    for i in range(0, len(tx_hashes), 30):
        batch = tx_hashes[i:i + 30]
        resps = rpc.batch([("eth_getTransactionByHash", [h]) for h in batch])
        for h, resp in zip(batch, resps):
            r = resp.get("result")
            if r:
                txs_raw[h] = r
        print(f"  tx bodies {i + len(batch)}/{len(tx_hashes)}")
        time.sleep(sleep)

    out: list[Transaction] = []
    for h in tx_hashes:
        ev = events[h]
        rcpt = receipts.get(h)
        tx_raw = txs_raw.get(h)
        if not rcpt or not tx_raw:
            continue

        transfer = None
        for log in rcpt["logs"]:
            if (
                log["address"].lower() == USDC
                and log["topics"][0] == TOPIC_TRANSFER
            ):
                transfer = log
                break
        if not transfer:
            continue

        sender = _topic_to_addr(transfer["topics"][1])
        recipient = _topic_to_addr(transfer["topics"][2])
        value_units = int(transfer["data"], 16)
        value_usd = value_units / 1e6

        bn = ev["block"]
        ts = block_ts.get(bn)
        if not ts:
            continue
        ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        gas_price_wei = _int_from_hex(
            rcpt.get("effectiveGasPrice") or tx_raw.get("gasPrice")
        )
        gas_price_gwei = gas_price_wei / 1e9

        input_hex = tx_raw.get("input", "0x")
        calldata_bytes = max((len(input_hex) - 2) // 2, 4)

        facilitator = (rcpt.get("from") or "0x").lower()

        out.append(Transaction(
            tx_id=h,
            agent_id=sender,
            ts=ts_iso,
            amount_usd=round(value_usd, 6),
            counterparty=recipient,
            calldata_bytes=calldata_bytes,
            gas_price_gwei=round(gas_price_gwei, 6),
            intent_action_delta=0.0,
            attestation_depth=0,
            block_number=bn,
            facilitator=facilitator,
        ))

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    txs = scrape(args.target, args.chunk, args.sleep)
    with open(OUT_PATH, "w") as f:
        for t in txs:
            f.write(json.dumps(asdict(t)) + "\n")

    unique_agents = len({t.agent_id for t in txs})
    unique_recipients = len({t.counterparty for t in txs})
    print(f"wrote {len(txs)} tx to {OUT_PATH}")
    print(f"  unique agents (payers):    {unique_agents}")
    print(f"  unique merchants:          {unique_recipients}")
    if txs:
        amts = sorted(t.amount_usd for t in txs)
        print(f"  amount USD p50 / p95:      "
              f"{amts[len(amts)//2]:.4f} / {amts[int(len(amts)*0.95)]:.4f}")


if __name__ == "__main__":
    main()
