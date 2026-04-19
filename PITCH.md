# Canopy: Command Center submission

## One-liner

A risk layer calibrated to machine-initiated transactions. Humans give you
vibes, agents give you receipts.

## The problem

Every fraud model running in a bank, PSP, or onchain risk engine was
trained on human behavior. The moment an AI agent signs a payment, those
models are out of distribution. Either the agent gets blocked (false
positive tax that kills adoption) or the adversarial agent slips through
because it looks nothing like a human fraudster.

The volume of machine-initiated transactions is not a future problem.
x402 is shipping. ERC-8004 is live. Subscription and treasury agents
are already moving money on Base every day. Risk infrastructure for this
stack does not exist.

## The product

Canopy is a scoring service that sits between the agent and the payment
rail. Every transaction returns a risk score with a four-signal breakdown
and an accept, review, or block decision.

The model was designed around a single insight: agents leave onchain
receipts that humans cannot. We combine those receipts into four scores:

1. **Behavior score.** HDBSCAN over 15 per-agent features (amount
   distribution, inter-arrival time, counterparty entropy, calldata
   shape, gas strategy, night ratio) discovers agent archetypes without
   labels. A Gaussian Mixture gives us a density, and agents off the
   manifold get a high behavior score.

2. **Identity score.** The ERC-8004 attestation declares an archetype.
   Canopy checks whether the observed cluster matches. If an agent
   declares `subscription_payer` but lives in the `arbitrage_bot`
   cluster, the mismatch alone is enough to block.

3. **Intent score.** x402 payment envelopes carry signed intent.
   Compromised or prompt-injected agents drift from signed intent,
   which we surface directly.

4. **Graph score.** Collusion rings are invisible to behavioral
   anomaly detection because the ring forms its own internally
   consistent cluster. We catch them by watching the fan-in of the
   ring's shared sink.

The recursive kicker: Canopy itself registers as an ERC-8004 agent.
Its outputs are signed, so other agents can query Canopy before
accepting payment from a counterparty. The risk score becomes a signed
onchain primitive.

## Results on real Base mainnet traffic

We scraped 2,880 real x402 settlements from Base USDC
(`AuthorizationUsed` events over a 70 minute window) giving 220 real
payer agents and 252 real merchants. On top of that real population we
injected 21 adversaries across four classes calibrated to the x402
micropayment distribution.

| class          | caught at threshold 0.5 |
|----------------|-------------------------|
| big_drainer    | 100%                    |
| spam_burst     | 100%                    |
| gas_anomaly    | 100%                    |
| ring_collusion | 100%                    |

False positive rate on real legitimate x402 agents: **1.75%** (1/57).

![confusion](dashboard/confusion.png)

We also run a fully-synthetic multi-archetype benchmark (230 agents,
245k tx): 100% TPR, 2.9% FPR. The model is unsupervised, so the same
pipeline transfers without retuning.

## Why this is a data business

The x402 facilitator logs and ERC-8004 attestation chain are public by
design. The moat is not access, it is labeling and interpretation, the
same model Chainalysis built on public chain data. Every new dev who
adopts the Canopy SDK feeds the data loop, and every honeypot agent
Canopy runs on mainnet surfaces free adversarial labels.

This is why we are building Canopy as infrastructure, not as a feature:
infrastructure compounds, features do not.

## Roadmap

| week 0 | hackathon MVP, synthetic data, live API                       |
| week 2 | ingest real Base x402 traffic, retrain, ship TypeScript SDK   |
| week 4 | honeypot agents on mainnet                                    |
| week 8 | first design partner on Base, onchain signed score attestations |

## Author

Ng Ju Peng, jupeng2015@gmail.com
