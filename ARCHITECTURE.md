# Canopy Architecture

```
                         ┌──────────────────────┐
  agent tx -------->     │   POST /score        │
  (x402 payload +        │   api/server.py      │
   ERC-8004 header)      └──────────┬───────────┘
                                    │
                                    v
                        ┌───────────────────────┐
                        │ per-agent rolling     │
                        │ window (size 64)      │
                        │ features/extract.py   │
                        └───────────┬───────────┘
                                    │ feature vector (15-d)
                                    v
                        ┌───────────────────────┐
                        │ StandardScaler        │
                        └───────────┬───────────┘
                                    │
                    ┌───────────────┼──────────────────┐
                    v               v                  v
           ┌────────────┐   ┌──────────────┐   ┌──────────────┐
           │ HDBSCAN    │   │ GMM density  │   │ graph fanin  │
           │ clusters   │   │ log-likelihood│   │ + top sink  │
           └─────┬──────┘   └──────┬───────┘   └──────┬───────┘
                 │                 │                  │
                 v                 v                  v
         identity_score     behavior_score      graph_score
                 │                 │                  │
                 └──────────┬──────┴──────────┬───────┘
                            v                 v
                      ┌──────────────────────────┐
                      │  risk_score =            │
                      │   0.55 * behavior        │
                      │ + 0.15 * identity        │
                      │ + 0.10 * intent          │
                      │ + 0.20 * graph           │
                      └────────────┬─────────────┘
                                   v
                   accept | review | block decision
```

## Feature set (15 dimensions)

| feature                      | intuition                                 |
|------------------------------|-------------------------------------------|
| amount_log_mu, sigma, cv     | payment size distribution                 |
| interval_log_mu, sigma       | inter-arrival time shape                  |
| counterparty_entropy         | how spread across receivers               |
| counterparty_unique_ratio    | fresh address every tx vs repeating       |
| calldata_mu, sigma           | call shape signature                      |
| gas_mu                       | gas strategy                              |
| night_ratio                  | fraction of tx between 0-6 UTC            |
| intent_action_delta_mu       | signed-intent vs executed action          |
| attestation_depth            | ERC-8004 identity chain length            |
| top_sink_shared_fanin        | distinct agents sharing your top sink     |
| top_sink_volume_share        | concentration of volume on top sink       |

## Model components

- **StandardScaler** fit on legitimate-only training split. Prevents
  adversaries from polluting the normalization.

- **HDBSCAN**(min_cluster_size=6, min_samples=3, selection=eom). Density
  clustering that handles variable-density archetypes without picking `k`.
  `prediction_data=True` enables streaming inference via
  `hdbscan.approximate_predict`.

- **Gaussian Mixture**(n_components = |clusters|, full covariance).
  Density estimator over scaled, clustered legit points. Its negative
  log-likelihood is our behavior score, normalized to 0-1 using the 5th
  and 99th percentile of training log-likelihoods.

- **Cluster -> archetype map**: for each HDBSCAN cluster, the dominant
  declared archetype among its legitimate members. Used to detect
  identity-behavior mismatch at inference.

## Streaming pipeline

Each agent has an `AgentWindow` (deque, capacity 64). Every incoming tx
appends to the window, and once the window has >=5 tx the API computes
features and scores. This means fresh agents stay in warmup for their
first 5 tx, which is the correct product behavior.

## Decision boundary

```
risk < 0.40   -> accept
0.40 - 0.60   -> review (human in the loop)
risk >= 0.60  -> block
```

## Known limits

- Low-frequency archetypes (treasury rebalancer, 1-2 tx / week) need a
  longer warmup and collect small-sample variance. Production would bias
  them toward review, never block, until the window saturates.

- Real x402 traffic is ingested directly from Base mainnet USDC
  `AuthorizationUsed` events via `data/scrape_base.py`. Synthetic
  generator remains available as an offline fallback and for
  architectural experiments.

- Identity score is neutral (0.0) when an agent has no ERC-8004
  attestation, rather than maximally suspicious. This reflects the
  current state where most Base x402 payers are unattested. As
  attestation coverage grows, the weight will shift from behavior
  toward identity without code changes (the model is already wired for
  it).

- Graph score fires only when fan-in exceeds the 95th percentile of
  the trained legitimate distribution AND concentration is >= 0.9.
  On real Base traffic this correctly ignores legitimate popular
  merchants (subscription APIs with 20+ distinct subscribers all
  sending share=1.0) while still catching tight collusion rings
  converging on a brand-new sink.

- Adaptive adversaries that exactly mimic a legit cluster are only
  caught by the graph score and intent-action drift. Continuous
  adversarial red-teaming is required to keep the model honest.
