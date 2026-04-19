"""Train the Canopy risk model.

Pipeline:
  1. Load per-agent features.
  2. Fit a StandardScaler on legitimate-only behavior so clusters are not
     contaminated by adversaries (density estimation is unsupervised; we
     only use the is_adversarial flag to hold out a small eval set).
  3. Fit HDBSCAN to discover behavioral archetypes.
  4. Fit a Gaussian Mixture on the clustered legitimate points. Its
     log-likelihood becomes our behavioral-density score: low likelihood
     means the agent sits off the learned manifold.
  5. Persist scaler + HDBSCAN + GMM for inference.

The final risk score blends:
    behavior_score  = -gmm.score_samples(z)        higher = more anomalous
    identity_score  = identity-behavior mismatch between the declared
                      archetype and the cluster the point assigns to
    intent_score    = mean signed-intent vs action delta (direct feature)

Calibration done with a held-out set of 20% legitimate agents + all
adversaries.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import hdbscan

ROOT = Path(__file__).resolve().parent.parent
FEAT_PATH = ROOT / "features" / "agent_features.csv"
ARTIFACT_PATH = ROOT / "model" / "canopy_model.pkl"
METRICS_PATH = ROOT / "model" / "metrics.json"

FEATURE_COLS = [
    "amount_log_mu", "amount_log_sigma", "amount_cv",
    "interval_log_mu", "interval_log_sigma",
    "counterparty_entropy", "counterparty_unique_ratio",
    "calldata_mu", "calldata_sigma",
    "gas_mu",
    "night_ratio",
    "intent_action_delta_mu",
    "attestation_depth",
    "top_sink_shared_fanin",
    "top_sink_volume_share",
]


def compute_cluster_archetype_map(df: pd.DataFrame, cluster_labels: np.ndarray) -> dict[int, str]:
    """For each discovered cluster, record the dominant declared archetype
    among its legitimate members. Used later for identity-behavior mismatch.
    """
    mapping: dict[int, str] = {}
    legit = df[~df["is_adversarial"]].copy()
    legit["cluster"] = cluster_labels[~df["is_adversarial"].to_numpy()]
    for cluster_id, sub in legit.groupby("cluster"):
        if cluster_id == -1:
            continue
        top_archetype = sub["true_archetype"].mode().iloc[0]
        mapping[int(cluster_id)] = top_archetype
    return mapping


def main() -> None:
    df = pd.read_csv(FEAT_PATH)
    df = df[df["tx_count"] >= 3].reset_index(drop=True)

    legit_mask = ~df["is_adversarial"]
    # Hold out 20% of legitimate agents to measure false-positive rate.
    rng = np.random.RandomState(0)
    legit_idx = df.index[legit_mask].to_numpy().copy()
    rng.shuffle(legit_idx)
    n_test = int(0.2 * len(legit_idx))
    test_legit = set(legit_idx[:n_test].tolist())
    train_mask = np.array([i not in test_legit and legit_mask.iloc[i] for i in df.index])

    X_all = df[FEATURE_COLS].to_numpy()
    X_train = df.loc[train_mask, FEATURE_COLS].to_numpy()

    scaler = StandardScaler().fit(X_train)
    Z_train = scaler.transform(X_train)
    Z_all = scaler.transform(X_all)

    # HDBSCAN over scaled legitimate behavior.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=4,
        min_samples=2,
        prediction_data=True,
        cluster_selection_method="eom",
    ).fit(Z_train)

    # Assign clusters to the full dataset via approximate_predict.
    labels_all, _ = hdbscan.approximate_predict(clusterer, Z_all)

    # GMM as density estimator on legitimate, non-noise training points.
    train_labels = clusterer.labels_
    good = train_labels >= 0
    n_components = max(len(set(train_labels[good])), 2)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=0,
        reg_covar=1e-3,
    ).fit(Z_train[good])

    # Behavior score = negative log-likelihood (higher = more anomalous).
    ll_all = gmm.score_samples(Z_all)
    behavior_score = -ll_all

    # Identity-behavior mismatch.
    cluster_archetype = compute_cluster_archetype_map(df, labels_all)
    identity_score = np.zeros(len(df))
    for i, row in df.iterrows():
        cluster_id = int(labels_all[i])
        declared = row["declared_archetype"] if pd.notna(row["declared_archetype"]) else None
        if declared is None:
            # ERC-8004 not yet universal. Treat missing attestation as
            # neutral rather than maximally suspicious.
            identity_score[i] = 0.0
            continue
        if cluster_id == -1:
            identity_score[i] = 0.6
            continue
        expected = cluster_archetype.get(cluster_id)
        if expected is None:
            identity_score[i] = 0.4
        elif expected != declared:
            identity_score[i] = 0.9
        else:
            identity_score[i] = 0.0

    intent_score = df["intent_action_delta_mu"].to_numpy()

    # Graph score: collusion rings share a sink with high concentration.
    # On real x402 traffic many legitimate merchants naturally have high
    # fan-in (subscription services, popular APIs). We calibrate fanin
    # relative to the legitimate training distribution and only fire
    # when BOTH fanin is unusually high AND concentration is near 1.
    fanin = df["top_sink_shared_fanin"].to_numpy()
    share = df["top_sink_volume_share"].to_numpy()
    train_fanin = fanin[train_mask]
    fanin_p95 = float(np.percentile(train_fanin, 95)) if len(train_fanin) else 1.0
    # fanin z against training: clipped to 0 when at or below p95 of legit.
    fanin_excess = np.clip((fanin - fanin_p95) / max(fanin_p95, 1.0), 0, 3) / 3.0
    # Both conditions required: unusually high fan-in AND share > 0.9.
    graph_score = np.where(share >= 0.9, fanin_excess, 0.0)
    graph_score = np.clip(graph_score, 0, 1)

    # Normalize behavior_score on train distribution to 0-1.
    train_behavior = behavior_score[train_mask]
    b_lo, b_hi = np.percentile(train_behavior, [5, 99])
    b_norm = np.clip((behavior_score - b_lo) / max(b_hi - b_lo, 1e-9), 0, 1)

    # When the ERC-8004 identity layer is sparse (most real agents are
    # unattested), we weight behavior more. As attestation coverage
    # grows this formula re-balances without code changes (identity_score
    # rises when attestations are informative).
    risk = (
        0.55 * b_norm
        + 0.15 * identity_score
        + 0.10 * intent_score
        + 0.20 * graph_score
    )
    risk = np.clip(risk, 0, 1)

    df["cluster"] = labels_all
    df["behavior_score"] = b_norm
    df["identity_score"] = identity_score
    df["intent_score"] = intent_score
    df["graph_score"] = graph_score
    df["risk_score"] = risk

    # Eval.
    threshold = 0.5
    test_legit_mask = np.array([i in test_legit for i in df.index])
    adv_mask = df["is_adversarial"].to_numpy()
    legit_eval_risk = risk[test_legit_mask]
    adv_eval_risk = risk[adv_mask]

    fpr = float((legit_eval_risk >= threshold).mean()) if len(legit_eval_risk) else 0.0
    tpr = float((adv_eval_risk >= threshold).mean()) if len(adv_eval_risk) else 0.0

    metrics = {
        "n_agents": int(len(df)),
        "n_train_legit": int(train_mask.sum()),
        "n_test_legit": int(test_legit_mask.sum()),
        "n_adversarial": int(adv_mask.sum()),
        "clusters_discovered": int(len(set(labels_all)) - (1 if -1 in labels_all else 0)),
        "noise_points": int((labels_all == -1).sum()),
        "threshold": threshold,
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "cluster_archetype_map": {str(k): v for k, v in cluster_archetype.items()},
    }

    with open(ARTIFACT_PATH, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "clusterer": clusterer,
            "gmm": gmm,
            "cluster_archetype_map": cluster_archetype,
            "b_lo": float(b_lo),
            "b_hi": float(b_hi),
            "feature_cols": FEATURE_COLS,
            "fanin_p95": fanin_p95,
        }, f)

    df[["agent_id", "declared_archetype", "true_archetype", "is_adversarial",
        "adversary_kind", "cluster", "behavior_score", "identity_score",
        "intent_score", "graph_score", "risk_score"]].to_csv(
        ROOT / "model" / "scored_agents.csv", index=False
    )

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
