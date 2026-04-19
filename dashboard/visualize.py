"""Generate cluster visualizations for the pitch deck.

Two figures:
  1. umap_clusters.png       agent embedding coloured by true archetype,
                              adversaries highlighted.
  2. risk_distribution.png   risk score histograms split by legit vs each
                              adversary kind.

Run: .venv/bin/python dashboard/visualize.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap

ROOT = Path(__file__).resolve().parent.parent
FEAT_PATH = ROOT / "features" / "agent_features.csv"
SCORED_PATH = ROOT / "model" / "scored_agents.csv"
MODEL_PATH = ROOT / "model" / "canopy_model.pkl"
OUT_DIR = ROOT / "dashboard"

sns.set_theme(style="whitegrid")


def main() -> None:
    feats = pd.read_csv(FEAT_PATH)
    scored = pd.read_csv(SCORED_PATH)
    df = feats.merge(scored[["agent_id", "risk_score", "cluster"]], on="agent_id")

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    scaler = artifact["scaler"]
    feature_cols = artifact["feature_cols"]

    X = df[feature_cols].to_numpy()
    Z = scaler.transform(X)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.05, random_state=0)
    emb = reducer.fit_transform(Z)
    df["u1"] = emb[:, 0]
    df["u2"] = emb[:, 1]

    # -------------------------- figure 1 --------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    legit = df[~df["is_adversarial"]]
    adv = df[df["is_adversarial"]]

    palette = sns.color_palette("tab10", n_colors=legit["true_archetype"].nunique())
    for color, (arch, sub) in zip(palette, legit.groupby("true_archetype")):
        ax.scatter(sub["u1"], sub["u2"], label=arch, s=55, alpha=0.75,
                   edgecolor="white", linewidth=0.3, color=color)

    marker_map = {
        "identity_mismatch_drainer": ("X", "crimson"),
        "compromised_trading_bot": ("P", "darkred"),
        "unattested_exfil": ("D", "purple"),
        "collusion_ring_member": ("*", "black"),
    }
    for kind, sub in adv.groupby("adversary_kind"):
        m, c = marker_map.get(kind, ("x", "red"))
        ax.scatter(sub["u1"], sub["u2"], marker=m, s=160, color=c, label=f"[ADV] {kind}",
                   edgecolor="white", linewidth=0.6, zorder=5)

    ax.set_title("Canopy agent embedding: legitimate clusters vs adversaries", fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "umap_clusters.png", dpi=150)
    plt.close(fig)

    # -------------------------- figure 2 --------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    order = ["legit"] + sorted(adv["adversary_kind"].dropna().unique().tolist())
    colors = sns.color_palette("Set2", n_colors=len(order))
    data = [df.loc[~df["is_adversarial"], "risk_score"]]
    data += [df.loc[df["adversary_kind"] == k, "risk_score"] for k in order[1:]]
    positions = np.arange(len(order))
    parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.75)
        pc.set_edgecolor("black")
    ax.axhline(0.6, color="red", linestyle="--", linewidth=1.2, label="block threshold (0.60)")
    ax.axhline(0.4, color="orange", linestyle="--", linewidth=1.2, label="review threshold (0.40)")
    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("risk score")
    ax.set_title("Risk score distribution: legitimate agents vs adversary classes", fontsize=13)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "risk_distribution.png", dpi=150)
    plt.close(fig)

    # -------------------------- figure 3: confusion ---------------
    threshold = 0.5
    tp = ((df["is_adversarial"]) & (df["risk_score"] >= threshold)).sum()
    fn = ((df["is_adversarial"]) & (df["risk_score"] < threshold)).sum()
    fp = ((~df["is_adversarial"]) & (df["risk_score"] >= threshold)).sum()
    tn = ((~df["is_adversarial"]) & (df["risk_score"] < threshold)).sum()

    fig, ax = plt.subplots(figsize=(5, 4.2))
    mat = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["pred: legit", "pred: flag"],
                yticklabels=["actual: legit", "actual: adv"], ax=ax)
    ax.set_title(f"Confusion @ threshold={threshold}  "
                 f"TPR={tp/(tp+fn):.0%}  FPR={fp/(fp+tn):.1%}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion.png", dpi=150)
    plt.close(fig)

    print("wrote:")
    for p in ["umap_clusters.png", "risk_distribution.png", "confusion.png"]:
        print("  ", OUT_DIR / p)


if __name__ == "__main__":
    main()
