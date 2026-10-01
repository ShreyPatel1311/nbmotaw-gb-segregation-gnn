"""Regenerate the report figures from gnn/runs/*.json (and, for the parity plot, a predictions.csv)."""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = os.path.join(ROOT, "gnn", "runs")
OUT = os.path.join(ROOT, "report", "figures")

GREY, INK, MUTED = "#9a9893", "#1f1e1c", "#6b6964"
COLORS = {"baseline": GREY, "ann": "#2a78d6", "ann_chem": "#eb6834", "gnn_inv": "#1baf7a", "gnn_eq": "#4a3aa7"}
LABELS = {"baseline": "Group-mean baseline", "ann": "ANN (SOAP, paper inputs)", "ann_chem": "ANN + solute + site element",
          "gnn_inv": "Invariant GNN", "gnn_eq": "Equivariant GNN"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def load(path):
    with open(os.path.join(RUNS, path)) as f:
        return json.load(f)


def bars(ax, groups, series, values, fmt):
    width = 0.8 / len(series)
    for i, s in enumerate(series):
        for j, g in enumerate(groups):
            v = values.get((g, s))
            x = j + (i - (len(series) - 1) / 2) * width
            if v is None:
                ax.text(x, 0.004, "not run", ha="center", va="bottom", fontsize=6.5, color=MUTED, rotation=90)
                continue
            ax.bar(x, v, width * 0.9, color=COLORS[s], edgecolor="white", linewidth=0.8, label=LABELS[s] if j == 0 else None)
            ax.text(x, v + 0.001, fmt.format(v), ha="center", va="bottom", fontsize=6.5, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_xticks(range(len(groups)))
    ax.yaxis.grid(True, color="#e6e4df", linewidth=0.6)
    ax.set_axisbelow(True)


def fig_segregation():
    paper, models = load("paper_l2_m24_L3_e100/result.json"), load("paper_models.json")["seg_paper"]
    site_ann, base_ann = load("site_l0_m32_L3/ann_comparison.json"), load("base_l0_m32_L3/ann_comparison.json")
    v = {
        ("paper", "baseline"): paper["baseline_group_mean"]["mae"],
        ("paper", "ann"): models["ann_paper_inputs"]["test"]["mae"],
        ("paper", "ann_chem"): models["ann_plus_chemistry"]["test"]["mae"],
        ("paper", "gnn_inv"): load("paper_l0_m32_L3_e100/result.json")["test"]["mae"],
        ("paper", "gnn_eq"): paper["test"]["mae"],
        ("site", "baseline"): load("site_l0_m32_L3/result.json")["baseline_group_mean"]["mae"],
        ("site", "ann"): site_ann["ann_soap"]["mae"],
        ("site", "ann_chem"): site_ann["ann_soap_chem"]["mae"],
        ("site", "gnn_inv"): site_ann["gnn"]["mae"],
        ("base", "baseline"): load("base_l0_m32_L3/result.json")["baseline_group_mean"]["mae"],
        ("base", "ann"): base_ann["ann_soap"]["mae"],
        ("base", "ann_chem"): base_ann["ann_soap_chem"]["mae"],
        ("base", "gnn_inv"): base_ann["gnn"]["mae"],
        ("base", "gnn_eq"): load("base_l2_m24_L3_e100/result.json")["test"]["mae"],
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    groups = ["paper", "site", "base"]
    bars(ax, groups, list(COLORS), v, "{:.3f}")
    ax.axhline(0.07, color=INK, linestyle=(0, (3, 3)), linewidth=0.8, label="Paper's reported ANN (0.07 eV)")
    ax.set_xticklabels(["Authors' split", "Unseen GB sites", "Unseen hosts\n(Ta, NbW, MoTaW)"])
    ax.set_ylabel("Test MAE of segregation energy (eV)")
    ax.set_xlim(-0.55, 2.55)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.18))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_segregation_mae.png"), bbox_inches="tight")


def fig_parity(pred_csv):
    d = pd.read_csv(pred_csv)
    d = d[d.w2 > 0]
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    hb = ax.hexbin(d.y, d.pred, gridsize=70, bins="log", cmap="Blues", mincnt=1, extent=(-1.2, 1.1, -1.2, 1.1))
    ax.plot([-1.2, 1.1], [-1.2, 1.1], color=INK, linewidth=0.7)
    mae = np.average(np.abs(d.pred - d.y), weights=d.w2)
    ax.text(-1.1, 0.95, f"MAE = {mae:.3f} eV\nn = {int(d.w2.sum()):,} test rows", fontsize=7.5, va="top", color=INK)
    ax.set_xlabel("Simulated segregation energy (eV)")
    ax.set_ylabel("Equivariant GNN prediction (eV)")
    ax.set_xlim(-1.2, 1.1)
    ax.set_ylim(-1.2, 1.1)
    fig.colorbar(hb, ax=ax, shrink=0.8, label="sites per bin")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_parity_equivariant.png"), bbox_inches="tight")


def fig_occupancy():
    m = load("paper_models.json")["occ_paper"]
    rows = {
        "XGBoost, paper protocol\n(SOAP + true E_seg)": m["xgb_as_published"],
        "ANN\n(SOAP + true E_seg)": m["ann_paper_inputs"],
        "XGBoost + solute + conc\n(SOAP + true E_seg)": m["xgb_plus_solute_conc"],
        "Invariant GNN\n(structure only)": load("occ_paper_l0_m32_L3_e100/result.json")["test"],
        "Equivariant GNN\n(structure only)": load("occ_paper_l2_m24_L3_e100/result.json")["test"],
    }
    colors = ["#2a78d6", "#86b6ef", "#eb6834", "#1baf7a", "#4a3aa7"]
    metrics = [("accuracy", "Accuracy"), ("tpr", "Occupied sites recovered (TPR)"), ("balanced_accuracy", "Balanced accuracy")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0))
    for ax, (key, title) in zip(axes, metrics):
        vals = [r[key] for r in rows.values()]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, vals, color=colors, edgecolor="white", height=0.75)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.1%}", va="center", fontsize=7, color=INK)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xlim(0, 1.12)
        ax.set_yticks(y)
        ax.set_yticklabels(list(rows) if ax is axes[0] else [], fontsize=7)
        ax.xaxis.grid(True, color="#e6e4df", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "50%", "100%"])
    fig.suptitle("Site occupancy after MC/MD at 300 K: authors' test set (76,336 labels, 5-25 at.%)", fontsize=8.5, color=INK, x=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_occupancy_paper_split.png"), bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="predictions.csv of the equivariant paper-split run (written by gnn/train.py)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fig_segregation()
    fig_occupancy()
    if args.pred:
        fig_parity(args.pred)
