"""Train the GNN to predict MC/MD site occupancy at 300 K from the undecorated base.

Input: relaxed base structure + solute + solute concentration. Target: 1 if a solute atom sits on
the GB site after MC/MD. Splits (all concentrations and solutes of a site stay together unless
the split is by concentration):
  site: stable hash of (base, atom_id) -> 70/10/20
  base: val {NbTa}, test {Ta, NbW, MoTaW}
  conc: train c <= 0.30, val c = 0.35, test 0.40 <= c <= 0.50 (concentration extrapolation; c > 0.5 unused)
  paper: the authors' stratified 80/20 split of M2_classification.npz (c = 0.05-0.25 only), with 1/8
         of their training rows held out for early stopping; c > 0.25 unused
"""
import argparse
import json
import os
import time
import zlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from data import ELEMENTS, attach_occupancy, build_graphs
from model import OccupancyGNN
from train import TEST_BASES, VAL_BASES, to_dev


def site_bucket(base, atom_ids):
    return np.array([zlib.crc32(f"{base}-{a}".encode()) % 100 for a in atom_ids])


def assign_split(graphs, mode):
    for g in graphs:
        n = len(g["occ_y"])
        if mode == "base":
            split = np.full(n, 2 if g["base"] in TEST_BASES else 1 if g["base"] in VAL_BASES else 0)
        elif mode == "conc":
            c = g["occ_conc"].numpy().astype(float).round(2)
            split = np.where(c <= 0.30, 0, np.where(c <= 0.35, 1, np.where(c <= 0.50, 2, -1)))
        elif mode == "paper":
            from paper_splits import m2_splits

            m2 = m2_splits()
            keys = pd.MultiIndex.from_arrays([[g["base"]] * n, [ELEMENTS[s] for s in g["occ_solute"].numpy()],
                                              g["occ_atom_id"].numpy(), g["occ_conc"].numpy().astype(float).round(2)])
            split = m2.reindex(keys).fillna(-1).astype(int).to_numpy()
        else:
            b = site_bucket(g["base"], g["occ_atom_id"].numpy())
            split = np.where(b < 70, 0, np.where(b < 80, 1, 2))
        g["occ_split"] = torch.tensor(split)


def cls_metrics(prob, y):
    pred = prob >= 0.5
    tp, tn = (pred & (y == 1)).sum(), (~pred & (y == 0)).sum()
    tpr, tnr = tp / max((y == 1).sum(), 1), tn / max((y == 0).sum(), 1)
    return dict(
        accuracy=float((pred == y).mean()), balanced_accuracy=float((tpr + tnr) / 2),
        tpr=float(tpr), tnr=float(tnr), f1=float(2 * tp / max(2 * tp + (pred & (y == 0)).sum() + (~pred & (y == 1)).sum(), 1)),
        auroc=float(roc_auc_score(y, prob)) if 0 < y.mean() < 1 else None, n=int(len(y)), pos_rate=float(y.mean()),
    )


def isotherm_mae(df):
    """Mean |predicted - true| GB solute fraction per (base, solute, concentration)."""
    agg = df.groupby(["base", "solute", "conc"]).agg(pred=("prob", "mean"), true=("y", "mean"))
    return float((agg.pred - agg.true).abs().mean())


def frame(graphs, probs=None):
    rows = []
    for i, g in enumerate(graphs):
        rows.append(pd.DataFrame(dict(
            base=g["base"], atom_id=g["occ_atom_id"].numpy(), solute=[ELEMENTS[s] for s in g["occ_solute"].numpy()],
            conc=g["occ_conc"].numpy().round(2), split=g["occ_split"].numpy(), y=g["occ_y"].numpy().astype(int),
            prob=probs[i] if probs is not None else np.nan,
        )))
    return pd.concat(rows, ignore_index=True)


def baseline(df, mode):
    """Training positive rate per (base, solute, conc); per (solute, conc) for unseen bases, per (base, solute) for unseen conc."""
    keys = {"base": ["solute", "conc"], "conc": ["base", "solute"], "site": ["base", "solute", "conc"], "paper": ["base", "solute", "conc"]}[mode]
    rate = df[df.split == 0].groupby(keys).y.mean().rename("p")
    test = df[df.split == 2].join(rate, on=keys)
    test["prob"] = test["p"].fillna(df[df.split == 0].y.mean())
    return dict(cls_metrics(test.prob.to_numpy(), test.y.to_numpy()), isotherm_mae=isotherm_mae(test))


def random_mixing(df):
    """Physics floor: with no segregation, a GB site holds solute with probability c."""
    test = df[df.split == 2].assign(prob=lambda d: d.conc)
    return dict(cls_metrics(test.prob.to_numpy(), test.y.to_numpy()), isotherm_mae=isotherm_mae(test))


@torch.no_grad()
def predict(model, graphs, dev):
    model.eval()
    return [torch.sigmoid(model(to_dev(g, dev))).cpu().numpy() for g in graphs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["site", "base", "conc", "paper"], default="site")
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--mul", type=int, default=24)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    here = os.path.dirname(__file__)
    name = f"occ_{args.split}_l{args.lmax}_m{args.mul}_L{args.layers}_e{args.epochs}"
    out_dir = os.path.join(here, "runs", name)
    os.makedirs(out_dir, exist_ok=True)

    graphs = attach_occupancy(build_graphs(cache=os.path.join(here, "cache", "graphs_rc5.0.pt")))
    assign_split(graphs, args.split)
    train_graphs = [g for g in graphs if (g["occ_split"] == 0).any()]
    model = OccupancyGNN(mul=args.mul, lmax=args.lmax, n_layers=args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_graphs), pct_start=0.05)
    print(f"{name}: params={sum(p.numel() for p in model.parameters())}", flush=True)

    best, t0, history = float("inf"), time.time(), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for i in np.random.permutation(len(train_graphs)):
            g = to_dev(train_graphs[i], dev)
            mask = g["occ_split"] == 0
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(g, mask), g["occ_y"][mask])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        df = frame(graphs, predict(model, graphs, dev))
        val = df[df.split == 1]
        eps = 1e-6
        val_bce = float(-(val.y * np.log(val.prob + eps) + (1 - val.y) * np.log(1 - val.prob + eps)).mean())
        history.append(dict(epoch=epoch, train_bce=float(np.mean(losses)), val_bce=val_bce))
        if val_bce < best:
            best = val_bce
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        if epoch % 5 == 0 or epoch == 1:
            print(f"ep {epoch:3d} train_bce={np.mean(losses):.4f} val_bce={val_bce:.4f} best={best:.4f} [{time.time() - t0:.0f}s]", flush=True)

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))
    df = frame(graphs, predict(model, graphs, dev))
    test = df[df.split == 2]
    result = dict(
        args=vars(args), train_time_s=time.time() - t0,
        test=dict(cls_metrics(test.prob.to_numpy(), test.y.to_numpy()), isotherm_mae=isotherm_mae(test)),
        test_by_conc={str(c): cls_metrics(d.prob.to_numpy(), d.y.to_numpy()) for c, d in test.groupby("conc")},
        baseline_group_rate=baseline(df, args.split),
        baseline_random_mixing=random_mixing(df),
    )
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    json.dump(result, open(os.path.join(out_dir, "result.json"), "w"), indent=2)
    print(json.dumps({k: result[k] for k in ["test", "baseline_group_rate", "baseline_random_mixing"]}, indent=2))


if __name__ == "__main__":
    main()
