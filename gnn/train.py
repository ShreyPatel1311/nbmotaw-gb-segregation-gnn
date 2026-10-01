"""Train the equivariant GNN on segregation energies.

Splits (all solutes of a GB site always stay together):
  site: random 70/10/20 split of unique (base, GB site) pairs; every base is seen in training
  base: whole bases held out -- val {NbTa}, test {Ta, NbW, MoTaW} (pure, binary, ternary host)
  paper: the authors' row-level 70/20/10 split of M1_regression.npz, where each site appears once
         per MC/MD concentration; a label is weighted by its number of rows in each split
Every label carries weights w[:, k] = number of its rows in split k (one-hot for site/base).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from e3nn import o3

from data import ELEMENTS, build_graphs
from model import SegregationGNN

VAL_BASES, TEST_BASES = {"NbTa"}, {"Ta", "NbW", "MoTaW"}


def assign_split(graphs, mode, seed=42):
    rng = np.random.default_rng(seed)
    for g in graphs:
        n = len(g["y"])
        if mode == "base":
            split = 2 if g["base"] in TEST_BASES else 1 if g["base"] in VAL_BASES else 0
            g["split"] = torch.full((n,), split)
        else:
            sites = np.unique(g["atom_id"].numpy())
            r = rng.random(len(sites))
            site_split = dict(zip(sites, np.where(r < 0.7, 0, np.where(r < 0.8, 1, 2))))
            g["split"] = torch.tensor([site_split[a] for a in g["atom_id"].numpy()])
    if mode == "paper":
        from paper_splits import m1_split_counts

        counts = m1_split_counts()
        for g in graphs:
            keys = pd.MultiIndex.from_arrays([[g["base"]] * len(g["y"]), [ELEMENTS[s] for s in g["solute"].numpy()], g["atom_id"].numpy()])
            g["w"] = torch.tensor(counts.loc[keys].to_numpy(), dtype=torch.float32)
            g["split"] = g["w"].argmax(1)
    else:
        for g in graphs:
            g["w"] = torch.nn.functional.one_hot(g["split"].long(), 3).float()


def metrics(pred, y, w=None):
    """MAE, RMSE, R2; with w, each label counts w times (the authors' duplicated rows)."""
    w = np.ones_like(y) if w is None else w
    m = w > 0
    pred, y, w = pred[m], y[m], w[m]
    err = pred - y
    mean_y = np.average(y, weights=w)
    return dict(
        mae=float(np.average(np.abs(err), weights=w)),
        rmse=float(np.sqrt(np.average(err**2, weights=w))),
        r2=float(1 - (w * err**2).sum() / (w * (y - mean_y) ** 2).sum()),
        n=int(w.sum()),
    )


def baseline(graphs, mode):
    """Mean of training labels per (base, solute, site element) -- or per (solute, site element) for unseen bases."""
    rows = []
    for g in graphs:
        species = g["species"][g["site_idx"]].numpy()
        w = g["w"].numpy()
        rows.append(pd.DataFrame(dict(base=g["base"], solute=g["solute"].numpy(), elem=species, y=g["y"].numpy(), w0=w[:, 0], w2=w[:, 2])))
    df = pd.concat(rows)
    keys = ["solute", "elem"] if mode == "base" else ["base", "solute", "elem"]
    train = df[df.w0 > 0]
    means = train.assign(wy=train.y * train.w0).groupby(keys)[["wy", "w0"]].sum().eval("wy / w0").rename("pred")
    test = df[df.w2 > 0].join(means, on=keys)
    test["pred"] = test["pred"].fillna(np.average(train.y, weights=train.w0))
    return metrics(test.pred.to_numpy(), test.y.to_numpy(), test.w2.to_numpy())


def to_dev(g, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in g.items()}


@torch.no_grad()
def predict(model, graphs, dev):
    model.eval()
    out = []
    for g in graphs:
        gd = to_dev(g, dev)
        w = g["w"].numpy()
        out.append(pd.DataFrame(dict(
            base=g["base"], atom_id=g["atom_id"].numpy(), solute=[ELEMENTS[s] for s in g["solute"].numpy()],
            split=g["split"].numpy(), w0=w[:, 0], w1=w[:, 1], w2=w[:, 2], y=g["y"].numpy(), pred=model(gd).cpu().numpy(),
        )))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["site", "base", "paper"], default="site")
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--mul", type=int, default=24)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.out or f"{args.split}_l{args.lmax}_m{args.mul}_L{args.layers}"
    out_dir = os.path.join(os.path.dirname(__file__), "runs", name)
    os.makedirs(out_dir, exist_ok=True)

    here = os.path.dirname(__file__)
    graphs = build_graphs(cutoff=args.cutoff, cache=os.path.join(here, "cache", f"graphs_rc{args.cutoff}.pt"))
    assign_split(graphs, args.split)
    train_graphs = [g for g in graphs if (g["w"][:, 0] > 0).any()]

    model = SegregationGNN(mul=args.mul, lmax=args.lmax, n_layers=args.layers, cutoff=args.cutoff).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    model.eval()
    with torch.no_grad():  # sanity check: predictions must not change under a global rotation
        g = to_dev(graphs[0], dev)
        rot = dict(g, edge_vec=g["edge_vec"] @ o3.rand_matrix().to(dev).T)
        inv_err = (model(g) - model(rot)).abs().max().item()
    print(f"{name}: params={n_params} rotation-invariance max |diff|={inv_err:.2e}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_graphs), pct_start=0.05)
    best, history = float("inf"), []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for i in np.random.permutation(len(train_graphs)):
            g = to_dev(train_graphs[i], dev)
            w = g["w"][:, 0]
            mask = w > 0
            loss = (w[mask] * (model(g, mask) - g["y"][mask]) ** 2).sum() / w[mask].sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        pred = predict(model, graphs, dev)
        val = metrics(pred.pred.to_numpy(), pred.y.to_numpy(), pred.w1.to_numpy())
        history.append(dict(epoch=epoch, train_rmse=float(np.sqrt(np.mean(losses))), val_mae=val["mae"], val_rmse=val["rmse"]))
        if val["mae"] < best:
            best = val["mae"]
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        if epoch % 5 == 0 or epoch == 1:
            print(f"ep {epoch:3d} train_rmse={history[-1]['train_rmse']:.4f} val_mae={val['mae']:.4f} best={best:.4f} [{time.time() - t0:.0f}s]", flush=True)

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))
    pred = predict(model, graphs, dev)
    test = pred[pred.w2 > 0]
    val = pred[pred.w1 > 0]
    result = dict(
        args=vars(args), params=n_params, rotation_invariance_err=inv_err, train_time_s=time.time() - t0,
        test=metrics(test.pred.to_numpy(), test.y.to_numpy(), test.w2.to_numpy()),
        val=metrics(val.pred.to_numpy(), val.y.to_numpy(), val.w1.to_numpy()),
        test_by_base={b: metrics(d.pred.to_numpy(), d.y.to_numpy(), d.w2.to_numpy()) for b, d in test.groupby("base")},
        test_by_solute={s: metrics(d.pred.to_numpy(), d.y.to_numpy(), d.w2.to_numpy()) for s, d in test.groupby("solute")},
        baseline_group_mean=baseline(graphs, args.split),
    )
    pred.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    json.dump(result, open(os.path.join(out_dir, "result.json"), "w"), indent=2)
    print(json.dumps({k: result[k] for k in ["test", "baseline_group_mean"]}, indent=2))


if __name__ == "__main__":
    main()
