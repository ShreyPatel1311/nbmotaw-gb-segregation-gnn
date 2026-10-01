"""Paper-style ANN (SOAP-PCA -> MLP) evaluated on exactly the same sites as a GNN run.

M1_regression.npz is ordered group-major: for each (base, solute) group in site-table order,
the group's rows repeat once per MC/MD concentration (19, or 10 for binaries). Inputs are
identical across repeats, so the first repeat gives one SOAP-PCA vector per site-table row.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from data import DATA_DIR, ELEMENTS
from train import metrics


def soap_features():
    table = pd.read_csv(os.path.join(DATA_DIR, "1_site_table", "gb_segregation_sites_50A4.csv"))
    npz = np.load(os.path.join(DATA_DIR, "5_ML_datasets", "M1_regression.npz"))
    X, y = npz["arr_0"], npz["arr_1"]
    feats, off = np.zeros((len(table), X.shape[1])), 0
    for _, g in table.groupby(["base", "solute"], sort=False):
        idx = g.index.to_numpy()
        feats[idx] = X[off : off + len(idx)]
        known = g["seg_energy_eV"].notna().to_numpy()
        assert np.allclose(y[off : off + len(idx)][known], g["seg_energy_eV"].to_numpy()[known], atol=1e-6)
        off += len(idx) * (10 if g["interaction"].iloc[0] == "binary" else 19)
    assert off == len(y)
    table[[f"soap{i}" for i in range(X.shape[1])]] = feats
    return table


def train_mlp(Xtr, ytr, Xva, yva, seed=0, epochs=300, patience=20):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr, ytr, Xva = map(lambda a: torch.tensor(a, dtype=torch.float32), (Xtr, ytr, Xva))
    best, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(Xtr)).split(256):
            loss = nn.functional.mse_loss(model(Xtr[b]).squeeze(1), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = np.abs(model(Xva).squeeze(1).numpy() - yva).mean()
        if val < best:
            best, best_state, wait = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="GNN run dir whose predictions.csv defines the split")
    args = ap.parse_args()
    run_dir = os.path.join(os.path.dirname(__file__), "runs", args.run)
    gnn = pd.read_csv(os.path.join(run_dir, "predictions.csv"))
    df = soap_features().merge(gnn.rename(columns={"pred": "gnn_pred", "y": "target"}), on=["base", "atom_id", "solute"])
    assert len(df) == len(gnn)
    soap = [c for c in df.columns if c.startswith("soap")]
    chem = pd.get_dummies(df["solute"]).reindex(columns=ELEMENTS, fill_value=0).add_prefix("sol_").join(
        pd.get_dummies(df["site_element"]).reindex(columns=ELEMENTS, fill_value=0).add_prefix("el_")).astype(float)
    variants = {"ann_soap": df[soap].to_numpy(), "ann_soap_chem": np.hstack([df[soap].to_numpy(), chem.to_numpy()])}
    tr, va, te = (df.split == 0).to_numpy(), (df.split == 1).to_numpy(), (df.split == 2).to_numpy()
    y = df["target"].to_numpy()
    out = {"gnn": metrics(df.gnn_pred.to_numpy()[te], y[te])}
    for name, X in variants.items():
        model = train_mlp(X[tr], y[tr], X[va], y[va])
        with torch.no_grad():
            pred = model(torch.tensor(X[te], dtype=torch.float32)).squeeze(1).numpy()
        out[name] = metrics(pred, y[te])
        df.loc[te, name] = pred
    for name in ["gnn_pred", "ann_soap", "ann_soap_chem"]:
        key = "gnn" if name == "gnn_pred" else name
        out[key]["mae_by_interaction"] = {k: float(np.abs(g[name] - g.target).mean()) for k, g in df[te].groupby("interaction")}
    json.dump(out, open(os.path.join(run_dir, "ann_comparison.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
