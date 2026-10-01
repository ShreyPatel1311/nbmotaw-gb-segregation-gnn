"""The authors' model families (ANN on SOAP-PCA, XGBoost on SOAP-PCA + true E_seg) on the same splits as the GNNs.

seg_paper: ANN on M1_regression.npz rows with the authors' 70/20/10 row split (duplicated rows kept).
occ_paper: XGBoost and ANN on M2_classification.npz rows with the authors' stratified 80/20 split.
           "as published" early-stops on the test set like ClassificationTask.py; "clean" uses our val rows.
occ_base / occ_conc: XGBoost and ANN on [SOAP-PCA, true E_seg, solute, concentration] with the GNN run's split.
Sites with no segregation energy (58) are dropped from the regression task, as in the GNN runs.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch import nn

from ann_baseline import soap_features
from data import DATA_DIR, ELEMENTS
from paper_splits import m1_rows, m2_rows
from train import metrics
from train_occupancy import cls_metrics, isotherm_mae

HERE = os.path.dirname(__file__)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def onehot(values):
    return pd.get_dummies(pd.Categorical(values, categories=ELEMENTS)).to_numpy(float)


def train_mlp(X, y, tr, va, classify, seed=0, max_epochs=60, patience=8):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 1),
    ).to(DEV)
    loss_fn = nn.functional.binary_cross_entropy_with_logits if classify else nn.functional.mse_loss
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt, yt = torch.tensor(X, dtype=torch.float32, device=DEV), torch.tensor(y, dtype=torch.float32, device=DEV)
    tr_idx, va_idx = torch.tensor(np.where(tr)[0], device=DEV), torch.tensor(np.where(va)[0], device=DEV)
    best, state, wait = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for b in tr_idx[torch.randperm(len(tr_idx), device=DEV)].split(4096):
            loss = loss_fn(model(Xt[b]).squeeze(1), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xt[va_idx]).squeeze(1), yt[va_idx]).item()
        if val < best:
            best, state, wait = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(Xt[i : i + 200000]).squeeze(1) for i in range(0, len(Xt), 200000)])
    return (torch.sigmoid(out) if classify else out).cpu().numpy()


def train_xgb(X, y, fit_rows, eval_rows):
    model = xgb.XGBClassifier(n_estimators=1000, early_stopping_rounds=10, eval_metric=["logloss", "error", "aucpr"],
                              objective="binary:logistic", tree_method="hist", n_jobs=8, random_state=42)
    model.fit(X[fit_rows], y[fit_rows], eval_set=[(X[eval_rows], y[eval_rows])], verbose=False)
    return model.predict_proba(X)[:, 1]


def cls_report(rows, prob, mask):
    d = rows[mask].assign(prob=prob[mask])
    return dict(cls_metrics(d.prob.to_numpy(), d.y.to_numpy()), isotherm_mae=isotherm_mae(d))


def seg_paper():
    npz = np.load(os.path.join(DATA_DIR, "5_ML_datasets", "M1_regression.npz"))
    rows = m1_rows()
    keep = rows["seg_energy_eV"].notna().to_numpy()
    X, y, rows = npz["arr_0"][keep], npz["arr_1"][keep], rows[keep].reset_index(drop=True)
    chem = np.hstack([onehot(rows.solute), onehot(rows.site_element)])
    tr, va, te = (rows.split == k for k in (0, 1, 2))
    out = {}
    for name, feats in {"ann_paper_inputs": X, "ann_plus_chemistry": np.hstack([X, chem])}.items():
        pred = train_mlp(feats, y, tr.to_numpy(), va.to_numpy(), classify=False)
        out[name] = dict(val=metrics(pred[va], y[va]), test=metrics(pred[te], y[te]))
        print("seg_paper", name, {k: round(v, 4) for k, v in out[name]["test"].items()}, flush=True)
    return out


def occ_paper():
    npz = np.load(os.path.join(DATA_DIR, "5_ML_datasets", "M2_classification.npz"))
    X, y = npz["arr_0"], npz["arr_1"]
    rows = m2_rows()
    tr, va, te = ((rows.split == k).to_numpy() for k in (0, 1, 2))
    extra = np.hstack([onehot(rows.solute), rows[["conc"]].to_numpy()])
    out = {
        "xgb_as_published": cls_report(rows, train_xgb(X, y, tr | va, te), te),
        "xgb_clean": cls_report(rows, train_xgb(X, y, tr, va), te),
        "xgb_plus_solute_conc": cls_report(rows, train_xgb(np.hstack([X, extra]), y, tr, va), te),
        "ann_paper_inputs": cls_report(rows, train_mlp(X, y, tr, va, classify=True), te),
    }
    for k, v in out.items():
        print("occ_paper", k, {m: round(x, 3) for m, x in v.items() if isinstance(x, float)}, flush=True)
    return out


def occ_split(run):
    """XGBoost and ANN on [SOAP, true E_seg, solute, conc] using the split of a GNN occupancy run."""
    gnn = pd.read_csv(os.path.join(HERE, "runs", run, "predictions.csv"))
    soap = soap_features()
    cols = [c for c in soap.columns if c.startswith("soap")]
    rows = gnn.merge(soap[["base", "solute", "atom_id", "seg_energy_eV"] + cols], on=["base", "solute", "atom_id"])
    assert len(rows) == len(gnn)
    rows = rows[rows.split >= 0].reset_index(drop=True)
    X = np.hstack([rows[cols].to_numpy(), rows[["seg_energy_eV"]].to_numpy(), onehot(rows.solute), rows[["conc"]].to_numpy()])
    y = rows.y.to_numpy().astype(float)
    tr, va, te = ((rows.split == k).to_numpy() for k in (0, 1, 2))
    X_ann = np.nan_to_num(X)
    out = {
        "xgb_soap_eseg_solute_conc": cls_report(rows, train_xgb(X, y, tr, va), te),
        "ann_soap_eseg_solute_conc": cls_report(rows, train_mlp(X_ann, y, tr, va, classify=True), te),
    }
    for k, v in out.items():
        print(run, k, {m: round(x, 3) for m, x in v.items() if isinstance(x, float)}, flush=True)
    return out


if __name__ == "__main__":
    results = dict(seg_paper=seg_paper(), occ_paper=occ_paper(),
                   occ_base=occ_split("occ_base_l2_m24_L3_e100"), occ_conc=occ_split("occ_conc_l2_m24_L3_e100"))
    json.dump(results, open(os.path.join(HERE, "runs", "paper_models.json"), "w"), indent=2)
