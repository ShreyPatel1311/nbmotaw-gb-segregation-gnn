"""Reproduce the authors' row-level train/val/test splits and map every row back to a GB site.

M1 (regression, RegressionTask.py): shuffle(random_state=42), then train_test_split(test_size=0.1,
random_state=42) and train_test_split(test_size=0.2/0.9, random_state=42) -> 70/20/10. The npz is
group-major: each (base, solute) group of the site table repeats once per MC/MD concentration, so a
site appears 10-19 times and its copies land in different splits.

M2 (classification, ClassificationTask.py): shuffle(random_state=42), then a stratified
train_test_split(test_size=0.2, random_state=42). The npz is group-major with five concentration
blocks per group in reverse order (0.25 ... 0.05). The authors had no validation set; we hold out
1/8 of their training rows (stable hash) for early stopping, leaving their test rows untouched.
"""
import os
import zlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from data import DATA_DIR

M2_CONCS = [0.25, 0.2, 0.15, 0.1, 0.05]


def _table():
    return pd.read_csv(os.path.join(DATA_DIR, "1_site_table", "gb_segregation_sites_50A4.csv"))


def m1_rows():
    """One row per line of M1_regression.npz: its site label and the authors' split (0/1/2)."""
    table = _table()
    rows = []
    for _, g in table.groupby(["base", "solute"], sort=False):
        reps = 10 if g["interaction"].iloc[0] == "binary" else 19
        rows.append(pd.concat([g[["base", "solute", "atom_id", "site_element", "interaction", "seg_energy_eV"]]] * reps, ignore_index=True))
    rows = pd.concat(rows, ignore_index=True)
    y = np.load(os.path.join(DATA_DIR, "5_ML_datasets", "M1_regression.npz"))["arr_1"]
    assert len(rows) == len(y)

    idx = shuffle(np.arange(len(rows)), random_state=42)
    trainval, test = train_test_split(idx, test_size=0.1, random_state=42)
    train, val = train_test_split(trainval, test_size=0.2 / 0.9, random_state=42)
    split = np.empty(len(rows), dtype=int)
    split[train], split[val], split[test] = 0, 1, 2
    rows["split"] = split
    return rows


def m1_split_counts():
    """Copies of each (base, solute, atom_id) label in the authors' train/val/test rows."""
    rows = m1_rows()
    counts = rows.groupby(["base", "solute", "atom_id", "split"]).size().unstack(fill_value=0)
    return counts.reindex(columns=[0, 1, 2], fill_value=0).rename(columns={0: "n_train", 1: "n_val", 2: "n_test"})


def m2_rows():
    """One row per line of M2_classification.npz: its label and split (0 train, 1 val carved from train, 2 test)."""
    table = _table()
    rows = []
    for _, g in table.groupby(["base", "solute"], sort=False):
        for c in M2_CONCS:
            rows.append(g[["base", "solute", "atom_id", "site_element"]].assign(conc=c, y=g[f"state_{c}"].to_numpy()))
    rows = pd.concat(rows, ignore_index=True)
    y = np.load(os.path.join(DATA_DIR, "5_ML_datasets", "M2_classification.npz"))["arr_1"]
    assert np.array_equal(rows["y"].to_numpy(), y), "M2 row order does not match the site table"

    idx, y_shuf = shuffle(np.arange(len(rows)), y, random_state=42)
    train, test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_shuf)
    split = np.zeros(len(rows), dtype=int)
    split[test] = 2
    key = rows["base"] + "-" + rows["solute"] + "-" + rows["atom_id"].astype(str) + "-" + rows["conc"].astype(str)
    carve = np.array([zlib.crc32(k.encode()) % 8 == 0 for k in key])
    split[(split == 0) & carve] = 1
    rows["split"] = split
    return rows


def m2_splits():
    """Split (0 train, 1 val, 2 test) of each (base, solute, atom_id, conc) label in the authors' M2 dataset."""
    return m2_rows().set_index(["base", "solute", "atom_id", "conc"])["split"]


if __name__ == "__main__":
    c = m1_split_counts()
    print("M1 rows per split:", c.sum().to_dict())
    print("M1 labels with >=1 test copy that also have >=1 train copy:", f"{((c.n_test > 0) & (c.n_train > 0)).sum() / (c.n_test > 0).sum():.1%}")
    s = m2_splits()
    print("M2 labels per split:", s.value_counts().sort_index().to_dict())
