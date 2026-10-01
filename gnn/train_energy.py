"""Train the GNN to predict per-atom potential energy on the 424 MC/MD final configurations.

Graphs are built on the fly by DataLoader workers (all 424 would not fit in RAM).
Splits: struct = stable hash of the file name -> 80/10/10; base = val {NbTa}, test {Ta, NbW, MoTaW}.
"""
import argparse
import json
import os
import time
import zlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data import ELEMENTS, mcmd_files, read_mcmd
from model import AtomEnergyGNN
from train import TEST_BASES, VAL_BASES, to_dev


class MCMDDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path, base, solute, conc = self.files[i]
        g = read_mcmd(path)
        g.update(base=base, solute=solute, conc=conc, name=os.path.basename(path))
        return g


def split_files(mode):
    out = {0: [], 1: [], 2: []}
    for f in mcmd_files():
        if mode == "base":
            s = 2 if f[1] in TEST_BASES else 1 if f[1] in VAL_BASES else 0
        else:
            b = zlib.crc32(os.path.basename(f[0]).encode()) % 10
            s = 2 if b == 0 else 1 if b == 1 else 0
        out[s].append(f)
    return out


def loader(files, shuffle):
    return DataLoader(MCMDDataset(files), batch_size=None, shuffle=shuffle, num_workers=4, persistent_workers=True, prefetch_factor=2)


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    rows, structs = [], []
    for g in dl:
        gd = to_dev(g, dev)
        pred = model(gd).cpu().numpy()
        y, sp = g["energy"].numpy(), g["species"].numpy()
        rows.append(pd.DataFrame(dict(err=pred - y, y=y, elem=sp)))
        structs.append(dict(name=g["name"], base=g["base"], solute=g["solute"], conc=g["conc"], n=len(y),
                            e_per_atom_true=float(y.mean()), e_per_atom_pred=float(pred.mean())))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(structs)


def summarize(atoms, structs, elem_means):
    base_err = atoms.y - atoms.elem.map(elem_means)
    return dict(
        atom_mae_meV=float(atoms.err.abs().mean() * 1e3), atom_rmse_meV=float(np.sqrt((atoms.err**2).mean()) * 1e3),
        atom_r2=float(1 - (atoms.err**2).sum() / ((atoms.y - atoms.y.mean()) ** 2).sum()),
        struct_energy_per_atom_mae_meV=float((structs.e_per_atom_pred - structs.e_per_atom_true).abs().mean() * 1e3),
        baseline_element_mean_atom_mae_meV=float(base_err.abs().mean() * 1e3),
        n_atoms=int(len(atoms)), n_structures=int(len(structs)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["struct", "base"], default="struct")
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--mul", type=int, default=24)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = f"energy_{args.split}_l{args.lmax}_m{args.mul}_L{args.layers}_e{args.epochs}"
    out_dir = os.path.join(os.path.dirname(__file__), "runs", name)
    os.makedirs(out_dir, exist_ok=True)

    files = split_files(args.split)
    train_dl, val_dl, test_dl = loader(files[0], True), loader(files[1], False), loader(files[2], False)
    model = AtomEnergyGNN(mul=args.mul, lmax=args.lmax, n_layers=args.layers).to(dev)

    # initialise element offsets at the training mean energy of each element
    sums, counts = torch.zeros(len(ELEMENTS)), torch.zeros(len(ELEMENTS))
    for f in files[0][:40]:
        g = read_mcmd(f[0])
        sums.index_add_(0, g["species"], g["energy"])
        counts.index_add_(0, g["species"], torch.ones_like(g["energy"]))
    elem_means = (sums / counts.clamp(min=1)).numpy()
    model.offset.weight.data = torch.tensor(elem_means, dtype=torch.float32)[:, None].to(dev)
    elem_means = pd.Series(elem_means)
    print(f"{name}: params={sum(p.numel() for p in model.parameters())} train/val/test structs={[len(files[s]) for s in (0, 1, 2)]}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(files[0]), pct_start=0.05)
    best, t0, history = float("inf"), time.time(), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for g in train_dl:
            g = to_dev(g, dev)
            loss = torch.nn.functional.mse_loss(model(g), g["energy"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        atoms, _ = evaluate(model, val_dl, dev)
        val_mae = float(atoms.err.abs().mean() * 1e3)
        history.append(dict(epoch=epoch, train_rmse_meV=float(np.sqrt(np.mean(losses)) * 1e3), val_mae_meV=val_mae))
        if val_mae < best:
            best = val_mae
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        print(f"ep {epoch:3d} train_rmse={history[-1]['train_rmse_meV']:.1f} meV val_mae={val_mae:.1f} meV best={best:.1f} [{time.time() - t0:.0f}s]", flush=True)

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))
    atoms, structs = evaluate(model, test_dl, dev)
    result = dict(args=vars(args), train_time_s=time.time() - t0, test=summarize(atoms, structs, elem_means))
    structs.to_csv(os.path.join(out_dir, "test_structures.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    json.dump(result, open(os.path.join(out_dir, "result.json"), "w"), indent=2)
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
