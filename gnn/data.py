"""Build periodic atom graphs of the 14 relaxed bases and attach segregation-energy labels.

Each base structure becomes one graph (all atoms, periodic radius graph). Labels are
per (GB site, solute): E_seg = E(solute at site) - E(solute at bulk reference of the same
element), where the reference atom ids come from the raw A2 energy files.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

ELEMENTS = ["Nb", "Mo", "Ta", "W"]  # LAMMPS types 1..4
DATA_DIR = os.environ.get(
    "GB_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "NbMoTaW_GB_Segregation_Dataset_MSMSE2024"),
)


def read_dump(path):
    with open(path) as f:
        lines = [next(f) for _ in range(9)]
    bounds = np.array([list(map(float, l.split()[:2])) for l in lines[5:8]])
    cols = lines[8].split()[2:]
    atoms = pd.read_csv(path, sep=r"\s+", skiprows=9, header=None, names=cols, usecols=["id", "type", "x", "y", "z"])
    return atoms.sort_values("id").reset_index(drop=True), bounds


def periodic_edges(pos, box_len, cutoff):
    """Directed edges (src -> dst) within cutoff under cubic PBC, with minimum-image vectors."""
    wrapped = np.mod(pos, box_len)
    tree = cKDTree(wrapped, boxsize=box_len)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    vec = wrapped[src] - wrapped[dst]
    vec -= box_len * np.round(vec / box_len)
    return src, dst, vec


def read_mcmd(path, cutoff=5.0):
    """One MC/MD final configuration as a graph with per-atom energy targets."""
    with open(path) as f:
        lines = [next(f) for _ in range(9)]
    bounds = np.array([list(map(float, l.split()[:2])) for l in lines[5:8]])
    atoms = pd.read_csv(path, sep=r"\s+", skiprows=9, header=None, names=["id", "type", "x", "y", "z", "c_eng"])
    box_len = bounds[:, 1] - bounds[:, 0]
    src, dst, vec = periodic_edges(atoms[["x", "y", "z"]].to_numpy() - bounds[:, 0], box_len, cutoff)
    return dict(
        species=torch.tensor(atoms["type"].to_numpy() - 1),
        edge_index=torch.tensor(np.stack([src, dst])),
        edge_vec=torch.tensor(vec, dtype=torch.float32),
        energy=torch.tensor(atoms["c_eng"].to_numpy(), dtype=torch.float32),
    )


def mcmd_files():
    """(path, base, solute, concentration) for every MC/MD final configuration."""
    out = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "4_A3_MCMD_final_configurations", "dump.MCMD_RSS_50A4_*"))):
        parts = os.path.basename(path).split("_")  # dump.MCMD RSS 50A4 <base> <fractions...> <solute> <conc> <temp>
        out.append((path, parts[3], parts[-3], float(parts[-2])))
    return out


def attach_occupancy(graphs):
    """Add MC/MD occupancy labels (site, solute, concentration, occupied) to each relaxed-base graph."""
    table = pd.read_csv(os.path.join(DATA_DIR, "1_site_table", "gb_segregation_sites_50A4.csv"))
    state_cols = [c for c in table.columns if c.startswith("state_")]
    long = table.melt(id_vars=["base", "solute", "atom_id"], value_vars=state_cols, var_name="conc", value_name="occupied").dropna()
    long["conc"] = long["conc"].str[len("state_"):].astype(float)
    for g in graphs:
        sub = long[long.base == g["base"]]
        idx = pd.Series(np.arange(len(g["species"])), index=g["all_atom_ids"])
        g["occ_site_idx"] = torch.tensor(idx[sub["atom_id"]].to_numpy())
        g["occ_atom_id"] = torch.tensor(sub["atom_id"].to_numpy())
        g["occ_solute"] = torch.tensor(sub["solute"].map(ELEMENTS.index).to_numpy())
        g["occ_conc"] = torch.tensor(sub["conc"].to_numpy(), dtype=torch.float32)
        g["occ_y"] = torch.tensor(sub["occupied"].to_numpy(), dtype=torch.float32)
    return graphs


def load_refs(base_comp, solute):
    path = os.path.join(DATA_DIR, "3_A2_segregation_energies_raw", f"SegEnergies_RSS_50A4_{base_comp}_{solute}.csv")
    raw = pd.read_csv(path, header=None, names=["id", "type", "E", "pe"])
    n_elem = len(re.findall(r"[A-Z][a-z]?", base_comp.split("_")[0]))
    refs = raw.iloc[1 : 1 + n_elem]
    return {int(t) - 1: int(i) for i, t in zip(refs["id"], refs["type"])}  # element index -> ref atom id


def build_graphs(cutoff=5.0, cache=None):
    if cache and os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    table = pd.read_csv(os.path.join(DATA_DIR, "1_site_table", "gb_segregation_sites_50A4.csv"))
    table = table.dropna(subset=["seg_energy_eV"])
    graphs = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "2_A1_relaxed_structures", "dump.RSS_50A4_*_sro_cna"))):
        base_comp = re.search(r"50A4_(.+)_sro_cna", path).group(1)
        base = base_comp.split("_")[0]
        atoms, bounds = read_dump(path)
        box_len = bounds[:, 1] - bounds[:, 0]
        assert np.allclose(box_len, box_len[0]), "expected cubic box"
        pos = atoms[["x", "y", "z"]].to_numpy() - bounds[:, 0]
        src, dst, vec = periodic_edges(pos, box_len, cutoff)
        id_to_idx = pd.Series(np.arange(len(atoms)), index=atoms["id"].to_numpy())

        sub = table[table.base == base]
        site_idx, ref_idx, solute, y, atom_id = [], [], [], [], []
        for s, grp in sub.groupby("solute"):
            refs = load_refs(base_comp, s)
            elem = grp["site_element"].map(ELEMENTS.index).to_numpy()
            site_idx.append(id_to_idx[grp["atom_id"]].to_numpy())
            ref_idx.append(id_to_idx[[refs[e] for e in elem]].to_numpy())
            solute.append(np.full(len(grp), ELEMENTS.index(s)))
            y.append(grp["seg_energy_eV"].to_numpy())
            atom_id.append(grp["atom_id"].to_numpy())
        graphs.append(
            dict(
                base=base,
                species=torch.tensor(atoms["type"].to_numpy() - 1),
                edge_index=torch.tensor(np.stack([src, dst])),
                edge_vec=torch.tensor(vec, dtype=torch.float32),
                site_idx=torch.tensor(np.concatenate(site_idx)),
                ref_idx=torch.tensor(np.concatenate(ref_idx)),
                solute=torch.tensor(np.concatenate(solute)),
                y=torch.tensor(np.concatenate(y), dtype=torch.float32),
                atom_id=torch.tensor(np.concatenate(atom_id)),
                all_atom_ids=atoms["id"].to_numpy(),
            )
        )
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save(graphs, cache)
    return graphs


if __name__ == "__main__":
    for g in build_graphs():
        n, e = len(g["species"]), g["edge_index"].shape[1]
        print(f"{g['base']:7s} atoms={n} edges={e} avg_deg={e / n:.1f} labels={len(g['y'])}")
