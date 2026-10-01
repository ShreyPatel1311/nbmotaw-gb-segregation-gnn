"""NequIP-style E(3)-equivariant message-passing network (e3nn) with a segregation-energy head.

The network embeds every atom of the undecorated base. A solute-conditioned head maps an
atom embedding to a substitution energy, and E_seg = head(site) - head(bulk reference).
With lmax=0 the same code is an invariant (distance-only) baseline.
"""
import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from torch import nn


def scatter_sum(src, index, n):
    return torch.zeros(n, src.shape[1], dtype=src.dtype, device=src.device).index_add_(0, index, src)


class Interaction(nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out, n_basis, radial_hidden, avg_degree):
        super().__init__()
        self.avg_degree = avg_degree
        mid, instructions = [], []
        for i, (mul, ir_in) in enumerate(irreps_in):
            for j, (_, ir_sh) in enumerate(irreps_sh):
                for ir_out in ir_in * ir_sh:
                    if ir_out in irreps_out:
                        instructions.append((i, j, len(mid), "uvu", True))
                        mid.append((mul, ir_out))
        mid = o3.Irreps(mid)
        mid_sorted, perm, _ = mid.sort()
        instructions = [(i, j, perm[k], m, t) for i, j, k, m, t in instructions]
        self.lin_in = o3.Linear(irreps_in, irreps_in)
        self.tp = o3.TensorProduct(irreps_in, irreps_sh, mid_sorted, instructions, shared_weights=False, internal_weights=False)
        self.radial = FullyConnectedNet([n_basis, radial_hidden, self.tp.weight_numel], torch.nn.functional.silu)
        self.lin_out = o3.Linear(mid_sorted, irreps_out)
        self.skip = o3.Linear(irreps_in, irreps_out)

    def forward(self, h, sh, edge_emb, src, dst):
        msg = self.tp(self.lin_in(h)[src], sh, self.radial(edge_emb))
        agg = scatter_sum(msg, dst, h.shape[0]) / self.avg_degree**0.5
        return self.lin_out(agg) + self.skip(h)


class SegregationGNN(nn.Module):
    def __init__(self, n_elements=4, mul=32, lmax=2, n_layers=3, cutoff=5.0, n_basis=8, radial_hidden=64, head_hidden=128, avg_degree=26.0):
        super().__init__()
        self.cutoff, self.n_basis, self.n_elements = cutoff, n_basis, n_elements
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        gated = o3.Irreps([(mul // 2, (l, (-1) ** l)) for l in range(1, lmax + 1)])
        self.embed = o3.Linear(o3.Irreps(f"{n_elements}x0e"), o3.Irreps(f"{mul}x0e"))
        layers, irreps = [], o3.Irreps(f"{mul}x0e")
        for _ in range(n_layers):
            gate = Gate(
                f"{mul}x0e", [torch.nn.functional.silu],
                f"{gated.num_irreps}x0e" if lmax > 0 else "", [torch.sigmoid] if lmax > 0 else [],
                gated,
            )
            layers.append(nn.ModuleList([Interaction(irreps, self.irreps_sh, gate.irreps_in, n_basis, radial_hidden, avg_degree), gate]))
            irreps = gate.irreps_out
        self.layers = nn.ModuleList(layers)
        self.to_scalar = o3.Linear(irreps, o3.Irreps(f"{mul}x0e"))
        self.head = nn.Sequential(
            nn.Linear(mul + n_elements, head_hidden), nn.SiLU(),
            nn.Linear(head_hidden, head_hidden), nn.SiLU(),
            nn.Linear(head_hidden, 1),
        )

    def embed_atoms(self, species, edge_index, edge_vec):
        src, dst = edge_index
        r = edge_vec.norm(dim=1)
        sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization="component")
        edge_emb = soft_one_hot_linspace(r, 0.0, self.cutoff, self.n_basis, basis="smooth_finite", cutoff=True) * self.n_basis**0.5
        h = self.embed(nn.functional.one_hot(species, self.n_elements).float())
        for conv, gate in self.layers:
            h = gate(conv(h, sh, edge_emb, src, dst))
        return self.to_scalar(h)

    def substitution_energy(self, z, solute):
        return self.head(torch.cat([z, nn.functional.one_hot(solute, self.n_elements).float()], dim=1)).squeeze(1)

    def forward(self, g, mask=None):
        z = self.embed_atoms(g["species"], g["edge_index"], g["edge_vec"])
        site, ref, sol = g["site_idx"], g["ref_idx"], g["solute"]
        if mask is not None:
            site, ref, sol = site[mask], ref[mask], sol[mask]
        return self.substitution_energy(z[site], sol) - self.substitution_energy(z[ref], sol)


class OccupancyGNN(SegregationGNN):
    """Logit that a solute occupies a GB site after MC/MD, from the undecorated base, the solute and its concentration."""

    def __init__(self, head_hidden=128, **kwargs):
        super().__init__(head_hidden=head_hidden, **kwargs)
        mul = self.to_scalar.irreps_out.dim
        self.head = nn.Sequential(
            nn.Linear(mul + self.n_elements + 1, head_hidden), nn.SiLU(),
            nn.Linear(head_hidden, head_hidden), nn.SiLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, g, mask=None):
        z = self.embed_atoms(g["species"], g["edge_index"], g["edge_vec"])
        site, sol, conc = g["occ_site_idx"], g["occ_solute"], g["occ_conc"]
        if mask is not None:
            site, sol, conc = site[mask], sol[mask], conc[mask]
        x = torch.cat([z[site], nn.functional.one_hot(sol, self.n_elements).float(), conc[:, None]], dim=1)
        return self.head(x).squeeze(1)


class AtomEnergyGNN(SegregationGNN):
    """Per-atom potential energy: learned element offset plus an MLP on the atom embedding."""

    def __init__(self, head_hidden=128, **kwargs):
        super().__init__(head_hidden=head_hidden, **kwargs)
        mul = self.to_scalar.irreps_out.dim
        self.head = nn.Sequential(nn.Linear(mul, head_hidden), nn.SiLU(), nn.Linear(head_hidden, 1))
        self.offset = nn.Embedding(self.n_elements, 1)

    def forward(self, g):
        z = self.embed_atoms(g["species"], g["edge_index"], g["edge_vec"])
        return self.head(z).squeeze(1) + self.offset(g["species"]).squeeze(1)
