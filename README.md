# Equivariant GNNs for grain-boundary segregation in NbMoTaW

Graph neural networks that read the relaxed atomic structure of an NbMoTaW polycrystal and predict how solutes
segregate to its grain boundaries. The models are compared with the ANN and XGBoost models of the original study,
retrained on identical splits.

**Short report:** [`report/REPORT.md`](report/REPORT.md)

## Data

This repository contains **no data**. The models are trained on the NbMoTaW grain-boundary segregation dataset of

> D. Aksoy, J. Luo, P. Cao, T. J. Rupert, "A machine learning framework for the prediction of grain boundary
> segregation in chemically complex environments," *Modelling Simul. Mater. Sci. Eng.* **32**, 065011 (2024).

For access to the dataset, see the paper or contact its authors. Place the unpacked folder at
`data/NbMoTaW_GB_Segregation_Dataset_MSMSE2024/`, or point the `GB_DATA_DIR` environment variable at it.

## What the models predict

| Task | Input | Target |
|---|---|---|
| Segregation energy | Relaxed host structure + solute | Dilute-limit segregation energy of every GB site |
| Site occupancy | Relaxed host structure + solute + concentration | Whether a solute occupies the GB site after MC/MD at 300 K |
| Per-atom energy | MC/MD final configurations | Per-atom potential energy |

The network is a NequIP-style E(3)-equivariant message-passing network ([e3nn](https://e3nn.org)). It runs on the
full periodic structure (about 7,000 atoms, 5 Å cutoff, 3 layers). The segregation energy is predicted as
`head(site) − head(bulk reference)`, which mirrors how the dataset defines it. Setting `--lmax 0` gives an
invariant, distance-only version of the same network.

## Headline results

| | Authors' model | Equivariant GNN (structure only) |
|---|---|---|
| Segregation energy, MAE on the authors' split | 0.07 eV (reported) | **0.057 eV** |
| Occupancy, accuracy / occupied sites recovered | 90.2 % / 45.2 % (reported) | **91.6 % / 57.1 %** |
| Segregation energy, MAE on unseen GB sites | 0.073 eV (ANN + chemistry) | **0.055 eV** (invariant GNN) |

The authors' occupancy model is given the simulated segregation energies as input; the GNN is not. All splits,
baselines and caveats are in the [report](report/REPORT.md).

## Repository layout

```
gnn/
  data.py              periodic graphs of the relaxed structures and MC/MD configurations, labels
  model.py             equivariant network and task heads
  train.py             segregation energy        (--split paper | site | base)
  train_occupancy.py   site occupancy            (--split paper | site | base | conc)
  train_energy.py      per-atom energy           (--split struct | base)
  paper_splits.py      the authors' exact row splits, mapped back to GB sites
  paper_models.py      authors' model families (ANN, XGBoost) retrained on identical splits
  ann_baseline.py      ANN on SOAP features with the GNN's site / host splits
  runs/                result.json, training history and checkpoint of every run
  logs/                training logs
report/
  REPORT.md            short report
  make_figures.py      regenerates report/figures from gnn/runs
```

## Reproducing

```bash
pip install -r requirements.txt
python3 gnn/train.py --split paper --lmax 2 --mul 24 --epochs 100            # segregation energy
python3 gnn/train_occupancy.py --split paper --lmax 2 --mul 24 --epochs 100  # occupancy
python3 gnn/paper_models.py                                                  # ANN / XGBoost baselines
python3 report/make_figures.py --pred gnn/runs/paper_l2_m24_L3_e100/predictions.csv
```

The equivariant model at width 24 fits on a 4 GB GPU, at about 30 minutes per 100-epoch run on an entry-level GPU.

## Citation

If you use this code, please also cite the dataset paper above.
