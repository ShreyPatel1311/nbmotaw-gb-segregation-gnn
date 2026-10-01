#!/bin/bash
# Authors' splits (paper), held-out bases and concentration extrapolation. Quick invariant (lmax=0)
# runs on the authors' splits first, then equivariant (lmax=2) runs.
cd "$(dirname "$0")"
mkdir -p logs
run() { name=$1; shift; echo "[$(date +%H:%M)] start $name"; python3 -u "$@" 2>&1 | grep --line-buffered -v Warning > "logs/$name.log"; echo "[$(date +%H:%M)] done  $name"; }
run seg_paper_l0       train.py --split paper --lmax 0 --mul 32 --epochs 100 --out paper_l0_m32_L3_e100
run occ_paper_l0       train_occupancy.py --split paper --lmax 0 --mul 32 --epochs 100
run seg_paper_l2       train.py --split paper --lmax 2 --mul 24 --epochs 100 --out paper_l2_m24_L3_e100
run occ_paper_l2       train_occupancy.py --split paper --lmax 2 --mul 24 --epochs 100
run seg_base_l2        train.py --split base --lmax 2 --mul 24 --epochs 100 --out base_l2_m24_L3_e100
run occ_base_l2        train_occupancy.py --split base --lmax 2 --mul 24 --epochs 100
run occ_conc_l2        train_occupancy.py --split conc --lmax 2 --mul 24 --epochs 100
run energy_struct_l2   train_energy.py --split struct --lmax 2 --mul 24 --epochs 15
echo ALL DONE
