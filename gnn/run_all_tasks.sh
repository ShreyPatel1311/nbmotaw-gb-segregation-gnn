#!/bin/bash
# All structure-based tasks. Fast invariant (lmax=0) runs first, then equivariant (lmax=2).
cd "$(dirname "$0")"
mkdir -p logs
run() { name=$1; shift; echo "[$(date +%H:%M)] start $name"; python3 -u "$@" 2>&1 | grep --line-buffered -v Warning > "logs/$name.log"; echo "[$(date +%H:%M)] done  $name"; }
# invariant
run seg_site_l0_e100   train.py --split site --lmax 0 --mul 32 --epochs 100 --out site_l0_m32_L3_e100
run seg_base_l0_e100   train.py --split base --lmax 0 --mul 32 --epochs 100 --out base_l0_m32_L3_e100
run occ_site_l0        train_occupancy.py --split site --lmax 0 --mul 32 --epochs 100
run occ_base_l0        train_occupancy.py --split base --lmax 0 --mul 32 --epochs 100
run occ_conc_l0        train_occupancy.py --split conc --lmax 0 --mul 32 --epochs 100
run energy_struct_l0   train_energy.py --split struct --lmax 0 --mul 32 --epochs 50
run energy_base_l0     train_energy.py --split base --lmax 0 --mul 32 --epochs 50
# equivariant
run seg_site_l2_e100   train.py --split site --lmax 2 --mul 24 --epochs 100 --out site_l2_m24_L3_e100
run seg_base_l2_e100   train.py --split base --lmax 2 --mul 24 --epochs 100 --out base_l2_m24_L3_e100
run occ_site_l2        train_occupancy.py --split site --lmax 2 --mul 24 --epochs 100
run occ_base_l2        train_occupancy.py --split base --lmax 2 --mul 24 --epochs 100
run occ_conc_l2        train_occupancy.py --split conc --lmax 2 --mul 24 --epochs 100
run energy_struct_l2   train_energy.py --split struct --lmax 2 --mul 24 --epochs 15
echo ALL DONE
