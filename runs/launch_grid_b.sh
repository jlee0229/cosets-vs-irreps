#!/bin/zsh
# width grid, Aff(F_11), widths above p^2 = 121
cd "$(dirname "$0")/.."
P=${PYTHON:-python}; export OMP_NUM_THREADS=2; mkdir -p runs/grid runs/logs
for m in 124 132 140; do for s in 1 2 3; do
  $P scripts/train.py --group aff11 --recipe wu --seed $s --hidden $m --epochs 100000 --device cpu --log_every 250 --out runs/grid/aff11_wu100k_m${m}_seed$s > runs/logs/grid_aff11_m${m}_seed$s.log 2>&1
done; done
echo GRID_B_DONE
