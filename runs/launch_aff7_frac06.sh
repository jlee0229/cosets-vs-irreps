#!/bin/zsh
# is the Aff(F_7) failure at widths <= 64 a data effect?  Same widths with 60% of the pairs instead of 40%.
cd "$(dirname "$0")/.."
P=${PYTHON:-python}; export OMP_NUM_THREADS=2; mkdir -p runs/grid runs/logs
for m in 40 48 56; do for s in 1 2; do
  $P scripts/train.py --group aff7 --recipe wu --seed $s --hidden $m --frac 0.6 --epochs 100000 --device cpu --log_every 250 --out runs/grid/aff7_wu100k_m${m}_frac06_seed$s > runs/logs/grid_aff7_m${m}_frac06_seed$s.log 2>&1
done; done
echo AFF7_FRAC06_DONE
