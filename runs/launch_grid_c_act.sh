#!/bin/zsh
# width grid, Aff(F_7), widths around p^2 = 49; then the activation runs on Aff(F_11) at width 128: |t|^q for q = 2 (quadratic), 1, 1.5, 3
cd "$(dirname "$0")/.."
P=${PYTHON:-python}; export OMP_NUM_THREADS=2; mkdir -p runs/grid runs/act runs/logs
for m in 40 48 56 64; do for s in 1 2 3; do
  $P scripts/train.py --group aff7 --recipe wu --seed $s --hidden $m --epochs 100000 --device cpu --log_every 250 --out runs/grid/aff7_wu100k_m${m}_seed$s > runs/logs/grid_aff7_m${m}_seed$s.log 2>&1
done; done
echo GRID_C_DONE
for q in 2 1 1.5 3; do for s in 1 2 3; do
  $P scripts/train.py --group aff11 --recipe wu --seed $s --hidden 128 --epochs 50000 --device cpu --act absq --q $q --log_every 250 --out runs/act/aff11_wu50k_absq${q}_m128_seed$s > runs/logs/act_aff11_absq${q}_seed$s.log 2>&1
done; done
echo ACT_DONE
