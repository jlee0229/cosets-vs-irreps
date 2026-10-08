#!/bin/zsh
# finer width grid: Aff(F_7) between 64 (fails) and 128 (groks), then Aff(F_13) between 128 (fails) and 256 (groks)
cd "$(dirname "$0")/.."
P=${PYTHON:-python}; export OMP_NUM_THREADS=2; mkdir -p runs/grid runs/logs
for m in 72 80 88 96 112; do for s in 1 2; do
  $P scripts/train.py --group aff7 --recipe wu --seed $s --hidden $m --epochs 100000 --device cpu --log_every 250 --out runs/grid/aff7_wu100k_m${m}_seed$s > runs/logs/grid_aff7_m${m}_seed$s.log 2>&1
done; done
for m in 160 192 224; do
  $P scripts/train.py --group aff13 --recipe wu --seed 1 --hidden $m --epochs 100000 --device cpu --log_every 250 --out runs/grid/aff13_wu100k_m${m}_seed1 > runs/logs/grid_aff13_m${m}_seed1.log 2>&1
done
echo GRID_D_DONE
