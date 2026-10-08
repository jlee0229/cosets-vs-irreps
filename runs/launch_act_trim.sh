#!/bin/zsh
# activation runs on Aff(F_11), width 128, 25k epochs (the quadratic model groks by epoch 2000): |t|^q for q = 2, 1, 1.5, 3
cd "$(dirname "$0")/.."
P=${PYTHON:-python}; export OMP_NUM_THREADS=2; mkdir -p runs/act runs/logs
for qs in "2 2" "2 3" "1 1" "1 2" "1.5 1" "3 1"; do set -- ${=qs}; q=$1; s=$2
  $P scripts/train.py --group aff11 --recipe wu --seed $s --hidden 128 --epochs 25000 --device cpu --act absq --q $q --log_every 250 --out runs/act/aff11_wu25k_absq${q}_m128_seed$s > runs/logs/act_aff11_absq${q}_seed$s.log 2>&1
done
echo ACT_TRIM_DONE
