# cosets vs irreps

Code, trained models and results for a pre-registered test of whether the neurons of one-hidden-layer MLPs trained on
finite-group multiplication are organized by cosets of subgroups (Stander et al., arXiv:2312.06581) or by irreducible
representations (Chughtai et al., arXiv:2302.03025; Wu et al., arXiv:2410.07476), on S5, D59/D61 and the affine groups
Aff(Z_n) for n = 11, 13, 15, 16, 21. The S5 and dihedral analyses run on Chughtai et al.'s released models; the affine
models were trained here. Pre-registrations were frozen before the corresponding models were analysed and are kept
unedited in `preregistration/`. The findings are written up in `writeup/cosets_irreps_s5.html` and analyzed in
`results/probe/REPORT.md` (S5, dihedral) and `results/affine/REPORT_affine.md` (affine family). In summary, the networks organize by plain point-stabilizer cosets in rank-one form on every group tested, never by the zero-slack single-frequency basis, and factor through CRT whenever n is composite.
Post hoc follow-ups on the prime moduli (what the circuit computes neuron by neuron, why weight decay prefers it, where
the width threshold sits, and what other activations do) are in `results/affine/circuit_census.md`, `maxmargin.md`,
`margins_analytic.md`, `width_grid.md` and `activation.md`.

## Setup

    uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
    git clone --depth 1 https://github.com/bilal-chughtai/rep-theory-mech-interp external/rep-theory-mech-interp   # for the S5/dihedral scripts
    .venv/bin/python -m pytest -q tests

Dependencies are numpy, pandas and torch. Everything ran on one M2 laptop (analyses in seconds, an affine model in
2 to 20 minutes; the exact training queues are `runs/launch_*.sh`).

## Reproducing the tables

| result | script | writes |
|---|---|---|
| S5 coset alignment, P1/P2 (`results/probe/REPORT.md`) | `scripts/s5_coset_probe.py --runs S5_MLP_seed1 ... --out results/probe/<phase>` | `results/probe/<phase>_per_{model,neuron}.csv` |
| random-init control | `scripts/s5_random_init_control.py` | `results/probe/randinit_*.csv` |
| single-irrep purity, P3 | `scripts/s5_purity.py` | `results/probe/p3_per_column.csv` |
| dihedral rank-1 pilot, P0 | `scripts/dihedral_rank1_pilot.py` | `results/probe/dihedral_pilot_*.csv` |
| affine nulls (attached to the pre-registration) | `scripts/affine_nulls.py` | `results/affine_nulls.md` |
| affine verdicts P-A to P-D (`results/affine/REPORT_affine.md`) | `scripts/affine_tally.py` | `results/affine/final_per_{model,column}.csv` |
| rank of block coefficient matrices | `scripts/affine_rank.py --runs runs/affine/* --out results/affine/all` | `results/affine/all_rank_*.csv` |
| which subgroup (index ≤ 16, all characters) | `scripts/affine_subgroup_sweep.py` | `results/affine/sweep_*.csv` |
| neuron census and ablations (`results/affine/circuit_census.md`) | `scripts/affine_census.py --runs runs/affine/aff11_* --out results/affine/census` | `results/affine/census_*.csv` |
| hand-built plain and twisted circuits | `scripts/affine_handbuilt.py` | `results/affine/handbuilt.txt` |
| closed-form margins, brute-force checks (`results/affine/margins_analytic.md`) | `scripts/affine_margins_check.py`, `scripts/affine_margins_split.py` | `results/affine/margins_*_output.txt` |
| numerical max-margin circuits (`results/affine/maxmargin.md`) | `scripts/affine_maxmargin.py --p 11 13 --free_from plain twisted --out results/affine/maxmargin/large` | `results/affine/maxmargin/*.csv`, `*.pt` |
| width grid and activation runs (`results/affine/width_grid.md`, `activation.md`) | `runs/launch_grid_*.sh`, `runs/launch_aff7_frac06.sh`, `runs/launch_act_trim.sh`, then `scripts/affine_grid_summary.py --runs runs/grid/*` | `results/affine/grid_summary.csv`, `act_summary.csv` |

`make s5` and `make affine` run the pre-registered ones in order; the last five rows are the post hoc follow-ups. The S5
subgroup ↔ irrep dictionary and the null distributions in `results/*.md` are produced by the self-test and
`scripts/affine_nulls.py`; none of them use model data.

## Models

`runs/affine/<group>_<recipe>_m<width>_seed<k>/` holds the final weights (`model.pt`, a state dict of W_x, W_y, W, W_U),
the config and the train/test curve of every affine run, including the ones that did not generalize. Architecture and
initialization are Chughtai et al.'s (embedding 256, hidden 128 unless `m512`, no bias, ReLU); the task is all |G|^2
ordered pairs with a 40% training split drawn by the seed. `wu` is Adam, lr 1e-2, weight decay 2e-4, 25k epochs
(100k for n = 11); `chughtai` is AdamW, lr 1e-3, weight decay 1.0, 250k epochs. `runs/validation/` is our retraining of
Chughtai's S5 seed 2, which reproduces their released model's statistics to two decimals. Group elements are indexed as
in `cosetprobe/affine.py`; the S5 indexing follows Chughtai's code (sympy 1.11.1 order, pinned in `data/`).
`runs/grid/` holds the width grid for p = 7, 11, 13 (and the 60% split at p = 7), `runs/act/` the |t|^q activation runs.

`scripts/train.py --group aff15 --recipe wu --seed 3 --out runs/affine/aff15_wu_m128_seed3` retrains one model;
`--hidden`, `--frac`, `--epochs` and `--act absq --q 2` vary width, split, schedule and activation.
Grokking of the prime groups at width 128 is fragile (Aff(Z_11)) or absent (Aff(Z_13)); the width grid puts the
threshold at about p^2 once the training split is large enough (`results/affine/width_grid.md`).

## Limitations

S5 is the only non-M-group, the dihedral pilot could only check rank one, two of the five affine moduli (15, 21) factor into direct products whose tensor block the networks never build, and n = 13 never generalizes at width 128. The slack comparison is therefore n = 11 against n = 16, and the seed-to-seed variety comes from five models of one group. All models are one-hidden-layer ReLU MLPs at width 128 under two optimizers; irrep selection is known to depend on the regime, so the preference for plain cosets is a claim about this one. The pre-registered families (point stabilizers with the trivial character; single frequencies on the translations) are narrower than what the models use, so several verdicts are 'empty' or 'neither' and were explained afterwards by a subgroup sweep whose nulls are weak for small blocks. P4 was withdrawn as tautological, the memorizer control in P2 was a design error, and P-C and P-D are inconclusive. The pre-registered statistics describe organization, not sufficiency; the ablations, the margin bounds and the width grid in the follow-up notes are post hoc and were not pre-registered. Five seeds per cell; Aff(Z_11) at width 128 groks or not depending on non-deterministic CPU arithmetic. The max-margin comparison is between hand-built circuits and local optima found by gradient ascent, under the two-layer weight norm rather than the trained architecture's, and is not a proof.

This repository was pushed as a single commit after the work was done, so git history does not show the pre-registrations being frozen before the models were trained; the freeze dates and amendment records inside each file are the only record of ordering, and a reader should weight them accordingly.

## Future directions

Two of the candidate explanations for why plain cosets win have now been tested on the primes: the activation function (|t|
keeps the plain circuit, the quadratic activation removes the preference without replacing it by the twisted basis;
`results/affine/activation.md`) and the max-margin basis (the plain circuit has the larger margin per weight norm at every
p tested, with the clock circuit's constant factor accounted for term by term; `maxmargin.md`, `margins_analytic.md`).
Which structure forms first in training is still open. Prime powers 25, 27 and 32 give non-factoring composites with
blocks of dimension at least 6; primes 17 and 19 would extend the width grid; SL(2,3) separates solvability from
monomiality. The action task (g, x) -> g(x) targets the permutation representation directly and has no published work.
A regime that uses the 6-dimensional irrep of S5 would show which orbit it picks; Chughtai's transformers and Stander's
weights would test architecture dependence.
