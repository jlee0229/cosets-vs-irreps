# Activation function and the plain-coset preference (Aff(F_11), width 128)

Runs: `runs/act/` from `runs/launch_grid_c_act.sh` (one quadratic run at 50k epochs) and `runs/launch_act_trim.sh` (25k epochs),
all R-Wu (Adam, lr 1e-2, weight decay 2e-4, 40% of pairs), hidden activation |t|^q from `scripts/train.py --act absq --q Q`.
Statistics: `scripts/affine_probe.py` (the pre-registered S1 to S3 and verdict rule, outputs `probe_act_*`), `scripts/affine_rank.py`
(Wu's rank-one test, `rank_act_*`), `scripts/affine_census.py` with the model's own activation in the ablation (`census_act_*`),
grokking from `act_summary.csv`. Post hoc; the prediction being tested is the one written in the README's future directions:
ReLU favours plain cosets because a pair of ReLUs turns a few-valued coset pre-activation into an exact indicator, while a
quadratic activation's cross term is exact in any basis, so quadratic networks should lose the plain preference and show
generic single-irrep directions rather than switch to the twisted basis.

| q | seeds | grokked (epoch) | F_plain | F_twist | verdict | rank-one columns | stabiliser types |
|---|---|---|---|---|---|---|---|
| 1 (|t|) | 1, 2 | yes, yes (1500, 2000) | 1.000, 1.000 | 0, 0 | plain, plain | 100%, 100% | point stabilisers (1/M, M/1) |
| 1.5 | 1 | yes (1000) | 0.349 | 0 | unclear | 65% | trivial |
| 2 | 1 (50k), 2, 3 | 0.989 (3250), yes, yes (1250, 1250) | 0, 0.227, 0 | 0.056, 0, 0 | neither x3 | 94%, 100%, 100% | trivial (1/1) |
| 3 | 1 | no (test acc 0.009 at 25k) | 0 | 0 | neither | 0% (ranks 7 to 9) | |

F_plain is the energy fraction of block-dominant columns with plain alpha >= 0.9 (same for twisted); the verdict rule is the frozen
one (plain if F_plain >= 0.5 and F_twist < 0.5, neither if both < 0.3, unclear otherwise). Stabiliser types are those of the
rank-one factors in the monomial basis: M means a point stabiliser (the plain structure), 1 means trivial (a generic direction).

Reading:

- |t| is the ReLU pair in one neuron (|t| = ReLU(t) + ReLU(-t)), and it gives the plain circuit exactly as ReLU does: every column
  a function of one point coordinate, rank one with point-stabiliser factors.
- The quadratic activation removes the plain preference completely. All three models are single-irrep and rank one, as Wu's
  theory and Morwani's max-margin result for quadratic activations require, but the rank-one factors point in generic directions:
  median plain alpha 0.37 to 0.50, no column at 0.9, and the twisted alpha is no higher (0.40 to 0.55, one model with 6% twisted).
  Replacing the sides by their coset means now costs accuracy (1.0 to 0.37 for the 50k model) and removing the plain component
  does not destroy the network (0.55), the reverse of the ReLU models. This is the predicted outcome: no switch to the twisted basis,
  just no basis preference inside the irrep.
- q = 1.5 is in between (35% plain energy, 65% rank one), consistent with the preference fading as the activation moves away from
  piecewise linear. q = 3 did not grok at 25k epochs, so its census describes a memorising model and says nothing here.
- All |t|^q models that grok do so within 1000 to 3250 epochs at width 128, where ReLU needs 25k to 100k and sometimes fails. The
  activation changes the optimisation as well as the basis, so the width-threshold question (`width_grid.md`) is ReLU-specific.

Limitations: one group and one width; one seed at q = 1.5 and 3; the verdict and alpha statistics were designed for the ReLU
pre-activation and are applied unchanged (they measure the sides, which is activation-independent, but the ablations are not).
