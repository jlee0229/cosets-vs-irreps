# What the Aff(F_p) models compute, neuron by neuron

Scripts: `scripts/affine_census.py` (per-neuron census and ablations, outputs `census_neurons.csv`, `census_ablation.csv`) and
`scripts/affine_handbuilt.py` (hand-built circuits, output `handbuilt.txt`). Models: the six grokked Aff(Z_11) runs at width 128
and the two Aff(Z_13) runs at width 256. Everything below is descriptive and post hoc; nothing here was pre-registered.

## The learned circuit

Every live neuron falls into one of two groups, with nothing in between:

- 112 to 120 neurons (p = 11) or 203 to 210 (p = 13) live in the (p-1)-dimensional irrep on both sides and are plain-aligned on both
  sides (alpha >= 0.9). The x side is a function of x^-1(c) for one point c (right cosets of the stabilizer of c), the y side a
  function of y(j0) for one point j0 (left cosets of the stabilizer of j0).
- 8 to 30 neurons live entirely in the linear characters. They compute the unit part a_z = a_x a_y.

For a plain neuron the two coset profiles are the same p values with opposite sign (sorted residual 0.001 to 0.003), so the
pre-activation is F(x^-1(c)) - F(y(j0)) up to an overall sign, with F injective on the p points. It is therefore zero exactly
when x^-1(c) = y(j0), that is when (xy)(j0) = c, and the ratio of |pre-activation| on those pairs to its overall mean is 0.002 to
0.004. The readout row is, in every one of 1105 neurons, the negative indicator of the set {z : z(j0) = c} for the same (j0, c):
the neuron fires on half of the pairs (active fraction 0.500) and pushes down the logits of the coset the product is not in.
Cells covered twice hold the pair F and -F (profile correlation -1.00 in every doubly covered cell), which together give -|F - F|.

F is graded, not an indicator: all p values distinct, top-coset energy share 0.23 to 0.28 where a one-hot profile would give
0.91. It is not a Fourier mode in the point index either (top-frequency share 0.13 to 0.21, flat would be 0.08 to 0.10), which is
why the twisted alpha of these neurons is 0.27 to 0.42 and not near 1. F varies somewhat from neuron to neuron within a model.

So the circuit is: pick a set J of point coordinates; for each j0 in J and each value c, a pair of neurons that penalise every
output z with z(j0) = c by |F(x^-1(c)) - F(y(j0))|; plus the unit part. Each coordinate costs 2p neurons. The models use

| model | plain neurons | cells (j0, c) covered | coordinates used | per cell |
|---|---|---|---|---|
| aff11 m128, 6 runs | 112 to 119 of 128 | 66 to 89 of 121 | 6 to 11 of 11 | 1 or 2 |
| aff13 m256, 2 runs | 203 to 210 of 256 | 120 to 121 of 169 | 10 to 11 of 13 | 1 or 2 |

At width 128 the Aff(Z_11) models fill the width with coordinate blocks; at width 256 the Aff(Z_13) models stop at 10 to 11
coordinates and leave 16 to 43 neurons dead.

## Ablations

Replacing every big-irrep side by its coset mean (its plain projection, constants kept) leaves accuracy at 1.0000 in all eight
models (0.9999 for the one model that was 0.9999). Removing the plain component instead (constants kept) drops accuracy to 0.03
to 0.09, chance being 1/|G| = 0.009 and 1/p = 0.08 to 0.09. The plain components are sufficient and necessary for these models.

## Hand-built circuits: how many neurons each basis needs

`affine_handbuilt.py` builds the circuits by hand and checks them on all |G|^2 pairs. The unit part is 2(p-1) neurons in all
cases (graded pairs on F_p^*, exact). The margin proxy is min over pairs of (correct logit - best wrong logit) divided by
sum_k ||(L_k, R_k)|| ||w_k||, the quantity L2 weight decay controls in a two-layer ReLU net once in and out norms balance. It
scores the specific constructions below, not the best possible circuit in each basis.

| circuit | neurons, p = 11 | accuracy | margin/cost (1e-3) | neurons, p = 13 | margin/cost (1e-3) |
|---|---|---|---|---|---|
| plain, 1 coordinate | 42 | exact | 0.164 | 50 | 0.089 |
| plain, 2 coordinates | 64 | exact | 0.214 | 76 | 0.116 |
| plain, p/2 coordinates | 130 | exact | 0.261 | 180 | 0.146 |
| plain, all p coordinates | 262 | exact | 0.517 | 362 | 0.290 |
| twisted, one frequency per fiber, 4 phases | 60 | 0.16 | negative | 72 | negative |
| twisted, one frequency per fiber, 6 phases | 80 | exact | 0.002 | 96 | 0.000 |
| twisted, one frequency per fiber, 8 phases | 100 | exact | 0.003 | 120 | 0.001 |
| twisted, all frequencies, 8 phases | 420 | exact | 0.025 | 600 | 0.012 |
| trained models (width 128 / 256) | 128 | 5 of 6 exact | 0.02 to 0.15 (one run negative, it misses one pair) | 256 | 0.05 to 0.13 |

The twisted circuit for Aff(F_p): the (p-1)-dimensional irrep is Ind from the translations of a non-trivial character, so its
twisted basis is "one fiber a_x = alpha, one frequency k in b_x". On each fiber the product's translation part is
b_z = b_x + alpha b_y, modular addition with y rescaled by alpha, so a twisted neuron is a clock neuron
ReLU(1[a_x = alpha] cos(k b_x + phi) + cos(k alpha b_y + phi)) silenced off its fiber, read out by cos(k b_z + 2 phi). One
frequency per fiber needs at least 6 phases to be exact, so 6(p-1) neurons, and its margin per unit norm is 40 to 100 times
below the plain circuit of the same size, because the clock's product term sits under a ReLU with harmonics that have to
cancel, while the plain pair -|F - F| is exact. The bilinear version (the full matrix-vector product rho(x) rho(y) 1 in the
induced basis) is (p-1)^2 complex products, 2(p-1)^2 real ones, so at least 4(p-1)^2 ReLU neurons, and still approximate.

## What this says about the width threshold

Neither basis needs p^2 neurons: the plain circuit is exact at 4p - 2 neurons (42 for p = 11, 50 for p = 13), and a twisted
circuit is exact at about 8p. So the failure of width-128 nets on p = 13 (169 > 128) is not a capacity limit, and it is not
that a cheaper twisted circuit exists and goes unused; under ReLU the plain circuit is the cheaper one on both counts. What
the models actually do is spend 2p neurons per coordinate and use as many coordinates as fit, which is what the margin proxy
favours: every wrong z disagrees with xy on at least |J| - 1 of the |J| coordinates, so penalties add across coordinates while
the norm cost adds only linearly, and margin per norm grows from 0.16 (one coordinate) to 0.52 (all eleven). The p^2 reading
of the threshold therefore becomes a claim about optimisation: how many coordinate blocks gradient descent needs room for
before the generalising circuit wins against memorisation. The census measures the number of coordinates directly, so a
width grid can read that number off rather than accuracy alone.
