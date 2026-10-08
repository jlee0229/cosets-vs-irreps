# Max-margin analysis: which basis does weight decay prefer on Aff(F_p)?

Scripts: `scripts/affine_maxmargin.py` (numerical max-margin within families, outputs in `maxmargin/`), `scripts/affine_margins_check.py`
and `affine_margins_split.py` (brute-force checks of the closed forms), `scripts/affine_handbuilt.py` (hand-built circuits).
The closed-form derivations are in `margins_analytic.md`. Everything here is post hoc.

## Question and figure of merit

Gradient descent with L2 weight decay on a homogeneous network drives the weights toward a maximum-margin solution (Morwani et al.
2023 use this to derive the Fourier features of modular addition with quadratic activations). For the one-hidden-layer ReLU model
f_z(x, y) = sum_k w_k(z) ReLU(L_k(x) + R_k(y)), the margin is gamma = min over pairs (x, y) and outputs z != xy of f_xy - f_z, and
the cost weight decay controls, once in and out norms balance, is C = sum_k ||(L_k, R_k)|| ||w_k||. The figure of merit is gamma / C,
which is scale invariant. The question is whether the plain coset circuit or a twisted (Fourier) circuit attains the larger value.

Two caveats on the cost. The trained architecture factorises the input side through a shared embedding (W_x W, W_y W) and is
3-homogeneous; the matching cost is 2 ||L||_* + 2 ||R||_* + ||W_U||_F^2 with normalisation gamma / cost^1.5, reported below as
"arch". All optimisation was done under the two-layer cost; the architecture cost is only evaluated on the result. And every number
below is a local optimum found by gradient ascent inside a parameterised family, not a proof of global optimality.

## Closed forms (hand-built circuits, verified by brute force at p = 7, 11, 13)

Plain coordinate circuit with coordinate set J and the evenly spaced profile F: margin exactly |J| for 1 <= |J| <= p-1 and
2(p-1) at |J| = p (the extra factor is an edge effect of the linear profile), cost 2p(p-1)[|J| sqrt(p(p^2-1)/6) + sqrt(p(p-1)(p-2)/6)],
so gamma / C scales as p^-3.5 for every |J| and grows with |J| from sqrt(6)/4 to sqrt(6) times p^-3.5. For one coordinate the
evenly spaced profile is optimal among injective profiles (it maximises the smallest gap per unit norm). The unit part is not
needed for |J| >= 2 (two coordinates determine an affine map) and is dead weight at |J| = p.

Twisted clock circuit (one neuron per fiber, frequency and phase, silenced off its fiber): the phase sum is exactly
(2K/3 pi) |cos((u-v)/2)| cos(u+v-theta) plus aliases of harmonics n = +-2 mod K, so K = 4 fails (the main term aliases with itself)
and K >= 6 works. One frequency has margin (2K/3pi)(1 - cos 2pi/p) sin(pi/2p), about 2K pi^2 / 3p^3, and gamma / C about 7.6 p^-6.
All (p-1)/2 frequencies give a margin 2K/(3 pi) times T(1) with T(1) about 2p/(3 pi), and a clock-part cost per unit margin of 9.6 p^3
against 0.41 p^3.5 for the plain coordinates. The twisted mechanism therefore scales half a power better in p but with a constant
23.6 / sqrt(p) worse (ReLU attenuation 3 pi/2, worst-case input factor 3 pi/4, silencing sqrt(3), and the plain circuit's own
gap-to-norm handicap 1.22 / sqrt(p)); the crossover is near p = 555, and the twisted circuit's unit part keeps it behind even there.
At p = 11 and 13 the hand-built plain circuit beats the hand-built twisted one by 21 and 24 times as built, 6.3 times after
rebalancing the two parts of the twisted circuit. The per-fiber duplication cannot be avoided in one hidden layer: the y-side
frequency depends on a_x, which R cannot see, and the induced irrep's matrix coefficients are fiber-supported by construction.

## Numerical max-margin within families

Soft-min ascent on gamma / C (Adam, 4000 steps, float32 on the GPU, exact margin of the best iterate recomputed in float64).
Families: plainJ (coordinates 0..J-1, two neurons per cell (j0, c), free profiles and readouts, hand-built init), plain (all
coordinates), twisted (one neuron per fiber, frequency and 8 phases, free fiber amplitudes and offsets, hand-built clock init),
free (unconstrained, started from an optimum of another family or from a trained model). Every family also has 4(p-1) unit-part
neurons. Random-init free runs never reach a separating solution with this optimiser and are omitted.

gamma / C in units of 1e-3, and "arch" gamma / cost^1.5 in units of 1e-6:

| p | plain, 1 coord | plain, 2 coords | plain, all coords (monotone init) | plain, all coords (reordered init) | twisted | free from plain | free from twisted | trained model, as is | free from trained |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 4.58 (193) | 5.59 (316) | 9.23 (497) | | 7.25 (246) | 9.23 (498) | 7.54 (254) | | |
| 7 | 1.37 (50) | 1.57 (75) | 3.10 (145) | 3.10 (145) | 2.78 (66) | 3.15 (142) | 2.81 (77) | | |
| 11 | 0.266 (8.2) | 0.296 (12) | 0.655 (27) | 0.977 (33) | 0.687 (5.9) | 0.699 (29) | 0.794 (8.3) | 0.052, 0.151 (2.1, 6.3) | 0.676, 0.654 (23, 24) |
| 13 | 0.143 (4.3) | 0.161 (6.1) | 0.346 (13) | 0.580 (20) | 0.246 (1.2) | 0.359 (13) | 0.390 (2.4) | 0.130 (4.8) | 0.469 (14) |

The "free from plain" column continues the monotone-init optimum; continuing the reordered-init optimum gives 3.12, 0.996 and 0.596
at p = 7, 11, 13 (`maxmargin/perm.csv`), still entirely plain.

Hand-built values for comparison (evenly spaced F, from `handbuilt.txt`): plain all coordinates 0.517 (p = 11) and 0.290 (p = 13);
twisted all frequencies at K = 8, 0.025 and 0.012. The optimiser improves the plain circuit by 1.2 to 1.6 times and the twisted
circuit by about 30 times, so the hand-built twisted clock is far from the best the twisted family can do.

What the optimised circuits look like (p = 11, `maxmargin/large_p11_*.pt`):

- Plain optimum: all 242 big-irrep neurons plain-aligned on both sides, readouts the indicators of {z : z(j0) = c} over all eleven
  coordinates, unit part dead (zero energy), no linear-character component.
- Twisted optimum: every big-irrep neuron twisted-aligned on both sides, 40% of their energy also plain-aligned (the intersection of
  the two families: functions of a point coordinate whose profile is a single Fourier mode). The x-side functions have spread from
  their single-fiber init to 9 or 10 of the 10 fibers. The readouts are functions of b_z alone (99% of their variance), i.e. of the
  single coordinate z(0). A third of the energy sits in linear-character neurons (silencing offsets and the unit part).
- Free from twisted: moves further into the intersection (94% of big-irrep energy plain-aligned, 99% twisted-aligned) and raises
  the margin; free from plain stays plain (99.9%) and barely moves.
- Free from a trained model: the two width-128 models start at 0.05 and 0.15 (they are trained with cross-entropy and decay 2e-4,
  not to the margin limit) and climb 4 to 13 times to the plain optimum's level without leaving the plain basis (99.7% and 100%).

At p = 5 and 7 the twisted family's optimum lies entirely in the intersection (97% and 99.8% of big-irrep energy plain-aligned) and
stays below the plain optimum. At p = 11 the twisted-initialised runs end above the plain optimum started from the monotone
profile (0.69 and 0.79 against 0.65 and 0.70) while drifting toward the intersection, but below the plain optimum started from a
reordered profile (0.98, see the next section); at p = 13 the plain optima lead under every start (0.58 against 0.25 and 0.39).
Under the architecture cost the plain optima are 3 to 17 times higher than the twisted ones at every p, because the twisted
circuits need more neurons and their side matrices have larger nuclear norm. The order of the values in the profile, not the
basis, was the largest lever the optimiser had.

## The profile F

For one coordinate the evenly spaced profile is optimal. With all coordinates the order of the values matters: the margin is the
minimum over affine maps sigma != id of sum_r |F(sigma r) - F(r)| while the cost depends only on the values. Per unit norm
(the figure of merit M below), the monotone evenly spaced profile gives 2.27, 1.91, 1.78 at p = 7, 11, 13; a local search over
orderings of the same values gives 2.27, 2.86, 3.11, and a continuous ascent 2.27, 2.87, 3.15. A cosine profile, the only kind the
twisted family can express as a point-coordinate function, scores at most 1.04, 0.84, 0.78. The trained models' profiles score
1.8 to 2.2 (p = 11) and 2.1 to 2.6 (p = 13): better than monotone at p = 13, between monotone and the optimum, and far above any
cosine. This is the quantitative version of "F is graded but not a Fourier mode": once all coordinates are in play a scrambled
order of evenly spaced values has a larger margin than the monotone one, and the trained profiles are partly scrambled.

## What this settles and what it does not

- The twisted clock as hand-built is not competitive at any trainable p; the closed forms say why factor by factor.
- Within this optimiser's reach the plain family holds the largest normalised margin at every p tested, by 1.2 to 2.4 times over
  the best twisted-initialised run under the two-layer cost and by 3 to 17 times under the trained architecture's cost, and the
  twisted optima are pulled into the plain family's intersection with it. This is not a proof that the plain circuit is the global
  maximum-margin solution: the free-from-random runs fail to separate, so the global optimum was not located independently of
  the structured starts, and the plain optimum itself moved by 1.5 times when its starting profile was reordered.
- The trained models are an order of magnitude below the max-margin value and are not at the margin limit; what they share with
  the max-margin solution is the basis, the spread over coordinates, and a partly scrambled graded profile.
- A proof in Morwani's style would need the max-margin problem for Aff(F_p) solved exactly; the closed forms above are the inputs
  it would need for the two candidate circuits.
