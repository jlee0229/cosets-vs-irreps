# Analytic margins: plain coordinate circuit vs twisted clock circuit on Aff(F_p)

Setting as specified: one hidden ReLU layer, logits f_z(x,y) = sum_k w_k(z) ReLU(L_k(x) + R_k(y)),
margin gamma = min over (x,y), z != xy of f_{xy} - f_z, cost C = sum_k ||(L_k,R_k)||_2 ||w_k||_2.
Group elements x = (a_x, b_x), x(j) = a_x j + b_x, (xy)(j) = x(y(j)), so a_{xy} = a_x a_y, b_{xy} = a_x b_y + b_x.
Every closed form below was checked by brute force over all |G|^2 input pairs and all |G| outputs for p = 7, 11, 13
(script `scripts/affine_margins_check.py` and `scripts/affine_margins_split.py`; their outputs are `results/affine/margins_check_output.txt` and `margins_split_output.txt`).
Where a formula is asymptotic or alias-free it is labelled as such.

Notation: g_F = min gap of F (1 for F(j) = j - (p-1)/2), g_f = min gap of f (1 for the evenly spaced unit function),
S_F = sum_j F(j)^2 = p(p^2-1)/12, S_f = sum_u f(u)^2 = p(p-1)(p-2)/12,
a_p = sqrt(p(p^2-1)/6) = sqrt(2 S_F), b_p = sqrt(p(p-1)(p-2)/6) = sqrt(2 S_f).

## 1. Circuit 1: plain coordinate circuit

### 1.1 What the logits are

Pair (j0, c): the two neurons sum to |F(x^{-1}(c)) - F(y(j0))| and are read out by -1[z(j0) = c]. With F(j) = j - (p-1)/2 this is
the ordinary integer distance |x^{-1}(c) - y(j0)|_Z between representatives in {0, ..., p-1} (not the cyclic distance).
The correct output has every contribution equal to zero, so for z != xy the gap is

    Delta(x,y,z) = sum_{j0 in J} |x^{-1}(z(j0)) - y(j0)|_Z + |f(a_x) - f(a_z / a_y)|,

and the key identity (x^{-1}(t) = (t - b_x)/a_x) is

    x^{-1}(z(j0)) - y(j0) = (z(j0) - (xy)(j0)) / a_x = ((a_z - a_{xy}) j0 + (b_z - b_{xy})) / a_x   (mod p).

### 1.2 Exact margin (case analysis)

Case A, a_z = a_{xy}, b_z != b_{xy}. The unit term is 0 and the mod-p shift delta = (b_z - b_{xy})/a_x is the same for all j0.
Writing delta' in {1, ..., p-1}, the coordinate term is delta' if y(j0) + delta' <= p-1 and p - delta' otherwise (the wrap).
Minimum: delta' = 1 (or p-1) with y(j0) != p-1 for every j0 in J, which is possible iff |J| <= p-1, giving |J| g_F.
For |J| = p exactly one coordinate wraps: (p-1) g_F + (p-1) g_F = 2(p-1) g_F. (Any delta' gives sum_r |F(r+delta') - F(r)| over a full
cycle >= 2(max F - min F) = 2(p-1) g_F, so delta' = +-1 is the worst.)

Case B, a_z != a_{xy}. The unit term is >= g_f. z and xy agree at exactly one point j* = (b_{xy} - b_z)/(a_z - a_{xy}); every other
coordinate contributes >= g_F. So Delta >= (|J| - 1[j* in J]) g_F + g_f >= (|J|-1) g_F + g_f, achievable.
For |J| = p the shift delta(j0) runs over all of F_p^* as j0 runs over F_p \ {j*}, so Delta >= g_F sum_{d=1}^{p-1} min(d, p-d) + g_f
= g_F (p^2-1)/4 + g_f, which exceeds 2(p-1) g_F for p >= 7 (brute force: 13 vs 12 at p = 7, 31 vs 20 at p = 11, 43 vs 24 at p = 13).

Result (exact, verified):

    gamma(J) = min( |J| g_F , (|J|-1) g_F + g_f )  for 1 <= |J| <= p-1      (= |J| when g_F = g_f = 1)
    gamma(J) = 2(p-1) g_F                           for |J| = p, p >= 7      (unit part irrelevant)

Why 20 (p=11) and 24 (p=13) and not p-1 or p: the all-coordinates margin is case A with b_z = b_{xy} +- a_x (brute-force minimiser:
x = y = identity, z = translation by 1). z disagrees with xy by one step on every coordinate; F is a line, not a circle, so the
coordinate where y(j0) = p-1 pays |F(0) - F(p-1)| = p-1 instead of 1. Total (p-1)*1 + (p-1) = 2(p-1).
The last coordinate therefore doubles the margin (p-1 -> 2(p-1)), which is a quirk of the 1-d embedding, not a feature of the circuit.

Side facts (verified): for |J| >= 2 the unit part is not needed for correctness (gamma = (|J|-1) g_F without it), and for |J| = p it is
dead weight (gamma unchanged, cost lower: gamma/C = 0.557e-3 instead of 0.517e-3 at p = 11).

### 1.3 Exact cost

Coordinate neuron: x^{-1}(c) and y(j0) take each value in F_p on exactly p-1 group elements, so ||L||^2 = ||R||^2 = (p-1) S_F;
w is the indicator of {z : z(j0) = c}, a set of size p-1. Per neuron: sqrt(2(p-1) S_F) sqrt(p-1) = (p-1) a_p. Count 2p|J|.
Unit neuron: f(a_x), f(v/a_y) take each value on p elements, ||L||^2 = ||R||^2 = p S_f; w indicates {a_z = v}, size p.
Per neuron p b_p. Count 2(p-1).

    C(J) = 2p(p-1) [ |J| a_p g_F + b_p g_f ] ,   a_p = sqrt(p(p^2-1)/6),  b_p = sqrt(p(p-1)(p-2)/6).

### 1.4 gamma / C

    |J| <= p-1 :  gamma/C = |J| / ( 2p(p-1) [ |J| a_p + b_p ] )
    |J| = p    :  gamma/C = 1 / ( p [ p a_p + b_p ] )

| p  | |J|=1 | |J|=2 | |J|=floor(p/2) | |J|=p-1 | |J|=p |
|----|-------|-------|----------------|---------|-------|
| 7  | gamma 1, C 1125.6, 0.889e-3 | 2, 1754.2, 1.140e-3 | 3, 2382.8, 1.259e-3 | 6, 4268.5, 1.406e-3 | 12, 4897.1, 2.450e-3 |
| 11 | 1, 6089.1, 0.164e-3 | 2, 9352.2, 0.214e-3 | 5, 19141.6, 0.261e-3 | 10, 35457.2, 0.282e-3 | 20, 38720.4, 0.517e-3 |
| 13 | 1, 11229.0, 0.0891e-3 | 2, 17181.6, 0.116e-3 | 6, 40991.9, 0.146e-3 | 12, 76707.4, 0.156e-3 | 24, 82660.0, 0.290e-3 |

All entries agree with the script's values (0.164, 0.214, 0.261, 0.517 for p = 11; 0.089, 0.116, 0.146, 0.290 for p = 13) to the
quoted digits, and the formulas reproduce the brute-force C to 1e-10.

Large p (a_p, b_p ~ p^{3/2}/sqrt(6)): every |J| scales as p^{-7/2}, only the constant moves:

    |J| = 1:        sqrt(6)/4 * p^{-7/2}
    |J| = p-1:      sqrt(6)/2 * p^{-7/2}
    |J| = p:        sqrt(6)   * p^{-7/2}   (exact: 1/(p[p a_p + b_p]); 5.55e-4 vs exact 5.17e-4 at p = 11)

The p^{-7/2} comes from (number of neuron pairs p^2) x (per-neuron norm p^{5/2}) / (margin 2p).

### 1.5 Optimising F (single coordinate)

For |J| = 1 the margin is exactly min(g_F, g_f) (case A gives g_F, case B with j* = j0 gives g_f; verified with g_f = 0.5 and 2).
A constant added to F cancels in L + R, so take F centred. For a centred injective F with sorted values v_1 < ... < v_p,

    S_F = (1/p) sum_{i<k} (v_k - v_i)^2 >= (g_F^2/p) sum_{i<k} (k-i)^2 = g_F^2 p(p^2-1)/12,

with equality iff the values are evenly spaced with gap g_F. So evenly spaced F maximises the min-gap-to-norm ratio
g_F/||F|| = sqrt(12/(p(p^2-1))), and the same for f on F_p^*. The two scales: gamma/C = min(g_F, g_f)/(2p(p-1)[a_p g_F + b_p g_f]) is
maximised on g_F = g_f (any imbalance adds cost without margin). The same holds for 2 <= |J| <= p-1 (the kink of
min(|J| g_F, (|J|-1) g_F + g_f) is the optimum because b_p/a_p = sqrt((p-2)/(p+1)) < |J|/(|J|-1)); for |J| = p the optimum is g_f = 0.
Conclusion: the script's evenly spaced, same-scale choice is optimal within this circuit family; the structural inefficiency of the
plain circuit is that ||F|| ~ p^{3/2} g_F while only the smallest gap enters the margin.

## 2. Circuit 2: twisted (clock) circuit

### 2.1 Structure

Neuron (alpha, k, i): on the fiber a_x = alpha the pre-activation is cos(u + phi_i) + cos(v + phi_i) with u = 2 pi k b_x/p,
v = 2 pi k alpha b_y/p, and u + v = 2 pi k b_{xy}/p since b_{xy} = b_x + alpha b_y. Off the fiber L = -1 and R <= 1, so the ReLU is 0.
The readout cos(theta + 2 phi_i), theta = 2 pi k b_z/p, depends on b_z only; the unit part depends on a_z only. Hence
f_z = Lambda(b_z) + U(a_z) and

    gamma = min( gamma_clock , gamma_unit ),   gamma_unit = g_f = 1,

where gamma_clock is the clock margin over wrong z with a_z = a_{xy}, and gamma_unit over wrong z with b_z = b_{xy}.
Because fiber alpha sees the pairs (b_x, alpha b_y), every fiber has the same margin structure; the clock margin can be computed on
one fiber (p^2 inputs), which the script also does and which agrees with the full brute force.

### 2.2 The phase sum S(u, v, theta)

ReLU(cos(u+phi) + cos(v+phi)) = 2|c| ReLU(cos(psi + phi + pi s)), c = cos((u-v)/2), psi = (u+v)/2, s = 1[c < 0].
Fourier series (checked numerically): ReLU(cos t) = sum_n a_n cos(nt), a_0 = 1/pi, a_1 = 1/2,
a_{2m} = (2/pi)(-1)^{m+1}/(4m^2-1) (a_2 = 2/(3 pi), a_4 = -2/(15 pi), a_6 = 2/(35 pi), ...), odd n >= 3 vanish.
Phase sum identity: sum_{i=0}^{K-1} cos(n phi_i + A) cos(2 phi_i + B) = (K/2) [ cos(A-B) 1[n = 2 mod K] + cos(A+B) 1[n = -2 mod K] ].
With A_n = n(psi + pi s), the n = 2 term has A_2 = u + v (mod 2 pi), the sign s drops out, and

    S_K(u,v,theta) = K |cos((u-v)/2)| sum_{m>=1} a_{2m} [ cos(m(u+v) - theta) 1[2m = 2 mod K] + cos(m(u+v) + theta) 1[2m = -2 mod K] ]
                     (+ an a_1 term if K | 3, + an a_0 term if K | 2).

This finite-K expression is exact (max deviation from the direct K-term sum < 1e-4 over random inputs, for K = 3, 4, 6, 8, 12, 2000).
For K -> infinity (sum -> (K/2 pi) integral) only m = 1 survives:

    S_inf(u,v,theta) = c_K |cos((u-v)/2)| cos(u + v - theta),   c_K = 2 (from 2|c|) x a_2 x (K/2) = K a_2 = 2K/(3 pi) = 0.2122 K.

The m = 1 main term has exactly this coefficient at every K >= 5, so c_K = 2K/(3 pi) is not an approximation; the only finite-K
effect is the aliased higher harmonics listed next. The factor |cos((u-v)/2)| = |cos(pi k (b_x - alpha b_y)/p)| depends on the input
(and on k), so margins must be taken at the worst input.

### 2.3 Aliasing at finite K

Harmonic n of ReLU(cos t) reaches the readout iff n = +-2 (mod K). Present harmonics: n = 0, 1 and all even n.

| K  | surviving n (channel)                        | lowest alias, size relative to main a_2 | outcome |
|----|----------------------------------------------|------------------------------------------|---------|
| 3  | 2; 1 (A+B), 4, 8, 10, ...                     | n = 1, a_1/a_2 = 3 pi/4 = 2.36           | fails |
| 4  | 2 in BOTH channels; 6, 10, ... likewise        | the main term aliases with itself         | fails |
| 6  | 2; 4 (A+B), 8 (A-B), 10 (A+B), ...             | n = 4, 1/5 = 0.20                         | works, marginal |
| 8  | 2; 6 (A+B), 10 (A-B), 14 (A+B), ...            | n = 6, 3/35 = 0.086                       | works |
| 12 | 2; 10 (A+B), 14 (A-B), 22, 26, ...             | n = 10, 3/99 = 0.030                      | works |

K = 4: both channels of n = 2 survive, so S_4 = 8|c| sum_{m odd} a_{2m} cos(m(u+v)) cos(theta): the logit is a function of u+v times
cos(theta) (verified to 1e-14), the argmax over b_z can only be 0 or (p+-1)/2, accuracy 0.18 at p = 11, margin -0.35.
For even K >= 6 the lowest alias is n = K-2 with relative size 3/((K-2)^2 - 1).
Why the aliases hurt the single-frequency margin far more than their relative size: the main term's gap between adjacent b_z is
(1 - cos(2 pi/p)) ~ 2 pi^2/p^2 of the amplitude, while an alias of relative size eps varies by up to 2 eps sin(pi/p) ~ 2 pi eps/p between
adjacent b_z, so the relative damage is ~ eps p/pi: 0.7, 0.3, 0.1 for K = 6, 8, 12 at p = 11 (observed 55 %, 21 %, 6 %).
Single-frequency clocks need K >~ 2 + 1.4 sqrt(p) to be alias-safe. With all frequencies the gap is O(p) and the aliases cost only
8 % (K = 8) and 2.5 % (K = 12).

### 2.4 Single frequency

On fiber alpha with m = b_x - alpha b_y, Lambda(b_z) = c_K |cos(pi k m/p)| cos(2 pi k (b_{xy} - b_z)/p) (alias-free).
Gap to the nearest wrong b_z: c_K |cos(pi k m/p)| (1 - cos(2 pi/p)). As m runs over F_p, |cos(pi k m/p)| takes its minimum at
k m = (p+-1)/2, value cos(pi(p-1)/(2p)) = sin(pi/(2p)) (brute-force worst inputs: b_x - b_y = 4, 5, 7 at p = 7, 11, 13, as predicted).

    gamma_clock(1 freq) = c_K (1 - cos(2 pi/p)) sin(pi/(2p)) = (4K/(3 pi)) sin^2(pi/p) sin(pi/(2p)) ~ 2 K pi^2 / (3 p^3)   (alias-free)

Cost per twisted neuron (note: off-fiber there are p(p-2) elements, not (p-1)p):
||L||^2 = p(p-2) + p/2, ||R||^2 = p(p-1)/2, ||w||^2 = p(p-1)/2, product of norms (p/2) sqrt((3p-4)(p-1)). Count (p-1) K |F|.

    C_clock = (p-1) |F| K (p/2) sqrt((3p-4)(p-1)),      C_unit = 2p(p-1) b_p   (same as circuit 1)
    gamma/C (1 freq) = gamma_clock / (C_clock + C_unit) ~ 4 pi^2 / (3 sqrt(3)) p^{-6} = 7.6 p^{-6}   (alias-free, unit cost dropped)

### 2.5 All frequencies F = {1, ..., (p-1)/2}

Lambda(b_z) = c_K sum_k |cos(pi k m/p)| cos(2 pi k d/p), d = b_{xy} - b_z. The attenuation factor is frequency dependent, so the
margin is NOT c_K (p/2) sin(pi/(2p)). Two cases:
m = 0 (b_x = alpha b_y): all factors 1, Dirichlet kernel sum_k cos(2 pi k d/p) = (p delta_{d,0} - 1)/2, gap c_K p/2.
m != 0: as k runs over 1..(p-1)/2, the factors {|cos(pi k m/p)|} are a permutation of {cos(pi j/p)}_{j=1}^{(p-1)/2}, so with
j = +-k m and e = d/m the gap is c_K T(e), T(e) = sum_{j=1}^{(p-1)/2} cos(pi j/p)(1 - cos(2 pi j e/p)). Numerically the minimum is at
e = +-1 (d = +-m) for every p tested, and in closed form

    T(1) = (1/4) [ csc(pi/(2p)) + csc(3 pi/(2p)) ] = 2p/(3 pi) + O(1/p)   (2.358 at p = 11, 2.779 at p = 13; 2p/(3 pi) = 2.334, 2.759)

    gamma_clock(all freq) = c_K T(1) ~ (2K/(3 pi)) (2p/(3 pi)) = 4 K p / (9 pi^2) = 0.045 K p     (alias-free)

So the effective worst-case attenuation with all frequencies is T(1)/(p/2) -> 4/(3 pi) = 0.42, a constant, instead of sin(pi/(2p)) ~ pi/(2p).
The exactly-1.000 margins in the script are the unit part: gamma = min(gamma_clock, 1) with gamma_clock > 1. The clock part's own margins
(brute force, finite K; alias-free formula in parentheses): p = 7, K = 8: 2.387 (2.588); p = 11, K = 8: 3.694 (4.004); K = 12: 5.855 (6.006);
p = 13, K = 8: 4.353 (4.718); K = 12: 6.899 (7.077).

Rebalanced normalised margin (scale the clock part by 1/gamma_clock, or equivalently the unit part up, so both parts have margin 1):

    gamma/C (rebalanced) = 1 / ( C_clock/gamma_clock + C_unit/gamma_unit ).

### 2.6 Numbers (brute force; formulas agree to the digits shown)

| p  | K  | freqs | gamma | gamma_clock (alias-free) | C_clock | C_unit | gamma/C as built | rebalanced |
|----|----|-------|-------|--------------------------|---------|--------|------------------|------------|
| 7  | 4  | 1     | -0.35 (acc 0.29) | -0.35 (0.071) | 848.4 | 497.0 | fails | fails |
| 7  | 6  | 1     | 0.0666 | 0.0666 (0.1067) | 1272.5 | 497.0 | 0.0376e-3 | 0.0510e-3 |
| 7  | 8  | 1     | 0.1215 | 0.1215 (0.1422) | 1696.7 | 497.0 | 0.0554e-3 | 0.0692e-3 |
| 7  | 12 | 1     | 0.2033 | 0.2033 (0.2133) | 2545.1 | 497.0 | 0.0668e-3 | 0.0768e-3 |
| 7  | 8  | all 3 | 1.000 (unit) | 2.387 (2.588) | 5090.2 | 497.0 | 0.179e-3 | 0.380e-3 |
| 11 | 4  | 1     | -0.35 (acc 0.18) | -0.35 (0.019) | 3746.5 | 2826.0 | fails | fails |
| 11 | 6  | 1     | 0.0129 | 0.0129 (0.0288) | 5619.7 | 2826.0 | 0.0015e-3 | 0.0023e-3 |
| 11 | 8  | 1     | 0.0303 | 0.0303 (0.0384) | 7492.9 | 2826.0 | 0.0029e-3 | 0.0040e-3 |
| 11 | 12 | 1     | 0.0539 | 0.0539 (0.0575) | 11239.4 | 2826.0 | 0.0038e-3 | 0.0047e-3 |
| 11 | 8  | all 5 | 1.000 (unit) | 3.694 (4.004) | 37464.7 | 2826.0 | 0.0248e-3 | 0.0771e-3 |
| 11 | 12 | all 5 | 1.000 (unit) | 5.855 (6.006) | 56197.0 | 2826.0 | 0.0169e-3 | 0.0805e-3 |
| 13 | 6  | 1     | 0.0061 | 0.0061 (0.0176) | 9591.1 | 5276.4 | 0.0004e-3 | 0.0006e-3 |
| 13 | 8  | 1     | 0.0177 | 0.0177 (0.0234) | 12788.2 | 5276.4 | 0.0010e-3 | 0.0014e-3 |
| 13 | 12 | 1     | 0.0326 | 0.0326 (0.0352) | 19182.3 | 5276.4 | 0.0013e-3 | 0.0017e-3 |
| 13 | 8  | all 6 | 1.000 (unit) | 4.353 (4.718) | 76729.2 | 5276.4 | 0.0122e-3 | 0.0437e-3 |
| 13 | 12 | all 6 | 1.000 (unit) | 6.899 (7.077) | 115093.8 | 5276.4 | 0.0083e-3 | 0.0455e-3 |

These reproduce the script's 0.013/0.030/0.054 (p = 11) and 0.006/0.018/0.033 (p = 13) single-frequency margins, the gamma/C values
0.002/0.003/0.004e-3 and 0.025e-3 (p = 11), 0.001e-3 and 0.012e-3 (p = 13), and the K = 4 failure.

### 2.7 Large-p scaling and where the gap comes from

Cost per unit of margin, leading order (K cancels in the twisted circuit; alias-free):

    plain, coordinate part, |J| = p:   C_coord/gamma = p^2 a_p                  ~ p^{7/2}/sqrt(6) = 0.408 p^{3.5}
    twisted clock part, all freqs:      C_clock/gamma_clock                      ~ (9 sqrt(3) pi^2/16) p^3 = 9.62 p^3
    unit part (both circuits):          C_unit/gamma_unit = 2p(p-1) b_p          ~ (2/sqrt(6)) p^{7/2} = 0.816 p^{3.5}
    twisted clock part, one freq:       C_clock/gamma_clock                      ~ (3 sqrt(3)/(4 pi^2)) p^6 = 0.13 p^6

Hence

    plain |J| = p:                 gamma/C ~ sqrt(6) p^{-7/2}                    (2.45 p^{-3.5})
    twisted all-freq, as built:    gamma/C = 1/(C_clock + C_unit) ~ 4/(sqrt(3) K p^4)   (unit-limited, margin 1, K = 8: 0.29 p^{-4})
    twisted all-freq, rebalanced:  gamma/C ~ 1/(9.62 p^3 + 0.816 p^{3.5})  -> (sqrt(6)/2) p^{-7/2} as p -> inf, half of plain
    twisted clock mechanism alone: gamma/C ~ 0.104 p^{-3}
    twisted one freq:              gamma/C ~ 7.6 p^{-6}

| p    | plain |J|=p | twisted as built (K=8) | twisted rebalanced | clock part alone |
|------|-------------|------------------------|--------------------|------------------|
| 11   | 5.17e-4     | 2.48e-5                | 8.21e-5            | 1.07e-4          |
| 13   | 2.90e-4     | 1.22e-5                | 4.64e-5            | 6.15e-5          |
| 31   | 1.43e-5     | 3.32e-7                | 2.61e-6            | 3.88e-6          |
| 101  | 2.34e-7     | 2.80e-9                | 5.60e-8            | 1.04e-7          |
| 547  | 6.39e-10    | 3.21e-12               | 2.14e-10           | 6.39e-10         |
| 1009 | 7.50e-11    | 2.77e-13               | 2.75e-11           | 1.02e-10         |

Scaling conclusion. The twisted clock mechanism scales one half power of p better than the plain coordinate mechanism
(p^{-3} vs p^{-3.5}): its features are bounded (|cos| <= 1) so a neuron costs ~ p^2 instead of ~ p^{5/2}, and the Dirichlet kernel
turns p/2 frequencies into a margin ~ p. But its constant is 23.6x worse, so the clock part only overtakes the coordinate part at
p ~ (9.62/0.408)^2 = 555, and the full twisted circuit as specified never does, because the unit part it needs (itself a coordinate
circuit on F_p^*, 0.816 p^{3.5}) becomes the bottleneck (plain/twisted -> 2 + 23.6/sqrt(p)). With a clock-style unit part on the
cyclic group F_p^* (cost per margin O(p^2)) the twisted circuit would win for p > 555. At p = 11, 13 the plain circuit wins by 21x
and 24x as built, by 6.3x and 6.3x after rebalancing.

Decomposition of the as-built gap at p = 11, K = 8 (factor 20.8):
1. unit-limited margin: gamma = 1 while gamma_clock = 3.69; rebalancing gives 3.1x.
2. remaining 6.7x: cost per unit margin 10142 (clock) + 2826 (unit) vs 1795 (coordinates) + 141 (unit, amortised over margin 20).
   The unit part is the same object in both circuits (2826); it is 20x cheaper per unit margin in the plain circuit only because the
   coordinate margin is 20 rather than 3.7.
3. The clock/coordinate ratio 5.65 (5.2 alias-free). Write each cost per margin as (neuron count) x (per-neuron cost) / (margin):
   plain 2p^2 x (p-1) a_p / 2(p-1);  twisted (p-1)^2 K/2 x (p/2) sqrt((3p-4)(p-1)) / (c_K T(1)). The ratio factorises as
   [count (p-1)^2 K/(4p^2) -> K/4] x [per-neuron (p/2)sqrt((3p-4)(p-1))/((p-1)a_p) -> sqrt(3) (sqrt(6)/2)/sqrt(p)] x
   [margin 2(p-1)/(c_K T(1)) -> (4/K)(3 pi/2)(3 pi/4)];  at p = 11, K = 8: 1.65 x 0.632 x 5.0 = 5.2.  K cancels, leaving
   23.6/sqrt(p) = (3 pi/2) x (3 pi/4) x sqrt(3) x (sqrt(6)/2)/sqrt(p):
   - 3 pi/2 = 4.71: ReLU attenuation (only the second phi-harmonic, weight a_2 = 2/(3 pi), reaches the readout);
   - 3 pi/4 = 2.36: worst-case |cos((u-v)/2)|, T(1)/(p/2) -> 4/(3 pi) (2.33 at p = 11; it would be 2p/pi = 7 with one frequency);
   - sqrt(3) = 1.73: silencing, ||(L,R)||^2 = p(3p-4)/2 instead of p^2/2 (1.62 at p = 11; 0.82 x this with the split constant of section 3);
   - (sqrt(6)/2)/sqrt(p) = 1.22/sqrt(p): bounded clock features (||L||^2 + ||R||^2 ~ p^2/2 without silencing) against the plain circuit's
     F with ||F||/g_F ~ p^{3/2}; this is the plain circuit's min-gap-to-norm handicap and the only p-dependent factor;
   - finite-K aliasing: 4.004/3.694 = 1.08 at K = 8 (1.03 at K = 12);
   - per-fiber duplication: the (p-1) fiber copies are matched one for one by the plain circuit's p value-copies (one neuron pair per
     (j0, c)), so both circuits have Theta(p^2) neurons and the duplication is not a factor relative to the plain circuit; it is a factor
     (p-1) relative to a hypothetical single-fiber clock, and it is what forces the silencing cost.

## 3. Cheaper silencing?

Smaller offset alone (L = -eta off-fiber, eta < 1, R unchanged) breaks exact silencing. An off-fiber neuron then contributes
sum_i ReLU(cos(v + phi_i) - eta) cos(theta + 2 phi_i) ~ (K/2) a_2(eta) cos(2v - theta) with the exact coefficient

    a_2(eta) = (2/(3 pi)) (1 - eta^2)^{3/2}   (verified; a_2(1) = 0, a_2(0) = 2/(3 pi)),

i.e. a spurious clock pointing at b_z = 2 alpha b_y. Summed over the p-2 inactive fibers and all frequencies: for b_y != 0 the sum over
alpha cancels to O(1) penalties on b_z = 0 and b_z = 2 a_x b_y (harmful only when one of them is the correct answer, penalty
~ (K/4) a_2 p); for b_y = 0 all inactive fibers are coherent and produce a Dirichlet spike ~ (K/4) a_2(eta) p(p-2) on b_z = 0.
Requiring this to stay below gamma_clock ~ 4Kp/(9 pi^2) gives a_2(eta) < 0.18/(p-2), i.e. 1 - eta^2 < (0.85/(p-2))^{2/3} ~ p^{-2/3},
so the admissible saving in ||L||^2 is O(p^{-2/3}) of the silencing term and vanishes with p. Brute force (p = 11, K = 8, all freqs):
eta = 0.95 saves 3 % of C_clock and cuts the clock margin from 3.69 to 1.50 (the narrow ReLU bump also has slowly decaying
harmonics that alias at finite K, so the damage exceeds the spike estimate); eta = 0.9 gives a negative margin. Not worth it.

Split the constant instead (exact silencing kept). Put L = cos(u + phi) + beta on the fiber, L = -(1 - beta) off it, R = cos(v + phi) - beta.
On the fiber L + R is unchanged, off it L + R = cos(v + phi) - 1 <= 0, so all activations and the margin are identical (verified).

    ||(L,R)||^2 = p(p-2)(1-beta)^2 + p(1/2 + beta^2) + p(p-1)(1/2 + beta^2),  minimised at beta* = (p-2)/(2(p-1)) ~ 1/2,
    value p^2 (2p-3)/(2(p-1)) ~ p^2   vs   p(3p-4)/2 ~ 3p^2/2 at beta = 0.

Cost factor sqrt(p(2p-3)/((3p-4)(p-1))) -> sqrt(2/3) = 0.816 (0.869 at p = 7, 0.849 at p = 11, brute force 0.8689, 0.8489).
This is optimal within exact silencing of an exact clock: the on-fiber constraint forces R = cos(v+phi) - beta, so max R = 1 - beta and
every off-fiber L must be <= -(1-beta); the constant 1 has to be paid on ~p^2 entries of L or of R, and (1-beta)^2 + beta^2 >= 1/2.
It shifts the twisted constants by 0.82 (clock part alone 0.127 p^{-3}, crossover with the plain circuit p ~ 370) but not the exponents.

## 4. Can the per-fiber duplication be avoided?

No, not in one additive hidden layer. The twisted neuron needs its y-side feature at frequency k a_x, and R(y) cannot see a_x; the only
way to get a_x-dependence into the pre-activation L(x) + R(y) is one copy of R per value alpha of a_x, i.e. one neuron family per fiber,
and then each family must be silenced off its fiber because ReLU is nonlinear (a zero L off-fiber would still pass ReLU(R(y)) through).
Moving the twist to the x side, L = cos(2 pi k b_x/(a_x p) + phi), R = cos(2 pi k b_y/p + phi), gives u + v = 2 pi k b_{xy}/(a_x p), which the
readout would have to undo with a_x; but w depends on z only and a_x is not a function of z = xy. Representation-theoretically the
(p-1)-dimensional irrep is induced from a character of the translation subgroup, so its matrix coefficients are each supported on a
single fiber a_x = const; any neuron that reads an irrep feature of x is fiber-local by construction, and the p-1 fibers are the p-1
rows of the representation matrix. The duplication (and with it the p(p-1)/2 entries of R and w stored p-1 times over) is the price
of the irrep being induced, not of this particular construction.

## 5. What was verified numerically

- Fourier coefficients of ReLU(cos t); the exact finite-K alias series for S_K (K = 3, 4, 6, 8, 12, 2000) against the direct phase sum;
  a_2(eta) in closed form; T(1) in closed form and T(e) minimised at e = +-1 (p = 7 ... 101).
- Circuit 1 built exactly as specified: gamma, C, gamma/C for |J| = 1, 2, floor(p/2), p-1, p at p = 7, 11, 13; case A vs case B minima
  separately; the |J| = p minimiser; no-unit-part and unequal-scale variants (g_f = 0.5, 2).
- Circuit 2 built exactly as specified: gamma, gamma_clock, gamma_unit, accuracy, C_clock, C_unit for K = 4, 6, 8, 12 (one frequency)
  and K = 8, 12 (all frequencies) at p = 7, 11, 13; the single-fiber computation; the worst single-frequency input; the eta < 1 and the
  split-constant silencing variants; the K = 4 collapse to (function of u+v) x cos(theta).
Approximations used only in the labelled places: alias-free (K -> inf) clock margins, the case B lower bound for |J| = p (valid p >= 7,
checked at 7, 11, 13), and the large-p leading orders.
