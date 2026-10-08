# Width grid: where does the ReLU threshold sit, and what fills the width?

Runs: `runs/grid/` from `runs/launch_grid_{a,b,d}.sh` and two one-off commands (Aff(F_7) at widths 128 and 256, and at width 64 with
60% of the pairs; Aff(F_13) at 192). R-Wu, 100k epochs so that "failed" means failed at the long schedule (one Aff(F_11) seed had
memorised at 25k and grokked at 100k in the earlier runs). Grokked means test accuracy >= 0.99 at the end; the epoch is the first
logged epoch at or above 0.99. Table `grid_summary.csv`; the census columns come from `scripts/affine_census.py` via
`scripts/affine_grid_summary.py`. Post hoc. Cells still running when this was written are marked.

## Grokking by width

| p | p^2 | width: outcome by seed (grok epoch) |
|---|---|---|
| 7 | 49 | 40: no, no, no. 48: no, no, no. 56: no, no, no. 64: no, no, no. 72: no, no. 80: 0.985, no. 88: no, no. 96: no, yes (2500). 112: 0.989, yes (1500). 128: yes (2250). 256: yes (1500) |
| 7, 60% of pairs | 49 | 40: no, no. 48: yes (4750), 0.955. 56: yes (1750), yes (1500). 64: yes (1000) |
| 11 | 121 | 100: no, no, no. 108: yes (67750), no, no. 116: yes (24750), no, no. 124: no, yes (7000), yes (92750). 128 (earlier runs): 5 of 6 at 100k. 132: yes (2000), yes (2750), 0.929. 140: yes (3750), yes (5000), yes (13250) |
| 13 | 169 | 128: no x 6 (earlier runs, both recipes). 160: yes (12000). 192: yes (3000). 224: yes (2250). 256: yes, yes (1750, 1500) |

At 40% of the pairs the p = 7 column is data-limited, not width-limited: every width up to 88 fails (80 reaches 0.985), 96 and
112 grok for one seed of two and 128 for the one seed run, while p^2 = 49; but width 64 groks in 1000 epochs as soon as the
training set is 60% of the pairs (1058 pairs instead of 705). With 60% of the pairs the p = 7 threshold falls between 40 (fails
twice) and 56 (groks twice, with 48 split one of two), i.e. at p^2. For p = 11 the fraction of seeds that grok within 100k epochs
rises from 0 of 3 at 100 through 1 of 3 at 108 and 116 and 2 of 3 at 124 and 132 to 3 of 3 at 140, so the transition is spread
over roughly 100 to 140 with p^2 = 121 in the middle. For p = 13 width 160 groks (12k epochs), so the threshold lies between 128
and 160, at or below p^2 = 169. Once data is sufficient, then, the threshold sits at p^2 for p = 7 and 11 and at or just under it
for p = 13; it is soft rather than sharp, and grokking time moves with width: at p = 11 the successful seeds take 25k to 93k
epochs at 108 to 124 and 2k to 5k at 132 and 140; at p = 13, 12k epochs at 160 against 1.5k to 3k at 192 and 256. The earlier
reading that the thresholds grow more slowly than p^2 rested on the data-limited p = 7 column and does not survive the 60% runs.

## What the grokked models contain

| run | width | live | plain big-irrep neurons | cells (j0, c) covered | doubly covered | coordinates | coset-mean accuracy | plain removed |
|---|---|---|---|---|---|---|---|---|
| aff7 width 48, 60% pairs | 48 | 48 | 39 | 27 of 49 | 12 | 4 | 0.999 | 0.117 |
| aff7 width 56, 60% pairs, seeds 1, 2 | 56 | 56 | 49, 43 | 48, 37 | 1, 6 | 7, 6 | 1.000, 0.999 | 0.05, 0.12 |
| aff7 width 64, 60% pairs | 64 | 64 | 54 | 35 of 49 | 19 | 7 | 1.000 | 0.126 |
| aff7 width 96, seed 2 | 96 | 96 | 76 | 48 of 49 | 28 | 7 | 0.999 | 0.170 |
| aff7 width 112, seed 2 | 112 | 83 | 70 | 35 of 49 | 35 | 7 | 0.998 | 0.143 |
| aff7 width 128 | 128 | 95 | 85 | 49 of 49 | 36 | 7 | 0.999 | 0.144 |
| aff7 width 256 | 256 | 96 | 86 | 49 of 49 | 37 | 7 | 0.998 | 0.141 |
| aff11 width 108, seed 1 | 108 | 108 | 99 | 72 of 121 | 27 | 7 | 1.000 | 0.094 |
| aff11 width 116, seed 1 | 116 | 116 | 95 | 81 | 14 | 11 | 1.000 | 0.091 |
| aff11 width 124, seeds 2, 3 | 124 | 124 | 109, 103 | 80, 75 | 29, 26 | 9, 7 | 1.000 | 0.04, 0.09 |
| aff11 width 128 (earlier, 6 runs) | 128 | 128 | 112 to 119 | 66 to 89 | 23 to 50 | 6 to 11 | 1.000 | 0.03 to 0.09 |
| aff11 width 132, seeds 1, 2 | 132 | 132 | 108, 116 | 85, 76 | 23, 40 | 11, 7 | 1.000 | 0.07, 0.06 |
| aff11 width 140, seeds 1, 2, 3 | 140 | 140 | 128, 123, 115 | 88, 90, 93 | 40, 33, 22 | 9, 11, 9 | 1.000 | 0.08, 0.09, 0.12 |
| aff13 width 160 | 160 | 160 | 138 | 105 of 169 | 33 | 11 | 1.000 | 0.048 |
| aff13 width 192 | 192 | 190 | 164 | 90 of 169 | 72 | 10 | 1.000 | 0.077 |
| aff13 width 224 | 224 | 209 | 179 | 100 of 169 | 79 | 10 | 1.000 | 0.077 |
| aff13 width 256 (earlier, 2 runs) | 256 | 213, 240 | 203, 210 | 121, 120 of 169 | 82, 90 | 11, 10 | 1.000 | 0.03, 0.08 |

Every grokked model is the same circuit as before (`circuit_census.md`): pairs of neurons per cell (j0, c) with graded profiles,
readouts the indicator of {z : z(j0) = c}, 100% cell match, accuracy unchanged under coset-mean replacement and at chance without
the plain component. The reviewer's stronger check, that nets just above threshold hold close to p^2 plain-aligned neurons, holds
at p = 11: at widths 108 to 140 the plain neurons number 95 to 128, i.e. the width minus about ten unit-part neurons, with no
dead neurons, and they cover 72 to 90 of the 121 cells; at p = 13 width 160 is filled the same way (138 plain neurons, 105 cells). At p = 7 the circuit saturates instead: widths 128 and 256 both use 85 to
86 plain neurons covering all 49 cells (36 to 37 of them twice) and leave the rest dead, and at p = 13 width 192 is filled (164
plain neurons, 90 cells, 72 of them twice) while width 256 leaves 16 to 43 dead with 120 cells covered. So the number of neurons the circuit wants is about p^2 cells times one to two, and the models fill
whatever width they have up to that, which is why the p = 11 models at 108 to 132 look width-limited and the p = 7 models do not.

## Reading

- Capacity is not the limit (a plain circuit is exact at 4p - 2 neurons, `circuit_census.md`). What the data support is more
  specific: the width at which grokking becomes reliable is about p^2, the number of cells (j0, c), and the model just above it
  holds one neuron per cell (width 56 at p = 7 with 60% of the pairs: 49 plain neurons covering 48 of 49 cells, one of them
  twice). Below that the plain circuit cannot be laid out over all cells, grokking becomes slow and seed-dependent, and below a
  further margin it fails within 100k epochs. The margin analysis (`maxmargin.md`) says why covering every cell pays: margin per
  unit norm rises with the number of coordinates used.
- The 40% column at p = 7 is confounded by data (705 training pairs); the 60% column removes the confound and puts the threshold
  at p^2. At p = 11 and 13 the 40% split already gives 4840 and 9734 pairs, so no such correction was needed there.
- `grid_summary.csv` holds every cell and is regenerated by `scripts/affine_grid_summary.py --runs runs/grid/*`.
