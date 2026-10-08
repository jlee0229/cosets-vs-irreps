"""Neuron census of grokked Aff(F_p) models: what the plain circuit is, neuron by neuron, and how many neurons it uses.
usage: scripts/affine_census.py --runs DIR [DIR ...] [--out PREFIX]

For every live neuron k with pre-activation L_k(x) + R_k(y):
  irrep content of L_k, R_k (the (p-1)-dim irrep vs the linear characters), best plain projector and its point stabilizer per side,
  the profile F of L_k over the p right cosets of stab(c) (indexed by x^-1(c)), likewise R_k over the left cosets of stab(j0) (by y(j0)),
  the set {z : z(j) = v} that the readout row W_U[k] picks out, the pre-activation on the pairs with (xy)(j) = v,
  the active fraction of ReLU(L_k + R_k) over all pairs, and the twisted-family alpha for comparison.
Then two ablations on the same model: every big-irrep side replaced by its coset-mean (plain projection), and the plain
component removed (constant kept).  Accuracy over all |G|^2 pairs in both cases."""
import json, argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from cosetprobe.affine import Affine

def sides(sd):
    Wx, Wy, W, WU = [sd[k].numpy().astype(np.float64) for k in ("W_x", "W_y", "W", "W_U")]
    E = Wx.shape[1]; return Wx @ W[:E], Wy @ W[E:], WU

def accuracy(A, Ex, Ey, WU, act=None):
    pre = Ex[:, None, :] + Ey[None, :, :]; logits = (np.maximum(pre, 0) if act is None else act(pre)) @ WU
    return (logits.argmax(2) == A.T).mean()

def point_profile(A, u, c, side):
    p = A.nmod; v = np.zeros(p); cnt = np.zeros(p)
    for i, (a, b) in enumerate(A.els):
        j = ((c - b) * pow(a, -1, p)) % p if side == "r" else (a * c + b) % p
        v[j] += u[i]; cnt[j] += 1
    return v / cnt

def census(run):
    run = Path(run); cfg = json.load(open(run / "cfg.json")); p = int(cfg["group"][3:]); A = Affine(p)
    Ex, Ey, WU = sides(torch.load(run / "model.pt", map_location="cpu")); Ux, Uy = Ex - Ex.mean(0), Ey - Ey.mean(0); WUc = WU - WU.mean(1, keepdims=True)
    m = Ux.shape[1]; n2 = (Ux ** 2).sum(0) + (Uy ** 2).sum(0); live = n2 > 1e-4 * n2.max()
    plain = A.plain_family(); twist = A.twisted_family(); big = A.block[p]
    Pbig = A.isotypic_projector(big).real; Plin = sum(A.isotypic_projector(r) for r in range(len(A.dims)) if A.dims[r] == 1).real
    zimg = np.array([[(a * j + b) % p for j in range(p)] for (a, b) in A.els])            # zimg[z, j] = z(j)
    ind = {(j, v): (lambda f: (f - f.mean()) / np.linalg.norm(f - f.mean()))((zimg[:, j] == v).astype(float)) for j in range(p) for v in range(p)}
    rows = []
    for k in range(m):
        r = dict(run=run.name, p=p, neuron=k, live=bool(live[k]), energy=n2[k] / n2.sum())
        for s, U in (("x", Ux), ("y", Uy)):
            u = U[:, k]; den = max((u ** 2).sum(), 1e-300)
            r[f"big_{s}"] = ((Pbig @ u) ** 2).sum() / den; r[f"lin_{s}"] = ((Plin @ u) ** 2).sum() / den
            sc = [((Q @ u) ** 2).sum() / den for _, Q in plain]; i = int(np.argmax(sc)); r[f"alpha_{s}"] = sc[i]; r[f"lab_{s}"] = plain[i][0]
            r[f"twist_{s}"] = max(((Q @ u) ** 2).sum() / den for _, Q in twist)
        cx, sx, jy, sy = int(r["lab_x"][1:-1]), r["lab_x"][-1], int(r["lab_y"][1:-1]), r["lab_y"][-1]
        Fx = point_profile(A, Ux[:, k], cx, sx); Fy = point_profile(A, Uy[:, k], jy, sy); Fx -= Fx.mean(); Fy -= Fy.mean()
        e = Fx ** 2 / max((Fx ** 2).sum(), 1e-300); r["prof_top1"] = e.max(); r["prof_distinct"] = len(np.unique(np.round(Fx / max(np.abs(Fx).max(), 1e-300), 2)))
        r["prof_xy_neg"] = np.linalg.norm(np.sort(Fx) + np.sort(Fy)[::-1]) / max(np.linalg.norm(Fx), 1e-300)
        P = np.abs(np.fft.fft(Fx))[1:] ** 2; r["prof_dft_top"] = P.max() / max(P.sum(), 1e-300)
        w = WUc[k]; wn = w / max(np.linalg.norm(w), 1e-300); sc = {jv: float(wn @ f) for jv, f in ind.items()}; (jr, vr) = max(sc, key=lambda jv: sc[jv] ** 2)
        r["out_j"], r["out_v"], r["out_score"], r["out_sign"] = jr, vr, sc[(jr, vr)] ** 2, np.sign(sc[(jr, vr)])
        r["cell_matches"] = bool(sx == "r" and sy == "l" and jr == jy and vr == cx)
        U = Ux[:, k][:, None] + Uy[:, k][None, :]; hit = zimg[A.T, jr] == vr; tot = max(np.abs(U).mean(), 1e-300)
        r["preact_on_cell"] = np.abs(U[hit]).mean() / tot; r["preact_off_cell"] = np.abs(U[~hit]).mean() / tot
        H = np.maximum(Ex[:, k][:, None] + Ey[:, k][None, :], 0); r["active_frac"] = (H > 0).mean()
        rows.append(r)
    df = pd.DataFrame(rows)
    # ablations
    isbig = live & (df.big_x.values > 0.9) & (df.big_y.values > 0.9)
    Qx = {lab: Q for lab, Q in plain}
    Ex1, Ey1, Ex2, Ey2 = Ex.copy(), Ey.copy(), Ex.copy(), Ey.copy()
    for k in np.where(isbig)[0]:
        for E1, E2, E0, lab in ((Ex1, Ex2, Ex, df.lab_x[k]), (Ey1, Ey2, Ey, df.lab_y[k])):
            q = Qx[lab] @ E0[:, k]; E1[:, k] = q; E2[:, k] = E0[:, k] - q + E0[:, k].mean()
    act = None if cfg.get("act", "relu") == "relu" else (lambda t: np.abs(t) ** cfg["q"])     # |t|^q models from train.py --act absq
    acc = dict(acc=accuracy(A, Ex, Ey, WU, act), acc_cosetmean=accuracy(A, Ex1, Ey1, WU, act), acc_plain_removed=accuracy(A, Ex2, Ey2, WU, act))
    return df, acc, cfg

def report(df, acc, cfg):
    L = df[df.live]; p = int(df.p.iloc[0]); m = len(df)
    big = L[(L.big_x > 0.9) & (L.big_y > 0.9)]; lin = L[(L.lin_x > 0.9) & (L.lin_y > 0.9)]
    pl = big[(big.alpha_x >= 0.9) & (big.alpha_y >= 0.9)]
    cells = pl.groupby(["out_j", "out_v"]).size()
    print(f"\n{df.run.iloc[0]}  p={p} m={m}  test acc {cfg['final']['test_acc']:.4f}  live {len(L)}")
    print(f"  big-irrep neurons {len(big)} (energy {big.energy.sum():.2f}), linear-character neurons {len(lin)} (energy {lin.energy.sum():.2f}), other {len(L) - len(big) - len(lin)}")
    print(f"  big-irrep neurons plain-aligned on both sides: {len(pl)}/{len(big)}; sides x {pl.lab_x.str[-1].value_counts().to_dict()} y {pl.lab_y.str[-1].value_counts().to_dict()}")
    print(f"  readout = one set {{z: z(j)=v}}: score median {pl.out_score.median():.3f} min {pl.out_score.min():.3f}; sign {pl.out_sign.value_counts().to_dict()}; cell (j,v) = (y point, x point): {int(pl.cell_matches.sum())}/{len(pl)}")
    print(f"  |pre-activation| on the readout cell / overall: median {pl.preact_on_cell.median():.3f} (off-cell {pl.preact_off_cell.median():.3f}); active fraction median {pl.active_frac.median():.3f}")
    print(f"  profile F: top-coset energy median {pl.prof_top1.median():.2f} (one-hot would be {(p-1)/p:.2f}), distinct values median {pl.prof_distinct.median():.0f}/{p}, DFT top-frequency share median {pl.prof_dft_top.median():.2f} (single mode 1.0), F_y = -F_x residual median {pl.prof_xy_neg.median():.3f}")
    print(f"  cells (j,v) covered: {len(cells)} of {p*p}; neurons per cell {cells.value_counts().sort_index().to_dict()}; coordinates j used: {len(pl.out_j.unique())} of {p}; 2p per coordinate = {2*p}")
    print(f"  twisted alpha of the same neurons: median {pl.twist_x.median():.3f} max {pl.twist_x.max():.3f}")
    print(f"  ablation: accuracy {acc['acc']:.4f}; big-irrep sides replaced by coset means {acc['acc_cosetmean']:.4f}; plain component removed {acc['acc_plain_removed']:.4f}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out"); a = ap.parse_args()
    dfs, accs = [], []
    for run in a.runs:
        df, acc, cfg = census(run); report(df, acc, cfg); dfs.append(df); accs.append(dict(run=Path(run).name, **acc))
    if a.out:
        pd.concat(dfs).to_csv(a.out + "_neurons.csv", index=False); pd.DataFrame(accs).to_csv(a.out + "_ablation.csv", index=False)

if __name__ == "__main__":
    main()
