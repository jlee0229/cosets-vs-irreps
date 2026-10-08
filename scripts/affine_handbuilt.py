"""Hand-built one-hidden-layer ReLU circuits for Aff(F_p) multiplication: neuron counts, exact accuracy over all |G|^2 pairs,
and a margin-per-norm proxy  min_(x,y) [logit(xy) - max_{z != xy} logit(z)] / sum_k ||(L_k, R_k)|| ||w_k||.
usage: scripts/affine_handbuilt.py [--p 11 13] [--runs DIR ...]

  a-part  : a_z = a_x a_y via graded pairs ReLU(+-(f(a_x) - f(v / a_y))) read out by -1[a_z = v];  2(p-1) neurons, exact.
  plain J : b-part via point coordinates j0 in J:  ReLU(+-(f(x^-1(c)) - f(y(j0)))) read out by -1[z(j0) = c];  2p|J| neurons, exact.
            This is the circuit the trained models use (scripts/affine_census.py).
  twisted : b-part via single-fiber, single-frequency clocks: for each fiber a_x = alpha and frequency k, K phases
            ReLU(1[a_x=alpha] cos(k b_x + phi) + cos(k alpha b_y + phi)) read out by cos(k b_z + 2 phi);  K(p-1)|F| neurons, approximate.
--runs scores trained models with the same proxy (uncentered side embeddings)."""
import argparse, numpy as np, torch
from pathlib import Path
from cosetprobe.affine import Affine

def evaluate(A, L, R, W, label):
    n = A.n; T = A.T
    logits = np.einsum("kxy,kz->xyz", np.maximum(L[:, :, None] + R[:, None, :], 0), W)
    correct = logits[np.arange(n)[:, None], np.arange(n)[None, :], T]
    logits[np.arange(n)[:, None], np.arange(n)[None, :], T] = -np.inf
    margin = (correct - logits.max(2)).min(); acc = (logits.max(2) < correct).mean()
    cost = (np.sqrt((L ** 2).sum(1) + (R ** 2).sum(1)) * np.sqrt((W ** 2).sum(1))).sum()
    print(f"  {label:36s} neurons {len(L):4d}  acc {acc:.4f}  min margin {margin:+8.3f}  margin/cost {1e3 * margin / cost:+.3f}e-3")

def a_part(A):
    p = A.nmod; f = {a: i - (len(A.units) - 1) / 2 for i, a in enumerate(A.units)}; L, R, W = [], [], []
    for v in A.units:
        l = np.array([f[a] for a, b in A.els]); r = np.array([-f[v * pow(a, -1, p) % p] for a, b in A.els]); w = -np.array([float(a == v) for a, b in A.els])
        for s in (1, -1): L.append(s * l); R.append(s * r); W.append(w)
    return np.array(L), np.array(R), np.array(W)

def plain_part(A, J, f=None):
    p = A.nmod; f = np.arange(p) - (p - 1) / 2 if f is None else np.asarray(f); L, R, W = [], [], []
    for j0 in J:
        for c in range(p):
            l = np.array([f[(c - b) * pow(a, -1, p) % p] for a, b in A.els]); r = np.array([-f[(a * j0 + b) % p] for a, b in A.els])
            w = -np.array([float((a * j0 + b) % p == c) for a, b in A.els])
            for s in (1, -1): L.append(s * l); R.append(s * r); W.append(w)
    return np.array(L), np.array(R), np.array(W)

def twisted_part(A, K, freqs):
    p = A.nmod; L, R, W = [], [], []
    for alpha in A.units:
        for k in freqs:
            for i in range(K):
                ph = 2 * np.pi * i / K
                L.append(np.array([np.cos(2 * np.pi * k * b / p + ph) if a == alpha else -1.0 for a, b in A.els]))   # silent off its fiber
                R.append(np.array([np.cos(2 * np.pi * k * alpha * b / p + ph) for a, b in A.els]))
                W.append(np.array([np.cos(2 * np.pi * k * b / p + 2 * ph) for a, b in A.els]))
    return np.array(L), np.array(R), np.array(W)

def cat(*parts): return tuple(np.concatenate([q[i] for q in parts]) for i in range(3))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--p", type=int, nargs="+", default=[11, 13]); ap.add_argument("--runs", nargs="*", default=[]); a = ap.parse_args()
    for p in a.p:
        A = Affine(p); print(f"\nAff(F_{p}), |G| = {A.n}, p^2 = {p * p}"); ap_ = a_part(A)
        evaluate(A, *ap_, "a-part alone")
        for J in ([0], [0, 1], list(range(p // 2)), list(range(p))): evaluate(A, *cat(ap_, plain_part(A, J)), f"plain, {len(J)} coordinate(s) + a-part")
        for K in (4, 6, 8, 12): evaluate(A, *cat(ap_, twisted_part(A, K, [1])), f"twisted, K={K}, one frequency + a-part")
        half = list(range(1, (p - 1) // 2 + 1)); evaluate(A, *cat(ap_, twisted_part(A, 8, half)), f"twisted, K=8, all {len(half)} frequencies + a-part")
        for run in a.runs:
            sd = torch.load(Path(run) / "model.pt", map_location="cpu")
            if sd["W_x"].shape[0] != A.n: continue
            Wx, Wy, W, WU = [sd[k].numpy().astype(np.float64) for k in ("W_x", "W_y", "W", "W_U")]; E = Wx.shape[1]
            evaluate(A, (Wx @ W[:E]).T, (Wy @ W[E:]).T, WU, f"trained {Path(run).name}")

if __name__ == "__main__":
    main()
