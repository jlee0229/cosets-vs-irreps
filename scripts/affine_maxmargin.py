"""Numerical max-margin circuits for Aff(F_p): the largest normalised margin each family of hidden neurons can reach.
usage: scripts/affine_maxmargin.py --p 11 [--families plain1 plain2 plain twisted free] [--steps 4000] [--seeds 1 2]
                                   [--free_from plain twisted] [--trained DIR ...] [--plain_init monotone|perm] [--device mps] --out PREFIX

Normalised margin  gamma / C  with  gamma = min over pairs (x, y) and z != xy of f_xy(x, y) - f_z(x, y)  and
C = sum_k ||(L_k, R_k)|| ||w_k||, the quantity L2 weight decay controls in a two-layer ReLU net once in and out norms balance.
Maximised by Adam on a soft-min surrogate inside a parameterised family (float32 on the device), then the exact (hard)
margin of the best iterate is computed in float64 on the CPU.
  plain    : L_k a function of x^-1(c_k), R_k a function of y(j0_k), two neurons per cell (j0, c), free profiles and readouts,
             initialised at the hand-built circuit (evenly spaced F); plainJ uses only the coordinates j0 < J.
  twisted  : L_k = u_a cos(2 pi f b_x / a / p) + v_a sin(..) + o_a on the fiber a_x = a (frequency f permuted by a, free fiber
             amplitudes and offsets), R_k = u'_a cos(2 pi f b_y / p) + v'_a sin(..) + o'_a (fixed frequency), free readouts;
             one neuron per (fiber, frequency, phase) with K = 8 phases, initialised at the hand-built single-fiber clock.
  free     : L_k, R_k, w_k unconstrained; random init, or (--free_from) from the plain / twisted optimum, or (--trained) from a
             trained model's side embeddings and readout.
Every family also gets 4(p-1) unit-part neurons (L a function of a_x, R a function of a_y).
Also reported: the cost of realising the circuit in the trained factorised architecture (W_x W, W_y W, W_U), which is
2 ||L||_* + 2 ||R||_* + ||W_U||_F^2, with the 3-homogeneous normalisation gamma / cost^1.5; and a census of the solution
(fraction of live hidden energy in the big irrep / the linear characters, and of the big-irrep part, the fraction with
plain alpha >= 0.9 or twisted alpha >= 0.9 on both sides)."""
import argparse, json, time, numpy as np, pandas as pd, torch
from pathlib import Path
from torch.nn import Parameter
from cosetprobe.affine import Affine


class Setup:
    def __init__(self, p, device, dtype):
        self.A = A = Affine(p); self.p, self.n, self.dev, self.dtype = p, A.n, device, dtype; els, units = A.els, A.units
        self.T = torch.tensor(A.T, device=device)
        self.aidx = torch.tensor([units.index(a) for a, b in els], device=device)
        self.inv_pt = torch.tensor([[(c - b) * pow(a, -1, p) % p for c in range(p)] for a, b in els], device=device)   # [x, c] = x^-1(c)
        self.img = torch.tensor([[(a * j + b) % p for j in range(p)] for a, b in els], device=device)                  # [z, j] = z(j)
        self.bdiv = torch.tensor([b * pow(a, -1, p) % p for a, b in els], device=device, dtype=dtype)
        self.bb = torch.tensor([b for a, b in els], device=device, dtype=dtype)
        ar = torch.arange(self.n, device=device)
        self.wrong = torch.ones(self.n, self.n, self.n, dtype=torch.bool, device=device); self.wrong[ar[:, None], ar[None, :], self.T] = False
    def param(self, *shape, scale=0.1): return Parameter(scale * torch.randn(*shape, dtype=self.dtype, device=self.dev))


def best_permutation(p, restarts=5, iters=3000, seed=0):
    """Reorder the evenly spaced profile to maximise  min over affine maps sigma != id of sum_r |F(sigma r) - F(r)|  (the margin of the
    all-coordinates plain circuit with indicator readouts; the cost depends only on the values, not their order)."""
    rng = np.random.default_rng(seed); ev = np.arange(p) - (p - 1) / 2
    S = [np.array([(c * r + d) % p for r in range(p)]) for c in range(p) for d in range(p) if not (c == 1 and d == 0)]
    M = lambda F: min(np.abs(F[s_] - F).sum() for s_ in S)
    best = (M(ev), ev)
    for _ in range(restarts):
        F = rng.permutation(ev); m = M(F)
        for _ in range(iters):
            i, j = rng.choice(p, 2, replace=False); G = F.copy(); G[i], G[j] = G[j], G[i]; mg = M(G)
            if mg >= m: F, m = G, mg
        if m > best[0]: best = (m, F)
    return best[1]


class Unit(torch.nn.Module):
    def __init__(self, S, m):
        super().__init__(); self.S = S; self.a, self.b, self.W = S.param(m, S.p - 1), S.param(m, S.p - 1), S.param(m, S.n)
    def forward(self): return self.a[:, self.S.aidx], self.b[:, self.S.aidx], self.W


class Plain(torch.nn.Module):
    def __init__(self, S, J, copies=2, F0=None):
        super().__init__(); self.S = S; p = S.p
        cells = [(j0, c, s) for j0 in J for c in range(p) for s in range(copies)]
        self.j0 = torch.tensor([j for j, c, s in cells], device=S.dev); self.c = torch.tensor([c for j, c, s in cells], device=S.dev); m = len(cells)
        self.Ap, self.Bp, self.W = S.param(m, p), S.param(m, p), S.param(m, S.n)
        F = torch.arange(p, dtype=S.dtype, device=S.dev) - (p - 1) / 2 if F0 is None else torch.tensor(F0, dtype=S.dtype, device=S.dev)
        with torch.no_grad():
            for k, (j0, c, s) in enumerate(cells):
                sg = 1.0 if s % 2 == 0 else -1.0; self.Ap[k] = sg * F + 0.01 * torch.randn_like(F); self.Bp[k] = -sg * F + 0.01 * torch.randn_like(F)
                self.W[k] = -(S.img[:, j0] == c).to(S.dtype)
    def forward(self):
        L = torch.gather(self.Ap, 1, self.S.inv_pt[:, self.c].T); R = torch.gather(self.Bp, 1, self.S.img[:, self.j0].T)
        return L, R, self.W


class Twisted(torch.nn.Module):
    def __init__(self, S, K=8):
        super().__init__(); self.S = S; p = S.p; units = S.A.units
        cells = [(ai, f, i) for ai in range(p - 1) for f in range(1, (p - 1) // 2 + 1) for i in range(K)]; m = len(cells)
        f = torch.tensor([f for ai, f, i in cells], dtype=S.dtype, device=S.dev)
        self.cL, self.sL = torch.cos(2 * np.pi * f[:, None] * S.bdiv[None, :] / p), torch.sin(2 * np.pi * f[:, None] * S.bdiv[None, :] / p)
        self.cR, self.sR = torch.cos(2 * np.pi * f[:, None] * S.bb[None, :] / p), torch.sin(2 * np.pi * f[:, None] * S.bb[None, :] / p)
        self.u, self.v, self.o, self.ur, self.vr, self.orr = [S.param(m, p - 1, scale=0.01) for _ in range(6)]; self.W = S.param(m, S.n, scale=0.01)
        with torch.no_grad():       # hand-built single-fiber clock: on its fiber cos(f b_x / a + phi), -1 elsewhere; R = cos(f b_y + phi); w = cos(f b_z / a + 2 phi)
            for k, (ai, ff, i) in enumerate(cells):
                ph = 2 * np.pi * i / K; alpha = units[ai]; inv = pow(alpha, -1, p)
                self.u[k, ai] += np.cos(ph); self.v[k, ai] += -np.sin(ph); self.o[k] += -1.0; self.o[k, ai] += 1.0
                self.ur[k] += np.cos(ph); self.vr[k] += -np.sin(ph)
                self.W[k] += torch.cos(2 * np.pi * ff * inv * S.bb / p + 2 * ph)
    def forward(self):
        a = self.S.aidx
        L = self.u[:, a] * self.cL + self.v[:, a] * self.sL + self.o[:, a]; R = self.ur[:, a] * self.cR + self.vr[:, a] * self.sR + self.orr[:, a]
        return L, R, self.W


class Free(torch.nn.Module):
    def __init__(self, S, m, init=None):
        super().__init__(); self.S = S
        if init is None: self.L, self.R, self.W = S.param(m, S.n), S.param(m, S.n), S.param(m, S.n)
        else:
            L, R, W = [q.to(S.dtype).to(S.dev) for q in init]; nz = 0.01 * L.abs().max()
            self.L = Parameter(L + nz * torch.randn_like(L)); self.R = Parameter(R + nz * torch.randn_like(R)); self.W = Parameter(W.clone())
    def forward(self): return self.L, self.R, self.W


class Circuit(torch.nn.Module):
    def __init__(self, parts): super().__init__(); self.parts = torch.nn.ModuleList(parts)
    def forward(self):
        Ls, Rs, Ws = zip(*[q() for q in self.parts]); return torch.cat(Ls), torch.cat(Rs), torch.cat(Ws)


def logits(L, R, W): return torch.relu(L.T[:, None, :] + R.T[None, :, :]) @ W

def margins(S, L, R, W):
    f = logits(L, R, W); ar = torch.arange(S.n, device=S.dev)
    return (f[ar[:, None], ar[None, :], S.T][:, :, None] - f)[S.wrong]

def cost(L, R, W): return (torch.sqrt((L ** 2).sum(1) + (R ** 2).sum(1) + 1e-30) * torch.sqrt((W ** 2).sum(1) + 1e-30)).sum()

def arch_cost(L, R, W): return 2 * torch.linalg.matrix_norm(L, "nuc") + 2 * torch.linalg.matrix_norm(R, "nuc") + (W ** 2).sum()

def optimise(S, circ, steps, lr, rel0, log=False):
    """Adam on  -softmin(margins) / C  with the temperature tied to the spread of the margins, annealed from rel0 to 0.003 of it.
    After every step the readouts are rescaled so that C keeps its initial value (the objective is scale invariant and Adam would
    otherwise drift the scale).  Last 20% of the steps at lr / 10.  Returns the best (hard margin / C) iterate and the final one."""
    opt = torch.optim.Adam(circ.parameters(), lr=lr); best = (-np.inf, None); t0 = time.time()
    with torch.no_grad(): C0 = float(cost(*circ()))
    for it in range(steps):
        if it == int(0.8 * steps):
            for g in opt.param_groups: g["lr"] = lr / 10
        L, R, W = circ(); mg = margins(S, L, R, W); C = cost(L, R, W)
        with torch.no_grad(): hard = float(mg.min()); tau = rel0 * ((0.003 / rel0) ** (it / max(steps - 1, 1))) * float(mg.std()) + 1e-12
        soft = -tau * torch.logsumexp(-mg / tau, 0)
        loss = -soft / C; opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            r = hard / float(C)
            if r > best[0]: best = (r, [q.detach().clone() for q in (L, R, W)])
            Cn = float(cost(*circ()))
            for q in circ.parts: q.W.mul_(C0 / Cn)
            if log and (it % 1000 == 0 or it == steps - 1): print(f"    it {it:5d} hard/C {r:+.4e} soft/C {float(soft / C):+.4e} tau {tau:.2e} ({time.time() - t0:.0f}s)", flush=True)
    with torch.no_grad(): final = [q.detach().clone() for q in circ()]
    return best, final

def census(A, p, L, R, W):
    Ln, Rn, Wn = [q.numpy() for q in (L, R, W)]
    plain = [Q for _, Q in A.plain_family()]; twist = [Q for _, Q in A.twisted_family()]
    Pbig = np.real(A.isotypic_projector(A.block[p])); Plin = np.real(sum(A.isotypic_projector(r) for r in range(len(A.dims)) if A.dims[r] == 1))
    e = np.sqrt((Ln ** 2).sum(1) + (Rn ** 2).sum(1)) * np.sqrt((Wn ** 2).sum(1)); live = e > 1e-3 * e.max(); sc = {}
    for name, fam in (("plain", plain), ("twist", twist), ("big", [Pbig]), ("lin", [Plin])):
        ok = np.ones(len(e), bool)
        for U in (Ln, Rn):
            Uc = U - U.mean(1, keepdims=True); den = np.maximum((Uc ** 2).sum(1), 1e-300)
            ok &= np.max([((Uc @ Q.T) ** 2).sum(1) / den for Q in fam], axis=0) >= 0.9
        sc[name] = ok
    big = live & sc["big"]; out = dict(frac_big=float(e[big].sum() / e[live].sum()), frac_lin=float(e[live & sc["lin"]].sum() / e[live].sum()))
    for name in ("plain", "twist"): out[f"frac_{name}_of_big"] = float(e[big & sc[name]].sum() / max(e[big].sum(), 1e-300))
    out["live"] = int(live.sum()); return out

def metrics(Se, L, R, W):
    mg = margins(Se, L, R, W); C = cost(L, R, W); ac = arch_cost(L, R, W)
    return dict(gamma=float(mg.min()), C=float(C), ratio=float(mg.min() / C), arch_cost=float(ac), arch_ratio=float(mg.min() / ac ** 1.5))

def run(S, Se, name, steps, seed, init=None, log=True, plain_init="monotone"):
    torch.manual_seed(seed); p = S.p; parts = [Unit(S, 4 * (p - 1))]; handbuilt = True
    if name.startswith("plain"): parts.append(Plain(S, list(range(p if name == "plain" else int(name[5:]))), F0=best_permutation(p) if plain_init == "perm" else None))
    elif name == "twisted": parts.append(Twisted(S))
    elif name == "free": parts = [Free(S, 2 * p * p + 4 * (p - 1), init=init)]; handbuilt = init is not None
    (r, sol), final = optimise(S, Circuit(parts), steps, lr=0.002 if handbuilt else 0.005, rel0=0.03 if handbuilt else 1.0, log=log)
    L, R, W = [q.cpu().double() for q in sol]          # two steps: .to("cpu", float64) in one call returns zeros on mps
    rec = dict(p=p, family=name, seed=seed, neurons=L.shape[0], **metrics(Se, L, R, W), **census(Se.A, p, L, R, W))
    rec["ratio_final"] = metrics(Se, *[q.cpu().double() for q in final])["ratio"]
    return rec, (L, R, W)

def trained_init(run):
    sd = torch.load(Path(run) / "model.pt", map_location="cpu"); Wx, Wy, W, WU = [sd[k].double() for k in ("W_x", "W_y", "W", "W_U")]; E = Wx.shape[1]
    return (Wx @ W[:E]).T.contiguous(), (Wy @ W[E:]).T.contiguous(), WU

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--p", type=int, nargs="+", default=[11]); ap.add_argument("--families", nargs="+", default=["plain1", "plain2", "plain", "twisted", "free"])
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--free_from", nargs="*", default=[], help="also run free initialised from these families' optima (e.g. plain twisted)")
    ap.add_argument("--trained", nargs="*", default=[], help="trained model dirs: run free initialised from each (matched to p by |G|)")
    ap.add_argument("--plain_init", choices=["monotone", "perm"], default="monotone", help="profile F at init: evenly spaced monotone, or its margin-maximising reordering")
    ap.add_argument("--out", required=True); a = ap.parse_args()
    rows = []; fmt = lambda rec: "  " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items())
    def record(rec): rows.append(rec); print(fmt(rec), flush=True); pd.DataFrame(rows).to_csv(a.out + ".csv", index=False)
    for p in a.p:
        S = Setup(p, a.device, torch.float32); Se = Setup(p, "cpu", torch.float64); sols = {}
        for fam in a.families:
            for seed in a.seeds:
                print(f"p={p} {fam} seed {seed}", flush=True); rec, sol = run(S, Se, fam, a.steps, seed, plain_init=a.plain_init); sols[(fam, seed)] = sol; record(rec); torch.save(sol, f"{a.out}_p{p}_{fam}_s{seed}.pt")
        for fam in a.free_from:
            for seed in a.seeds:
                if (fam, seed) not in sols: continue
                print(f"p={p} free from {fam} seed {seed}", flush=True); rec, sol = run(S, Se, "free", a.steps, seed, init=sols[(fam, seed)]); rec["family"] = f"free_from_{fam}"; record(rec); torch.save(sol, f"{a.out}_p{p}_free_from_{fam}_s{seed}.pt")
        for t in a.trained:
            init = trained_init(t)
            if init[0].shape[1] != S.n: continue
            rec0 = dict(p=p, family=f"trained_{Path(t).name}", seed=0, neurons=init[0].shape[0], **metrics(Se, *init), **census(Se.A, p, *init)); record(rec0)
            print(f"p={p} free from trained {Path(t).name}", flush=True); rec, _ = run(S, Se, "free", a.steps, 1, init=init); rec["family"] = f"free_from_trained_{Path(t).name}"; record(rec)

if __name__ == "__main__":
    main()
