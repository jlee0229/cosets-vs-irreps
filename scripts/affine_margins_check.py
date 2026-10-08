"""Brute-force checks for the analytic margin/cost formulas of the two Aff(F_p) circuits."""
import numpy as np
from math import pi, sin, cos, sqrt
np.set_printoptions(precision=5, suppress=True)

# ---------------------------------------------------------------- group
def group(p):
    els = [(a, b) for a in range(1, p) for b in range(p)]
    idx = {g: i for i, g in enumerate(els)}
    n = len(els)
    A = np.array([e[0] for e in els]); B = np.array([e[1] for e in els])
    mul = np.zeros((n, n), dtype=np.int64)
    for i, (a1, b1) in enumerate(els):
        for j, (a2, b2) in enumerate(els):
            mul[i, j] = idx[((a1 * a2) % p, (a1 * b2 + b1) % p)]   # (xy)(j) = x(y(j))
    return els, idx, mul, A, B

def gap_table(logits, mul, A=None, which="all"):
    """gap[x,y] = f_{xy} - max_{z wrong (in class)} f_z."""
    n = logits.shape[0]
    X = np.arange(n)[:, None]; Y = np.arange(n)[None, :]
    correct = logits[X, Y, mul]
    L = logits.copy()
    L[X, Y, mul] = -np.inf
    if which != "all":
        a_xy = A[mul]
        same = (A[None, None, :] == a_xy[:, :, None])
        L = np.where(same if which == "same_a" else ~same, L, -np.inf)
    return correct - L.max(axis=2)

def cost(Ls, Rs, Ws):
    return float(np.sum(np.sqrt((Ls**2).sum(1) + (Rs**2).sum(1)) * np.sqrt((Ws**2).sum(1))))

def run(Ls, Rs, Ws):
    pre = Ls[:, :, None] + Rs[:, None, :]
    return np.einsum('kxy,kz->xyz', np.maximum(pre, 0.0), Ws)

# ---------------------------------------------------------------- unit part (shared)
def unit_part(p, A, gf=1.0):
    units = list(range(1, p)); fidx = {u: i for i, u in enumerate(units)}
    f = lambda u: gf * (fidx[u] - (p - 2) / 2)
    Ls, Rs, Ws = [], [], []
    fa = np.array([f(int(ai)) for ai in A])
    for v in range(1, p):
        fv = np.array([f((v * pow(int(ai), -1, p)) % p) for ai in A])
        w = -(A == v).astype(float)
        for s in (+1, -1):
            Ls.append(s * fa); Rs.append(-s * fv); Ws.append(w)
    return Ls, Rs, Ws

# ---------------------------------------------------------------- circuit 1
def circuit1(p, J, gF=1.0, gf=1.0, unit=True):
    els, idx, mul, A, B = group(p)
    Ainv = np.array([pow(int(ai), -1, p) for ai in A])
    F = lambda j: gF * (j - (p - 1) / 2)
    Ls, Rs, Ws = [], [], []
    for j0 in J:
        yj0 = (A * j0 + B) % p
        for c in range(p):
            xinv_c = ((c - B) * Ainv) % p
            w = -(yj0 == c).astype(float)       # z(j0) == c
            for s in (+1, -1):
                Ls.append(s * F(xinv_c)); Rs.append(-s * F(yj0)); Ws.append(w)
    if unit:
        l, r, w = unit_part(p, A, gf); Ls += l; Rs += r; Ws += w
    Ls, Rs, Ws = map(np.array, (Ls, Rs, Ws))
    return run(Ls, Rs, Ws), cost(Ls, Rs, Ws), mul, A, B

def C1_formula(p, J, gF=1.0, gf=1.0, unit=True):
    a = sqrt(p * (p * p - 1) / 6); b = sqrt(p * (p - 1) * (p - 2) / 6)
    return 2 * p * (p - 1) * (J * a * gF + (b * gf if unit else 0))

def gamma1_formula(p, J, gF=1.0, gf=1.0, unit=True):
    if J == p:
        return 2 * (p - 1) * gF            # p >= 7
    if unit:
        return min(J * gF, (J - 1) * gF + gf)
    return (J - 1) * gF

# ---------------------------------------------------------------- circuit 2
def circuit2(p, K, freqs, eta=1.0, gf=1.0, unit=True):
    els, idx, mul, A, B = group(p)
    Ls, Rs, Ws = [], [], []
    for alpha in range(1, p):
        for k in freqs:
            for i in range(K):
                phi = 2 * pi * i / K
                Ls.append(np.where(A == alpha, np.cos(2 * pi * k * B / p + phi), -eta))
                Rs.append(np.cos(2 * pi * k * alpha * B / p + phi))
                Ws.append(np.cos(2 * pi * k * B / p + 2 * phi))
    Ls, Rs, Ws = map(np.array, (Ls, Rs, Ws))
    clock = run(Ls, Rs, Ws); Cc = cost(Ls, Rs, Ws)
    if unit:
        l, r, w = unit_part(p, A, gf); l, r, w = map(np.array, (l, r, w))
        un = run(l, r, w); Cu = cost(l, r, w)
    else:
        un = 0 * clock; Cu = 0.0
    return clock, un, Cc, Cu, mul, A, B

def C2_formula(p, K, nf, eta=1.0):
    per = sqrt(p * (p - 2) * eta**2 + p / 2 + p * (p - 1) / 2) * sqrt(p * (p - 1) / 2)
    return (p - 1) * nf * K * per

def Cunit_formula(p, gf=1.0):
    return 2 * p * (p - 1) * sqrt(p * (p - 1) * (p - 2) / 6) * gf

cK = lambda K: 2 * K / (3 * pi)
def gclock_1freq(p, K): return cK(K) * (1 - cos(2 * pi / p)) * sin(pi / (2 * p))
def T1(p): return 0.25 * (1 / sin(pi / (2 * p)) + 1 / sin(3 * pi / (2 * p)))
def gclock_all(p, K): return cK(K) * min(p / 2, T1(p))

# alias-including finite-K clock logits on one fiber (semi-analytic)
def a_relu(n):
    if n == 0: return 1 / pi
    if n == 1: return 0.5
    if n % 2: return 0.0
    m = n // 2
    return (2 / pi) * (-1) ** (m + 1) / (4 * m * m - 1)

def S_K_series(u, v, th, K, nmax=4000):
    c = np.abs(np.cos((u - v) / 2)); psi = (u + v) / 2
    s = (np.cos((u - v) / 2) < 0).astype(float)         # sign flip -> phase pi
    out = 0.0
    for n in range(nmax + 1):
        a = a_relu(n)
        if a == 0: continue
        An = n * (psi + pi * s)
        if n % K == 2 % K: out = out + (K / 2) * a * np.cos(An - th)
        if (-n) % K == 2 % K: out = out + (K / 2) * a * np.cos(An + th)
    return 2 * c * out

def S_K_direct(u, v, th, K):
    out = 0.0
    for i in range(K):
        phi = 2 * pi * i / K
        out = out + np.maximum(np.cos(u + phi) + np.cos(v + phi), 0) * np.cos(th + 2 * phi)
    return out

def clock_fiber_margin(p, K, freqs, use_series=False):
    bx = np.arange(p)[:, None, None]; by = np.arange(p)[None, :, None]; bz = np.arange(p)[None, None, :]
    Lam = np.zeros((p, p, p))
    for k in freqs:
        u = 2 * pi * k * bx / p; v = 2 * pi * k * by / p; th = 2 * pi * k * bz / p
        Lam = Lam + (S_K_series(u, v, th, K) if use_series else S_K_direct(u, v, th, K))
    corr = (bx + by) % p
    correct = np.take_along_axis(Lam, corr, axis=2)[:, :, 0]
    L = Lam.copy(); np.put_along_axis(L, corr, -np.inf, axis=2)
    gap = correct - L.max(axis=2)
    return gap.min(), gap

def T(p, e):
    j = np.arange(1, (p - 1) // 2 + 1)
    return float(np.sum(np.cos(pi * j / p) * (1 - np.cos(2 * pi * j * e / p))))

# ================================================================= checks
if __name__ == "__main__":
    print("=== Fourier coefficients of ReLU(cos t) ===")
    t = np.linspace(0, 2 * pi, 200001)[:-1]
    r = np.maximum(np.cos(t), 0)
    for n in range(0, 9):
        num = (np.mean(r * np.cos(n * t)) * (1 if n == 0 else 2))
        print(f"  n={n}: numeric {num:+.6f}  formula {a_relu(n):+.6f}")

    print("\n=== S_K closed form, K->inf (K=2000) and finite K vs alias series ===")
    rng = np.random.default_rng(0)
    for K in (2000, 3, 4, 6, 8, 12):
        err = 0.0; err_main = 0.0
        for _ in range(200):
            u, v, th = rng.uniform(0, 2 * pi, 3)
            d = S_K_direct(u, v, th, K); s = S_K_series(u, v, th, K)
            main = cK(K) * abs(cos((u - v) / 2)) * cos(u + v - th)
            err = max(err, abs(d - s)); err_main = max(err_main, abs(d - main))
        print(f"  K={K:5d}: max|direct - alias series| = {err:.2e}; max|direct - main term| = {err_main:.3e} (c_K={cK(K):.4f})")

    print("\n=== a_2(eta) for ReLU(cos t - eta): formula (2/3pi)(1-eta^2)^{3/2} ===")
    for eta in (0, 0.5, 0.8, 0.9, 0.99):
        num = 2 * np.mean(np.maximum(np.cos(t) - eta, 0) * np.cos(2 * t))
        print(f"  eta={eta}: numeric {num:.6f}  formula {2 / (3 * pi) * (1 - eta**2)**1.5:.6f}")

    print("\n=== T(e) = sum_j cos(pi j/p)(1-cos(2 pi j e/p)); claim min at e=1, T(1)=(csc(pi/2p)+csc(3pi/2p))/4 ===")
    for p in (7, 11, 13, 31, 101):
        Ts = [T(p, e) for e in range(1, (p - 1) // 2 + 1)]
        print(f"  p={p}: T(1)={Ts[0]:.5f} formula {T1(p):.5f}  min_e T = {min(Ts):.5f} at e={1 + int(np.argmin(Ts))}; p/2={p / 2}; 2p/(3pi)={2 * p / (3 * pi):.4f}")

    print("\n=== CIRCUIT 1 (plain coordinate) ===")
    for p in (7, 11, 13):
        Js = sorted(set([1, 2, p // 2, p - 1, p]))
        for J in Js:
            logits, C, mul, A, B = circuit1(p, range(J))
            g = gap_table(logits, mul); gA = gap_table(logits, mul, A, "same_a"); gB = gap_table(logits, mul, A, "diff_a")
            print(f"  p={p:2d} |J|={J:2d}: gamma={g.min():7.3f} (same-a {gA.min():6.2f}, diff-a {gB.min():6.2f}) formula {gamma1_formula(p, J):6.2f} | "
                  f"C={C:10.2f} formula {C1_formula(p, J):10.2f} | gamma/C={g.min() / C * 1e3:.4f}e-3")
        # no unit part, and unequal scales
        for J in (2, p):
            logits, C, mul, A, B = circuit1(p, range(J), unit=False)
            g = gap_table(logits, mul).min()
            print(f"  p={p:2d} |J|={J:2d} NO unit part: gamma={g:.3f} (formula {gamma1_formula(p, J, unit=False):.3f}), C={C:.2f}, gamma/C={g / C * 1e3:.4f}e-3")
        for J, gf in ((1, 0.5), (1, 2.0), (3, 0.5), (3, 2.0)):
            logits, C, mul, A, B = circuit1(p, range(J), gf=gf)
            g = gap_table(logits, mul).min()
            print(f"  p={p:2d} |J|={J} g_f={gf}: gamma={g:.3f} formula {gamma1_formula(p, J, gf=gf):.3f}; C={C:.1f} formula {C1_formula(p, J, gf=gf):.1f}")
        # explain the |J|=p minimiser
        logits, C, mul, A, B = circuit1(p, range(p))
        g = gap_table(logits, mul)
        x, y = np.unravel_index(np.argmin(g), g.shape)
        xy = mul[x, y]
        row = logits[x, y].copy(); row[xy] = -np.inf; z = int(np.argmax(row))
        print(f"  p={p} |J|=p argmin: x=(a={A[x]},b={B[x]}) y=(a={A[y]},b={B[y]}) xy=(a={A[xy]},b={B[xy]}) worst z=(a={A[z]},b={B[z]}); "
              f"(b_z-b_xy)/a_x mod p = {((B[z] - B[xy]) * pow(int(A[x]), -1, p)) % p}")

    print("\n=== CIRCUIT 2 (twisted clock) ===")
    for p in (7, 11, 13):
        nf_all = (p - 1) // 2
        for K, freqs in ((4, [1]), (6, [1]), (8, [1]), (12, [1]), (8, list(range(1, nf_all + 1))), (12, list(range(1, nf_all + 1)))):
            clock, un, Cc, Cu, mul, A, B = circuit2(p, K, freqs)
            tot = clock + un
            g = gap_table(tot, mul).min()
            gclk = gap_table(tot, mul, A, "same_a").min()
            gunit = gap_table(un, mul, A, "diff_a").min()
            acc = float(np.mean(tot.argmax(axis=2) == mul))
            fib, _ = clock_fiber_margin(p, K, freqs)
            nf = len(freqs)
            af = gclock_1freq(p, K) if nf == 1 else gclock_all(p, K)
            Cbal = Cc / gclk + Cu / gunit if gclk > 0 else np.inf
            print(f"  p={p:2d} K={K:2d} nf={nf}: acc={acc:.3f} gamma={g:7.4f} clock={gclk:7.4f} (fiber calc {fib:7.4f}; alias-free formula {af:7.4f}) unit={gunit:.3f} | "
                  f"C_clock={Cc:9.1f} (formula {C2_formula(p, K, nf):9.1f}) C_unit={Cu:8.1f} (formula {Cunit_formula(p):8.1f}) | "
                  f"gamma/C={g / (Cc + Cu) * 1e3:.4f}e-3 | rebalanced 1/(Cc/gc+Cu/gu)={1 / Cbal * 1e3:.4f}e-3")
        # worst-case input for one frequency: check it is |cos((u-v)/2)| = sin(pi/2p)
        _, gap = clock_fiber_margin(p, 12, [1])
        bx, by = np.unravel_index(np.argmin(gap), gap.shape)
        m = (bx - by) % p
        print(f"  p={p} K=12 1-freq worst fiber input: b_x - b_y = {m} -> |cos(pi m/p)| = {abs(cos(pi * m / p)):.4f}, sin(pi/2p) = {sin(pi / (2 * p)):.4f}")

    print("\n=== silencing offset eta < 1, p=7, all freqs, K=8 ===")
    p = 7; freqs = [1, 2, 3]
    for eta in (1.0, 0.95, 0.9, 0.8, 0.5, 0.0):
        clock, un, Cc, Cu, mul, A, B = circuit2(p, 8, freqs, eta=eta)
        gclk = gap_table(clock + un, mul, A, "same_a").min()
        acc = float(np.mean((clock + un).argmax(axis=2) == mul))
        print(f"  eta={eta:4.2f}: clock margin {gclk:8.4f}  acc {acc:.3f}  C_clock {Cc:8.1f} (formula {C2_formula(p, 8, 3, eta):8.1f})")
    p = 11; freqs = list(range(1, 6))
    for eta in (1.0, 0.95, 0.9, 0.8):
        clock, un, Cc, Cu, mul, A, B = circuit2(p, 8, freqs, eta=eta)
        gclk = gap_table(clock + un, mul, A, "same_a").min()
        print(f"  p=11 eta={eta:4.2f}: clock margin {gclk:8.4f}  C_clock {Cc:8.1f}; a_2(eta)={2 / (3 * pi) * (1 - eta**2)**1.5:.4f}, spike bound K a_2 p(p-2)/4 = {8 * 2 / (3 * pi) * (1 - eta**2)**1.5 * p * (p - 2) / 4:.3f}")

    print("\n=== large-p table of analytic gamma/C ===")
    for p in (7, 11, 13, 31, 101, 547, 1009):
        plain = gamma1_formula(p, p) / C1_formula(p, p)
        plain1 = gamma1_formula(p, 1) / C1_formula(p, 1)
        K = 8
        tw_built = 1.0 / (C2_formula(p, K, (p - 1) // 2) + Cunit_formula(p))
        tw_bal = 1.0 / (C2_formula(p, K, (p - 1) // 2) / gclock_all(p, K) + Cunit_formula(p))
        tw_clock_only = gclock_all(p, K) / C2_formula(p, K, (p - 1) // 2)
        tw_1f = gclock_1freq(p, K) / (C2_formula(p, K, 1) + Cunit_formula(p))
        print(f"  p={p:5d}: plain|J|=1 {plain1:.3e}  plain|J|=p {plain:.3e} (sqrt6 p^-3.5 = {sqrt(6) * p**-3.5:.3e})  "
              f"twisted-all as built {tw_built:.3e}  rebalanced {tw_bal:.3e}  clock-only {tw_clock_only:.3e} (0.104 p^-3 = {16 / (9 * sqrt(3) * pi**2) * p**-3.0:.3e})  "
              f"twisted-1freq {tw_1f:.3e}")
