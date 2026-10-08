import numpy as np
from math import pi, sqrt, cos, sin
from affine_margins_check import group, gap_table, cost, run, unit_part, C2_formula

def circuit2_split(p, K, freqs, beta):
    els, idx, mul, A, B = group(p)
    Ls, Rs, Ws = [], [], []
    for alpha in range(1, p):
        for k in freqs:
            for i in range(K):
                phi = 2*pi*i/K
                Ls.append(np.where(A == alpha, np.cos(2*pi*k*B/p + phi) + beta, -(1 - beta)))
                Rs.append(np.cos(2*pi*k*alpha*B/p + phi) - beta)
                Ws.append(np.cos(2*pi*k*B/p + 2*phi))
    Ls, Rs, Ws = map(np.array, (Ls, Rs, Ws))
    return run(Ls, Rs, Ws), cost(Ls, Rs, Ws), mul, A, B

def C2_split_formula(p, K, nf, beta):
    LR2 = p*(p-2)*(1-beta)**2 + p*(0.5 + beta**2) + p*(p-1)*(0.5 + beta**2)
    return (p-1)*nf*K*sqrt(LR2)*sqrt(p*(p-1)/2)

print("=== split silencing: L_on = cos+beta, L_off = -(1-beta), R = cos-beta ===")
for p in (7, 11):
    freqs = list(range(1, (p-1)//2 + 1)); K = 8
    l, r, w = unit_part(p, group(p)[3]); l, r, w = map(np.array, (l, r, w))
    un = run(l, r, w)
    bopt = (p-2)/(2*(p-1))
    for beta in (0.0, 0.5, bopt, 1.0):
        clock, Cc, mul, A, B = circuit2_split(p, K, freqs, beta)
        gclk = gap_table(clock + un, mul, A, "same_a").min()
        print(f"  p={p} beta={beta:.4f}: clock margin {gclk:.4f}  C_clock {Cc:.1f} (formula {C2_split_formula(p,K,len(freqs),beta):.1f}; beta=0 ref {C2_formula(p,K,len(freqs)):.1f})  ratio to beta=0: {Cc/C2_formula(p,K,len(freqs)):.4f}  sqrt((2p-3)/(3p-4)*p/(p-1))={sqrt((2*p-3)/(3*p-4)*p/(p-1)):.4f}")
    # one frequency too
    clock, Cc, mul, A, B = circuit2_split(p, 8, [1], 0.5)
    gclk = gap_table(clock + un, mul, A, "same_a").min()
    print(f"  p={p} 1 freq beta=0.5: clock margin {gclk:.4f} C_clock {Cc:.1f}")

print("\n=== K=4: logit on a fiber is (function of u+v) * cos(theta) ===")
p = 11
bx = np.arange(p)[:,None,None]; by = np.arange(p)[None,:,None]; bz = np.arange(p)[None,None,:]
u = 2*pi*bx/p; v = 2*pi*by/p; th = 2*pi*bz/p
Lam = sum(np.maximum(np.cos(u+2*pi*i/4)+np.cos(v+2*pi*i/4),0)*np.cos(th+2*2*pi*i/4) for i in range(4))
# check Lam(bz) / cos(th) is independent of bz (where cos th != 0)
ratio = Lam / np.cos(th)
print("  max over (bx,by) of spread of Lam/cos(theta) over bz:", float(np.max(ratio.max(2)-ratio.min(2))))
print("  argmax b_z values taken for K=4:", sorted(set(Lam.argmax(2).ravel().tolist())))
