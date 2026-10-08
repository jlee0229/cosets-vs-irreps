"""One-hidden-layer MLP on group composition, Chughtai's architecture and init.  Recipes from preregistration/affine_family.md.
usage: scripts/train.py --group aff15|s5|c113|d59 --recipe wu|chughtai --seed 1 [--hidden 128] [--epochs N] [--frac 0.4] --out DIR"""
import argparse, json, time, numpy as np, torch
from pathlib import Path

def group_table(name):
    if name == "s5":
        from cosetprobe import S5; return S5().table
    if name.startswith("aff"):
        from cosetprobe.affine import Affine; return Affine(int(name[3:])).T
    if name.startswith("c"):
        n = int(name[1:]); return (np.arange(n)[:, None] + np.arange(n)[None, :]) % n
    if name.startswith("d"):
        from cosetprobe.dihedral import dihedral_table; return dihedral_table(int(name[1:]))
    raise ValueError(name)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--group", required=True); ap.add_argument("--recipe", choices=["wu", "chughtai"], required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--hidden", type=int, default=128); ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--epochs", type=int); ap.add_argument("--frac", type=float, default=0.4); ap.add_argument("--out", required=True); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--act", choices=["relu", "absq"], default="relu"); ap.add_argument("--q", type=float, default=2.0, help="exponent for --act absq: |t|^q")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    act = torch.relu if a.act == "relu" else (lambda t: t.abs() ** a.q)
    T = torch.tensor(group_table(a.group)); N = len(T); E, m = a.embed, a.hidden
    # data split exactly as Chughtai: all pairs in row-major order, randperm with the seed, first frac -> train
    X, Y = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij"); X, Y = X.reshape(-1), Y.reshape(-1); Z = T[X, Y]
    torch.manual_seed(a.seed); perm = torch.randperm(N * N); ntr = int(a.frac * N * N); tr, te = perm[:ntr], perm[ntr:]
    # init exactly as Chughtai (torch.manual_seed(seed) then randn/sqrt(fan_in) in this order)
    torch.manual_seed(a.seed)
    W_x = torch.nn.Parameter(torch.randn(N, E) / np.sqrt(E)); W_y = torch.nn.Parameter(torch.randn(N, E) / np.sqrt(E))
    W = torch.nn.Parameter(torch.randn(2 * E, m) / np.sqrt(2 * E)); W_U = torch.nn.Parameter(torch.randn(m, N) / np.sqrt(m))
    params = [W_x, W_y, W, W_U]; dev = torch.device(a.device)
    for p in params: p.data = p.data.to(dev)
    X, Y, Z = X.to(dev), Y.to(dev), Z.to(dev)
    if a.recipe == "wu":
        opt = torch.optim.Adam(params, lr=1e-2, betas=(0.9, 0.98), weight_decay=2e-4); epochs = a.epochs or 25000
    else:
        opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0); epochs = a.epochs or 250000
    def logits(idx):
        # identical to relu(cat(W_x[x], W_y[y]) @ W) @ W_U, but the (N x m) side-embeddings are formed once per step
        Ex = W_x @ W[:E]; Ey = W_y @ W[E:]
        return act(Ex[X[idx]] + Ey[Y[idx]]) @ W_U
    curve = []; t0 = time.time()
    for ep in range(epochs + 1):
        if ep % a.log_every == 0 or ep == epochs:
            with torch.no_grad():
                lt = logits(tr); le = logits(te)
                rec = dict(epoch=ep, train_loss=torch.nn.functional.cross_entropy(lt, Z[tr]).item(), test_loss=torch.nn.functional.cross_entropy(le, Z[te]).item(),
                           train_acc=(lt.argmax(-1) == Z[tr]).float().mean().item(), test_acc=(le.argmax(-1) == Z[te]).float().mean().item(), t=time.time() - t0)
                curve.append(rec)
                if ep % (a.log_every * 50) == 0 or ep == epochs: print(f"  ep {ep:7d}  train {rec['train_loss']:.2e}/{rec['train_acc']:.3f}  test {rec['test_loss']:.2e}/{rec['test_acc']:.3f}  ({rec['t']:.0f}s)", flush=True)
        if ep == epochs: break
        opt.zero_grad(set_to_none=True); loss = torch.nn.functional.cross_entropy(logits(tr), Z[tr]); loss.backward(); opt.step()
    sd = {k: v.detach().cpu() for k, v in zip(["W_x", "W_y", "W", "W_U"], params)}
    torch.save(sd, out / "model.pt"); json.dump(curve, open(out / "curve.json", "w"))
    json.dump(dict(group=a.group, recipe=a.recipe, seed=a.seed, embed=E, hidden=m, epochs=epochs, frac=a.frac, n=N, act=a.act, q=a.q if a.act == "absq" else None, final=curve[-1]), open(out / "cfg.json", "w"), indent=1)
    print(f"done {a.group} {a.recipe} seed{a.seed}: test_acc={curve[-1]['test_acc']:.4f} in {curve[-1]['t']:.0f}s -> {out}")

if __name__ == "__main__": main()
