"""Summarise the width grid and activation runs: did each run grok, when, and (for grokked prime-modulus models) how many
coordinate blocks the plain circuit uses.  usage: scripts/affine_grid_summary.py --runs DIR [DIR ...] --out PREFIX"""
import json, argparse, numpy as np, pandas as pd
from pathlib import Path
from affine_census import census

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    rows = []
    for run in a.runs:
        run = Path(run); cfg = json.load(open(run / "cfg.json")); curve = json.load(open(run / "curve.json"))
        first = next((r["epoch"] for r in curve if r["test_acc"] >= 0.99), None); best = max(r["test_acc"] for r in curve)
        rec = dict(run=run.name, group=cfg["group"], width=cfg["hidden"], seed=cfg["seed"], frac=cfg["frac"], act=cfg.get("act", "relu"), q=cfg.get("q"),
                   epochs=cfg["epochs"], test_acc=cfg["final"]["test_acc"], best_test_acc=best, grok_epoch=first, grokked=cfg["final"]["test_acc"] >= 0.99)
        if rec["grokked"] and cfg.get("act", "relu") == "relu":
            df, acc, _ = census(run); L = df[df.live]; big = L[(L.big_x > 0.9) & (L.big_y > 0.9)]; lin = L[(L.lin_x > 0.9) & (L.lin_y > 0.9)]
            pl = big[(big.alpha_x >= 0.9) & (big.alpha_y >= 0.9)]; cells = pl.groupby(["out_j", "out_v"]).size()
            rec.update(live=len(L), big=len(big), lin=len(lin), plain=len(pl), cells=len(cells), cells_double=int((cells == 2).sum()), coords=len(pl.out_j.unique()),
                       cell_match=float(pl.cell_matches.mean()) if len(pl) else np.nan, acc_cosetmean=acc["acc_cosetmean"], acc_plain_removed=acc["acc_plain_removed"])
        rows.append(rec); print(" ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.out + ".csv", index=False)
    print("\n" + df.to_string(index=False))

if __name__ == "__main__":
    main()
