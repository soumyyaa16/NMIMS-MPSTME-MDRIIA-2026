"""Test the learned filter in the REAL MuJoCo needle trials (same physics, same paired tremor draws as run_trials.py).
    python src\\rl_eval.py              (50 seeds x 6 methods, about 15-20 min)       python src\\rl_eval.py --n 10   (quick look)
Methods (all use the same robot, plan, tremor and tissue; only the filter in front of the controller differs):
  BASE         no filter, plain pseudo-inverse IK                       (the 'no assistance' of the film)
  NOFILTER     no filter, damped IK + null space (same controller as the filtered ones)
  FIXED_3.5    fixed 3.5 Hz low-pass on the raw hand signal             (a normal filter: does NOT know the plan)
  RULE         hand-written speed rule on the filter bank               (no learning, does NOT know the plan)
  RL           PPO-learned filter schedule                              (does NOT know the plan)
  PROP         our plan-relative filter                                 (KNOWS the plan: upper reference, not a fair rival)
Seeds 1000.. are the SAME as the N=50 batch. The RL policy was trained on separate simulated hands (seeds 50000+).
Writes analytics/rl/eval_trials.csv, analytics/rl/eval_summary.csv and docs/figs/rl_filter_eval.png."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import rl_filter as rf
import surgical_tremor_controller as sc

ROOT = sc.ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--policy", default=str(ROOT / "analytics" / "rl" / "policy.npz"))
    ap.add_argument("--out", default=str(ROOT / "analytics" / "rl"))
    a = ap.parse_args()
    policy = rf.NumpyPolicy(a.policy)
    model = sc.load_model()
    orig = sc.StreamingLowpass
    methods = [("BASE", "BASE", None), ("NOFILTER", "DLSONLY", None), ("FIXED_3.5", "LPFRAW", orig), ("RULE", "LPFRAW", lambda: rf.AdaptiveFilter("rule")),
               ("RL", "LPFRAW", lambda: rf.AdaptiveFilter("rl", policy)), ("PROP", "PROP", None)]
    rows, t0 = [], time.time()
    try:
        for i in range(a.n):
            seed = a.seed0 + i
            rng = np.random.default_rng(seed)
            amp = float(rng.uniform(50, 250))
            k_scale = float(rng.uniform(0.7, 1.3))
            for name, cond, fac in methods:
                if fac is not None:
                    sc.StreamingLowpass = fac
                else:
                    sc.StreamingLowpass = orig
                r = sc.run_trial(model, cond, seed, amp, k_scale)["summary"]
                r["method"] = name
                rows.append(r)
            sc.StreamingLowpass = orig
            print(f"seed {seed} done ({i + 1}/{a.n})  {time.time() - t0:.0f}s", flush=True)
    finally:
        sc.StreamingLowpass = orig
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "eval_trials.csv", index=False)
    cols = ["rmse_track_mm", "hold_rms_mm", "hold_max_mm", "place_err_end_insert_mm"]
    order = [m[0] for m in methods]
    g = df.groupby("method")[cols].mean().loc[order]
    g["pass_hold_<0.5mm_%"] = 100 * df.groupby("method")["pass_0p5mm_hold"].mean().loc[order]
    g["aborted_%"] = 100 * df.groupby("method")["aborted"].mean().loc[order]
    # paired differences vs the fixed 3.5 Hz filter (same seeds), with bootstrap 95% CI
    piv = {c: df.pivot(index="seed", columns="method", values=c) for c in cols}
    rng = np.random.default_rng(0)
    lines = []
    for ref in ("FIXED_3.5", "RULE"):
        for c in ("rmse_track_mm", "hold_rms_mm"):
            d = (piv[c]["RL"] - piv[c][ref]).dropna().values
            bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
            lines.append(dict(comparison=f"RL minus {ref}", metric=c, mean_diff_mm=float(d.mean()),
                              ci95_lo=float(np.percentile(bs, 2.5)), ci95_hi=float(np.percentile(bs, 97.5)),
                              rl_better_in_pct_of_runs=float(100 * (d < 0).mean())))
    cmp_ = pd.DataFrame(lines)
    g.round(4).to_csv(out / "eval_summary.csv")
    cmp_.round(4).to_csv(out / "eval_paired.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nMEAN OVER", a.n, "PAIRED RUNS (lower is better except pass %)")
    print(g.round(3).to_string())
    print("\nPAIRED DIFFERENCES (negative = RL better; mm)")
    print(cmp_.round(4).to_string(index=False))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for k, (c, t) in enumerate((("rmse_track_mm", "Tracking error while inserting + holding (mm RMS)"),
                                    ("hold_rms_mm", "Error while holding at target (mm RMS)"))):
            data = [piv[c][m].dropna().values for m in order]
            ax[k].boxplot(data, tick_labels=order, showfliers=False)
            ax[k].set_title(t, fontsize=9)
            ax[k].tick_params(axis="x", rotation=25)
            ax[k].grid(alpha=0.3)
        fig.suptitle(f"Learned (RL) filter vs baselines, {a.n} paired MuJoCo runs (PROP knows the plan; the others do not)", fontsize=10)
        fig.tight_layout()
        fd = ROOT / "docs" / "figs"
        fd.mkdir(parents=True, exist_ok=True)
        fig.savefig(fd / "rl_filter_eval.png", dpi=160)
        print("figure:", fd / "rl_filter_eval.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()