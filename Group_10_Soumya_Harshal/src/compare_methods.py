"""compare_methods.py - the full comparative study: every filter, every hand type, paired MuJoCo trials.
    python src\\compare_methods.py --tune                                  (once: tunes Kalman / WFLC / BMFLC on held-out simulated hands, ~5 min)
    python src\\compare_methods.py --n 30 --rl first=analytics\\rl\\policy.npz --rl dr1=analytics\\rl_dr_s1\\policy.npz  (about 1 h)
Methods (all LPFRAW-style: they see only the noisy hand signal and do NOT know the plan, except PROP):
  BASE no filter | FIXED_3.5 fixed low-pass | RULE speed rule | KALMAN | WFLC | BMFLC | RL_<name> learned schedule | PROP plan-relative filter (reference)
Hand types (same four tiers as run_tiers.py): A typical, B typical + measured drift, C enhanced tremor, D enhanced + measured drift.
Fairness: baselines are tuned on simulated hands (seeds 60000+) with the SAME objective the RL policy is trained on; MuJoCo test seeds are 1000+.
Writes analytics/compare_trials.csv, compare_summary.csv, compare_paired.csv, docs/figs/compare_methods.png"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import rl_filter as rf
import surgical_tremor_controller as sc
import tremor_baselines as tb

ROOT = sc.ROOT
TIERS = [("A typical hand", 24, 50, 40.0), ("B typical + measured drift", 24, 50, 280.0),
         ("C enhanced tremor", 50, 250, 40.0), ("D enhanced + measured drift", 50, 250, 280.0)]
GRIDS = {
    "bmflc": [dict(mu=m) for m in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)],
    "wflc": [dict(mu=m, mu0=m0) for m in (0.01, 0.03, 0.1, 0.3, 1.0) for m0 in (0.0, 3e-4, 3e-3, 3e-2)],
    "kalman": [dict(q_i=a, q_t=b, r=c) for a in (1e4, 1e5, 1e6, 1e7, 1e8) for b in (1e-3, 1e-2, 1e-1) for c in (1e-4, 1e-3, 1e-2, 1e-1)],
}


def tune(n_eps):
    """Pick the setting with the lowest mean clipped squared error (100 um units) on held-out simulated hands (domain-randomised)."""
    rg = np.random.default_rng(60_000)
    eps = [rf.make_episode(rg, dr=True) for _ in range(n_eps)]
    best = {}
    for kind, grid in GRIDS.items():
        res = []
        for pr in grid:
            J = []
            for intent, r, _, _ in eps:
                f = tb.KINDS[kind](float(r[0]), **pr)
                y = np.array([f(float(v)) for v in r])
                J.append(np.mean(np.clip(((y - intent) / rf.SCALE_X) ** 2, 0, 25.0)))
            res.append((float(np.mean(J)), pr))
            print(f"  {kind:6s} {pr}  J = {res[-1][0]:.3f}", flush=True)
        best[kind] = min(res, key=lambda t: t[0])[1]
        print(f"=> {kind}: best {best[kind]}", flush=True)
    J0 = np.mean([np.mean(np.clip(((r - intent) / rf.SCALE_X) ** 2, 0, 25.0)) for intent, r, _, _ in eps])
    print(f"(reference: no filter J = {J0:.3f})")
    (ROOT / "analytics").mkdir(exist_ok=True)
    (ROOT / "analytics" / "baselines_tuned.json").write_text(json.dumps(best, indent=1))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tune-eps", type=int, default=16)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--rl", action="append", default=[], help="name=path/to/policy.npz (repeatable)")
    ap.add_argument("--tiers", default="ABCD")
    ap.add_argument("--no-prop", action="store_true")
    ap.add_argument("--kp", type=float, default=None, help="position-loop gain KP (default 100); KI is set to 0.003*KP^2")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _kp50")
    a = ap.parse_args()
    if a.kp:
        sc.KP, sc.KI = a.kp, 0.003 * a.kp ** 2
        print(f"controller gains: KP={sc.KP}  KI={sc.KI:.2f}")
    tpath = ROOT / "analytics" / "baselines_tuned.json"
    if a.tune or not tpath.exists():
        best = tune(a.tune_eps)
        if a.tune:
            return
    else:
        best = json.loads(tpath.read_text())
    print("baseline settings:", best)
    orig = sc.StreamingLowpass
    methods = [("BASE", "BASE", None), ("FIXED_3.5", "LPFRAW", orig), ("RULE", "LPFRAW", lambda: rf.AdaptiveFilter("rule")),
               ("KALMAN", "LPFRAW", lambda: tb.BaselineFilter("kalman", **best["kalman"])),
               ("WFLC", "LPFRAW", lambda: tb.BaselineFilter("wflc", **best["wflc"])),
               ("BMFLC", "LPFRAW", lambda: tb.BaselineFilter("bmflc", **best["bmflc"]))]
    for spec in a.rl:
        name, path = spec.split("=", 1)
        pol = rf.NumpyPolicy(path)
        methods.append((f"RL_{name}", "LPFRAW", (lambda pol=pol: rf.AdaptiveFilter("rl", pol))))
    if not a.no_prop:
        methods.append(("PROP", "PROP", None))
    model = sc.load_model()
    rows, t0 = [], time.time()
    try:
        for name, lo, hi, drift in [t for t in TIERS if t[0][0] in a.tiers]:
            sc.TremorModel.__init__.__defaults__ = (drift,)
            for i in range(a.n):
                seed = a.seed0 + i
                rng = np.random.default_rng(seed)
                amp, k_scale = float(rng.uniform(lo, hi)), float(rng.uniform(0.7, 1.3))
                for mname, cond, fac in methods:
                    sc.StreamingLowpass = fac if fac is not None else orig
                    r = sc.run_trial(model, cond, seed, amp, k_scale)["summary"]
                    r["method"], r["tier"], r["amp_um"], r["drift_um"] = mname, name, amp, drift
                    rows.append(r)
                sc.StreamingLowpass = orig
            print(f"{name} done  {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(ROOT / "analytics" / f"compare_trials{a.tag}.csv", index=False)
    finally:
        sc.StreamingLowpass = orig
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "analytics" / f"compare_trials{a.tag}.csv", index=False)
    order = [m[0] for m in methods]
    rng = np.random.default_rng(0)
    summ, paired = [], []
    for tier, d in df.groupby("tier", sort=True):
        piv = {c: d.pivot(index="seed", columns="method", values=c)[order] for c in ("rmse_track_mm", "hold_rms_mm", "hold_max_mm")}
        for m in order:
            summ.append(dict(tier=tier, method=m, n=len(piv["hold_rms_mm"]), track_mm=piv["rmse_track_mm"][m].mean(), hold_rms_mm=piv["hold_rms_mm"][m].mean(),
                             hold_max_mm=piv["hold_max_mm"][m].mean(), pass_pct=100 * (piv["hold_max_mm"][m] < 0.5).mean()))
            if m != "BASE":
                dlt = (piv["hold_rms_mm"][m] - piv["hold_rms_mm"]["BASE"]).values
                bs = [rng.choice(dlt, len(dlt)).mean() for _ in range(3000)]
                paired.append(dict(tier=tier, method=m, hold_vs_BASE_pct=100 * dlt.mean() / piv["hold_rms_mm"]["BASE"].mean(),
                                   ci95_lo_mm=np.percentile(bs, 2.5), ci95_hi_mm=np.percentile(bs, 97.5), better_in_pct_runs=100 * (dlt < 0).mean()))
    S, Pd = pd.DataFrame(summ), pd.DataFrame(paired)
    S.round(4).to_csv(ROOT / "analytics" / f"compare_summary{a.tag}.csv", index=False)
    Pd.round(4).to_csv(ROOT / "analytics" / f"compare_paired{a.tag}.csv", index=False)
    pd.set_option("display.width", 220)
    for col, title in (("hold_rms_mm", "HOLD ERROR (mm RMS, lower is better)"), ("track_mm", "TRACKING ERROR while inserting (mm RMS)"), ("pass_pct", "% of runs with hold max < 0.5 mm")):
        print("\n" + title)
        print(S.pivot(index="method", columns="tier", values=col).loc[order].round(3).to_string())
    print("\nHOLD ERROR change vs no filter (%, negative = better)")
    print(Pd.pivot(index="method", columns="tier", values="hold_vs_BASE_pct").loc[[m for m in order if m != "BASE"]].round(1).to_string())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tiers = sorted(df.tier.unique())
        fig, ax = plt.subplots(2, len(tiers), figsize=(3.4 * len(tiers), 6.4), squeeze=False)
        for j, tier in enumerate(tiers):
            d = df[df.tier == tier]
            for i, (c, lab) in enumerate((("rmse_track_mm", "tracking error (mm)"), ("hold_rms_mm", "hold error (mm)"))):
                ax[i][j].boxplot([d[d.method == m][c].values for m in order], tick_labels=order, showfliers=False)
                ax[i][j].tick_params(axis="x", rotation=70, labelsize=7)
                ax[i][j].set_title(tier if i == 0 else "", fontsize=8)
                ax[i][j].set_ylabel(lab if j == 0 else "")
                ax[i][j].grid(alpha=0.3)
                if c == "rmse_track_mm":
                    ax[i][j].set_ylim(0, 0.3)
        fig.tight_layout()
        fd = ROOT / "docs" / "figs"
        fd.mkdir(parents=True, exist_ok=True)
        fig.savefig(fd / f"compare_methods{a.tag}.png", dpi=150)
        print("figure:", fd / f"compare_methods{a.tag}.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()