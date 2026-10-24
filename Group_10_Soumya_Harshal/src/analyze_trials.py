"""Stats + figures from real MuJoCo runs.
python src/analyze_trials.py [--csv analytics/surgical_real_trials.csv]
Writes analytics/stats_summary.csv and docs/figures/*.png.  Everything comes from the CSV/telemetry."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=str(ROOT / "analytics" / "surgical_real_trials.csv"))
ap.add_argument("--a", default="BASE")
ap.add_argument("--b", default="PROP")
args = ap.parse_args()

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
df = pd.read_csv(args.csv)
fig_dir = ROOT / "docs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

METRICS = [("rmse_track_mm", "Tracking RMSE (mm)"), ("hold_rms_mm", "Hold RMS error (mm)"),
           ("hold_max_mm", "Hold max error (mm)"), ("tremor_rejection_dB", "Tremor rejection 7-12 Hz (dB)"),
           ("control_effort", "Control effort (sum tau^2 dt)"), ("max_cond", "Max Jacobian condition no.")]

print("\n=== per-condition mean +/- sd (n per condition) ===")
g = df.groupby("cond")
summ = g[[m for m, _ in METRICS]].agg(["mean", "std"]).round(4)
print(summ.T.to_string())
print("\nn per condition:", g.size().to_dict())
print("pass < 0.5 mm (hold max):", g["pass_0p5mm_hold"].mean().round(3).to_dict(), " aborted:", g["aborted"].sum().to_dict())

A = df[df.cond == args.a].sort_values("seed")
B = df[df.cond == args.b].sort_values("seed")
rows = []
print(f"\n=== {args.a} vs {args.b}: Welch t-test, Cohen's d, 95% CI of mean difference ({args.a} - {args.b}) ===")
for m, label in METRICS:
    x, y = A[m].dropna().values, B[m].dropna().values
    t, p = stats.ttest_ind(x, y, equal_var=False)
    sp = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    d = (x.mean() - y.mean()) / sp
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    dfw = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    se = np.sqrt(vx + vy)
    diff = x.mean() - y.mean()
    ci = (diff - stats.t.ppf(0.975, dfw) * se, diff + stats.t.ppf(0.975, dfw) * se)
    tp, pp = stats.ttest_rel(x, y) if len(x) == len(y) else (np.nan, np.nan)
    pct = 100 * diff / x.mean() if x.mean() else np.nan
    rows.append(dict(metric=m, n_a=len(x), n_b=len(y), mean_a=x.mean(), mean_b=y.mean(), diff=diff,
                     pct_change_vs_a=pct, ci95_lo=ci[0], ci95_hi=ci[1], welch_t=t, welch_p=p,
                     cohens_d=d, paired_t=tp, paired_p=pp))
st = pd.DataFrame(rows)
print(st.round(5).to_string(index=False))
st.to_csv(ROOT / "analytics" / "stats_summary.csv", index=False)

# --- box plots
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
for a_, (m, label) in zip(ax, [METRICS[0], METRICS[2], METRICS[3]]):
    data = [df[df.cond == c][m].dropna().values for c in ["NT", args.a, args.b] if c in set(df.cond)]
    names = [c for c in ["NT", args.a, args.b] if c in set(df.cond)]
    a_.boxplot(data, tick_labels=names, showmeans=True)
    a_.set_title(label, fontsize=10)
    a_.grid(alpha=0.3)
ax[1].axhline(0.5, color="r", ls="--", lw=1)
ax[1].text(0.6, 0.5, " 0.5 mm limit", color="r", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(fig_dir / "fig_boxplots.png", dpi=200)

# --- error vs time, one seed, from logged telemetry
seed = int(A.seed.iloc[0])
tel = ROOT / "analytics" / "telemetry"
if (tel / f"{args.a}_{seed}.csv.gz").exists() and (tel / f"{args.b}_{seed}.csv.gz").exists():
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for c, col in [(args.a, "tab:red"), (args.b, "tab:blue")]:
        T = pd.read_csv(tel / f"{c}_{seed}.csv.gz")
        w = (T.fsm == 3) | (T.fsm == 2)
        ax.plot(T.t[w], T.err_norm[w] * 1e3, color=col, lw=0.8, label=c)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tip error vs plan (mm)")
    ax.set_title(f"INSERT + HOLD, seed {seed}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_error_vs_time.png", dpi=200)
print("\nsaved: analytics/stats_summary.csv, docs/figures/fig_boxplots.png, fig_error_vs_time.png")