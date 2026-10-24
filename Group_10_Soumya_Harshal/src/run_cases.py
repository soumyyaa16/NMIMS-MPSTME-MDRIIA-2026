"""Run planner + closed-loop tremor experiment on every selected case, then summarise across cases.
    python src/run_cases.py                       # cases marked 'selected' in cases/summary.csv, 3 seeds each
    python src/run_cases.py --cases c160,c040 --n 5
    python src/run_cases.py --summary-only        # just rebuild the tables/figures from existing results
Writes analytics/cases_summary.csv, analytics/cases_trials_all.csv, docs/figures/fig_cases_*.png"""
import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="", help="comma list; default = 'selected' cases in cases/summary.csv")
ap.add_argument("--n", type=int, default=3, help="seeds per condition per scenario")
ap.add_argument("--summary-only", action="store_true")
a = ap.parse_args()

summ = pd.read_csv(ROOT / "cases" / "summary.csv")
if a.cases:
    cases = a.cases.split(",")
elif "selected" in summ:
    cases = summ.case_id[summ.selected].tolist()
else:
    cases = summ.case_id.tolist()
print("cases:", cases, flush=True)

if not a.summary_only:
    for c in cases:
        print(f"\n===== {c}: planner =====", flush=True)
        subprocess.run([sys.executable, str(HERE / "plan_scenarios.py"), "--case", c], check=True)
        print(f"\n===== {c}: closed-loop runs =====", flush=True)
        subprocess.run([sys.executable, str(HERE / "run_scenarios.py"), "--case", c, "--n", str(a.n)], check=True)

out, trials, rows = ROOT / "analytics", [], []
order = ["superior", "anterior", "posterior", "left", "right", "inferior"]
for c in cases:
    d = out / "cases" / c
    po = pd.read_csv(d / "scenario_outcomes.csv")
    tr = pd.read_csv(d / "scenario_trials.csv")
    trials.append(tr)
    g = tr.groupby("cond")
    s = summ[summ.case_id == c].iloc[0]
    rows.append(dict(case=c, vol_enh_cm3=s.vol_enh_cm3, depth_below_surface_mm=s.depth_below_surface_mm,
                     n_scenarios=len(po), reachable=int((po.status == "REACHABLE").sum()),
                     flagged=int((po.status == "FLAGGED").sum()), blocked=int((po.status == "BLOCKED").sum()),
                     mean_pct_paths_ok=po.pct_candidates_reachable.mean(),
                     runs=len(tr), aborts=int(tr.aborted.sum()),
                     pass_BASE=g.pass_0p5mm_hold.mean().get("BASE", np.nan), pass_PROP=g.pass_0p5mm_hold.mean().get("PROP", np.nan),
                     hold_rms_BASE=g.hold_rms_mm.mean().get("BASE", np.nan), hold_rms_PROP=g.hold_rms_mm.mean().get("PROP", np.nan),
                     hold_max_BASE=g.hold_max_mm.mean().get("BASE", np.nan), hold_max_PROP=g.hold_max_mm.mean().get("PROP", np.nan),
                     rej_dB_BASE=g.tremor_rejection_dB.mean().get("BASE", np.nan), rej_dB_PROP=g.tremor_rejection_dB.mean().get("PROP", np.nan),
                     floor_NT_hold_max=g.hold_max_mm.mean().get("NT", np.nan)))
res = pd.DataFrame(rows)
allt = pd.concat(trials, ignore_index=True)
allt.to_csv(out / "cases_trials_all.csv", index=False)
res.to_csv(out / "cases_summary.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print("\n=== per-case summary ===")
print(res.round(3).to_string(index=False))

# pooled paired comparison BASE vs PROP across every case/scenario/seed
key = ["case", "scenario", "seed"]
B = allt[allt.cond == "BASE"].set_index(key).sort_index()
P = allt[allt.cond == "PROP"].set_index(key).sort_index()
print("\n=== pooled BASE vs PROP over", len(B), "paired runs ===")
for m in ("hold_rms_mm", "hold_max_mm", "tremor_rejection_dB"):
    x, y = B[m].values, P[m].values
    t, p = stats.ttest_rel(x, y)
    dz = (x - y).mean() / (x - y).std(ddof=1)
    print(f"{m:22s} BASE {x.mean():.3f}  PROP {y.mean():.3f}  diff {(x - y).mean():+.3f}  paired t {t:.1f}  p {p:.2g}  d_z {dz:.2f}")
print("runs under 0.5 mm:", allt.groupby("cond").pass_0p5mm_hold.mean().round(3).to_dict(), " aborted runs:", int(allt.aborted.sum()))

# ---- figures
figdir = ROOT / "docs" / "figures"
figdir.mkdir(parents=True, exist_ok=True)
color = {"REACHABLE": "#2a9d5c", "FLAGGED": "#e0a100", "BLOCKED": "#c0392b"}
fig, ax = plt.subplots(figsize=(13, 0.55 * len(cases) + 1.6))
cols = [(l, z) for l in ("enhancing", "necrotic", "edema") for z in order]
for i, c in enumerate(cases):
    po = pd.read_csv(out / "cases" / c / "scenario_outcomes.csv").set_index(["lesion", "zone"])
    for j, (l, z) in enumerate(cols):
        st = po.status.get((l, z), None)
        if st is None:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color="#dddddd", ec="white"))
            continue
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color[st], ec="white", lw=1.5))
        ax.text(j + .5, i + .5, st[0], ha="center", va="center", color="white", fontsize=8, fontweight="bold")
ax.set_xlim(0, len(cols)); ax.set_ylim(len(cases), 0)
ax.set_xticks([j + .5 for j in range(len(cols))])
ax.set_xticklabels([f"{l[:3]}\n{z[:4]}" for l, z in cols], fontsize=7)
ax.set_yticks([i + .5 for i in range(len(cases))]); ax.set_yticklabels(cases)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("Planner verdict per case x lesion region x approach zone  (R reachable, F flagged, B blocked)", fontsize=10)
fig.tight_layout(); fig.savefig(figdir / "fig_cases_matrix.png", dpi=200)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
x = np.arange(len(cases)); w = 0.38
for k, (m, lab) in enumerate((("hold_rms", "Hold RMS error (mm)"), ("rej_dB", "Tremor rejection 7-12 Hz (dB)"))):
    ax[k].bar(x - w / 2, res[f"{m}_BASE"], w, label="BASE", color="#c0392b")
    ax[k].bar(x + w / 2, res[f"{m}_PROP"], w, label="PROP", color="#2a6fb0")
    ax[k].set_xticks(x); ax[k].set_xticklabels(cases); ax[k].set_ylabel(lab); ax[k].grid(axis="y", alpha=.3); ax[k].legend()
fig.tight_layout(); fig.savefig(figdir / "fig_cases_performance.png", dpi=200)
print("\nsaved analytics/cases_summary.csv, analytics/cases_trials_all.csv, docs/figures/fig_cases_matrix.png, fig_cases_performance.png")