"""Section 4 tables from the real MuJoCo trials: baseline (no filter) vs proposed (learned filter).
Proposed = mean of the 10 trained policies on each hand (so n = 30 independent hands, not 300 correlated rows).
Paired two-sided t-test on the same seeds. Run from the repo root:  python src\\section4_tables.py"""
import sys, shutil
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
TAG = sys.argv[1] if len(sys.argv) > 1 else "_kp50_10s"
SRC = ROOT / "analytics" / f"compare_trials{TAG}.csv"
df = pd.read_csv(SRC)
obj = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
mcol = "method" if "method" in df.columns else None
tcol = "tier" if "tier" in df.columns else None
if mcol is None or tcol is None or "seed" not in df.columns:
    sys.exit(f"cannot find method/tier/seed columns. columns = {list(df.columns)}\n{df.head(3)}")
MEAS = [("rmse_track_mm", "Tracking RMSE while inserting", "mm", "low"),
        ("hold_rms_mm", "Hold error RMS (primary)", "mm", "low"),
        ("place_err_end_insert_mm", "Needle placement error at end of insertion", "mm", "low"),
        ("tremor_rejection_dB", "Tremor rejection 20log10(A_in/A_out)", "dB", "high"),
        ("control_effort", "Control effort integral tau^2 dt", "N^2 m^2 s", "low"),
        ("max_cond", "Max Jacobian condition number", "-", "low"),
        ("max_force_N", "Max tissue insertion force", "N", "low")]
rows = []
for tier in sorted(df[tcol].unique()):
    d = df[df[tcol] == tier]
    b = d[d[mcol] == "BASE"].groupby("seed").mean(numeric_only=True)
    p = d[d[mcol].astype(str).str.startswith("RL_dr")].groupby("seed").mean(numeric_only=True)
    seeds = b.index.intersection(p.index)
    for col, name, unit, good in MEAS:
        if col not in b.columns:
            continue
        x, y = b.loc[seeds, col].astype(float), p.loc[seeds, col].astype(float)
        ok = x.notna() & y.notna()
        x, y = x[ok], y[ok]
        n = len(x)
        if n < 3:
            continue
        diff = x - y
        t, pv = stats.ttest_rel(x, y)
        sp = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        d_c = (x.mean() - y.mean()) / sp if sp > 0 else np.nan
        half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
        better = (y.mean() < x.mean()) if good == "low" else (y.mean() > x.mean())
        rows.append(dict(tier=tier, measure=name, unit=unit, n=n, base_mean=x.mean(), base_sd=x.std(ddof=1),
                         prop_mean=y.mean(), prop_sd=y.std(ddof=1), better="proposed" if better else "baseline",
                         t=t, df=n - 1, p=pv, cohens_d=d_c, diff_mean=diff.mean(),
                         ci95_lo=diff.mean() - half, ci95_hi=diff.mean() + half))
out = pd.DataFrame(rows)
out.to_csv(ROOT / "analytics" / f"section4_tables{TAG}.csv", index=False)
shutil.copy(SRC, ROOT / "analytics" / f"compare_real_trials{TAG}.csv")   # name matches the template's *_real_trials.csv rule
pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", lambda v: f"{v:.4g}")
for tier, g in out.groupby("tier"):
    print(f"\n=== tier {tier} ===")
    print(g.drop(columns=["tier", "unit"]).to_string(index=False))
print("\nsaved analytics/section4_tables.csv and analytics/compare_real_trials_kp50_10s.csv")
print("hands:", df['seed'].nunique(), "first seed", df['seed'].min())
