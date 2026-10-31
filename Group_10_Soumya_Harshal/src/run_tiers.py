"""run_tiers.py - how big is the filter's benefit if the hand is a NORMAL surgeon's hand, or a tired / enhanced-tremor one?
    python src\\run_tiers.py            (30 seeds x 4 tiers x 2 conditions, about 10 min)       python src\\run_tiers.py --n 5   (quick look)
Amplitude convention: our `amp` is the peak per-axis amplitude of 3 weighted sinusoids, so  vector RMS = 0.755 x amp  (per-axis RMS = 0.436 x amp).
  A  typical hand      amp 24-50 um  -> vector RMS 18-38 um   (literature: Ladha 2023 median 18 um; Singh & Riviere 2002: 38 um RMS)   drift 40 um
  B  typical + drift   same tremor, drift amplitude ~210 um on average (Ladha 2023: measured manual drift median 212 um)
  C  enhanced tremor   amp 50-250 um -> vector RMS 38-189 um  (our N=50 batch; Elble 2005: enhanced tremor is 5-20x normal)           drift 40 um
  D  enhanced + drift  same as C with drift ~210 um
Same seeds for BASE (no filter) and PROP (our filter). Writes analytics/tiers.csv and analytics/tiers_summary.csv"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats

import surgical_tremor_controller as sc

TIERS = [("A typical hand", 24, 50, 40.0), ("B typical + measured drift", 24, 50, 280.0),
         ("C enhanced tremor (our N=50 range)", 50, 250, 40.0), ("D enhanced + measured drift", 50, 250, 280.0)]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed0", type=int, default=1000)
a = ap.parse_args()
model = sc.load_model()
rows, t0 = [], time.time()
for name, lo, hi, drift in TIERS:
    sc.TremorModel.__init__.__defaults__ = (drift,)          # drift_um default for every TremorModel built inside run_trial
    for i in range(a.n):
        seed = a.seed0 + i
        rng = np.random.default_rng(seed)
        amp = float(rng.uniform(lo, hi))
        k_scale = float(rng.uniform(0.7, 1.3))
        for c in ("BASE", "PROP"):
            r = sc.run_trial(model, c, seed, amp, k_scale)["summary"]
            r["tier"], r["amp_um"], r["drift_um"] = name, amp, drift
            rows.append(r)
    print(f"{name} done  {time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
df.to_csv(sc.ROOT / "analytics" / "tiers.csv", index=False)
out = []
for name, *_ in TIERS:
    d = df[df.tier == name]
    b = d[d.cond == "BASE"].set_index("seed")
    p = d[d.cond == "PROP"].set_index("seed")
    row = dict(tier=name, n=len(b))
    for m in ("hold_rms_mm", "hold_max_mm"):
        row[f"BASE_{m}"], row[f"PROP_{m}"] = b[m].mean(), p[m].mean()
    row["reduction_hold_rms_%"] = 100 * (1 - p["hold_rms_mm"].mean() / b["hold_rms_mm"].mean())
    row["paired_p"] = stats.ttest_rel(b["hold_rms_mm"], p["hold_rms_mm"]).pvalue
    row["pass_BASE_%"] = 100 * (b["hold_max_mm"] < 0.5).mean()
    row["pass_PROP_%"] = 100 * (p["hold_max_mm"] < 0.5).mean()
    out.append(row)
S = pd.DataFrame(out)
S.round(4).to_csv(sc.ROOT / "analytics" / "tiers_summary.csv", index=False)
pd.set_option("display.width", 220)
print(S.round(3).to_string(index=False))