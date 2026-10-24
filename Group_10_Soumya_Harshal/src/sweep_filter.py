"""Filter cut-off sweep: tremor rejection vs lag cost.  python src/sweep_filter.py [--n 10]
PROP  = plan-relative filter (rejection benefit).  LPFRAW = same filter on the raw command (lag cost).
Writes analytics/filter_sweep.csv and docs/figures/fig_filter_pareto.png"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, group_delay

import surgical_tremor_controller as sc

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--fcs", default="1,2,3.5,5,7,10")
a = ap.parse_args()

Orig = sc.StreamingLowpass
model = sc.load_model()
fcs = [float(x) for x in a.fcs.split(",")]
rows = []
for fc in fcs:
    sc.StreamingLowpass = lambda fc=fc: Orig(fc=fc)
    b, aa = butter(sc.FILTER_ORDER, fc, fs=1.0 / sc.DT_C)
    w, gd = group_delay((b, aa), w=[0.5], fs=1.0 / sc.DT_C)       # group delay at 0.5 Hz ~ DC value
    delay_ms = float(gd[0]) * sc.DT_C * 1e3
    for i in range(a.n):
        seed = 1000 + i
        rng = np.random.default_rng(seed)
        amp, ks = float(rng.uniform(50, 250)), float(rng.uniform(0.7, 1.3))
        for c in ("PROP", "LPFRAW"):
            r = sc.run_trial(model, c, seed, amp, ks, record=True)["summary"]
            rows.append(dict(fc_hz=fc, delay_ms=delay_ms, cond=c, seed=seed, rmse_track_mm=r["rmse_track_mm"],
                             hold_max_mm=r["hold_max_mm"], tremor_rejection_dB=r["tremor_rejection_dB"]))
    print(f"fc {fc} Hz done (group delay {delay_ms:.0f} ms)", flush=True)
df = pd.DataFrame(rows)
out = Path(sc.ROOT)
df.to_csv(out / "analytics" / "filter_sweep.csv", index=False)
s = df.groupby(["cond", "fc_hz"])[["delay_ms", "rmse_track_mm", "hold_max_mm", "tremor_rejection_dB"]].mean().round(3)
pd.set_option("display.width", 200)
print(s.to_string())

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for c, col in (("PROP", "tab:blue"), ("LPFRAW", "tab:red")):
    d = df[df.cond == c].groupby("fc_hz").mean(numeric_only=True)
    ax[0].plot(d.index, d.tremor_rejection_dB, "o-", color=col, label=c)
    ax[1].plot(d.index, d.rmse_track_mm, "o-", color=col, label=c)
ax[0].set_xlabel("cut-off (Hz)"); ax[0].set_ylabel("tremor rejection (dB)"); ax[0].legend()
ax[1].set_xlabel("cut-off (Hz)"); ax[1].set_ylabel("tracking RMSE (mm)"); ax[1].set_yscale("log")
ax[1].axhline(0.5, color="k", ls="--", lw=1)
d = df[df.cond == "PROP"].groupby("fc_hz").mean(numeric_only=True)
ax[2].plot(d.delay_ms, d.tremor_rejection_dB, "o-")
for fc_, x_, y_ in zip(d.index, d.delay_ms, d.tremor_rejection_dB):
    ax[2].annotate(f"{fc_:g} Hz", (x_, y_), fontsize=8, xytext=(3, 3), textcoords="offset points")
ax[2].set_xlabel("filter group delay (ms)"); ax[2].set_ylabel("tremor rejection (dB)")
for x in ax: x.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out / "docs" / "figures" / "fig_filter_pareto.png", dpi=200)
print("saved analytics/filter_sweep.csv, docs/figures/fig_filter_pareto.png")