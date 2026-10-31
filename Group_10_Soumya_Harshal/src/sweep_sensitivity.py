"""One-at-a-time sensitivity of the BASE vs PROP result to the modelling assumptions.
    python src/sweep_sensitivity.py [--n 20] [--workers 1]
Nominal point: tremor amplitude 150 um, tissue stiffness x1 (15 N/m), damping 3 N s/m, drift 40 um.
Each parameter is swept alone over a wide range; the SAME seeds (paired) are used for BASE and PROP.
Writes analytics/sensitivity.csv and docs/figures/fig_sensitivity.png"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import surgical_tremor_controller as sc

NOM = dict(amp=150.0, k=1.0, c=3.0, drift=40.0)
SWEEPS = {
    "amp":   ("tremor amplitude, per-axis peak (um)", [30, 50, 100, 150, 250, 400, 600]),
    "drift": ("slow drift amplitude (um)",            [0, 40, 100, 200, 400]),
    "k":     ("tissue stiffness (x 15 N/m)",          [0.1, 0.3, 1, 3, 10]),
    "c":     ("tissue damping (N s/m)",               [0.3, 3, 10, 30]),
}

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
a = ap.parse_args()

model = sc.load_model()
orig_defaults = sc.TremorModel.__init__.__defaults__
rows = []
t0 = time.time()
settings = [(p, v) for p, (_, vals) in SWEEPS.items() for v in vals]
for si, (param, val) in enumerate(settings):
    cfg = dict(NOM)
    cfg[param] = val
    sc.TISSUE_C = cfg["c"]
    sc.TremorModel.__init__.__defaults__ = (cfg["drift"],)
    for i in range(a.n):
        seed = 3000 + i
        for c in ("BASE", "PROP"):
            r = sc.run_trial(model, c, seed, cfg["amp"], cfg["k"], record=True)["summary"]
            rows.append(dict(param=param, value=val, cond=c, seed=seed,
                             hold_rms_mm=r["hold_rms_mm"], hold_max_mm=r["hold_max_mm"],
                             rmse_track_mm=r["rmse_track_mm"], rejection_dB=r["tremor_rejection_dB"],
                             aborted=r["aborted"]))
    print(f"[{si + 1}/{len(settings)}] {param}={val} done, {time.time() - t0:.0f}s", flush=True)
sc.TremorModel.__init__.__defaults__ = orig_defaults

df = pd.DataFrame(rows)
ROOT = Path(sc.ROOT)
df.to_csv(ROOT / "analytics" / "sensitivity.csv", index=False)

g = df.groupby(["param", "value", "cond"]).agg(
    hold_rms=("hold_rms_mm", "mean"), hold_max=("hold_max_mm", "mean"),
    worst=("hold_max_mm", "max"), rej=("rejection_dB", "mean"),
    pass_rate=("hold_max_mm", lambda x: float((x < 0.5).mean())),
    aborts=("aborted", "sum")).round(4)
pd.set_option("display.width", 200)
print(g.to_string())
print("\nPROP better than BASE (hold RMS, paired, per setting):")
for (p, v), d in df.groupby(["param", "value"]):
    b = d[d.cond == "BASE"].sort_values("seed").hold_rms_mm.values
    q = d[d.cond == "PROP"].sort_values("seed").hold_rms_mm.values
    print(f"  {p:6s} {v:>6}: PROP lower in {int((q < b).sum())}/{len(b)} seeds, "
          f"mean BASE {b.mean():.3f} -> PROP {q.mean():.3f} mm")

fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
for k_, (param, (label, vals)) in enumerate(SWEEPS.items()):
    for c, col in (("BASE", "tab:red"), ("PROP", "tab:blue")):
        d = df[(df.param == param) & (df.cond == c)].groupby("value").hold_max_mm
        m, s = d.mean(), d.std()
        ax[k_].errorbar(m.index, m.values, yerr=s.values, fmt="o-", color=col, capsize=3, label=c)
    ax[k_].axhline(0.5, color="k", ls="--", lw=1)
    ax[k_].set_xlabel(label, fontsize=9)
    if param in ("amp", "k", "c"):
        ax[k_].set_xscale("log")
    ax[k_].grid(alpha=0.3)
ax[0].set_ylabel("hold max error (mm), mean +/- sd")
ax[0].legend()
ax[0].set_yscale("log")
fig.suptitle("Sensitivity of the result to the modelling assumptions (dashed = 0.5 mm target)", fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "docs" / "figures" / "fig_sensitivity.png", dpi=200)
print("saved analytics/sensitivity.csv, docs/figures/fig_sensitivity.png")