import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import surgical_tremor_controller as sc
import rl_filter as rf

ROOT = sc.ROOT
sc.KP, sc.KI = 50.0, 0.003 * 50.0 ** 2
sc.TremorModel.__init__.__defaults__ = (40.0,)          # tier C drift
seed = 1000
rng = np.random.default_rng(seed)
amp, k_scale = float(rng.uniform(50, 250)), float(rng.uniform(0.7, 1.3))
model = sc.load_model()
orig = sc.StreamingLowpass
pol = rf.NumpyPolicy(str(ROOT / "analytics" / "rl_dr_s1" / "policy.npz"))
runs = {}
for name, cond, fac in [("BASE", "BASE", None), ("Learned", "LPFRAW", lambda: rf.AdaptiveFilter("rl", pol))]:
    sc.StreamingLowpass = fac if fac is not None else orig
    rows, events = sc.run_trial(model, cond, seed, amp, k_scale)["telemetry"]
    df = pd.DataFrame(np.array(rows), columns=sc.TELEMETRY_COLUMNS)
    df.to_csv(ROOT / "analytics" / f"telemetry_C{seed}_{name}.csv", index=False)
    runs[name] = df
sc.StreamingLowpass = orig
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
t = runs["BASE"].t.values
i0, i1 = np.argmin(abs(t - 3.0)), np.argmin(abs(t - 8.0))
ref = runs["BASE"][["ref_x", "ref_y", "ref_z"]].values
u = ref[i1] - ref[i0]
u /= np.linalg.norm(u)
ax[0].plot(t, (ref - ref[i0]) @ u * 1e3, "k--", label="reference")
for n, c in (("BASE", "C3"), ("Learned", "C0")):
    tip = runs[n][["tip_x", "tip_y", "tip_z"]].values
    ax[0].plot(t, (tip - ref[i0]) @ u * 1e3, c, lw=1, label=n)
    ax[1].plot(t, runs[n].err_norm * 1e6, c, lw=1, label=n)
    h = (t >= 8.0) & (t < 10.0)
    print(n, "hold RMS (um):", round(float(np.sqrt(np.mean((runs[n].err_norm[h] * 1e6) ** 2))), 1))
ax[0].set(xlim=(3, 10), xlabel="time (s)", ylabel="position along insertion axis (mm)", title="Reference vs actual")
ax[1].set(xlim=(3, 10), xlabel="time (s)", ylabel="tip error (um)", title="Tip error (hold: 8-10 s)")
for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
fig.suptitle(f"One run: tier C, seed {seed}, tremor {amp:.0f} um, tissue scale {k_scale:.2f}", fontsize=9)
fig.tight_layout()
(ROOT / "docs" / "figs").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "docs" / "figs" / "section4_trajectory.png", dpi=170)
print("saved docs/figs/section4_trajectory.png and two telemetry CSVs in analytics/")
