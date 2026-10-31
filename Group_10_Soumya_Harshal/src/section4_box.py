"""Box-plot of the primary metric (hold error RMS), baseline vs proposed, N = 30 hands per tier.
Reads analytics/compare_real_trials_kp50_10s.csv. Run from the repo root: python src\\section4_box.py"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "analytics" / "compare_real_trials_kp50_10s.csv")
tiers = sorted(df["tier"].unique())
fig, axes = plt.subplots(1, len(tiers), figsize=(3.6 * len(tiers), 4.2))
rng = np.random.default_rng(0)
for ax, tier in zip(np.atleast_1d(axes), tiers):
    d = df[df["tier"] == tier]
    b = d[d["method"] == "BASE"].groupby("seed")["hold_rms_mm"].mean()
    p = d[d["method"].astype(str).str.startswith("RL_dr")].groupby("seed")["hold_rms_mm"].mean()
    s = b.index.intersection(p.index)
    bp = ax.boxplot([b[s], p[s]], widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#111"))
    for patch, c in zip(bp["boxes"], ["#9aa7b2", "#1f9e8f"]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    for i, v in enumerate([b[s], p[s]], start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=9, color="#222", alpha=0.55, zorder=3)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["no filter\n(baseline)", "learned filter\n(proposed)"])
    ax.set_title(tier, fontsize=9); ax.grid(axis="y", alpha=0.3)
    if ax is np.atleast_1d(axes)[0]:
        ax.set_ylabel("hold error, RMS (mm)")
fig.suptitle(f"Hold error while holding at the target: N = {len(s)} hands per tier (proposed = mean of 10 trained policies)", fontsize=10)
fig.tight_layout()
out = ROOT / "docs" / "figs" / "section4_boxplot.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180)
print("saved", out)
