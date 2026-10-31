"""sensor_sweep.py - do the filters survive a real arm? Sweeps encoder noise and feedback delay (swept, NOT claimed values).
    python src\\sensor_sweep.py --n 10 --rl dr1=analytics\\rl_dr_s1\\policy.npz --rl dr2=analytics\\rl_dr_s2\\policy.npz --rl dr3=analytics\\rl_dr_s3\\policy.npz   (~40 min)
Enhanced-tremor hands (tier C, amp 50-250 um, drift 40 um), paired seeds 3000+ (unused elsewhere).
Writes analytics/sensor_trials.csv, sensor_summary.csv, docs/figs/sensor_sweep.png"""
import argparse, json, time
import numpy as np, pandas as pd
import rl_filter as rf
import surgical_tremor_controller as sc
import tremor_baselines as tb

ROOT = sc.ROOT
NOISE_UM = [0, 10, 25, 50, 100]          # rms per axis on measured tip position
DELAY_MS = [0, 2, 4, 6, 8]             # feedback delay (2 ms control tick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--rl", action="append", default=[])
    a = ap.parse_args()
    best = json.loads((ROOT / "analytics" / "baselines_tuned.json").read_text())
    orig = sc.StreamingLowpass
    methods = [("BASE", "BASE", None), ("KALMAN", "LPFRAW", lambda: tb.BaselineFilter("kalman", **best["kalman"]))]
    for spec in a.rl:
        name, path = spec.split("=", 1)
        pol = rf.NumpyPolicy(path)
        methods.append((f"RL_{name}", "LPFRAW", (lambda pol=pol: rf.AdaptiveFilter("rl", pol))))
    methods.append(("PROP", "PROP", None))
    settings = [("noise", v, 0) for v in NOISE_UM] + [("delay", v, v // 2) for v in DELAY_MS if v > 0]
    model = sc.load_model()
    sc.TremorModel.__init__.__defaults__ = (40.0,)
    rows, t0 = [], time.time()
    try:
        for axis, val, ticks in settings:
            sc.MEAS_NOISE_UM = float(val) if axis == "noise" else 0.0
            sc.MEAS_DELAY_TICKS = int(ticks)
            for i in range(a.n):
                seed = a.seed0 + i
                rng = np.random.default_rng(seed)
                amp, k_scale = float(rng.uniform(50, 250)), float(rng.uniform(0.7, 1.3))
                for mname, cond, fac in methods:
                    sc.StreamingLowpass = fac if fac is not None else orig
                    r = sc.run_trial(model, cond, seed, amp, k_scale)["summary"]
                    r.update(method=mname, axis=axis, value=val)
                    rows.append(r)
                sc.StreamingLowpass = orig
            print(f"{axis} {val}  done  {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(ROOT / "analytics" / "sensor_trials.csv", index=False)
    finally:
        sc.StreamingLowpass, sc.MEAS_NOISE_UM, sc.MEAS_DELAY_TICKS = orig, 0.0, 0
    df = pd.DataFrame(rows)
    order = [m[0] for m in methods]
    g = df.groupby(["axis", "value", "method"])[["hold_rms_mm", "rmse_track_mm", "hold_max_mm"]].mean().reset_index()
    g["abort_pct"] = df.groupby(["axis", "value", "method"])["aborted"].mean().values * 100 if "aborted" in df else 0
    g.round(4).to_csv(ROOT / "analytics" / "sensor_summary.csv", index=False)
    pd.set_option("display.width", 220)
    for axis in ("noise", "delay"):
        for col, ttl in (("hold_rms_mm", "HOLD ERROR (mm)"), ("rmse_track_mm", "TRACKING ERROR (mm)")):
            print(f"\n{ttl} vs {axis} ({'um rms' if axis == 'noise' else 'ms delay'})")
            print(g[g.axis == axis].pivot(index="method", columns="value", values=col).loc[order].round(3).to_string())
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(9, 6.5))
        for j, axis in enumerate(("noise", "delay")):
            for i, (col, lab) in enumerate((("hold_rms_mm", "hold error (mm)"), ("rmse_track_mm", "tracking error (mm)"))):
                for m in order:
                    d = g[(g.axis == axis) & (g.method == m)].sort_values("value")
                    ax[i][j].plot(d.value, d[col], marker="o", label=m)
                ax[i][j].set_xlabel("encoder noise (um rms)" if axis == "noise" else "feedback delay (ms)"); ax[i][j].set_ylabel(lab); ax[i][j].grid(alpha=.3)
                if axis == "delay": ax[i][j].set_yscale("log")
        ax[0][0].legend(fontsize=7); fig.tight_layout()
        (ROOT / "docs" / "figs").mkdir(parents=True, exist_ok=True); fig.savefig(ROOT / "docs" / "figs" / "sensor_sweep.png", dpi=150); print("figure saved")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
