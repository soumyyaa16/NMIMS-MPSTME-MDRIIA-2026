"""Batch runner: python src/run_trials.py --n 30 --out analytics
Writes analytics/surgical_real_trials.csv (one row per run) and telemetry/*.csv.gz (500 Hz)."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from surgical_tremor_controller import CONDITIONS, TELEMETRY_COLUMNS, load_model, run_trial, ROOT

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="runs per condition")
    ap.add_argument("--conds", default="NT,BASE,PROP", help="comma list from " + ",".join(CONDITIONS))
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "analytics"))
    ap.add_argument("--no-telemetry", action="store_true")
    a = ap.parse_args()

    model = load_model()
    out = Path(a.out)
    tel_dir = out / "telemetry"
    tel_dir.mkdir(parents=True, exist_ok=True)
    conds = a.conds.split(",")

    rows = []
    t0 = time.time()
    for i in range(a.n):
        seed = a.seed0 + i
        rng = np.random.default_rng(seed)           # SAME disturbance draw for every condition (paired design)
        amp_um = float(rng.uniform(50, 250))
        k_scale = float(rng.uniform(0.7, 1.3))
        for c in conds:
            r = run_trial(model, c, seed, amp_um, k_scale)
            r["summary"]["run_id"] = f"{c}_{seed}"
            rows.append(r["summary"])
            if r["telemetry"] is not None and not a.no_telemetry:
                tel, ev = r["telemetry"]
                df = pd.DataFrame(tel, columns=TELEMETRY_COLUMNS)
                df.to_csv(tel_dir / f"{c}_{seed}.csv.gz", index=False, float_format="%.7g")
        print(f"seed {seed} done ({i + 1}/{a.n})  elapsed {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "surgical_real_trials.csv", index=False)
    print(df.groupby("cond")[["rmse_track_mm", "hold_rms_mm", "hold_max_mm", "tremor_rejection_dB",
                              "pass_0p5mm_hold", "aborted"]].mean().round(3))