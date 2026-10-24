"""Run the tremor experiment on every planned scenario.
    python src/run_scenarios.py [--n 5] [--ids enhancing_superior,edema_anterior]
Needs analytics/scenarios.json from plan_scenarios.py.  Writes analytics/scenario_trials.csv."""
import argparse
import json
import time

import numpy as np
import pandas as pd

import surgical_tremor_controller as sc

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5, help="seeds per condition per scenario")
ap.add_argument("--ids", default="", help="comma list of scenario ids (default: all)")
ap.add_argument("--seed0", type=int, default=2000)
a = ap.parse_args()

scns = json.loads((sc.ROOT / "analytics" / "scenarios.json").read_text())
if a.ids:
    keep = a.ids.split(",")
    scns = [s for s in scns if s["id"] in keep]
model = sc.load_model()
rows, t0 = [], time.time()
for s in scns:
    plan = sc.plan_from_scenario(s)
    for i in range(a.n):
        seed = a.seed0 + i
        rng = np.random.default_rng(seed)
        amp, ks = float(rng.uniform(50, 250)), float(rng.uniform(0.7, 1.3))
        for c in ("NT", "BASE", "PROP"):
            r = sc.run_trial(model, c, seed, amp, ks, plan=plan)["summary"]
            r.update(scenario=s["id"], planner_status=s["status"], path_len_mm=s["path_len_m"] * 1e3,
                     plan_min_w=s["min_w"])
            rows.append(r)
    print(f"{s['id']:22s} done  elapsed {time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
df.to_csv(sc.ROOT / "analytics" / "scenario_trials.csv", index=False)
pd.set_option("display.width", 220)
g = df.pivot_table(index="scenario", columns="cond",
                   values=["hold_rms_mm", "hold_max_mm", "tremor_rejection_dB"], aggfunc="mean").round(3)
print(g.to_string())
print("\nrun failures (plan_ok=0 or aborted):", int((df.plan_ok == 0).sum()), int(df.aborted.sum()))
print("pass < 0.5 mm:", df.groupby("cond").pass_0p5mm_hold.mean().round(3).to_dict())