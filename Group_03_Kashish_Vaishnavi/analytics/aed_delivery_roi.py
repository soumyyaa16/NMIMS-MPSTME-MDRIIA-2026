"""CSBS business script for the capstone dossier (Section 3). Dimensionless; no currency anywhere.

    python analytics/aed_delivery_roi.py

Reads ``analytics/aed_delivery_real_trials.csv`` (the MuJoCo Monte Carlo) so the pod speed and the
FMEA occurrence scores are measured, not typed in. Writes the numbers, the FMEA, the sensitivity
tables and the three dossier graphs next to this file. Every output is recomputed on each run.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import aedrover  # noqa: F401
except ImportError:                      # exported course folder: the package is vendored under ../src
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402
import math  # noqa: E402
from dataclasses import replace  # noqa: E402

import pandas as pd  # noqa: E402

from aedrover.capstone.business import (  # noqa: E402
    Inputs,
    break_even,
    evaluate,
    fmea_table,
    input_records,
    required_standdown_calls_per_pod_day,
    scenario_table,
    sweep,
    tornado,
    with_measured_speed,
)
from aedrover.capstone.figures import fmea_plot, queue_plot, sensitivity_compact, sensitivity_plot  # noqa: E402

HERE = Path(__file__).resolve().parent
SWEEP_INPUT = "ohca_incidence_per_100k_yr"      # the most uncertain input: no verified national value
TORNADO_INPUTS = ["ohca_incidence_per_100k_yr", "p_standdown", "h_als_per_call", "pod_opex", "hub_opex",
                  "suspected_call_factor", "hub_radius_m", "v_amr_mps", "turnaround_h", "pop_density_per_km2"]


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None if math.isnan(v) else ("inf" if v > 0 else "-inf")
    return v


def main(trials_csv: Path | None = None, out_dir: Path | None = None) -> dict:
    trials_csv = trials_csv or HERE / "aed_delivery_real_trials.csv"
    out = out_dir or HERE
    figs = out.parent / "docs" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    trials = pd.read_csv(trials_csv)

    x = with_measured_speed(Inputs(), trials)
    res = evaluate(x)
    base = getattr(x, SWEEP_INPUT)
    grid = [base * f for f in _geom(0.5, 3000.0, 90)]
    sw = sweep(x, SWEEP_INPUT, grid)
    be = {
        "kappa=1 (cost parity)": break_even(x, SWEEP_INPUT, "kappa", 1.0, lo=base, hi=base * 1e4),
        "kappa=0.25 (course target)": break_even(x, SWEEP_INPUT, "kappa", 0.25, lo=base, hi=base * 1e5),
    }
    tor = tornado(x, TORNADO_INPUTS)
    fm = fmea_table(trials)

    scen = scenario_table(x)
    scen.to_csv(out / "roi_scenarios.csv", index=False, float_format="%.6g")
    need = {"base_costs": required_standdown_calls_per_pod_day(x),
            "cheap_pod_shared_hub": required_standdown_calls_per_pod_day(replace(x, pod_opex=0.03, hub_opex=0.005)),
            "dual_use_marginal": required_standdown_calls_per_pod_day(replace(x, pod_opex=0.005, hub_opex=0.0)),
            "achieved_base": res["calls_per_day"] * x.p_standdown / res["pods"]}
    pd.DataFrame(input_records(x)).to_csv(out / "roi_inputs.csv", index=False)
    sw.to_csv(out / "roi_sensitivity.csv", index=False, float_format="%.6g")
    tor.to_csv(out / "roi_tornado.csv", index=False, float_format="%.6g")
    fm.to_csv(out / "fmea.csv", index=False, float_format="%.6g")
    sensitivity_plot(sw, SWEEP_INPUT, base, be, figs / "business_sensitivity.png")
    sensitivity_compact(sw, SWEEP_INPUT, base, be, figs / "business_sensitivity_compact.png")
    queue_plot(res["lambda_per_hub_h"], res["mu_per_h"], figs / "business_queue.png")
    fmea_plot(fm, figs / "business_fmea.png")

    payload = {"inputs": {r["input"]: r["value"] for r in input_records(x)}, "results": res,
               "sensitivity_input": SWEEP_INPUT, "break_even": be, "standdown_calls_per_pod_day_needed_for_kappa_0.25": need}
    (out / "roi_results.json").write_text(json.dumps(payload, indent=2, default=_clean), encoding="utf-8")

    print("MDRIIA Group 03 - dimensionless business model (all costs in ALS vehicle-years)")
    for k in ("hubs", "pods", "calls_per_day", "H_reclaimed_veh_h_per_day", "delta_FTE", "kappa", "NCRI",
              "T_payback_months", "P_payback", "lambda_per_hub_h", "mu_per_h", "rho", "p_wait", "Wq_min",
              "delta_survival", "delta_QALY_per_encounter", "QALY_per_ALS_vehicle_year_of_cost"):
        print(f"  {k:36s} {res[k]:.6g}")
    for k, v in be.items():
        print(f"  break-even {SWEEP_INPUT} for {k}: {v:.6g}")
    print(scen[["scenario", "pods", "kappa", "NCRI"]].round(3).to_string(index=False))
    print("stand-down calls per pod per day needed for kappa <= 0.25:", {k: round(v, 3) for k, v in need.items()})
    print(fm[["failure_mode", "S", "O_pre", "D_pre", "RPN_pre", "O_post", "D_post", "RPN_post"]].to_string(index=False))
    return payload


def _geom(lo: float, hi: float, n: int) -> list[float]:
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
