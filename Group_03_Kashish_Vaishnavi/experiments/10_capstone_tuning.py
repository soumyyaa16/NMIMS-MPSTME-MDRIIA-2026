"""Tune the capstone controller on TUNING seeds, never on the evaluation seeds.

    python experiments/10_capstone_tuning.py --n 20

The Monte Carlo in ``09_capstone_monte_carlo.py`` evaluates on seeds from 70000; this script tunes on seeds from 60000, a disjoint
range, so the evaluation result is not selected on its own data. Each configuration is a set of FSM and safety-filter options;
the table written to ``results/capstone_tuning.csv`` is what the chosen settings in ``capstone/monte_carlo.py`` are read from.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import aedrover  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402
import functools  # noqa: E402

import pandas as pd  # noqa: E402

import aedrover.capstone.fsm  # noqa: E402,F401  (registers fsm_dwa)
from aedrover.capstone.monte_carlo import CONDITIONS, CRUISE_MPS  # noqa: E402
from aedrover.control.safety_filter import SafetyFilter, SafetyParams  # noqa: E402
from aedrover.nav.base import make_controller, run_episode  # noqa: E402
from aedrover.parallel import pmap  # noqa: E402
from aedrover.sim.env import AEDRoverEnv  # noqa: E402
from aedrover.sim.vehicle_mjcf import VehicleParams  # noqa: E402

TUNING_SEED_BASE = 60_000
TUNING_CONDITIONS = ("lane_splitting", "slippery", "kerb", "crowd")
CONFIGS: dict[str, dict] = {
    "current": {},
    "assist": {"climb_assist": True},
    "mu_belief 0.35": {"mu_belief": 0.35},          # adopted: the lowest friction the scenario sampler draws
    "margin 0.30": {"climb_margin": 0.30},
    "margin 0.05": {"climb_margin": 0.05},
    "assist + margin 0.30": {"climb_assist": True, "climb_margin": 0.30},
}


# Safety-filter variants that close the recorded standards deviations (reaction time 100 ms -> 50 ms, pedestrian-zone speed
# 1.4 -> 0.8 m/s, clearance bubble 0.5 m). They are evaluated on top of the chosen FSM configuration to measure what compliance costs.
SAFETY_CONFIGS: dict[str, dict] = {
    "safety: default": {},
    "safety: t_react 0.05 s": {"t_react": 0.05},
    "safety: ped cap 0.8 m/s": {"ped_speed_cap": 0.8},
    "safety: 0.5 m clearance": {"front_margin": 0.5, "side_margin": 0.5, "ped_extra": 0.4},
    "safety: all three": {"t_react": 0.05, "ped_speed_cap": 0.8, "front_margin": 0.5, "side_margin": 0.5, "ped_extra": 0.4},
}


@functools.lru_cache(maxsize=1)
def _env() -> AEDRoverEnv:
    return AEDRoverEnv(obs_mode="dict", veh=VehicleParams.by_name("course"))


def run_cfg(job: tuple[str, str, int, str]) -> dict:
    cfg, cond, seed, base = job
    safety = SAFETY_CONFIGS.get(cfg, {})
    fsm_kw = {"v_cruise": CRUISE_MPS, **CONFIGS[base if cfg in SAFETY_CONFIGS else cfg]}
    ctrl = make_controller("fsm_dwa", v_cruise=CRUISE_MPS, fsm_kwargs=fsm_kw, safety_kwargs=safety)
    ep = run_episode(_env(), ctrl, SafetyFilter(SafetyParams(**safety)), seed=seed, options={"family": CONDITIONS[cond]})
    return {"config": cfg, "condition": cond, "seed": seed, "outcome": ep["outcome"], "success": int(ep["success"]),
            "over3g": int(ep["shock_over_budget"]), "peak_g": ep["peak_shock_g"], "time_s": ep["time_s"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--grid", choices=("fsm", "safety"), default="fsm")
    ap.add_argument("--base", default="current", help="FSM configuration the safety grid is evaluated on")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    names = CONFIGS if a.grid == "fsm" else SAFETY_CONFIGS
    jobs = [(c, cond, TUNING_SEED_BASE + 1000 * ci + k, a.base) for c in names for ci, cond in enumerate(TUNING_CONDITIONS) for k in range(a.n)]
    df = pd.DataFrame(pmap(run_cfg, jobs, workers=a.workers, desc="tuning"))
    root = Path(__file__).resolve().parents[1]
    out = Path(a.out or root / ("analytics" if Path(__file__).resolve().parent.name == "analytics" else "results") / f"capstone_tuning_{a.grid}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.5g")
    tab = df.groupby(["config", "condition"]).agg(success=("success", "mean"), over3g=("over3g", "mean")).unstack("condition")
    tab[("success", "mean")] = df.groupby("config").success.mean()
    tab[("over3g", "mean")] = df.groupby("config").over3g.mean()
    print(tab.round(2).to_string())


if __name__ == "__main__":
    main()
