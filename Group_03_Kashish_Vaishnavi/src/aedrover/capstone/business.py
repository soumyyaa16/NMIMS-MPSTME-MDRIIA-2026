"""CSBS techno-economic model for the capstone dossier (Section 3): strictly dimensionless.

Money never appears. Every cost is a multiple of one reference quantity, the annual operating
cost of one ALS ambulance ("ALS vehicle-year", a unit, not a price). Every number in the result
is computed from the inputs in :data:`INPUT_TABLE`; each input is either tied to a source key in
``docs/references.json`` or labelled ``ASSUMPTION`` and varied in the sensitivity analysis. The
model is a template for the *structure* of the argument; the defaults are placeholders, and the
break-even values show how far each one would have to move for the case to change.

Quantities (course definitions, ``ACADEMIC_AND_INDUSTRY_STANDARDS_MANUAL.md`` section 3)
-----------------------------------------------------------------------------------------
``H_reclaimed``  ALS-ambulance vehicle-hours/day not dispatched for calls the AED pod resolves on its
                 own (stand-down):  ``calls/day * p_standdown * h_per_call``
``kappa``        ``C_robot / C_baseline``, both in ALS vehicle-years per year
                 ``C_robot = pods * pod_opex + hubs * hub_opex``;  ``C_baseline = H_reclaimed * 365 / 8760``
``NCRI``         ``1 - kappa``  (negative means the robot costs more than the ambulance hours it frees)
``P_payback``    ``CapEx / (T_life * dOpEx)`` with ``dOpEx = C_baseline - C_robot``; < 1 pays back inside life
``T_payback``    ``CapEx / dOpEx`` in months (``inf`` when dOpEx <= 0)
``dFTE``         ``H_reclaimed * 365 / 2080``
queue            M/M/c per hub (Erlang C), ``rho = lambda / (c * mu) < 1``;  M/M/1 is the ``c = 1`` case

The labour lens alone often cannot justify a pod, because one hub sees few cardiac-arrest calls. The
``qaly_*`` fields therefore also report the health benefit per ALS vehicle-year of cost, still
without currency.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ..clinical.decision import DispatchTimes
from ..clinical.economics import HOURS_PER_FTE, KAPPA_TARGET
from ..clinical.ems_delay import NAESS_URBAN
from ..clinical.survival import LarsenModel, delta_survival

HOURS_PER_VEHICLE_YEAR = 8760.0     # 24 h x 365 d (calendar)
DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class Inputs:
    # ---- geography and demand
    zone_area_km2: float = 50.0            # urban zone whose arrests are counted                  ASSUMPTION
    coverage_fraction: float = 1.0         # share of the zone that gets pods (1 = uniform grid)  design choice, see SCENARIOS
    hotspot_factor: float = 1.0            # arrest rate in the covered part / zone average        ASSUMPTION (1 = no clustering)
    pop_density_per_km2: float = 20_000.0  # dense metropolitan core                              ASSUMPTION
    ohca_incidence_per_100k_yr: float = 55.0   # OHCA events per 100,000 per year                 ASSUMPTION
    suspected_call_factor: float = 2.5     # dispatched suspected-arrest calls per true OHCA      ASSUMPTION
    hub_radius_m: float = 500.0            # response radius of a micro-hub                       README RQ3 (break-even 0.46-0.58 km)
    route_factor: float = 1.54             # path / straight-line                                 docs/OSM_ROUTES.md (measured)
    v_amr_mps: float = 1.9                 # mean pod speed; replaced by the measured value when trials are supplied
    # ---- service (queue)
    turnaround_h: float = 0.25             # recharge + AED re-arm between missions [h]            ASSUMPTION
    p_wait_max: float = 0.05               # design target: P(a call waits for a free pod)         ASSUMPTION
    # ---- labour reallocation
    p_standdown: float = 0.15              # share of suspected calls resolved without ALS dispatch   ASSUMPTION (most uncertain)
    h_als_per_call: float = 1.5            # ALS vehicle-hours consumed by one dispatch [h]        ASSUMPTION
    # ---- cost (units: ALS vehicle-years)
    pod_opex: float = 0.10                 # annual opex of one pod / annual opex of one ALS ambulance   ASSUMPTION
    hub_opex: float = 0.03                 # annual opex of one hub (dock, comms) / same reference       ASSUMPTION
    pod_capex: float = 0.35                # pod up-front cost / annual opex of one ALS ambulance        ASSUMPTION
    hub_capex: float = 0.10                # hub up-front cost / same reference                          ASSUMPTION
    life_years: float = 8.0                # service life of a pod                                  ASSUMPTION
    # ---- health benefit
    life_expectancy_yr: float = 12.0       # post-arrest life expectancy                            course scaffold ASSUMPTION
    qol: float = 0.85                      # quality-of-life weight                                 course scaffold ASSUMPTION
    t_cpr_min: float = 3.0                 # collapse -> bystander CPR                              DispatchTimes default (ASSUMPTION)


# name -> (unit, source). SOURCE is a docs/references.json key, a repo document, or ASSUMPTION.
INPUT_TABLE: dict[str, tuple[str, str]] = {
    "zone_area_km2": ("km^2", "ASSUMPTION"),
    "coverage_fraction": ("-", "design choice (scenario)"),
    "hotspot_factor": ("-", "ASSUMPTION (scenario)"),
    "pop_density_per_km2": ("persons/km^2", "ASSUMPTION"),
    "ohca_incidence_per_100k_yr": ("events/100k/yr", "ASSUMPTION (unverified order of magnitude)"),
    "suspected_call_factor": ("-", "ASSUMPTION"),
    "hub_radius_m": ("m", "README RQ3 clinical break-even"),
    "route_factor": ("-", "docs/OSM_ROUTES.md (Mumbai OSM geometry)"),
    "v_amr_mps": ("m/s", "measured: analytics/aed_delivery_real_trials.csv"),
    "turnaround_h": ("h", "ASSUMPTION"),
    "p_wait_max": ("-", "ASSUMPTION"),
    "p_standdown": ("-", "ASSUMPTION"),
    "h_als_per_call": ("vehicle-h", "ASSUMPTION"),
    "pod_opex": ("ALS vehicle-yr/yr", "ASSUMPTION"),
    "hub_opex": ("ALS vehicle-yr/yr", "ASSUMPTION"),
    "pod_capex": ("ALS vehicle-yr", "ASSUMPTION"),
    "hub_capex": ("ALS vehicle-yr", "ASSUMPTION"),
    "life_years": ("yr", "ASSUMPTION"),
    "life_expectancy_yr": ("yr", "course scaffold ASSUMPTION"),
    "qol": ("-", "course scaffold ASSUMPTION"),
    "t_cpr_min": ("min", "ASSUMPTION (clinical.DispatchTimes)"),
}


# ------------------------------------------------------------------ queueing
def erlang_c(c: int, a: float) -> float:
    """Probability that an arriving call must wait in an M/M/c queue with offered load ``a = lambda/mu``."""
    if c < 1:
        raise ValueError("c must be >= 1")
    if a < 0:
        raise ValueError("offered load must be >= 0")
    if a >= c:
        return 1.0
    if a == 0:
        return 0.0
    # numerically stable: sum a^k/k! by recurrence
    term, s = 1.0, 1.0
    for k in range(1, c):
        term *= a / k
        s += term
    top = term * a / c * c / (c - a)           # a^c/c! * c/(c-a)
    return top / (s + top)


def mmc_wait(lam: float, mu: float, c: int) -> dict[str, float]:
    """M/M/c metrics. ``lam``, ``mu`` per hour. ``Wq`` and ``W`` in hours; unstable queues return ``inf``."""
    if lam < 0 or mu <= 0:
        raise ValueError("need lam >= 0 and mu > 0")
    rho = lam / (c * mu)
    if rho >= 1.0:
        return {"rho": rho, "p_wait": 1.0, "Wq_h": math.inf, "W_h": math.inf, "Lq": math.inf}
    pw = erlang_c(c, lam / mu)
    wq = pw / (c * mu - lam)
    return {"rho": rho, "p_wait": pw, "Wq_h": wq, "W_h": wq + 1.0 / mu, "Lq": lam * wq}


def pods_needed(lam: float, mu: float, p_wait_max: float, c_max: int = 50) -> int:
    for c in range(1, c_max + 1):
        if lam / (c * mu) < 1.0 and erlang_c(c, lam / mu) <= p_wait_max:
            return c
    return c_max


# --------------------------------------------------------------- the model
def evaluate(x: Inputs) -> dict[str, float]:
    """All dimensionless results for one input set."""
    hub_area = 3 * math.sqrt(3) / 2 * (x.hub_radius_m / 1000.0) ** 2         # regular hexagon of circumradius r
    served_km2 = x.zone_area_km2 * x.coverage_fraction
    hubs = max(1, math.ceil(served_km2 / hub_area))
    population = served_km2 * x.pop_density_per_km2
    ohca_per_yr = population * x.ohca_incidence_per_100k_yr / 1e5 * x.hotspot_factor
    calls_per_day = ohca_per_yr * x.suspected_call_factor / DAYS_PER_YEAR

    one_way_h = x.hub_radius_m * x.route_factor / x.v_amr_mps / 3600.0
    service_h = 2.0 * one_way_h + x.turnaround_h                              # out, back, recharge
    mu = 1.0 / service_h
    lam_hub = calls_per_day / hubs / 24.0
    c = pods_needed(lam_hub, mu, x.p_wait_max)
    q = mmc_wait(lam_hub, mu, c)
    q1 = mmc_wait(lam_hub, mu, 1)

    pods = hubs * c
    h_reclaimed = calls_per_day * x.p_standdown * x.h_als_per_call            # vehicle-h/day
    c_base = h_reclaimed * DAYS_PER_YEAR / HOURS_PER_VEHICLE_YEAR             # ALS vehicle-years / yr
    c_robot = pods * x.pod_opex + hubs * x.hub_opex
    kappa = c_robot / c_base if c_base > 0 else math.inf
    d_opex = c_base - c_robot
    capex = pods * x.pod_capex + hubs * x.hub_capex
    t_pay_months = capex / d_opex * 12.0 if d_opex > 0 else math.inf
    p_pay = capex / (x.life_years * d_opex) if d_opex > 0 else math.inf

    # health lens: the pod is dispatched with the ambulance, so the first shock is min(t_pod, t_ambulance);
    # expected survival gain over the Naess urban response-time distribution (Larsen model, shared CPR and ACLS times)
    d = DispatchTimes()
    t_amb_q = d.collapse_to_call_min + NAESS_URBAN.quantile_grid(400) + d.ems_arrival_to_shock_min
    t_amb = d.collapse_to_call_min + NAESS_URBAN.median_min + d.ems_arrival_to_shock_min
    t_pod = d.collapse_to_call_min + d.call_to_alert_min + one_way_h * 60.0 + d.handoff_min
    t_cpr = np.minimum(x.t_cpr_min, t_amb_q)
    ds = float(np.mean(delta_survival(LarsenModel(), np.minimum(t_pod, t_amb_q), t_amb_q, t_cpr, t_amb_q)))
    dq = ds * x.life_expectancy_yr * x.qol
    qaly_yr = ohca_per_yr * dq
    return {
        "hubs": hubs, "pods_per_hub": c, "pods": pods, "population": population,
        "ohca_per_year": ohca_per_yr, "calls_per_day": calls_per_day,
        "lambda_per_hub_h": lam_hub, "mu_per_h": mu, "service_time_min": service_h * 60.0,
        "rho": q["rho"], "p_wait": q["p_wait"], "Wq_min": q["Wq_h"] * 60.0,
        "rho_mm1": q1["rho"], "Wq_mm1_min": q1["Wq_h"] * 60.0,
        "H_reclaimed_veh_h_per_day": h_reclaimed, "delta_FTE": h_reclaimed * DAYS_PER_YEAR / HOURS_PER_FTE,
        "C_baseline": c_base, "C_robot": c_robot, "kappa": kappa, "NCRI": 1.0 - kappa,
        "meets_kappa_target": bool(kappa <= KAPPA_TARGET), "capex": capex,
        "T_payback_months": t_pay_months, "P_payback": p_pay,
        "t_first_shock_pod_min": t_pod, "t_first_shock_ambulance_min": t_amb,
        "delta_survival": ds, "delta_QALY_per_encounter": dq, "QALY_per_year": qaly_yr,
        "QALY_per_ALS_vehicle_year_of_cost": qaly_yr / c_robot if c_robot > 0 else math.inf,
    }


# ------------------------------------------------------------- sensitivity
def sweep(x: Inputs, name: str, values) -> pd.DataFrame:
    rows = []
    for v in values:
        r = evaluate(replace(x, **{name: float(v)}))
        rows.append({name: float(v), "kappa": r["kappa"], "NCRI": r["NCRI"], "P_payback": r["P_payback"],
                     "T_payback_months": r["T_payback_months"], "Wq_min": r["Wq_min"], "rho": r["rho"]})
    return pd.DataFrame(rows)


def break_even(x: Inputs, name: str, target: str = "kappa", level: float = 1.0,
               lo: float | None = None, hi: float | None = None) -> float:
    """Value of input ``name`` at which result ``target`` equals ``level`` (``nan`` if not bracketed)."""
    v0 = getattr(x, name)
    lo = v0 * 0.01 if lo is None else lo
    hi = v0 * 100.0 if hi is None else hi

    def f(v: float) -> float:
        r = evaluate(replace(x, **{name: v}))[target]
        return (r if math.isfinite(r) else 1e12) - level

    grid = np.geomspace(lo, hi, 400)
    vals = [f(v) for v in grid]
    for i in range(len(grid) - 1):
        if vals[i] == 0:
            return float(grid[i])
        if vals[i] * vals[i + 1] < 0:
            return float(brentq(f, grid[i], grid[i + 1], xtol=1e-12))
    return float("nan")


def tornado(x: Inputs, names: list[str], frac: float = 0.30, target: str = "kappa") -> pd.DataFrame:
    base = evaluate(x)[target]
    rows = []
    for n in names:
        v0 = getattr(x, n)
        lo = evaluate(replace(x, **{n: v0 * (1 - frac)}))[target]
        hi = evaluate(replace(x, **{n: v0 * (1 + frac)}))[target]
        rows.append({"input": n, "base": base, "minus_30pct": lo, "plus_30pct": hi, "swing": abs(hi - lo)})
    return pd.DataFrame(rows).sort_values("swing", ascending=False, ignore_index=True)


# -------------------------------------------------------------------- FMEA
OCCURRENCE_SCALE = [(0.001, 1), (0.005, 2), (0.01, 3), (0.02, 4), (0.05, 5), (0.10, 6), (0.20, 7), (0.30, 8), (0.50, 9)]


def occurrence_score(rate: float) -> int:
    """Map a measured per-mission failure rate to a 1-10 occurrence score (scale documented in docs/CAPSTONE.md)."""
    for upper, score in OCCURRENCE_SCALE:
        if rate < upper:
            return score
    return 10


# Severity and detection (pre, post) are engineering judgement. Detection only changes where the mitigation adds a way to see the failure\n# (the safety filter logs every intervention); a prevention measure changes occurrence, which comes from the Monte Carlo.
FMEA_MODES = [
    {"mode": "Wheel slip or entrapment on a wet kerb", "effect": "Late arrival or stranded pod: delayed first shock",
     "S": 8, "D": (5, 5), "pre": ("baseline_fast", "not_success_slippery"), "post": ("proposed", "not_success_slippery"),
     "mitigation": "Climb speed from the kerb table at a conservative friction belief; planner replans around the obstruction",
     "shown_in": "Both"},
    {"mode": "Payload shock above 3 g", "effect": "AED capacitor or electrodes damaged, device unusable at the scene",
     "S": 9, "D": (5, 5), "pre": ("baseline_fast", "shock_over_3g"), "post": ("proposed", "shock_over_3g"),
     "mitigation": "FSM CURB_SIDEWALK speed cap from the measured kerb table and a dynamic-window planner that slows for the kerb",
     "shown_in": "Both"},
    {"mode": "Pedestrian or obstacle collision", "effect": "Injury to a bystander, mission abort, loss of trust",
     "S": 9, "D": (6, 3), "pre": ("baseline_unshielded", "collision"), "post": ("proposed", "collision"),
     "mitigation": "CBF-style speed-and-separation safety filter on the swept footprint; DWA plans against predicted pedestrian motion",
     "shown_in": "Both"},
    {"mode": "Stall: no forward progress for 15 s", "effect": "Mission aborted; ambulance remains the only response",
     "S": 7, "D": (4, 4), "pre": ("baseline_fast", "stall"), "post": ("proposed", "stall"),
     "mitigation": "Dynamic-window planner replaces constant-speed pure pursuit; FSM keeps a creep speed in every state",
     "shown_in": "Simulation"},
    {"mode": "Rollover on an oblique kerb approach", "effect": "Loss of vehicle and payload, possible injury",
     "S": 10, "D": (5, 5), "pre": ("baseline_fast", "rollover"), "post": ("proposed", "rollover"),
     "mitigation": "Roll-limit monitored as the stability index (18 deg); speed cap in CURB_SIDEWALK",
     "shown_in": "Simulation"},
]


def _rate(df: pd.DataFrame, label: str, what: str) -> tuple[float, int]:
    sub = df[df["controller"] == label]
    if what == "not_success_slippery":
        sub = sub[sub["condition"] == "slippery"]
        return float((sub["success"] == 0).mean()), len(sub)
    if what == "shock_over_3g":
        return float(sub["shock_over_3g"].mean()), len(sub)
    return float((sub["outcome"] == what).mean()), len(sub)


def fmea_table(trials: pd.DataFrame) -> pd.DataFrame:
    """FMEA with RPN = S x O x D before and after mitigation; occurrence comes from the measured rates."""
    rows = []
    for m in FMEA_MODES:
        r_pre, n_pre = _rate(trials, *m["pre"])
        r_post, n_post = _rate(trials, *m["post"])
        o_pre, o_post = occurrence_score(r_pre), occurrence_score(r_post)
        d_pre, d_post = m["D"]
        rows.append({
            "failure_mode": m["mode"], "effect": m["effect"], "S": m["S"], "O_pre": o_pre, "D_pre": d_pre,
            "RPN_pre": m["S"] * o_pre * d_pre, "rate_pre": r_pre, "n_pre": n_pre, "pre_arm": m["pre"][0],
            "mitigation": m["mitigation"], "O_post": o_post, "D_post": d_post,
            "RPN_post": m["S"] * o_post * d_post, "rate_post": r_post, "n_post": n_post, "post_arm": m["post"][0],
            "shown_in": m["shown_in"],
            "RPN_reduction_pct": 100.0 * (1.0 - m["S"] * o_post * d_post / (m["S"] * o_pre * d_pre)),
        })
    return pd.DataFrame(rows)


def with_measured_speed(x: Inputs, trials: pd.DataFrame, label: str = "proposed") -> Inputs:
    """Replace the speed assumption by the mean speed measured in the Monte Carlo (successful runs)."""
    v = trials[(trials["controller"] == label) & (trials["success"] == 1)]["mean_speed_mps"].mean()
    return replace(x, v_amr_mps=float(v)) if np.isfinite(v) and v > 0 else x


def input_records(x: Inputs) -> list[dict]:
    d = asdict(x)
    return [{"input": k, "value": d[k], "unit": INPUT_TABLE[k][0], "source": INPUT_TABLE[k][1]} for k in INPUT_TABLE]


# ----------------------------------------------------- what would have to be true
# Scenarios are labelled design alternatives. Their inputs are ASSUMPTIONS: the point of the table is to show how far the
# inputs must move, not to claim that any of them is the likely case.
SCENARIOS: dict[str, dict] = {
    "base: uniform grid over the zone": {},
    "siting: pods only in the densest 10 % of the zone, 4x arrest rate there": {"coverage_fraction": 0.10, "hotspot_factor": 4.0},
    "siting + unmanned pod at 3 % and shared hubs at 0.5 % of an ambulance-year": {
        "coverage_fraction": 0.10, "hotspot_factor": 4.0, "pod_opex": 0.03, "hub_opex": 0.005},
    "siting + cheap pods + half of suspected calls resolved without an ambulance": {
        "coverage_fraction": 0.10, "hotspot_factor": 4.0, "pod_opex": 0.03, "hub_opex": 0.005, "p_standdown": 0.5},
    "dual-use fleet: AED added to existing delivery robots (marginal 0.5 % per pod, no new hubs), siting, half stood down": {
        "coverage_fraction": 0.10, "hotspot_factor": 4.0, "pod_opex": 0.005, "hub_opex": 0.0, "p_standdown": 0.5},
}


def scenario_table(x: Inputs) -> pd.DataFrame:
    rows = []
    for name, over in SCENARIOS.items():
        r = evaluate(replace(x, **over))
        rows.append({"scenario": name, "pods": r["pods"], "H_reclaimed_veh_h_per_day": r["H_reclaimed_veh_h_per_day"],
                     "kappa": r["kappa"], "NCRI": r["NCRI"], "T_payback_months": r["T_payback_months"],
                     "meets_kappa_target": r["meets_kappa_target"]})
    return pd.DataFrame(rows)


def required_standdown_calls_per_pod_day(x: Inputs, target: float = KAPPA_TARGET) -> float:
    """Stand-down calls per pod per day needed for ``kappa <= target`` at this pod and hub cost (hubs shared one per pod).

    Each pod-year costs ``pod_opex + hub_opex`` ambulance-years; each stand-down call frees ``h_als_per_call`` ambulance-hours.
    A pod that serves fewer calls than this cannot meet the target whatever the arrest rate of the zone it sits in.
    """
    cost_veh_years = x.pod_opex + x.hub_opex
    hours_needed_per_day = cost_veh_years * HOURS_PER_VEHICLE_YEAR / DAYS_PER_YEAR / target
    return hours_needed_per_day / x.h_als_per_call
