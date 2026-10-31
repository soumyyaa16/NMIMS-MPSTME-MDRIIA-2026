"""Capstone deliverables: FSM, 500 Hz telemetry, business model, standards, and the zero-currency rule.

The synthetic frames below are unit-test fixtures for the arithmetic; no test result is reported anywhere.
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aedrover.capstone.business import (
    SCENARIOS,
    Inputs,
    break_even,
    erlang_c,
    evaluate,
    fmea_table,
    mmc_wait,
    occurrence_score,
    pods_needed,
    required_standdown_calls_per_pod_day,
    scenario_table,
    sweep,
    tornado,
)
from aedrover.capstone.fsm import STATES, FSMParams, FSMSupervisor, MissionFSM, State
from aedrover.capstone.monte_carlo import CONDITIONS, CONTROLLERS, McJob, run_mc_job, ttfs_minutes
from aedrover.capstone.standards import braking_distance, standards_report
from aedrover.capstone.telemetry import COLUMNS, summarise_frame
from aedrover.clinical.economics import payback_months
from aedrover.nav.base import make_controller
from aedrover.sim.scenario import sample_scenario

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------- FSM
def _drive(fsm: MissionFSM, sc, dt=0.05, v=1.5):
    fsm.reset(sc)
    t, x, seen = 0.0, 0.0, []
    while x < sc.x_goal:
        fsm.update({"t": t, "x": x})
        if not seen or seen[-1] != fsm.state:
            seen.append(fsm.state)
        t, x = t + dt, x + v * dt
    return seen


def test_fsm_visits_the_documented_states_in_order_on_a_kerb_route():
    sc = sample_scenario("kerb", 70001)
    seen = _drive(MissionFSM(), sc)
    assert seen[0] is State.DISPATCH and seen[-1] is State.ARRIVE
    assert State.CURB_SIDEWALK in seen
    assert seen.count(State.CURB_SIDEWALK) == 2                    # kerb up and kerb down
    assert all(s.value in STATES for s in seen)


def test_fsm_without_a_kerb_never_enters_the_curb_state():
    sc = sample_scenario("flat_clear", 70001)
    assert State.CURB_SIDEWALK not in _drive(MissionFSM(), sc)


def test_fsm_records_a_reason_for_every_transition():
    fsm = MissionFSM()
    _drive(fsm, sample_scenario("kerb", 70002))
    assert fsm.transitions and all(t.reason for t in fsm.transitions)
    assert [t.dst for t in fsm.transitions][-1] == "ARRIVE"


def test_fsm_speed_caps_ramp_and_brake():
    fsm = MissionFSM(FSMParams())
    fsm.reset(sample_scenario("flat_clear", 1))
    assert fsm.speed_cap({"t": 0.0, "x": 0.0}) == pytest.approx(FSMParams().v_floor)
    fsm.update({"t": 0.0, "x": 0.0})
    assert fsm.speed_cap({"t": 1.0, "x": 1.0}) == pytest.approx(FSMParams().a_launch + FSMParams().v_floor)
    near = {"t": 30.0, "x": fsm._sc.x_goal - 1.0}
    fsm.update(near)
    assert fsm.state is State.ARRIVE
    assert fsm.speed_cap(near) < fsm.speed_cap({"t": 30.0, "x": fsm._sc.x_goal - 2.5})


def test_supervisor_never_raises_the_planner_speed():
    class Fast:
        name = "fast"

        def reset(self, env):
            pass

        def act(self, obs):
            return 9.9, 0.1

    class Env:
        scenario = sample_scenario("flat_clear", 3)

        class veh0:
            name = "nominal"
            wheel_radius = 0.15

    sup = FSMSupervisor(Fast())
    sup.reset(Env())
    v, delta = sup.act({"t": 0.0, "x": 0.0})
    assert v < 9.9 and delta == 0.1
    assert sup.name == "fsm_fast"


def test_fsm_dwa_is_registered():
    assert make_controller("fsm_dwa").name == "fsm_dwa"


# ----------------------------------------------------------------- telemetry
def test_summarise_frame_computes_rmse_and_effort_exactly():
    n = 1001
    df = pd.DataFrame({c: np.zeros(n) for c in COLUMNS})
    df["t_s"] = np.arange(n) * 0.002
    df["fsm_state"] = "ROAD_CRUISE"
    df["e_y_m"] = 0.1
    df["tau_fl_Nm"] = 2.0
    df["tau_steer_fl_Nm"] = 1.0
    s = summarise_frame(df, 0.002)
    assert s.log_rate_hz == pytest.approx(500.0)
    assert s.rmse_m == pytest.approx(0.1)
    assert s.effort_Nm2s == pytest.approx(n * (4.0 + 1.0) * 0.002)


def test_handover_rows_are_excluded_from_rmse_and_effort():
    n = 100
    df = pd.DataFrame({c: np.zeros(n) for c in COLUMNS})
    df["t_s"] = np.arange(n) * 0.002
    df["fsm_state"] = ["ROAD_CRUISE"] * 50 + ["AED_HANDOVER"] * 50
    df["e_y_m"] = [0.2] * 50 + [5.0] * 50
    df["tau_fl_Nm"] = 1.0
    s = summarise_frame(df, 0.002)
    assert s.rmse_m == pytest.approx(0.2)
    assert s.effort_Nm2s == pytest.approx(50 * 0.002)


# ------------------------------------------------------------------ queueing
def test_erlang_c_matches_textbook_values():
    assert erlang_c(1, 0.6) == pytest.approx(0.6)                 # M/M/1: P(wait) = rho
    assert erlang_c(2, 1.0) == pytest.approx(1.0 / 3.0)           # M/M/2 at offered load 1
    assert erlang_c(3, 0.0) == 0.0 and erlang_c(2, 2.5) == 1.0


def test_mm1_wait_matches_closed_form_and_mmc_with_c1():
    lam, mu = 3.0, 5.0
    q = mmc_wait(lam, mu, 1)
    rho = lam / mu
    assert q["Wq_h"] == pytest.approx(rho / (mu - lam))
    assert q["W_h"] == pytest.approx(q["Wq_h"] + 1.0 / mu)
    assert q["Lq"] == pytest.approx(lam * q["Wq_h"])                 # Little's law


def test_unstable_queue_reports_infinite_wait():
    assert math.isinf(mmc_wait(10.0, 2.0, 2)["Wq_h"])


def test_pods_needed_grows_with_load_and_meets_the_wait_target():
    mu = 2.0
    c = pods_needed(5.0, mu, 0.05)
    assert c >= 4 and erlang_c(c, 5.0 / mu) <= 0.05 and erlang_c(c - 1, 5.0 / mu) > 0.05


# ------------------------------------------------------------------ business
def test_ncri_is_one_minus_kappa_and_kappa_is_a_cost_ratio():
    r = evaluate(Inputs())
    assert r["NCRI"] == pytest.approx(1.0 - r["kappa"])
    assert r["kappa"] == pytest.approx(r["C_robot"] / r["C_baseline"])


def test_payback_agrees_with_the_course_formula_when_it_exists():
    x = Inputs(ohca_incidence_per_100k_yr=Inputs().ohca_incidence_per_100k_yr * 2000.0)
    r = evaluate(x)
    assert r["kappa"] < 1.0
    expected = payback_months(r["kappa"], r["capex"] / r["C_baseline"])
    assert r["T_payback_months"] == pytest.approx(float(expected))
    assert r["P_payback"] == pytest.approx(r["capex"] / (x.life_years * (r["C_baseline"] - r["C_robot"])))


def test_no_payback_when_the_robot_costs_more_than_it_saves():
    r = evaluate(Inputs(p_standdown=1e-6))
    assert r["kappa"] > 1.0 and math.isinf(r["T_payback_months"]) and math.isinf(r["P_payback"])


def test_h_reclaimed_scales_linearly_with_standdown_share():
    a, b = evaluate(Inputs(p_standdown=0.1)), evaluate(Inputs(p_standdown=0.2))
    assert b["H_reclaimed_veh_h_per_day"] == pytest.approx(2 * a["H_reclaimed_veh_h_per_day"])


def test_break_even_is_a_root_of_kappa():
    x = Inputs()
    v = break_even(x, "ohca_incidence_per_100k_yr", "kappa", 1.0, lo=x.ohca_incidence_per_100k_yr,
                   hi=x.ohca_incidence_per_100k_yr * 1e5)
    assert math.isfinite(v)
    from dataclasses import replace
    assert evaluate(replace(x, ohca_incidence_per_100k_yr=v))["kappa"] == pytest.approx(1.0, rel=1e-3)


def test_break_even_is_nan_when_unreachable():
    assert math.isnan(break_even(Inputs(), "p_standdown", "kappa", 1.0, lo=0.01, hi=1.0))


def test_sweep_and_tornado_have_the_expected_shape():
    x = Inputs()
    s = sweep(x, "p_standdown", [0.05, 0.1, 0.2])
    assert list(s["p_standdown"]) == [0.05, 0.1, 0.2] and s["kappa"].is_monotonic_decreasing
    t = tornado(x, ["p_standdown", "pod_opex"])
    assert t["swing"].is_monotonic_decreasing and set(t["input"]) == {"p_standdown", "pod_opex"}


def test_scenarios_follow_their_stated_inputs_and_none_is_hidden():
    t = scenario_table(Inputs())
    assert len(t) == len(SCENARIOS) and t["kappa"].iloc[0] == pytest.approx(evaluate(Inputs())["kappa"])
    assert t["kappa"].iloc[-1] < t["kappa"].iloc[0]                      # every added assumption is an improvement, and is named
    assert not t["meets_kappa_target"].any()                             # the repository reports it when no scenario reaches the target


def test_required_standdown_demand_scales_with_cost_and_target():
    x = Inputs()
    base = required_standdown_calls_per_pod_day(x)
    assert required_standdown_calls_per_pod_day(x, 0.5) == pytest.approx(base / 2)
    assert required_standdown_calls_per_pod_day(Inputs(pod_opex=0.05, hub_opex=0.0)) < base


def test_occurrence_score_is_monotone_and_bounded():
    rates = [0.0, 0.0005, 0.003, 0.015, 0.04, 0.08, 0.15, 0.25, 0.4, 0.9]
    scores = [occurrence_score(r) for r in rates]
    assert scores == sorted(scores) and scores[0] == 1 and scores[-1] == 10


def test_fmea_has_at_least_four_modes_and_rpn_is_s_times_o_times_d():
    rng = np.random.default_rng(0)                                  # arithmetic fixture, not a result
    rows = []
    for label in CONTROLLERS:
        for cond in CONDITIONS:
            for _ in range(20):
                rows.append({"controller": label, "condition": cond, "success": int(rng.random() < 0.6),
                             "outcome": rng.choice(["goal", "collision", "stall", "rollover"]),
                             "shock_over_3g": int(rng.random() < 0.2)})
    f = fmea_table(pd.DataFrame(rows))
    assert len(f) >= 4
    assert (f["RPN_pre"] == f["S"] * f["O_pre"] * f["D_pre"]).all()
    assert (f["RPN_post"] == f["S"] * f["O_post"] * f["D_post"]).all()


# ------------------------------------------------------------------ standards
def test_braking_distance_formula():
    assert braking_distance(2.0, 1.0, 0.05) == pytest.approx(4.0 / (2 * 9.81) + 0.1)


@pytest.mark.slow
def test_standards_report_measures_the_emergency_stop():
    rep = standards_report()
    row = rep[rep["requirement"] == "Emergency braking deceleration"].iloc[0]
    assert row["status"] == "MET"
    assert set(rep["status"]) <= {"MET", "DEVIATION", "NOT MEASURED"}


# ------------------------------------------------------------------- pipeline
def test_ttfs_decreases_with_speed_and_is_undefined_without_progress():
    assert ttfs_minutes(2.0) < ttfs_minutes(1.0)
    assert math.isnan(ttfs_minutes(0.0)) and math.isnan(ttfs_minutes(float("nan")))


@pytest.mark.slow
def test_one_monte_carlo_run_logs_at_500_hz_with_the_recorded_seed():
    row = run_mc_job(McJob("free_flow", "proposed", 70123, handover_s=1.0))
    assert row["seed"] == 70123 and row["controller"] == "proposed" and row["condition"] == "free_flow"
    assert row["log_rate_hz"] == pytest.approx(500.0, rel=1e-6)
    assert row["success"] in (0, 1) and row["tracking_rmse_m"] >= 0.0 and row["control_effort_Nm2s"] > 0.0
    assert row["fsm_transitions"] >= 2


# ----------------------------------------------------------------- zero currency
CURRENCY = [r"\\\$", r"\bRs\b\.?\s*\d+", r"\bINR\b\s*\d+", r"\bUSD\b\s*\d+", r"\bEUR\b\s*\d+", r"\brupees\b", r"\bdollars\b"]


def test_capstone_sources_contain_no_currency_amounts():
    files = list((ROOT / "src" / "aedrover" / "capstone").glob("*.py")) + [ROOT / "analytics" / "aed_delivery_roi.py"]
    bad = [f"{f.name}:{i}" for f in files for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
           if any(re.search(p, line, re.IGNORECASE) for p in CURRENCY)]
    assert not bad, bad
