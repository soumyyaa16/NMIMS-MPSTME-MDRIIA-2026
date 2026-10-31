"""surgical_or_economics.py - CSBS techno-economic model (v2): robotic vs frame-based stereotactic DBS, dimensionless money.
    python analytics\\surgical_or_economics.py
Money is a RATIO to C_baseline = cost of one frame-based DBS case (= 1). No currency in any result.
Every input is tagged  CITED (a paper, number copied from it) / DERIVED (calculated from cited numbers, formula shown) /
ASSUMPTION (we could not source it: it is varied in the sensitivity graph and its break-even is reported) / SIM (our own MuJoCo result).
Sources: [O26] Owens et al., Stereotact Funct Neurosurg 2026 (Utah, 256 bilateral DBS, 2012-2023; capital cost EXCLUDED)
         [C18] Childers & Maggard-Gibbons, JAMA Surg 2018;153(4):e176233 (OR cost per minute, California FY2014)
         [C18b] Childers & Maggard-Gibbons, JAMA 2018;320(8):835-836 (robot acquisition vs maintenance per procedure)
         [K22] Koester et al., J Clin Outcomes Manag 2022;29(6):208-213 (neurosurgical turnover, Vanderbilt)
         [R25] R2 Surgical vendor blog, 27 Jun 2025 (reseller, NOT peer reviewed): neurosurgery robots USD 0.6-1.5 M
Outputs: analytics/economics_inputs.csv, economics_results.csv, docs/figures/fig_sensitivity.png, fig_volume.png, fig_queue.png, fig_tornado.png"""
import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# name: (value, unit, tag, source / formula)
INPUTS = {
    "N_total":   (256,    "patients",      "CITED",      "[O26] bilateral DBS patients at one centre"),
    "years":     (12,     "years",         "CITED",      "[O26] 2012-2023"),
    "T_man":     (220,    "min",           "CITED",      "[O26] median operative time, frame + C-arm (cheapest frame method)"),
    "T_rob":     (129,    "min",           "CITED",      "[O26] median operative time, robot + intra-op CT"),
    "c_man":     (106.6,  "% of ref",      "CITED",      "[O26] median total cost, frame + C-arm (capital excluded)"),
    "c_rob":     (117.5,  "% of ref",      "CITED",      "[O26] median total cost, robot + iCT (capital excluded; vs C-arm p = 0.37, not significant)"),
    "fac_pct":   (20.8,   "% of ref",      "CITED",      "[O26] facility cost, frame + C-arm"),
    "turn":      (85,     "min",           "CITED",      "[K22] mean neurosurgical turnover (83-88 min across periods)"),
    "cost_min":  (36.795, "USD/min",       "CITED",      "[C18] mean OR cost per minute, California FY2014 (36.14 ambulatory, 37.45 inpatient); only used to size CapEx as a ratio (approximation: treated as the facility share of [O26] total cost)"),
    "capex_usd": (1.05e6, "USD",           "CITED",      "[R25] midpoint of 0.6-1.5 M (reseller blog estimate, not peer reviewed): swept"),
    "life_fac":  (1.639,  "x CapEx",       "DERIVED",    "[C18b] 1 + 663/1038 = 1 + maintenance/acquisition (da Vinci benchmark, not a neuro robot)"),
    "T_life":    (7.0,    "years",         "ASSUMPTION", "useful life, no source: swept 5-10"),
    "m":         (0.5,    "x C_baseline",  "ASSUMPTION", "margin earned by one EXTRA case the freed OR time makes possible: the one number nobody publishes; swept 0-1"),
    "weeks":     (48,     "weeks/year",    "ASSUMPTION", "operating weeks (queue only)"),
    "block":     (2400,   "OR-min/week",   "ASSUMPTION", "one dedicated room, 5 x 8 h (queue only)"),
}


def defaults():
    return {k: v[0] for k, v in INPUTS.items()}


def erlang_c(c, a):
    s, term = 0.0, 1.0
    for k in range(c):
        if k > 0:
            term *= a / k
        s += term
    top = term * (a / c) * (c / (c - a))
    return top / (s + top)


def wq_weeks(lam, mu, c=1):
    return math.inf if lam >= c * mu else erlang_c(c, lam / mu) / (c * mu - lam)


def model(p, N=None):
    N = p["N_total"] / p["years"] if N is None else N                     # DBS cases per year
    dT = p["T_man"] - p["T_rob"]
    H = dT * N / 60.0                                                     # OR-hours reclaimed per year
    ncri = (p["c_man"] - p["c_rob"]) / p["c_man"]                         # per-case cost reduction (capital excluded)
    kappa = 1.0 - ncri
    g = dT / (p["T_rob"] + p["turn"])                                     # extra cases per case, from freed time
    dN = N * g                                                            # extra cases per year the freed time allows
    benefit = N * ncri + dN * p["m"]                                      # Delta OpEx annual (x C_baseline / yr)
    C_b_usd = p["cost_min"] * p["T_man"] / (p["fac_pct"] / p["c_man"])    # USD of one frame case (sizes the CapEx ratio only)
    K = p["capex_usd"] / C_b_usd
    capex = K * p["life_fac"]                                             # purchase + lifetime maintenance, x C_baseline
    if benefit > 0:
        P, Tm = capex / (p["T_life"] * benefit), 12 * capex / benefit
    else:
        P, Tm = math.inf, math.inf
    need = capex / p["T_life"]
    m_floor = -ncri / g if g > 0 else math.inf                            # margin below which each extra case still loses money
    m_star = (need - N * ncri) / dN if dN > 0 else math.inf               # margin needed for P = 1 at this volume
    den = ncri + g * p["m"]
    N_star = need / den if den > 0 else math.inf                          # volume needed for P = 1 at this margin
    lam_w = N / p["weeks"]
    mu_m, mu_r = p["block"] / (p["T_man"] + p["turn"]), p["block"] / (p["T_rob"] + p["turn"])
    return dict(N=N, dT=dT, H=H, ncri=ncri, kappa=kappa, g=g, dN=dN, benefit=benefit, C_b_usd=C_b_usd, K=K, capex=capex, P=P, T_months=Tm,
                m_floor=m_floor, m_star=m_star, N_star=N_star, lam_w=lam_w, mu_m=mu_m, mu_r=mu_r, rho_m=lam_w / mu_m, rho_r=lam_w / mu_r,
                wq_m_h=wq_weeks(lam_w, mu_m) * 7 * 24, wq_r_h=wq_weeks(lam_w, mu_r) * 7 * 24)


FMEA = [  # failure mode, effect, S, O, D, mitigation, S2, O2, D2, shown in
    ("Singular configuration -> joint-speed spike", "arm jerks near a singular pose; needle leaves the plan", 8, 4, 4,
     "adaptive damping lambda = lambda0 (1 - w/w_th), null space, abort at W_ABORT", 8, 2, 3, "Simulation + code (singularity test)"),
    ("Filter phase lag -> overshoot at target", "a fixed low-pass lags the moving hand (0.83 mm tracking error)", 6, 5, 4,
     "learned filter keeps tracking error near 0.06 mm (fixed filter 0.83 mm)", 6, 2, 3, "Simulation (compare_methods, KP=50, N=30 per hand type, 10 RL seeds)"),
    ("Drift dominates tremor", "slow drift (~0.2 mm, not filterable) is the largest error at measured values", 8, 7, 5,
     "reported as a limit; needs feedback (imaging) to remove: future work", 8, 7, 4, "Simulation (compare_methods, KP=50, 10 RL seeds: mean benefit 1-3 % at measured drift, single seeds -5 to +2 %)"),
    ("RL policy fails on an unseen hand", "wrong filter mix, extra error", 7, 3, 6,
     "output is a mix of fixed safe filters; trained on randomised hands, tested on unseen seeds", 7, 2, 4, "Simulation (compare_methods, KP=50, 10 RL seeds: 90-100 % of runs hold < 0.5 mm, 100 % without drift)"),
    ("Feedback delay destabilises the position loop", "latency in tip measurement; error grows x5 at 4 ms and the loop diverges beyond", 9, 5, 4,
     "lower loop gain (KP 100 -> 50); measure real latency before any hardware claim", 9, 3, 3, "Simulation (sensor_sweep, N=10, KP=100: stable up to 2 ms, unstable from 6 ms)"),
    ("Demand below break-even volume", "robot never pays back; capacity idle", 7, 8, 3,
     "pool robot across DBS / biopsy / sEEG; report N* before purchase", 7, 5, 3, "This script (N*, fig_volume)"),
    ("Image-to-robot registration error", "whole plan shifted: not visible to the simulation", 9, 5, 6,
     "NONE in this work: stated limitation", 9, 5, 6, "NOT shown"),
]


def solve(f, lo, hi):
    a, b = f(lo), f(hi)
    if not (np.isfinite(a) and np.isfinite(b)) or a * b > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        v = f(mid)
        if a * v <= 0:
            hi, b = mid, v
        else:
            lo, a = mid, v
    return 0.5 * (lo + hi)


def main():
    p = defaults()
    m0 = model(p)
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "analytics" / "economics_inputs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["input", "value", "unit", "tag", "source_or_formula"])
        for k, (v, u, t, s) in INPUTS.items():
            w.writerow([k, v, u, t, s])
    rows = [("DBS cases / year  N = N_total / years", m0["N"], "cases/yr"), ("minutes saved per case  dT = T_man - T_rob", m0["dT"], "min"),
            ("H_reclaimed = dT * N / 60", m0["H"], "OR-h/yr"), ("NCRI = (c_man - c_rob) / c_man", m0["ncri"], "-"), ("kappa = C_robot / C_baseline", m0["kappa"], "-"),
            ("extra cases the freed time allows  dN = N * dT/(T_rob+turn)", m0["dN"], "cases/yr"), ("annual benefit  N*NCRI + dN*m", m0["benefit"], "x C_baseline / yr"),
            ("C_baseline (USD, only to size CapEx)", m0["C_b_usd"], "USD"), ("CapEx ratio K = CapEx / C_baseline", m0["K"], "x C_baseline"),
            ("CapEx + lifetime maintenance", m0["capex"], "x C_baseline"), ("P_payback = CapEx_total / (T_life * benefit)", m0["P"], "-"),
            ("T_payback", m0["T_months"], "months"), ("margin floor m_floor (extra case must beat this to be worth doing)", m0["m_floor"], "x C_baseline"),
            ("break-even margin m* at this volume (P = 1)", m0["m_star"], "x C_baseline"), ("break-even volume N* at m = 0.5 (P = 1)", m0["N_star"], "cases/yr"),
            ("lambda", m0["lam_w"], "cases/week"), ("mu manual", m0["mu_m"], "cases/week"), ("mu robot", m0["mu_r"], "cases/week"),
            ("rho manual", m0["rho_m"], "-"), ("rho robot", m0["rho_r"], "-"), ("Wq manual", m0["wq_m_h"], "hours"), ("Wq robot", m0["wq_r_h"], "hours")]
    with open(ROOT / "analytics" / "economics_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value", "unit"])
        for a, b, c in rows:
            w.writerow([a, round(b, 5) if np.isfinite(b) else "inf", c])
    print("INPUTS")
    for k, (v, u, t, s) in INPUTS.items():
        print(f"  {k:9s} = {v:<10} {u:14s} [{t}] {s}")
    print("\nRESULTS (DBS, capital included via CapEx ratio)")
    for a, b, c in rows:
        print(f"  {a:66s} {b:12.4f} {c}")
    # tremor error budget from the simulation
    print("\nERROR BUDGET: whole-system DBS radial error 1.06 mm (Giridharan 2022) / 1.01-1.32 mm (Barcelona 2025);"
          " tremor-induced hold error 0.041-0.045 mm (ideal arm, tuned controller, N=30 per hand type) -> 3.9-4.2 % of the budget;"
          " the learned filter removes 0.005-0.008 mm = 0.5-0.8 %; measured-scale drift alone leaves 0.28 mm = 26 %.")

    # sensitivity on m (most uncertain for P), and on robot cost ratio (most uncertain for NCRI)
    def Pm(m):
        return model(dict(p, m=m))["P"]
    ms = np.linspace(0.0, 1.0, 201)
    print("\nSENSITIVITY on m (+-30 % around 0.5):", [(round(x, 2), round(Pm(x), 2)) for x in (0.35, 0.5, 0.65)])
    c_star = p["c_man"]
    N_stars = {m_: model(dict(p, m=m_))["N_star"] for m_ in (0.35, 0.5, 0.65, 0.8)}
    print("break-even volume N* by margin m:", {k: round(v, 1) for k, v in N_stars.items()})

    # tornado on P
    tor = []
    for k in ("capex_usd", "T_life", "m", "T_rob", "c_rob", "turn", "life_fac", "N_total"):
        lo, hi = model(dict(p, **{k: p[k] * 0.7}))["P"], model(dict(p, **{k: p[k] * 1.3}))["P"]
        tor.append((k, lo, hi))
    tor.sort(key=lambda t: -abs(min(t[1], 8) - min(t[2], 8)))
    print("\nTORNADO P_payback at -30 % / +30 %:")
    for k, a, b in tor:
        print(f"  {k:10s} {a:8.2f} / {b:8.2f}")
    print("\nFMEA (S, O, D are team judgements)")
    for f in FMEA:
        print(f"  {f[0][:46]:46s} RPN {f[2]*f[3]*f[4]:3d} -> {f[6]*f[7]*f[8]:3d}  [{f[9]}]")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = ROOT / "docs" / "figures"
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ms, [min(Pm(x), 8) for x in ms], "C0", lw=2)
    ax[0].axhline(1, color="grey", ls="--")
    ax[0].axvspan(0.35, 0.65, color="C0", alpha=0.12, label="+-30 % of m = 0.5")
    ax[0].axvline(m0["m_floor"], color="C1", ls=":", label=f"margin floor {m0['m_floor']:.2f}")
    if m0["m_star"] < 1.2:
        ax[0].axvline(m0["m_star"], color="C3", ls=":", label=f"break-even m* = {m0['m_star']:.2f}")
    ax[0].set(ylim=(0, 6), xlabel="margin m of one extra case (assumption)", ylabel="P_payback  (<1 = pays back in life)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    cr = np.linspace(0.7 * p["c_rob"], 1.3 * p["c_rob"], 100)
    ax[1].plot(cr, [model(dict(p, c_rob=x))["ncri"] for x in cr], "C2", lw=2)
    ax[1].axhline(0, color="grey", ls="--")
    ax[1].axvline(p["c_man"], color="C3", ls=":", label=f"cost parity: robot case = frame case ({p['c_man']})")
    ax[1].axvline(p["c_rob"], color="k", ls="-", lw=1, label=f"measured {p['c_rob']}")
    ax[1].set(xlabel="robot per-case cost (% of reference), +-30 %", ylabel="NCRI = 1 - kappa")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle("Sensitivity of P_payback and NCRI to the most uncertain inputs", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / "fig_sensitivity.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4))
    Ns = np.linspace(5, 120, 200)
    for m_, col in ((0.35, "C1"), (0.5, "C0"), (0.65, "C2")):
        ax.plot(Ns, [min(model(dict(p, m=m_), N=n)["P"], 6) for n in Ns], col, lw=2, label=f"m = {m_}   (break-even {N_stars[m_]:.0f} cases/yr)")
    ax.axhline(1, color="grey", ls="--")
    ax.axvline(m0["N"], color="k", lw=1)
    ax.text(m0["N"] + 1, 5.5, f"measured centre: {m0['N']:.0f}/yr", fontsize=8)
    ax.set(ylim=(0, 6), xlabel="DBS cases per year", ylabel="P_payback")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fd / "fig_volume.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    rho = np.linspace(0.05, 0.95, 200)
    for mu, lab, col in ((m0["mu_m"], "frame", "C1"), (m0["mu_r"], "robot", "C0")):
        ax.plot(rho, [wq_weeks(r * mu, mu) * 7 * 24 for r in rho], col, lw=2, label=f"{lab} (mu = {mu:.2f}/wk)")
    for r_, w_, col in ((m0["rho_m"], m0["wq_m_h"], "C1"), (m0["rho_r"], m0["wq_r_h"], "C0")):
        ax.plot([r_], [w_], "o", color=col, ms=8)
    ax.set(ylim=(0, 400), xlabel="utilisation rho = lambda / mu", ylabel="mean wait W_q (hours)")
    ax.set_title(f"M/M/1, lambda = {m0['lam_w']:.2f}/week: queue is not the bottleneck at this volume", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fd / "fig_queue.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    top = tor[::-1]
    for i, (k, a, b) in enumerate(top):
        lo_, hi_ = sorted((min(a, 8), min(b, 8)))
        ax.barh(i, hi_ - lo_, left=lo_, color="C0", alpha=0.7)
    ax.axvline(min(m0["P"], 8), color="k", lw=1)
    ax.axvline(1, color="C3", ls="--", lw=1)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([t[0] for t in top])
    ax.set(xlabel="P_payback for the input -30 % ... +30 %  (red = 1; capped at 8)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(fd / "fig_tornado.png", dpi=170)
    plt.close(fig)
    print("\nsaved analytics/economics_inputs.csv, economics_results.csv; docs/figures/fig_sensitivity.png, fig_volume.png, fig_queue.png, fig_tornado.png")


if __name__ == "__main__":
    main()