"""Figures for the dossier, regenerated from ``analytics/aed_delivery_real_trials.csv`` and the
representative 500 Hz run. No synthetic data is drawn anywhere in this module."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 300
COLORS = {"baseline_unshielded": "#9a9a9a", "baseline": "#c0622d", "proposed": "#1f6f8b"}


def trajectory_plot(telemetry_csv: str | Path, out: str | Path) -> Path:
    """Reference vs actual path and the FSM state timeline for one 500 Hz run."""
    df = pd.read_csv(telemetry_csv)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.2), height_ratios=[2, 1], constrained_layout=True)
    ax1.plot(df["x_m"], df["y_ref_m"], "k--", lw=1.2, label="planned centreline")
    ax1.plot(df["x_m"], df["y_m"], color=COLORS["proposed"], lw=1.6, label="MuJoCo chassis")
    ax1.set_xlabel("x [m]"), ax1.set_ylabel("y [m]")
    ax1.set_title(f"Reference vs actual path (500 Hz, {len(df)} samples)")
    ax1.legend(frameon=False)
    states = list(dict.fromkeys(df["fsm_state"].fillna("")))
    pal = plt.get_cmap("tab10")
    for i, s in enumerate(states):
        m = df["fsm_state"].fillna("") == s
        ax2.plot(df.loc[m, "t_s"], np.full(m.sum(), i), ".", ms=2, color=pal(i), label=s or "-")
    ax2.set_yticks(range(len(states)), states)
    ax2.set_xlabel("t [s]"), ax2.set_title("Mission FSM state")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return Path(out)


def sensitivity_plot(sweep_df: pd.DataFrame, name: str, base_value: float, breakevens: dict[str, float],
                     out: str | Path, band: float = 0.30) -> Path:
    """kappa and NCRI against the most uncertain input, +/-30 % band shaded, break-even values marked."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    x = sweep_df[name]
    ax1.loglog(x, sweep_df["kappa"], color=COLORS["proposed"], lw=1.8)
    ax1.axhline(1.0, color="k", lw=0.8, ls=":")
    ax1.axhline(0.25, color="#b03a2e", lw=0.8, ls="--")
    ax1.text(x.iloc[0], 0.25, " kappa = 0.25 target", va="bottom", fontsize=8, color="#b03a2e")
    ax2.semilogx(x, sweep_df["NCRI"].clip(lower=-5), color=COLORS["baseline"], lw=1.8)
    ax2.axhline(0.0, color="k", lw=0.8, ls=":")
    for ax in (ax1, ax2):
        ax.axvspan(base_value * (1 - band), base_value * (1 + band), color="#1f6f8b", alpha=0.15)
        ax.axvline(base_value, color="#1f6f8b", lw=0.8)
        for v in breakevens.values():
            if np.isfinite(v):
                ax.axvline(v, color="#b03a2e", lw=0.9, ls="--")
        ax.set_xlabel(name)
    ax1.set_ylabel("kappa = C_robot / C_baseline")
    ax2.set_ylabel("NCRI = 1 - kappa (clipped at -5)")
    be = ", ".join(f"{k}: {v:.3g}" for k, v in breakevens.items() if np.isfinite(v)) or "none in range"
    fig.suptitle(f"Sensitivity to {name} (shaded: +/-30 %); break-even {be}", fontsize=9)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return Path(out)


def sensitivity_compact(sweep_df: pd.DataFrame, name: str, base_value: float, breakevens: dict[str, float],
                        out: str | Path, band: float = 0.30) -> Path:
    """The same sensitivity as ``sensitivity_plot`` as two stacked panels in a portrait figure: readable at half a page width."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.2, 5.4), sharex=True, constrained_layout=True)
    x = sweep_df[name]
    ax1.loglog(x, sweep_df["kappa"], color=COLORS["proposed"], lw=2)
    ax1.axhline(1.0, color="k", lw=0.8, ls=":")
    ax1.axhline(0.25, color="#b03a2e", lw=0.9, ls="--")
    ax1.text(x.iloc[0], 0.27, "target 0.25", fontsize=8, color="#b03a2e")
    ax2.semilogx(x, sweep_df["NCRI"].clip(lower=-5), color=COLORS["baseline"], lw=2)
    ax2.axhline(0.0, color="k", lw=0.8, ls=":")
    for ax in (ax1, ax2):
        ax.axvspan(base_value * (1 - band), base_value * (1 + band), color="#1f6f8b", alpha=0.2)
        for v in breakevens.values():
            if np.isfinite(v):
                ax.axvline(v, color="#b03a2e", lw=1.0, ls="--")
        ax.tick_params(labelsize=8)
    ax1.set_ylabel("kappa = C_robot / C_baseline", fontsize=8)
    ax2.set_ylabel("NCRI = 1 - kappa (clipped at -5)", fontsize=8)
    ax2.set_xlabel("arrests per 100,000 per year", fontsize=8)
    fig.suptitle("Sensitivity (shaded: base +/-30 %; dashed: break-even)", fontsize=8.5)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return Path(out)


def queue_plot(lam: float, mu: float, out: str | Path, c_values=(1, 2, 3)) -> Path:
    """Mean wait W_q against utilisation rho for several pod counts (M/M/c, Erlang C)."""
    from .business import mmc_wait

    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    for c in c_values:
        rho = np.linspace(0.02, 0.97, 80)
        wq = [mmc_wait(r * c * mu, mu, c)["Wq_h"] * 60 for r in rho]
        ax.plot(rho, wq, label=f"c = {c}")
    ax.set_yscale("log")
    ax.set_xlabel("utilisation rho = lambda / (c mu)"), ax.set_ylabel("mean wait W_q [min]")
    ax.set_title(f"M/M/c delay at mu = {mu:.2f}/h")
    ax.legend(frameon=False)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return Path(out)


def fmea_plot(fmea: pd.DataFrame, out: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)
    y = np.arange(len(fmea))
    ax.barh(y + 0.18, fmea["RPN_pre"], height=0.34, color=COLORS["baseline"], label="RPN before")
    ax.barh(y - 0.18, fmea["RPN_post"], height=0.34, color=COLORS["proposed"], label="RPN after")
    ax.set_yticks(y, fmea["failure_mode"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("RPN = S x O x D")
    ax.legend(frameon=False)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return Path(out)


def primary_boxplot(df: pd.DataFrame, out: str | Path, metric: str = "ttfs_min",
                    condition: str | tuple[str, ...] | None = None) -> Path:
    """Box plot of the primary metric, baseline vs proposed, successful runs only.

    ``condition`` may be one condition, several (pooled), or ``None`` for all of them.
    """
    conds = (condition,) if isinstance(condition, str) else condition
    sub = df if conds is None else df[df["condition"].isin(conds)]
    sub = sub[sub["success"] == 1]
    labels = ["baseline", "proposed"]
    data = [sub[sub["controller"] == lab][metric].dropna().to_numpy() for lab in labels]
    fig, ax = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
    bp = ax.boxplot(data, tick_labels=[f"{lab}\n(n={len(d)})" for lab, d in zip(labels, data, strict=True)], patch_artist=True)
    for patch, lab in zip(bp["boxes"], labels, strict=True):
        patch.set_facecolor(COLORS[lab]), patch.set_alpha(0.6)
    ax.set_ylabel(metric.replace("_", " "))
    label = conds[0] if conds is not None and len(conds) == 1 else ("protocol conditions pooled" if conds is not None else "all conditions")
    ax.set_title(f"{metric}: {label}")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return Path(out)
