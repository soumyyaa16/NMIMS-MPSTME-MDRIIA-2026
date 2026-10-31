"""Capstone Monte Carlo (course dossier Section 4): baseline vs proposed, 500 Hz, recorded seeds.

    python experiments/09_capstone_monte_carlo.py --n 50
    python experiments/09_capstone_monte_carlo.py --from-csv      # recompute statistics and figures only

Writes ``analytics/aed_delivery_real_trials.csv`` (one row per MuJoCo episode, seed included),
``analytics/capstone_statistics.csv`` (Welch t, Cohen's d, 95 % CI per measure and condition),
``analytics/capstone_success_rates.csv`` and the dossier figures. Runs are paired: every controller
sees the same ``n`` scenarios per condition.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import aedrover  # noqa: F401
except ImportError:                      # exported course folder: the package is vendored under ../src
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402

from aedrover.capstone.figures import primary_boxplot, trajectory_plot  # noqa: E402
from aedrover.capstone.monte_carlo import (  # noqa: E402
    CONDITIONS,
    CONTROLLERS,
    PRIMARY_METRIC,
    PROTOCOL_CONDITIONS,
    SEED_BASE,
    run_monte_carlo,
    statistics_table,
    success_table,
)


def write_outputs(df: pd.DataFrame, out: Path) -> None:
    parts = []
    for b in ("proposed", "proposed_codesigned"):
        if b in set(df["controller"]):
            t = statistics_table(df, b=b)
            t.insert(0, "comparison", f"baseline vs {b}")
            parts.append(t.rename(columns=lambda c, b=b: c.replace(f"_{b}", "_proposed")))
    pd.concat(parts, ignore_index=True).to_csv(out / "capstone_statistics.csv", index=False, float_format="%.6g")
    success_table(df).to_csv(out / "capstone_success_rates.csv", index=False, float_format="%.6g")
    figs = out.parent / "docs" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    rep = out / "telemetry" / "representative_run_500hz.csv"
    if rep.exists():
        trajectory_plot(rep, figs / "capstone_trajectory.png")
    primary_boxplot(df, figs / "capstone_ttfs_boxplot.png", PRIMARY_METRIC, PROTOCOL_CONDITIONS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="runs per condition and controller (course minimum 30)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "analytics"))
    ap.add_argument("--conditions", nargs="*", default=None, choices=list(CONDITIONS))
    ap.add_argument("--from-csv", action="store_true", help="skip the simulation; recompute tables and figures")
    ap.add_argument("--labels", nargs="*", default=None, choices=list(CONTROLLERS), help="run only these controller arms")
    ap.add_argument("--append", action="store_true", help="replace the rows of --labels in the existing CSV instead of overwriting it")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        write_outputs(pd.read_csv(out / "aed_delivery_real_trials.csv"), out)
        return
    t0 = time.time()
    df = run_monte_carlo(args.n, args.workers, telemetry_dir=out / "telemetry", conditions=args.conditions, labels=args.labels)
    if args.append:
        old = pd.read_csv(out / "aed_delivery_real_trials.csv")
        df = pd.concat([old[~old["controller"].isin(set(df["controller"]))], df], ignore_index=True)
    df.to_csv(out / "aed_delivery_real_trials.csv", index=False, float_format="%.6g")
    write_outputs(df, out)
    meta = {"n_per_condition": args.n, "seed_base": SEED_BASE, "conditions": CONDITIONS,
            "controllers": {k: {"planner": v[0], "safety_filter": v[1], "vehicle": v[2], "planner_kwargs": v[3]}
                                 for k, v in CONTROLLERS.items()},
            "episodes": len(df), "log_rate_hz_min": float(df["log_rate_hz"].min()),
            "log_rate_hz_max": float(df["log_rate_hz"].max()), "wall_s": time.time() - t0}
    (out / "capstone_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {len(df)} episodes in {meta['wall_s']:.0f}s  ->  {out}")


if __name__ == "__main__":
    main()
