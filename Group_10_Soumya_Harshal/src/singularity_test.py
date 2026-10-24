"""Singularity stress test (kinematic, no tremor): drive the needle tip radially outward from the shoulder
towards full arm extension, holding needle orientation, with plain pseudoinverse vs adaptive DLS.
python src/singularity_test.py   -> analytics/singularity_test.csv, docs/figures/fig_singularity.png"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import surgical_tremor_controller as sc

model = sc.load_model()
robot = sc.NeedleRobot(model)
plan = sc.load_plan()
ok, q_s = robot.solve_pose(plan["standoff"], plan["ins"], sc.HOME)
assert ok
J, p0, ax0 = robot.jac(q_s)
import mujoco
d = mujoco.MjData(model); d.qpos[:7] = q_s; mujoco.mj_forward(model, d)
shoulder = d.xpos[model.body("p_link1").id].copy()
u = (p0 - shoulder); r0 = np.linalg.norm(u); u /= r0
print(f"start reach from shoulder {r0*1e3:.0f} mm")

T, dt = 4.0, sc.DT_C
rows = []
for r_goal in [0.60, 0.70, 0.76, 0.80, 0.83]:
    for mode, null in (("pinv", False), ("dls", True)):
        q = q_s.copy()
        n = int(T / dt)
        wmin = 1e9; err_max = 0.0; dq_dem_max = 0.0; clipped = 0
        for k in range(n + 1):
            tau = k / n
            s, ds = sc.min_jerk(np.array(tau))
            p_ref = shoulder + u * (r0 + (r_goal - r0) * s)
            v_ref = u * (r_goal - r0) * ds / T
            J, p, ax = robot.jac(q)
            w = robot.manipulability(J)
            wmin = min(wmin, w)
            e = p_ref - p
            err_max = max(err_max, np.linalg.norm(e))
            dx = v_ref * dt + sc.KP * dt * e
            e_o = np.cross(ax, plan["ins"])
            dq, lam = sc.ik_step(J, np.r_[dx, sc.KO * dt * e_o], q, q_s, mode, null, w)
            dq_dem_max = max(dq_dem_max, np.max(np.abs(dq)) / dt)
            lim = sc.V_JOINT_MAX * dt
            if np.any(np.abs(dq) > lim): clipped += 1
            q = np.clip(q + np.clip(dq, -lim, lim), robot.lo, robot.hi)
        rows.append(dict(r_goal_mm=r_goal * 1e3, ik=mode, min_w=wmin, max_tracking_err_mm=err_max * 1e3,
                         peak_joint_speed_demand_rad_s=dq_dem_max, steps_speed_clipped=clipped,
                         final_err_mm=np.linalg.norm(e) * 1e3))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
df.to_csv(sc.ROOT / "analytics" / "singularity_test.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for m, col in (("pinv", "tab:red"), ("dls", "tab:blue")):
    x = df[df.ik == m]
    ax[0].plot(x.r_goal_mm, x.peak_joint_speed_demand_rad_s, "o-", color=col, label=m)
    ax[1].plot(x.r_goal_mm, x.max_tracking_err_mm, "o-", color=col, label=m)
ax[0].set_ylabel("peak joint speed demand (rad/s)"); ax[0].set_yscale("log"); ax[0].axhline(sc.V_JOINT_MAX, color="k", ls="--", lw=1)
ax[1].set_ylabel("max tracking error (mm)"); ax[1].set_yscale("log")
for a_ in ax: a_.set_xlabel("goal reach from shoulder (mm)"); a_.grid(alpha=0.3); a_.legend()
fig.tight_layout(); fig.savefig(sc.ROOT / "docs" / "figures" / "fig_singularity.png", dpi=200)