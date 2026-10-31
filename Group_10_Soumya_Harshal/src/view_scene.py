"""Open the interactive 3D scene.
    python src\\view_scene.py --case c160 --scenario enhancing_superior --nifti D:\\robotics\\brats_cases
Pick a camera in the viewer sidebar: Rendering > Camera (octant, octant_close, octant_top, overview, hero, left, right, top, front, close, room).
Keys: double-click a body = centre on it; hold left mouse = orbit; right mouse = pan; scroll = zoom.
Geom groups (keys 0-5): 4 toggles the un-cut brain.  --depth 0.03 poses the needle 30 mm beyond the entry point."""
import argparse
import json
import time

import mujoco
import mujoco.viewer
import numpy as np

import scene_build as sb
import surgical_tremor_controller as sc

ap = argparse.ArgumentParser()
ap.add_argument("--case", default=None)
ap.add_argument("--scenario", default=None)
ap.add_argument("--nifti", default=r"D:\robotics\brats_cases")
ap.add_argument("--depth", type=float, default=0.0)
ap.add_argument("--cam", default="octant")
ap.add_argument("--no-cutaway", action="store_true")
a = ap.parse_args()

plan = None
if a.scenario:
    f = sc.ROOT / "analytics" / "cases" / a.case / "scenarios.json"
    plan = sc.plan_from_scenario([s for s in json.loads(f.read_text()) if s["id"] == a.scenario][0])
m, info = sb.load_visual(a.case, plan, a.nifti, not a.no_cutaway)
d = mujoco.MjData(m)
plan = plan or sc.load_plan()
robot = sc.NeedleRobot(m)
goal = plan["standoff"] if a.depth <= 0 else plan["entry"] + plan["ins"] * a.depth
q = np.array(sc.HOME, float)
for _ in range(500):
    d.qpos[:7] = q
    mujoco.mj_forward(m, d)
    err = goal - d.site_xpos[robot.tip]
    J = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, J, None, robot.tip)
    J = J[:, :7]
    q = q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err) * 0.7
    if np.linalg.norm(err) < 1e-4:
        break
d.qpos[:7] = q
d.ctrl[:7] = q
mujoco.mj_forward(m, d)
print(f"tip error to goal: {np.linalg.norm(goal - d.site_xpos[robot.tip]) * 1e3:.2f} mm   (cut-away octant {info['sgn']})")
with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    v.cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, a.cam)
    while v.is_running():
        v.sync()
        time.sleep(0.03)