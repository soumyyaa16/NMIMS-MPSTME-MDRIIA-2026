"""Render stills of the visual scene.  python src/render_scene.py --case c160 --nifti <brats root> --cams overview,hero,left
Optional: --scenario <id> uses analytics/cases/<case>/scenarios.json entry; --pose standoff|inserted."""
import argparse, json
from pathlib import Path
import mujoco, numpy as np
from PIL import Image
import scene_build as sb
import surgical_tremor_controller as sc

ap = argparse.ArgumentParser()
ap.add_argument("--case", default=None)
ap.add_argument("--nifti", default=None)
ap.add_argument("--scenario", default=None)
ap.add_argument("--cams", default="octant,octant_close,overview,left,top,room")
ap.add_argument("--out", default=str(sc.ROOT / "docs" / "figures" / "scene"))
ap.add_argument("--w", type=int, default=960); ap.add_argument("--h", type=int, default=600)
ap.add_argument("--no-cutaway", action="store_true")
ap.add_argument("--depth", type=float, default=0.0, help="needle depth beyond entry in m (pose the arm for the still)")
a = ap.parse_args()
plan = None
if a.scenario:
    f = sc.ROOT / "analytics" / "cases" / a.case / "scenarios.json" if a.case else sc.ROOT / "analytics" / "scenarios.json"
    scn = [s for s in json.loads(f.read_text()) if s["id"] == a.scenario][0]
    plan = sc.plan_from_scenario(scn)
m, info = sb.load_visual(a.case, plan, a.nifti, not a.no_cutaway)
d = mujoco.MjData(m)
if plan is None:
    plan = sc.load_plan()
# pose the arm: IK to the standoff (or deeper) using the controller's own solver
robot = sc.NeedleRobot(m)
tip_goal = plan["standoff"] if a.depth <= 0 else plan["entry"] + plan["ins"] * a.depth
q = np.array(sc.HOME, float)
for _ in range(400):
    d.qpos[:7] = q; mujoco.mj_forward(m, d)
    err = tip_goal - d.site_xpos[robot.tip]
    J = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, J, None, robot.tip)
    J = J[:, :7]
    q = q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err) * 0.7
    if np.linalg.norm(err) < 1e-4: break
d.qpos[:7] = q; mujoco.mj_forward(m, d)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
r = mujoco.Renderer(m, a.h, a.w)
opt = mujoco.MjvOption()      # default groups: 0,1,2 visible (robot visual = 2, collision = 3 hidden)
for cam in a.cams.split(","):
    r.update_scene(d, camera=cam, scene_option=opt)
    Image.fromarray(r.render()).save(out / f"{a.case or 'orig'}_{cam}.png")
    print("saved", out / f"{a.case or 'orig'}_{cam}.png")