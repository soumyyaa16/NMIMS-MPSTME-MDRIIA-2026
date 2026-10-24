"""Shows the Panda aiming the needle at the lesion: standoff -> slow insertion -> hold -> loop.
Joint poses come from an offline IK search (reachable, >100 mm arm-to-brain clearance).
Run from the group folder:  python src\\view_pose.py
"""
import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "surgical_7dof_robot.xml"))
d = mujoco.MjData(m)

STANDOFF = np.array([0.15396, 0.33471, 0.11872, -1.60091, -0.23256, 1.38441, -0.48519])
INSERTED = np.array([0.11002, 0.24632, 0.09760, -2.00608, -0.24983, 1.70564, -0.46927])
tip = m.site("p_needle_tip").id
target = m.site("target_site").id


def run(viewer=None, loops=1, realtime=True):
    for _ in range(loops):
        d.qpos[:7] = STANDOFF
        d.ctrl[:7] = STANDOFF
        mujoco.mj_forward(m, d)
        # 2 s hold, 6 s insertion, 2 s hold
        for k in range(10000):
            s = np.clip((k * m.opt.timestep - 2.0) / 6.0, 0.0, 1.0)
            s = s * s * (3 - 2 * s)  # smoothstep
            d.ctrl[:7] = STANDOFF + s * (INSERTED - STANDOFF)
            mujoco.mj_step(m, d)
            if viewer is not None and k % 16 == 0:
                viewer.sync()
                if realtime:
                    time.sleep(0.016)
        err = np.linalg.norm(d.site_xpos[tip] - d.site_xpos[target]) * 1000
        print(f"needle tip to target: {err:.2f} mm (servo tracking only, no feedback yet)")


if __name__ == "__main__":
    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
        v.cam.lookat[:] = [-0.15, 0.02, 0.0]
        v.cam.distance = 1.35
        v.cam.azimuth = 125
        v.cam.elevation = -18
        while v.is_running():
            run(v)