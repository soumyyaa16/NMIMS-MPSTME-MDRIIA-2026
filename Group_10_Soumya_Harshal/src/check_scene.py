import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(r"models\surgical_7dof_robot.xml")
d = mujoco.MjData(m)
home = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
d.qpos[:7] = home; d.ctrl[:7] = home
mujoco.mj_forward(m, d)
tip = m.site("p_needle_tip").id
p0 = d.site_xpos[tip].copy()
for _ in range(3000): mujoco.mj_step(m, d)
print("nq", m.nq, "nu", m.nu, "ngravcomp", m.ngravcomp)
print("tip drift (mm):", np.linalg.norm(d.site_xpos[tip] - p0) * 1000)