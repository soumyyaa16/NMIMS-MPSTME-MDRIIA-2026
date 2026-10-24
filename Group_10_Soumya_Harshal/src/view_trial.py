"""Watch one trial live:  python src/view_trial.py PROP   (or BASE / NT)   [amp_um]
Tremor is real here: the needle visibly jitters under BASE and is steadier under PROP."""
import sys

import mujoco
import mujoco.viewer

from surgical_tremor_controller import load_model, run_trial

cond = sys.argv[1] if len(sys.argv) > 1 else "PROP"
amp = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
model = load_model()
data = mujoco.MjData(model)
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
    v.cam.lookat[:] = [0.0, 0.03, 0.02]
    v.cam.distance = 0.55
    v.cam.azimuth = 150
    v.cam.elevation = -20
    r = run_trial(model, cond, 1000, amp, 1.0, viewer=v, realtime=True, data=data)
    s = r["summary"]
    print(f"{cond}: tracking RMSE {s['rmse_track_mm']:.3f} mm | hold max {s['hold_max_mm']:.3f} mm"
          f" | tremor rejection {s['tremor_rejection_dB']:.1f} dB")
    while v.is_running():
        v.sync()