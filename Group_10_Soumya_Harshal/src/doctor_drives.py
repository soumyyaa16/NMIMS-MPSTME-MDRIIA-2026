"""Doctor-drives mode: YOU advance the needle with the keyboard, the simulated hand tremor is injected,
and you switch our assistant (filter) on and off LIVE. Parking-sensor beeps get faster as the tip nears the target;
warnings for sideways deviation, too deep and near-singularity. Everything is written to an event log + telemetry CSV
+ a dashboard picture (docs/doctor_log/) that you can cite in the dossier.

    python src\\doctor_drives.py --case c160 --scenario enhancing_superior --nifti D:\\robotics\\brats_cases

KEYS   W / S  needle forward / back 1 mm (5 mm outside the head)     E / D  fine: 0.1 mm forward / back
       SPACE  stop where it is     R  pull the needle all the way out
       F      assistant ON / OFF (3.5 Hz low-pass + damped IK  vs  raw tremor + plain pseudo-inverse IK)
       J L I K  nudge the hand sideways 0.1 mm (simulates the doctor drifting off the line)     Z  centre again
       [ ]    tremor smaller / bigger (25 um steps)           1..6  camera: wide, close, side, top, follow needle, tip close-up
       T      skin see-through / solid     H  help on/off      Q or Esc  finish (saves the log)
HONESTY: the hand is a model (tremor + drift added to the needle command); the robot executes your motion intent
(your key-driven depth and nudges) and the assistant removes only the fast shake. Not a clinical device."""
import argparse
import csv
import datetime
import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np

import scene_build as sb
import surgical_tremor_controller as sc

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="c160")
ap.add_argument("--scenario", default="enhancing_superior")
ap.add_argument("--nifti", default=r"D:\robotics\brats_cases")
ap.add_argument("--amp", type=float, default=150.0, help="tremor amplitude, um per axis (change live with [ and ])")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--w", type=int, default=1100)
ap.add_argument("--h", type=int, default=620, help="height of the 3D view (the dashboard is added below)")
ap.add_argument("--mute", action="store_true")
ap.add_argument("--record", default=None, help="also write the whole session to this mp4 (for the LinkedIn video)")
ap.add_argument("--selftest", action="store_true", help="scripted keys, no window (for testing)")
ap.add_argument("--stills", default=None)
a = ap.parse_args()

W_, H_ = a.w, a.h
DASH_H = int(H_ * 0.36)
FRAME_STEPS = 32                         # physics steps (1 ms) per displayed frame -> 32 ms
FRAME_DT = FRAME_STEPS * sc.DT_SIM
T_AL = 3.0                               # s: automatic alignment at the start
SPEEDS = [-15.0, -5.0, -2.0, 0.0, 2.0, 5.0, 15.0]    # mm/s
TISSUE_VMAX = 5.0                        # mm/s safety limit once the tip is inside the head
LAT_WARN_UM, LAT_LIMIT_UM, HIT_MM = 300.0, 500.0, 0.5

# ------------------------------------------------------------------ model and plan (same as the film)
f = sc.ROOT / "analytics" / "cases" / a.case / "scenarios.json"
_all = json.loads(f.read_text())
_hit = [s for s in _all if s["id"] == a.scenario]
if not _hit:
    raise SystemExit(f"scenario '{a.scenario}' not in {a.case}. Available: " + ", ".join(s["id"] for s in _all))
scn = _hit[0]
if scn["status"] != "REACHABLE":
    raise SystemExit(f"{a.case}/{a.scenario} is {scn['status']} - pick a REACHABLE one")
plan = sc.plan_from_scenario(scn)
m, info = sb.load_visual(a.case, plan, a.nifti, True)
nd = sb.find_nifti_dir(a.case, a.nifti)
meta = sb.case_paths(a.case)[2]
vol = sb.load_volume(nd) if nd is not None else None
sgn = np.array(info["sgn"], float)
target = np.asarray(plan["target"], float)
ins = np.asarray(plan["ins"], float)
entry = np.asarray(plan["entry"], float)
standoff = np.asarray(plan["standoff"], float)
robot = sc.NeedleRobot(m)
tip_id = robot.tip
tip_body = int(m.site_bodyid[tip_id])
ctrl_lo, ctrl_hi = m.actuator_ctrlrange[:7].T
STAND_LEN = float(np.linalg.norm(standoff - entry))
PATH_LEN = float(np.dot(target - entry, ins))
U_MAX = PATH_LEN + 0.003
SKIN_MATS = [m.material(n).id for n in ("skin", "hair", "eye")]
mat_path, mat_ent, mat_tgt = (m.material(n).id for n in ("mark_path", "mark_entry", "mark_target"))
_nid = m.geom('p_needle_shaft').id
m.geom_size[_nid, 0] = 0.0024
m.geom_matid[_nid] = -1
m.geom_rgba[_nid] = (1.0, 0.55, 0.05, 1.0)
m.site_size[tip_id] = 0.0045
m.site_rgba[tip_id] = (1.0, 0.1, 0.1, 1.0)
m.geom_size[m.geom('traj_line').id, 0] = 0.0006
m.geom_size[m.geom('tgt_ring').id, 0] = 0.0055          # bigger green target ball (visual only)
m.geom_size[m.geom('entry_mark').id, 0] = 0.0045
renderer = mujoco.Renderer(m, H_, W_)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
skin_solid = False


def set_skin(alpha):
    for _i in SKIN_MATS:
        m.mat_rgba[_i, 3] = alpha


set_skin(0.26)

_E1 = np.cross(ins, [0, 0, 1.0])
_E1 = _E1 / np.linalg.norm(_E1) if np.linalg.norm(_E1) > 1e-6 else np.array([1.0, 0, 0])
_E2 = np.cross(ins, _E1)

# ------------------------------------------------------------------ drawing helpers (same look as the film)
FONT = cv2.FONT_HERSHEY_DUPLEX
K = H_ / 720.0


def txt(img, s, org, scale=0.6, col=(255, 255, 255), th=1):
    th = max(1, int(round(th * H_ / 720)))
    o = max(1, th)
    for dx, dy in ((-o, 0), (o, 0), (0, -o), (0, o), (-o, -o), (o, o), (-o, o), (o, -o)):
        cv2.putText(img, s, (org[0] + dx, org[1] + dy), FONT, scale, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, col, th, cv2.LINE_AA)


def bar(img, x0, y0, x1, y1, col, alpha):
    ov = img.copy()
    cv2.rectangle(ov, (int(x0), int(y0)), (int(x1), int(y1)), col, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def look(pos, tgt):
    pos, tgt = np.asarray(pos, float), np.asarray(tgt, float)
    fwd = tgt - pos
    dist = np.linalg.norm(fwd)
    fwd /= dist
    cam.lookat[:] = tgt
    cam.distance = dist
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    cam.elevation = float(np.degrees(np.arcsin(fwd[2])))


def orbit(az, dist, el_deg):
    el = np.radians(el_deg)
    return target + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def _arc(a0, a1, u):
    d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    return a0 + d * u


AZ_SIDE = float(np.arctan2(sgn[1], sgn[0]))
AZ_FACE = float(np.arctan2(0.9, 0.7))
AZ_END = _arc(AZ_FACE, AZ_SIDE, 0.45)


def preset(key, tip):
    if key == "1":
        return orbit(AZ_FACE, 0.72, 18.0), target
    if key == "2":
        return orbit(AZ_END, 0.46, 24.0), target
    if key == "3":
        return orbit(AZ_SIDE, 0.42, 10.0), target
    if key == "4":
        return target + np.array([0.0, -0.12, 0.50]), target
    if key == "5":
        return tip - ins * 0.10 + _E1 * 0.09, tip + ins * 0.02
    return tip - ins * 0.035 + _E1 * 0.035 + _E2 * 0.01, tip + ins * 0.012


def proj_from_scene():
    c0 = renderer.scene.camera[0]
    return np.array(c0.pos, float), np.array(c0.forward, float), np.array(c0.up, float)


def to_px(proj, p, w, h):
    pos, fwd, up = proj
    right = np.cross(fwd, up)
    q = np.asarray(p, float) - pos
    z = q @ fwd
    if z < 0.05:
        return None
    fpx = (h / 2) / np.tan(np.radians(m.vis.global_.fovy) / 2)
    return int(w / 2 + fpx * (q @ right) / z), int(h / 2 - fpx * (q @ up) / z)


def tag(img, proj, p, text, col, off=(40, -40)):
    h, w = img.shape[:2]
    xy = to_px(proj, p, w, h)
    if xy is None or not (0 <= xy[0] < w and 0 <= xy[1] < h):
        return
    o = (int(xy[0] + off[0] * K), int(xy[1] + off[1] * K))
    cv2.circle(img, xy, int(7 * K), col, 2, cv2.LINE_AA)
    cv2.line(img, xy, o, col, 2, cv2.LINE_AA)
    txt(img, text, (o[0] + 4, o[1] - 4), 0.5 * K + 0.1, col)


def needle_eye(tip, S_out):
    """MRI re-slice through the tip along the needle axis (real scan)."""
    S = 160
    w1 = _E1
    L = 0.06
    rgba = sb.sample_plane(vol, meta, tip, w1, -ins, L, S, alpha_outside=False)[..., :3]
    img = cv2.cvtColor(rgba, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (S_out, S_out), interpolation=cv2.INTER_CUBIC)
    sc_ = S_out / L
    cx = cy = S_out // 2
    rel = target - tip
    px, py = int(cx + (rel @ w1) * sc_), int(cy + (rel @ (-ins)) * sc_)
    cv2.line(img, (cx, 0), (cx, cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 5, (60, 60, 255), -1, cv2.LINE_AA)
    if 0 <= px < S_out and 0 <= py < S_out:
        cv2.drawMarker(img, (px, py), (90, 255, 120), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        txt(img, "TARGET", (min(px + 11, S_out - 62), max(py - 8, 14)), 0.4, (90, 255, 120))
    txt(img, "TIP", (cx + 9, cy + 5), 0.4, (80, 80, 255))
    cv2.rectangle(img, (0, 0), (S_out - 1, S_out - 1), (255, 255, 255), 1)
    return img


def micro_view(S_out, trail, tip_xy):
    img = np.full((S_out, S_out, 3), (35, 28, 22), np.uint8)
    c = S_out // 2
    k = (S_out / 2) / 0.6
    cv2.circle(img, (c, c), int(0.5 * k), (60, 160, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (c, c), int(0.1 * k), (90, 90, 90), 1, cv2.LINE_AA)
    cv2.line(img, (c, 0), (c, S_out), (70, 70, 70), 1)
    cv2.line(img, (0, c), (S_out, c), (70, 70, 70), 1)
    pts = [(int(c + x * k), int(c - y * k)) for x, y in trail]
    for i in range(1, len(pts)):
        al = i / len(pts)
        cv2.line(img, pts[i - 1], pts[i], (int(80 * al + 20), int(210 * al + 20), int(255 * al)), 1, cv2.LINE_AA)
    px, py = int(c + tip_xy[0] * k), int(c - tip_xy[1] * k)
    cv2.circle(img, (int(np.clip(px, 3, S_out - 4)), int(np.clip(py, 3, S_out - 4))), 4, (80, 230, 255), -1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (S_out - 1, S_out - 1), (255, 255, 255), 1)
    return img


def legend(img, x, y, k=1.0):
    items = [("needle (held by the robot)", (30, 140, 255), "line"), ("planned path", (255, 230, 60), "line"),
             ("entry point", (60, 220, 255), "dot"), ("target (green cross)", (110, 255, 130), "cross"),
             ("tumour - enhancing", (50, 50, 240), "box"), ("necrotic core", (50, 215, 250), "box"),
             ("oedema (swelling)", (250, 140, 80), "box")]
    row = int(21 * k)
    bar(img, x, y, x + int(250 * k), y + row * len(items) + int(26 * k), (35, 28, 22), 0.75)
    txt(img, "LEGEND", (x + 8, y + int(17 * k)), 0.42 * k + 0.1)
    for i, (lab, col, kind) in enumerate(items):
        yy = y + int(26 * k) + i * row + row // 2
        if kind == "line":
            cv2.line(img, (x + 10, yy), (x + int(34 * k), yy), col, 3, cv2.LINE_AA)
        elif kind == "dot":
            cv2.circle(img, (x + int(22 * k), yy), int(5 * k), col, -1, cv2.LINE_AA)
        elif kind == "cross":
            cv2.drawMarker(img, (x + int(22 * k), yy), col, cv2.MARKER_CROSS, int(12 * k), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x + int(12 * k), yy - int(6 * k)), (x + int(32 * k), yy + int(6 * k)), col, -1)
        txt(img, lab, (x + int(42 * k), yy + int(5 * k)), 0.38 * k + 0.1)


# ------------------------------------------------------------------ beeps (parking sensor)
class Beeper(threading.Thread):
    """Runs in its own thread so the simulation never waits for a beep. Priority: too deep > sideways > singularity >
    at target (steady tone) > proximity (beeps get faster as the tip nears the target)."""

    def __init__(self, mute):
        super().__init__(daemon=True)
        self.mode, self.dist, self.mute, self.alive = "off", 99.0, mute, True
        self.flash = 0.0
        self.count = 0
        try:
            import winsound
            self._beep = winsound.Beep
        except ImportError:
            self._beep = None

    def beep(self, hz, ms):
        self.flash = time.time() + ms / 1000.0
        self.count += 1
        if self._beep is not None and not self.mute:
            self._beep(int(hz), int(ms))
        else:
            time.sleep(ms / 1000.0)

    def run(self):
        while self.alive:
            md, ds = self.mode, self.dist
            if md == "deep":
                self.beep(300, 300)
                time.sleep(0.05)
            elif md == "lateral":
                self.beep(450, 110)
                time.sleep(0.05)
                self.beep(450, 110)
                time.sleep(0.35)
            elif md == "sing":
                self.beep(250, 200)
                time.sleep(0.4)
            elif md == "target":
                self.beep(1300, 120)
            elif md == "prox":
                self.beep(900, 40)
                time.sleep(max(0.03, 0.05 + 0.9 * min(ds, 30.0) / 30.0))
            else:
                time.sleep(0.03)


# ------------------------------------------------------------------ the drive simulation (same controller as the trials)
class Drive:
    def __init__(self):
        self.d = mujoco.MjData(m)
        ok_s, self.q_s = robot.solve_pose(standoff, ins, sc.HOME)
        if not ok_s:
            raise SystemExit("planner pose not reachable")
        self.d.qpos[:7] = sc.HOME
        self.d.ctrl[:7] = sc.HOME
        mujoco.mj_forward(m, self.d)
        self.q_cmd = sc.HOME.copy()
        self.e_int = np.zeros(3)
        self.step = 0
        self.u = -STAND_LEN                  # needle depth along the planned line (m), 0 = entry point
        self.goal = -STAND_LEN               # where the needle is heading (m); it STOPS there, never runs away
        self.off = np.zeros(2)               # manual sideways nudge (m) along E1, E2
        self.filt = True
        self.amp = a.amp
        self.hand = sc.TremorModel(a.seed, self.amp)
        self.lp = sc.StreamingLowpass()
        self.lp.reset(np.zeros(3))
        self.F = 0.0
        self.w = 0.1
        self.v_tip = np.zeros(6)
        self.phase = "ALIGN"
        self.u_cmd_v = 0.0

    @property
    def t(self):
        return self.step * sc.DT_SIM

    def set_filter(self, on):
        if on and not self.filt:
            dev, _ = self.hand(self.t)
            self.lp.reset(dev)
        self.filt = on

    def set_amp(self, amp):
        self.amp = float(np.clip(amp, 0.0, 300.0))
        self.hand = sc.TremorModel(a.seed, self.amp)

    def move_goal(self, delta_m):
        self.goal = float(np.clip(self.goal + delta_m, -STAND_LEN, U_MAX))

    def speed_mm_s(self):
        vmax = 20.0 if self.u < -0.008 else TISSUE_VMAX
        return float(np.clip(8.0 * (self.goal - self.u) * 1e3, -vmax, vmax))

    def control_tick(self):
        t = self.t
        d = self.d
        mujoco.mj_kinematics(m, d)
        p_meas = d.site_xpos[tip_id].copy()
        ax_meas = d.site_xmat[tip_id].reshape(3, 3)[:, 2].copy()
        J, _, _ = robot.jac(d.qpos[:7])
        self.w = robot.manipulability(J)
        if self.phase == "ALIGN":
            if t >= T_AL:
                self.phase = "DRIVE"
                self.e_int[:] = 0
            else:
                s, _ = sc.min_jerk(t / T_AL)
                self.q_cmd = sc.HOME + s * (self.q_s - sc.HOME)
                d.ctrl[:7] = np.clip(self.q_cmd, ctrl_lo, ctrl_hi)
                return
        v = self.speed_mm_s() * 1e-3
        self.u = float(np.clip(self.u + v * sc.DT_C, -STAND_LEN, U_MAX))
        if self.u in (-STAND_LEN, U_MAX):
            v = 0.0
        p_ref = entry + ins * self.u + _E1 * self.off[0] + _E2 * self.off[1]
        dev, _ = self.hand(t)
        p_des = p_ref + self.lp(dev) if self.filt else p_ref + dev
        e = p_des - p_meas
        self.e_int = np.clip(self.e_int + e * sc.DT_C, -sc.I_CLAMP, sc.I_CLAMP)
        dx = ins * v * sc.DT_C + sc.KP * sc.DT_C * e + sc.KI * sc.DT_C * self.e_int
        e_o = np.cross(ax_meas, ins)
        dx6 = np.r_[dx, sc.KO * sc.DT_C * e_o]
        dq, _ = sc.ik_step(J, dx6, d.qpos[:7], self.q_s, "dls" if self.filt else "pinv", self.filt, self.w)
        lim = sc.V_JOINT_MAX * sc.DT_C
        dq = np.clip(dq, -lim, lim)
        self.q_cmd = np.clip(self.q_cmd + dq, robot.lo, robot.hi)
        d.ctrl[:7] = np.clip(self.q_cmd, ctrl_lo, ctrl_hi)

    def physics_step(self):
        d = self.d
        if self.step % sc.CTRL_EVERY == 0:
            self.control_tick()
        d.qfrc_applied[:] = 0.0
        p_tip = d.site_xpos[tip_id]
        depth = float(np.dot(p_tip - entry, ins))
        if self.phase == "DRIVE" and depth > 0.0:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, tip_id, self.v_tip, 0)
            v_ax = float(np.dot(self.v_tip[3:], ins))
            self.F = max(sc.TISSUE_K * depth + sc.TISSUE_C * v_ax, 0.0)
            mujoco.mj_applyFT(m, d, -self.F * ins, np.zeros(3), p_tip, tip_body, d.qfrc_applied)
        else:
            self.F = 0.0
        mujoco.mj_step(m, d)
        self.step += 1


# ------------------------------------------------------------------ session log, alerts, dashboard
class Session:
    def __init__(self):
        self.t, self.u, self.lat, self.dist, self.filt = [], [], [], [], []
        self.rows = []
        self.events = []
        self.act = {"lateral": False, "deep": False, "sing": False, "target": False}
        self.sums = {True: [0.0, 0], False: [0.0, 0]}            # lateral error sums while the tip is inside the head
        self.n_in_tissue = 0
        self.n_hit = 0
        self.max_lat_in = 0.0
        self.max_u = -1.0
        self.warn_counts = {"lateral": 0, "deep": 0, "sing": 0}

    def event(self, t, name, detail=""):
        self.events.append((round(t, 3), name, detail))

    def update(self, t, u, lat, dist, w, filt, amp, F, tip):
        self.t.append(t); self.u.append(u * 1e3); self.lat.append(lat); self.dist.append(dist); self.filt.append(filt)
        self.rows.append((round(t, 3), round(u * 1e3, 3), round(lat, 1), round(dist, 4), int(filt), amp, round(w, 4), round(F, 3), *np.round(tip, 6)))
        inside = u > 0
        if inside:
            self.n_in_tissue += 1
            self.sums[filt][0] += lat
            self.sums[filt][1] += 1
            self.max_lat_in = max(self.max_lat_in, lat)
            if dist < HIT_MM:
                self.n_hit += 1
        self.max_u = max(self.max_u, u * 1e3)
        new = {"lateral": inside and lat > (0.85 if self.act["lateral"] else 1.0) * LAT_WARN_UM, "deep": u > PATH_LEN + 0.0005,
               "sing": t > T_AL and w < sc.W_DAMP, "target": inside and dist < HIT_MM}
        for k_, v_ in new.items():
            if v_ and not self.act[k_]:
                if k_ in self.warn_counts:
                    self.warn_counts[k_] += 1
                self.event(t, {"lateral": "WARN_LATERAL", "deep": "WARN_TOO_DEEP", "sing": "WARN_SINGULARITY", "target": "TARGET_REACHED"}[k_],
                           f"lat={lat:.0f}um dist={dist:.2f}mm depth={u * 1e3:.1f}mm")
            if (not v_) and self.act[k_]:
                self.event(t, {"lateral": "CLEAR_LATERAL", "deep": "CLEAR_TOO_DEPTH", "sing": "CLEAR_SINGULARITY", "target": "LEFT_TARGET"}[k_], "")
            self.act[k_] = v_
        return new

    def mean(self, filt):
        s, n = self.sums[filt]
        return s / n if n else float("nan")


def draw_series(img, x0, y0, x1, y1, ts, ys, filts, tnow, span, ymax, title, unit, col, limit=None, limit_label=""):
    bar(img, x0, y0, x1, y1, (30, 25, 20), 0.85)
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 80, 70), 1)
    txt(img, title, (x0 + 6, y0 + int(17 * K)), 0.42 * K + 0.1)
    px0, px1, py0, py1 = x0 + 6, x1 - 6, y0 + int(24 * K), y1 - 8
    if len(ts) < 2:
        return
    ts_a = np.asarray(ts)
    sel = ts_a >= tnow - span
    tt, yy, ff = ts_a[sel], np.asarray(ys)[sel], np.asarray(filts)[sel]
    X = px0 + ((tt - (tnow - span)) / span * (px1 - px0)).astype(int)
    Y = py1 - (np.clip(yy, 0, ymax) / ymax * (py1 - py0)).astype(int)
    # filter ON (green) / OFF (red) strip at the bottom
    for i in range(len(X) - 1):
        cv2.line(img, (X[i], py1 + 3), (X[i + 1], py1 + 3), (90, 200, 90) if ff[i] else (70, 70, 230), 4)
    if limit is not None:
        yl = py1 - int(limit / ymax * (py1 - py0))
        cv2.line(img, (px0, yl), (px1, yl), (60, 160, 255), 1, cv2.LINE_AA)
        txt(img, limit_label, (px0 + 4, yl - 4), 0.36 * K + 0.1, (60, 160, 255))
    cv2.polylines(img, [np.stack([X, Y], 1).astype(np.int32)], False, col, 2, cv2.LINE_AA)
    txt(img, f"{yy[-1]:.0f} {unit}" if unit == "um" else f"{yy[-1]:.1f} {unit}", (x1 - int(84 * K), y0 + int(17 * K)), 0.5 * K + 0.1, col)


def dashboard(sess, drive, tnow):
    img = np.full((DASH_H, W_, 3), (24, 20, 16), np.uint8)
    w1, w2 = int(0.40 * W_), int(0.27 * W_)
    draw_series(img, 4, 4, w1, DASH_H - 4, sess.t, sess.lat, sess.filt, tnow, 25.0, 700.0,
                "deviation from planned line   (strip: green = ON, red = OFF)", "um", (80, 200, 255), LAT_LIMIT_UM, "limit 500 um")
    draw_series(img, w1 + 8, 4, w1 + w2, DASH_H - 4, sess.t, sess.dist, sess.filt, tnow, 25.0, 80.0,
                "distance to target (mm)", "mm", (120, 255, 150), HIT_MM, "")
    x0 = w1 + w2 + 8
    bar(img, x0, 4, W_ - 4, DASH_H - 4, (30, 25, 20), 0.85)
    cv2.rectangle(img, (x0, 4), (W_ - 4, DASH_H - 4), (90, 80, 70), 1)
    txt(img, "SESSION (from the logged telemetry)", (x0 + 6, int(21 * K)), 0.42 * K + 0.1)
    mon, mof = sess.mean(True), sess.mean(False)
    tin = sess.n_in_tissue * FRAME_DT
    rows = [(f"mean deviation, assistant ON : {mon:5.0f} um" if mon == mon else "mean deviation, assistant ON :   -", (120, 255, 150)),
            (f"mean deviation, assistant OFF: {mof:5.0f} um" if mof == mof else "mean deviation, assistant OFF:   -", (110, 110, 255)),
            (f"time inside head: {tin:5.1f} s   at target (<0.5 mm): {sess.n_hit * FRAME_DT:4.1f} s", (230, 230, 230)),
            (f"warnings: sideways {sess.warn_counts['lateral']}  deep {sess.warn_counts['deep']}  singular {sess.warn_counts['sing']}", (230, 230, 230))]
    for i, (s, c) in enumerate(rows):
        txt(img, s, (x0 + 8, int(44 * K) + i * int(19 * K)), 0.40 * K + 0.1, c)
    for i, (te, nm, dt) in enumerate(list(sess.events)[-4:]):
        txt(img, f"{te:6.1f}s  {nm}", (x0 + 8, int(44 * K) + (4 + i) * int(17 * K) + 6), 0.36 * K + 0.1, (200, 200, 120))
    return img


HELP = ["KEYS", "W / S   forward / back (1 mm)", "E / D   fine 0.1 mm", "R   pull out    SPACE   stop", "F   assistant ON/OFF", "J L I K   nudge hand 0.1 mm", "Z   centre", "[ ]   tremor -/+",
        "1-6   camera", "T   skin", "H   hide help", "Q   finish + save"]


def compose(img, drive, sess, alerts, beeper, proj, tip, lat, dist, u, tnow, inset, show_help):
    h, w = img.shape[:2]
    if proj is not None:
        tag(img, proj, entry, "ENTRY", (60, 220, 255), (-70, -45))
        tag(img, proj, target, "TARGET (inside brain)", (110, 255, 130), (60, 50))
        tag(img, proj, tip, "NEEDLE TIP", (80, 80, 255), (70, -60))
    bar(img, 0, 0, w, int(36 * K), (40, 30, 20), 0.65)
    txt(img, f"{a.case} | {a.scenario} | DOCTOR DRIVES | t = {tnow:5.1f} s", (int(12 * K), int(25 * K)), 0.55 * K + 0.1)
    on = drive.filt
    badge = "ASSISTANT ON  (3.5 Hz filter + damped IK)" if on else "ASSISTANT OFF  (raw tremor, plain IK)"
    (tw_, _), _ = cv2.getTextSize(badge, FONT, 0.5 * K + 0.1, 1)
    bx1 = w - int(12 * K)
    bar(img, bx1 - tw_ - int(16 * K), int(4 * K), bx1, int(32 * K), (50, 140, 50) if on else (50, 50, 190), 0.9)
    txt(img, badge, (bx1 - tw_ - int(8 * K), int(25 * K)), 0.5 * K + 0.1)
    # depth bar: where the needle is, where it must go
    bx0, bx2 = int(0.29 * w), int(0.71 * w)
    by0, by1 = int(46 * K), int(66 * K)
    span = STAND_LEN + U_MAX

    def X(uu):
        return int(bx0 + (uu + STAND_LEN) / span * (bx2 - bx0))
    bar(img, X(-STAND_LEN), by0, X(0), by1, (110, 110, 110), 0.85)
    bar(img, X(0), by0, X(PATH_LEN), by1, (150, 120, 220), 0.85)
    bar(img, X(PATH_LEN), by0, X(U_MAX), by1, (60, 60, 220), 0.85)
    cv2.line(img, (X(0), by0 - 4), (X(0), by1 + 4), (60, 220, 255), 2)
    cv2.line(img, (X(PATH_LEN), by0 - 4), (X(PATH_LEN), by1 + 4), (110, 255, 130), 2)
    txt(img, "outside", (X(-STAND_LEN) + 4, by1 + int(14 * K)), 0.36 * K + 0.1, (220, 220, 220))
    txt(img, "ENTRY", (X(0) - int(20 * K), by1 + int(14 * K)), 0.36 * K + 0.1, (60, 220, 255))
    txt(img, "TARGET", (X(PATH_LEN) - int(26 * K), by1 + int(14 * K)), 0.36 * K + 0.1, (110, 255, 130))
    txt(img, "too deep", (X(U_MAX) - int(52 * K), by0 - int(5 * K)), 0.32 * K + 0.1, (140, 140, 255))
    xt = int(np.clip(X(u), bx0, bx2))
    cv2.circle(img, (xt, (by0 + by1) // 2), int(7 * K), (60, 60, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (xt, (by0 + by1) // 2), int(7 * K), (255, 255, 255), 1, cv2.LINE_AA)
    sp = drive.speed_mm_s()
    info1 = (f"depth {max(u, 0) * 1e3:4.1f} mm of {PATH_LEN * 1e3:.0f}   to target {dist:5.1f} mm   "
             + ("moving" if abs(sp) > 0.05 else "stopped"))
    (tw2, _), _ = cv2.getTextSize(info1, FONT, 0.42 * K + 0.1, 1)
    txt(img, info1, ((bx0 + bx2 - tw2) // 2, by1 + int(34 * K)), 0.42 * K + 0.1, (230, 230, 230))
    # warning banner + beep light
    if alerts["deep"]:
        msg, col = "TOO DEEP - back the needle out", (40, 40, 200)
    elif alerts["lateral"]:
        msg, col = f"OFF THE LINE: {lat:.0f} um (limit {LAT_LIMIT_UM:.0f})", (30, 110, 220)
    elif alerts["sing"]:
        msg, col = "robot arm near a singular pose", (30, 130, 200)
    elif alerts["target"]:
        msg, col = f"ON TARGET: {dist * 1000:.0f} um from the centre (limit 500 um)", (50, 130, 50)
    elif drive.phase == "ALIGN":
        msg, col = "Robot aligning the needle with the planned path", (110, 80, 30)
    else:
        msg, col = ("press W to move the needle toward the head" if u < 0 else "inside the head: W/S 1 mm, E/D 0.1 mm - watch the beeps"), (110, 80, 30)
    bar(img, bx0, by1 + int(42 * K), bx2, by1 + int(68 * K), col, 0.9)
    (tw3, _), _ = cv2.getTextSize(msg, FONT, 0.5 * K + 0.1, 1)
    txt(img, msg, ((bx0 + bx2 - tw3) // 2, by1 + int(61 * K)), 0.5 * K + 0.1)
    lit = time.time() < beeper.flash
    cv2.circle(img, (bx2 + int(18 * K), by1 + int(55 * K)), int(9 * K), (0, 230, 255) if lit else (70, 70, 70), -1, cv2.LINE_AA)
    txt(img, "beep", (bx2 + int(32 * K), by1 + int(60 * K)), 0.36 * K + 0.1)
    # micro view (left) and help
    ms = int(0.17 * w)
    rel = tip - target
    xy = (float(rel @ _E1) * 1e3, float(rel @ _E2) * 1e3)
    drive_trail.append(xy)
    img[int(46 * K):int(46 * K) + ms, int(12 * K):int(12 * K) + ms] = micro_view(ms, list(drive_trail), xy)
    bar(img, int(12 * K), int(46 * K) + ms, int(12 * K) + ms, int(46 * K) + ms + int(22 * K), (40, 30, 20), 0.75)
    txt(img, "tip in target plane (orange 0.5 mm)", (int(16 * K), int(46 * K) + ms + int(16 * K)), 0.34 * K + 0.1)
    if show_help:
        hy = int(46 * K) + ms + int(34 * K)
        bar(img, int(12 * K), hy, int(12 * K) + ms + int(40 * K), hy + int(17 * K) * len(HELP) + 8, (35, 28, 22), 0.75)
        for i, s in enumerate(HELP):
            txt(img, s, (int(18 * K), hy + int(15 * K) + i * int(17 * K)), 0.36 * K + 0.1, (255, 230, 140) if i == 0 else (230, 230, 230))
    txt(img, f"tremor {drive.amp:.0f} um   skin {'solid' if skin_solid else 'see-through'}", (int(14 * K), h - int(10 * K)), 0.42 * K + 0.1)
    mis = "MISSION: bring the red TIP dot onto the green TARGET ball (within 0.5 mm).   W/S move   F assistant on/off   R pull out"
    bar(img, int(10 * K), h - int(44 * K), w - int(10 * K), h - int(22 * K), (110, 80, 30), 0.85)
    (tw4, _), _ = cv2.getTextSize(mis, FONT, 0.46 * K + 0.1, 1)
    txt(img, mis, ((w - tw4) // 2, h - int(28 * K)), 0.46 * K + 0.1)
    # needle's-eye inset + legend
    isz = int(0.20 * w)
    ix, iy = w - isz - int(12 * K), int(46 * K)
    if inset is not None:
        img[iy:iy + isz, ix:ix + isz] = inset
        bar(img, ix, iy + isz, ix + isz, iy + isz + int(26 * K), (40, 30, 20), 0.78)
        txt(img, "NEEDLE'S-EYE (real MRI)", (ix + 5, iy + isz + int(18 * K)), 0.40 * K + 0.1)
    legend(img, w - int(262 * K), iy + isz + int(36 * K), K)
    return img


drive_trail = deque(maxlen=60)


def save_outputs(sess, drive, last_frame):
    out = sc.ROOT / "docs" / "doctor_log"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out / f"{a.case}_{a.scenario}_{stamp}"
    with open(str(base) + "_telemetry.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t_s", "depth_mm", "lateral_um", "dist_to_target_mm", "assistant_on", "tremor_um", "manipulability", "needle_force_N", "tip_x", "tip_y", "tip_z"])
        wr.writerows(sess.rows)
    with open(str(base) + "_events.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t_s", "event", "detail"])
        wr.writerows(sess.events)
    summ = dict(case=a.case, scenario=a.scenario, seed=a.seed, session_s=round(drive.t, 1),
                time_inside_head_s=round(sess.n_in_tissue * FRAME_DT, 1), time_on_target_s=round(sess.n_hit * FRAME_DT, 1),
                max_depth_mm=round(sess.max_u, 2), max_lateral_in_head_um=round(sess.max_lat_in, 0),
                mean_lateral_assistant_on_um=None if sess.mean(True) != sess.mean(True) else round(sess.mean(True), 1),
                mean_lateral_assistant_off_um=None if sess.mean(False) != sess.mean(False) else round(sess.mean(False), 1),
                warnings=sess.warn_counts, n_events=len(sess.events))
    Path(str(base) + "_summary.json").write_text(json.dumps(summ, indent=2))
    if last_frame is not None:
        cv2.imwrite(str(base) + "_dashboard.png", last_frame)
    print("\nSession saved:")
    for s in ("_telemetry.csv", "_events.csv", "_summary.json", "_dashboard.png"):
        print("  ", str(base) + s)
    print(json.dumps(summ, indent=2))


# ------------------------------------------------------------------ main loop
def scripted_keys(t, done):
    """--selftest: a scripted doctor (so the mode can be tested without a window)."""
    plan_ = ([(3.2 + 0.06 * i, "w") for i in range(12)] + [(5.5 + 0.08 * i, "w") for i in range(6)] + [(7.0, "s"), (7.1, "s"), (7.3, "e"), (7.5, "d"),
             (8.0, "l"), (8.1, "l"), (8.2, "l"), (8.3, "l"), (9.5, "z"), (10.0, "f"), (11.0, " "), (12.0, "q")])
    for te, k_ in plan_:
        if t >= te and (te, k_) not in done:
            done.add((te, k_))
            return k_
    return None


def main():
    drive = Drive()
    sess = Session()
    beeper = Beeper(a.mute or a.selftest)
    beeper.start()
    writer = None
    if a.record:
        Path(a.record).parent.mkdir(parents=True, exist_ok=True)
    cam_pos, cam_tgt = preset("2", target)
    cam_key = "2"
    show_help, last_frame, inset, nframe, done = True, None, None, 0, set()
    global skin_solid
    win = "Doctor drives - the needle is yours"
    if not a.selftest:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print(__doc__.split("KEYS")[1].split("HONESTY")[0].replace("\n       ", "\n  "))
    sess.event(0.0, "SESSION_START", f"{a.case}/{a.scenario} seed={a.seed} tremor={a.amp:.0f}um")
    sess.event(0.0, "ASSISTANT_ON", "default")
    t_wall0 = time.time()
    running = True
    while running:
        tf = time.time()
        for _ in range(FRAME_STEPS):
            drive.physics_step()
        t = drive.t
        d = drive.d
        tip = d.site_xpos[tip_id].copy()
        rel = tip - target
        lat = float(np.linalg.norm(rel - (rel @ ins) * ins)) * 1e6
        dist = float(np.linalg.norm(rel)) * 1e3
        u = float((tip - entry) @ ins)
        alerts = {k_: False for k_ in ("lateral", "deep", "sing", "target")}
        if drive.phase == "DRIVE":
            alerts = sess.update(t, u, lat, dist, drive.w, drive.filt, drive.amp, drive.F, tip)
        beeper.dist = dist
        beeper.mode = ("deep" if alerts["deep"] else "lateral" if alerts["lateral"] else "sing" if alerts["sing"] else
                       "target" if alerts["target"] else "prox" if (drive.phase == "DRIVE" and u > 0) else "off")
        # camera: smooth glide to the chosen preset
        dp, dt_ = preset(cam_key, tip)
        cam_pos = cam_pos + (dp - cam_pos) * 0.12
        cam_tgt = cam_tgt + (dt_ - cam_tgt) * 0.12
        look(cam_pos, cam_tgt)
        renderer.update_scene(d, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        proj = proj_from_scene()
        if vol is not None and nframe % 3 == 0 and drive.phase == "DRIVE":
            inset = needle_eye(tip, int(0.20 * W_))
        top = compose(img, drive, sess, alerts, beeper, proj, tip, lat, dist, u, t, inset, show_help and t < 600)
        frame = np.concatenate([top, dashboard(sess, drive, t)], 0)
        last_frame = frame
        if a.record:
            if writer is None:
                writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / FRAME_DT, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        if a.stills and nframe % int(3.0 / FRAME_DT) == 0:
            Path(a.stills).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(a.stills) / f"drive_{t:05.1f}.png"), frame)
        nframe += 1
        # keys
        if a.selftest:
            k = scripted_keys(t, done)
        else:
            cv2.imshow(win, frame)
            kk = cv2.waitKey(1) & 0xFF
            k = chr(kk).lower() if kk != 255 else None
            if kk == 27:
                k = "q"
            if kk == 32:
                k = " "
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                k = "q"
        if k:
            if k == "q":
                running = False
            elif k in ("w", "s"):
                g = drive.goal
                if k == "w":
                    new = min(g + 5e-3, 0.0) if g < -1e-6 else g + 1e-3      # outside: 5 mm steps that land exactly on the entry point
                else:
                    new = max(g - 1e-3, 0.0) if g > 1e-6 else g - 5e-3
                drive.goal = float(np.clip(new, -STAND_LEN, U_MAX))
                sess.event(t, "ADVANCE" if k == "w" else "BACK", f"goal depth={drive.goal * 1e3:.1f}mm")
            elif k in ("e", "d"):
                drive.move_goal(1e-4 if k == "e" else -1e-4)
                sess.event(t, "FINE_ADVANCE" if k == "e" else "FINE_BACK", f"goal depth={drive.goal * 1e3:.1f}mm")
            elif k == "r":
                drive.goal = -STAND_LEN
                sess.event(t, "PULL_OUT", "")
            elif k == " ":
                drive.goal = drive.u
                sess.event(t, "STOP", f"depth={drive.u * 1e3:.1f}mm")
            elif k == "f":
                drive.set_filter(not drive.filt)
                sess.event(t, "ASSISTANT_ON" if drive.filt else "ASSISTANT_OFF", f"depth={u * 1e3:.1f}mm")
            elif k in "jlik":
                drive.off += {"j": (-1e-4, 0), "l": (1e-4, 0), "i": (0, 1e-4), "k": (0, -1e-4)}[k]
                sess.event(t, "NUDGE", f"offset=({drive.off[0] * 1e3:+.1f},{drive.off[1] * 1e3:+.1f}) mm")
            elif k == "z":
                drive.off[:] = 0
                sess.event(t, "CENTRE", "")
            elif k == "[":
                drive.set_amp(drive.amp - 25)
                sess.event(t, "TREMOR", f"{drive.amp:.0f}um")
            elif k == "]":
                drive.set_amp(drive.amp + 25)
                sess.event(t, "TREMOR", f"{drive.amp:.0f}um")
            elif k in "123456":
                cam_key = k
            elif k == "t":
                skin_solid = not skin_solid
                set_skin(0.95 if skin_solid else 0.26)
            elif k == "h":
                show_help = not show_help
        # real-time pacing
        if not a.selftest:
            left = FRAME_DT - (time.time() - tf)
            if left > 0:
                time.sleep(left)
        if a.selftest and t > 40:
            running = False
    beeper.alive = False
    if writer is not None:
        writer.release()
        print("session video:", a.record)
    sess.event(drive.t, "SESSION_END", "")
    if not a.selftest:
        cv2.destroyAllWindows()
    save_outputs(sess, drive, last_frame)


if __name__ == "__main__":
    main()