"""Render the demo film (MP4) from the SAME physics used for the experiments.
    python src\\make_film.py --case c160 --scenario enhancing_superior --nifti D:\\robotics\\brats_cases
Options:  --list (show all cases/scenarios)   --quick (small and fast, to check the flow)   --live (play in the MuJoCo window)
          --conds BASE,PROP   --amp 150 --seed 2000   --out path.mp4   --no-play
Storyboard: title -> operating room, robot arrives -> camera closes on the patient -> head turns see-through, lesion shown ->
planned path -> explainer card -> run WITHOUT assistance -> run WITH assistance (same tremor, same seed) -> results card.
Every number on screen is computed (live physics, or read from analytics/surgical_real_trials.csv)."""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd

import scene_build as sb
import surgical_tremor_controller as sc

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="c160")
ap.add_argument("--scenario", default="enhancing_superior")
ap.add_argument("--nifti", default=r"D:\robotics\brats_cases")
ap.add_argument("--conds", default="BASE,PROP")
ap.add_argument("--amp", type=float, default=150.0, help="tremor amplitude, um (per-axis peak)")
ap.add_argument("--seed", type=int, default=2000)
ap.add_argument("--w", type=int, default=1280)
ap.add_argument("--h", type=int, default=720)
ap.add_argument("--stride", type=int, default=2, help="render every n-th 16 ms physics tick (2 = 31 fps)")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--no-intro", action="store_true")
ap.add_argument("--list", action="store_true", help="list the cases and scenarios that can be played, then exit")
ap.add_argument("--slow", type=float, default=1.5, help="slow-motion factor (1 = real time, 2 = half speed)")
ap.add_argument("--live", action="store_true", help="play in the interactive MuJoCo window (real time) instead of writing an mp4")
ap.add_argument("--no-play", action="store_true", help="do not open the film when it is finished")
ap.add_argument("--out", default=str(sc.ROOT / "docs" / "film" / "demo_film.mp4"))
ap.add_argument("--stills", default=None, help="folder: also save a still every 2 s (for checking)")
a = ap.parse_args()
if a.list:
    for fn in sorted((sc.ROOT / "analytics" / "cases").glob("*/scenarios.json")):
        sl = json.loads(fn.read_text())
        ok = [x["id"] for x in sl if x["status"] == "REACHABLE"]
        print(f"{fn.parent.name}: {len(ok)} reachable   e.g. " + ", ".join(ok[:6]))
    raise SystemExit(0)
if a.quick:
    a.w, a.h, a.stride = 640, 360, 6
W_, H_ = a.w, a.h
FRAME_DT = 0.016 * a.stride
FPS = 1.0 / (FRAME_DT * a.slow)
INTRO_T = 14.0                       # seconds of intro animation (simulated time)

# ------------------------------------------------------------------ model, plan, volume
f = sc.ROOT / "analytics" / "cases" / a.case / "scenarios.json"
_all = json.loads(f.read_text())
_hit = [s for s in _all if s["id"] == a.scenario]
if not _hit:
    raise SystemExit(f"scenario '{a.scenario}' not in {a.case}. Available: " + ", ".join(s["id"] for s in _all))
scn = _hit[0]
if scn["status"] != "REACHABLE":
    raise SystemExit(f"{a.case}/{a.scenario} is {scn['status']} (planner rejected it) - pick a REACHABLE one (see --list)")
plan = sc.plan_from_scenario(scn)
m, info = sb.load_visual(a.case, plan, a.nifti, True)
nd = sb.find_nifti_dir(a.case, a.nifti)
meta = sb.case_paths(a.case)[2]
vol = sb.load_volume(nd) if nd is not None else None
sgn = np.array(info["sgn"], float)
target = np.asarray(plan["target"], float)
ins = np.asarray(plan["ins"], float)
entry = np.asarray(plan["entry"], float)
robot = sc.NeedleRobot(m)
tip_id = robot.tip
link0 = m.body("p_link0").id
base0 = m.body_pos[link0].copy()
cart_ids = [m.geom(n).id for n in info["cart_geoms"]]
cart0 = m.geom_pos[cart_ids].copy()
mat_skin = m.material("skin").id
SKIN_MATS = [m.material(n).id for n in ("skin", "hair", "eye")]       # head, hair and eyes fade together
SKIN_RUN = 0.26                                                      # see-through level during the runs


def set_skin(alpha):
    for _i in SKIN_MATS:
        m.mat_rgba[_i, 3] = alpha


mat_path, mat_ent, mat_tgt = (m.material(n).id for n in ("mark_path", "mark_entry", "mark_target"))
ALPHA0 = {mat_skin: 1.0, mat_path: float(m.mat_rgba[mat_path, 3]),
          mat_ent: 1.0, mat_tgt: 1.0}
_nid = m.geom('p_needle_shaft').id
m.geom_size[_nid, 0] = 0.0024                              # drawn thicker so it is visible on screen (visual only)
m.geom_matid[_nid] = -1
m.geom_rgba[_nid] = (1.0, 0.55, 0.05, 1.0)                 # bright orange needle
m.site_size[tip_id] = 0.0045                               # big red tip marker (visual only)
m.site_rgba[tip_id] = (1.0, 0.1, 0.1, 1.0)
m.geom_size[m.geom('traj_line').id, 0] = 0.0006
renderer = mujoco.Renderer(m, H_, W_)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE

# ------------------------------------------------------------------ drawing helpers
FONT = cv2.FONT_HERSHEY_DUPLEX
TEAL = (150, 130, 15)       # BGR
ORNG = (60, 160, 255)


def txt(img, s, org, scale=0.6, col=(255, 255, 255), th=1):
    th = max(1, int(round(th * img.shape[0] / 720)))
    o = max(1, th)                      # outline = same-thickness text shifted (keeps glyph advance identical)
    for dx, dy in ((-o, 0), (o, 0), (0, -o), (0, o), (-o, -o), (o, o), (-o, o), (o, -o)):
        cv2.putText(img, s, (org[0] + dx, org[1] + dy), FONT, scale, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, col, th, cv2.LINE_AA)


def bar(img, x0, y0, x1, y1, col, alpha):
    ov = img.copy()
    cv2.rectangle(ov, (x0, y0), (x1, y1), col, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def caption(img, main, sub=None):
    h, w = img.shape[:2]
    y0 = h - int(h * 0.15)
    bar(img, 0, y0, int(w * 0.80), h - int(h * 0.04), TEAL, 0.80)
    txt(img, main, (int(w * 0.02), y0 + int(h * 0.055)), h / 1000 * 1.05, (255, 255, 255), 1)
    if sub:
        txt(img, sub, (int(w * 0.02), y0 + int(h * 0.105)), h / 1000 * 0.72, (225, 245, 245), 1)


def look(pos, tgt, c=None):
    c = cam if c is None else c
    pos, tgt = np.asarray(pos, float), np.asarray(tgt, float)
    fwd = tgt - pos
    dist = np.linalg.norm(fwd)
    fwd /= dist
    c.lookat[:] = tgt
    c.distance = dist
    c.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    c.elevation = float(np.degrees(np.arcsin(fwd[2])))


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def lerp(p, q, u):
    return (1 - u) * np.asarray(p, float) + u * np.asarray(q, float)


def needle_eye(tip, S_out):
    """MRI re-slice through the tip along the needle axis (real scan), 'what the needle sees'."""
    S = 200
    w1 = np.cross(ins, [0, 0, 1.0])
    if np.linalg.norm(w1) < 1e-6:
        w1 = np.array([1.0, 0, 0])
    w1 /= np.linalg.norm(w1)
    L = 0.06
    rgba = sb.sample_plane(vol, meta, tip, w1, -ins, L, S, alpha_outside=False)[..., :3]
    img = cv2.cvtColor(rgba, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (S_out, S_out), interpolation=cv2.INTER_CUBIC)
    sc_ = S_out / L
    cx = cy = S_out // 2
    rel = target - tip
    px, py = int(cx + (rel @ w1) * sc_), int(cy + (rel @ (-ins)) * sc_)
    cv2.line(img, (cx, 0), (cx, cy), (255, 255, 255), 1, cv2.LINE_AA)               # needle
    cv2.circle(img, (cx, cy), 5, (60, 60, 255), -1, cv2.LINE_AA)                     # tip
    if 0 <= px < S_out and 0 <= py < S_out:
        cv2.drawMarker(img, (px, py), (90, 255, 120), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)   # target
    txt(img, "TIP", (cx + 9, cy + 5), 0.4, (80, 80, 255))
    if 0 <= px < S_out and 0 <= py < S_out:
        txt(img, "TARGET", (min(px + 11, S_out - 62), max(py - 8, 14)), 0.4, (90, 255, 120))
    cv2.rectangle(img, (0, 0), (S_out - 1, S_out - 1), (255, 255, 255), 1)
    return img


_E1 = np.cross(ins, [0, 0, 1.0])
_E1 = _E1 / np.linalg.norm(_E1) if np.linalg.norm(_E1) > 1e-6 else np.array([1.0, 0, 0])
_E2 = np.cross(ins, _E1)


def micro_view(S_out, trail, tip_xy):
    """Tip position in the plane perpendicular to the planned line, +-0.6 mm window, 0.5 mm acceptance circle."""
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
    cv2.circle(img, (int(c + tip_xy[0] * k), int(c - tip_xy[1] * k)), 4, (80, 230, 255), -1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (S_out - 1, S_out - 1), (255, 255, 255), 1)
    return img


def proj_from_scene():
    """Camera of the frame that was just rendered (used to pin text labels onto 3D points)."""
    c0 = renderer.scene.camera[0]
    return np.array(c0.pos, float), np.array(c0.forward, float), np.array(c0.up, float)


def to_px(proj, p, w, h):
    pos, fwd, up = proj
    right = np.cross(fwd, up)
    q = np.asarray(p, float) - pos
    z = q @ fwd
    if z < 0.05:
        return None
    f = (h / 2) / np.tan(np.radians(m.vis.global_.fovy) / 2)
    return int(w / 2 + f * (q @ right) / z), int(h / 2 - f * (q @ up) / z)


def tag(img, proj, p, text, col, off=(40, -40)):
    h, w = img.shape[:2]
    k = h / 720.0
    xy = to_px(proj, p, w, h)
    if xy is None or not (0 <= xy[0] < w and 0 <= xy[1] < h):
        return
    o = (int(xy[0] + off[0] * k), int(xy[1] + off[1] * k))
    cv2.circle(img, xy, int(7 * k), col, 2, cv2.LINE_AA)
    cv2.line(img, xy, o, col, 2, cv2.LINE_AA)
    txt(img, text, (o[0] + 4, o[1] - 4), 0.5 * k + 0.1, col)


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


class Sink:
    """Receives a call every 16 ms of simulated time (run_trial's viewer.sync) and writes film frames."""

    def __init__(self, writer, d, cond, cam_fn, cap_main, cap_sub):
        self.wr, self.d, self.cond, self.cam_fn = writer, d, cond, cam_fn
        self.cap_main, self.cap_sub = cap_main, cap_sub
        self.n = 0
        self.lat = deque(maxlen=int(3.0 / FRAME_DT))
        self.trail = deque(maxlen=int(2.0 / FRAME_DT))
        self.last = None
        self.proj = None

    def sync(self):
        self.n += 1
        if self.n % a.stride:
            return
        d = self.d
        t = d.time
        tip = d.site_xpos[tip_id].copy()
        rel = tip - target
        lat = np.linalg.norm(rel - (rel @ ins) * ins) * 1e6                          # um from the planned line
        dist = np.linalg.norm(rel) * 1e3
        depth = max(0.0, float((tip - entry) @ ins)) * 1e3
        self.lat.append(lat)
        look(*self.cam_fn(t))
        renderer.update_scene(d, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        self.proj = proj_from_scene()
        self.last = (img.copy(), t, lat, dist, depth)
        self.wr.write(self.compose(img.copy(), t, lat, dist, depth))
        if int(t * 1000) % 1600 < int(FRAME_DT * 1000):
            print(f"  {self.cond}: {t:4.1f} s of 12 s  ({time.time() - t0:.0f}s elapsed)", flush=True)
        if a.stills and int(t * 1000) % 2000 < int(FRAME_DT * 1000):
            cv2.imwrite(str(Path(a.stills) / f"{self.cond}_{t:05.2f}.png"), self.compose(img.copy(), t, lat, dist, depth))

    def compose(self, img, t, lat, dist, depth, final=None):
        h, w = img.shape[:2]
        k_ = h / 720.0
        state = ("ALIGN" if t < sc.T_ALIGN else "APPROACH" if t < sc.T_ALIGN + sc.T_APPROACH else
                 "INSERT" if t < sc.T_ALIGN + sc.T_APPROACH + sc.T_INSERT else
                 "HOLD" if t < sc.T_ALIGN + sc.T_APPROACH + sc.T_INSERT + sc.T_HOLD else "RETRACT")
        if self.proj is not None:
            tag(img, self.proj, entry, "ENTRY", (60, 220, 255), (-70, -45))
            tag(img, self.proj, target, "TARGET", (110, 255, 130), (60, 50))
            tag(img, self.proj, self.d.site_xpos[tip_id], "NEEDLE TIP", (80, 80, 255), (70, -60))
        caption(img, self.cap_main, final or self.cap_sub)
        bar(img, 0, 0, w, int(36 * k_), (40, 30, 20), 0.65)
        txt(img, f"{a.case}  |  {a.scenario}  |  planner: {scn['status']}  |  state: {state}  |  t = {t:4.1f} s",
            (int(12 * k_), int(25 * k_)), 0.55 * k_ + 0.1)
        # live deviation trace (top-left)
        pw, ph = int(0.26 * w), int(0.16 * h)
        x0, y0 = int(12 * k_), int(46 * k_)
        bar(img, x0, y0, x0 + pw, y0 + ph, (30, 25, 20), 0.72)
        txt(img, "deviation from line", (x0 + 6, y0 + int(17 * k_)), 0.42 * k_ + 0.1)
        vals = np.array(self.lat) if len(self.lat) > 1 else np.zeros(2)
        top = 200.0
        pts = [(x0 + 8 + int((pw - 16) * i / max(1, self.lat.maxlen - 1)),
                y0 + ph - 8 - int((ph - int(30 * k_)) * min(v, top) / top)) for i, v in enumerate(vals)]
        cv2.polylines(img, [np.array(pts, np.int32)], False, (80, 200, 255), 2, cv2.LINE_AA)
        txt(img, f"{lat:4.0f} um" if lat < 3000 else "approaching", (x0 + pw - int(110 * k_), y0 + int(17 * k_)), 0.5 * k_ + 0.1, (80, 200, 255))
        # micro view (below the trace)
        ms = int(0.20 * w)
        my0 = y0 + ph + int(8 * k_)
        rel = self.d.site_xpos[tip_id] - target
        xy = (float(rel @ _E1) * 1e3, float(rel @ _E2) * 1e3)
        self.trail.append(xy)
        mv = micro_view(ms, list(self.trail), xy)
        img[my0:my0 + ms, x0:x0 + ms] = mv
        bar(img, x0, my0 + ms, x0 + ms, my0 + ms + int(22 * k_), (40, 30, 20), 0.75)
        txt(img, "tip in target plane (orange = 0.5 mm)", (x0 + 4, my0 + ms + int(16 * k_)), 0.38 * k_ + 0.1)
        # live status banner (what the needle is doing right now)
        path_mm = float(np.linalg.norm(target - entry)) * 1e3
        if state == "APPROACH":
            msg, bc = "Needle moving to the entry point", (110, 80, 30)
        elif state == "INSERT":
            msg, bc = f"INSERTING  depth {depth:4.1f} mm of {path_mm:.0f} mm  -  {dist:4.1f} mm to go", (110, 80, 30)
        elif state == "HOLD":
            ok_ = dist < 0.5
            msg, bc = (f"AT TARGET  tip is {dist * 1000:.0f} um from the target centre   (limit 500 um)" if ok_ else
                       f"holding near target ({dist:.2f} mm away)"), (40, 120, 40)
        elif state == "RETRACT":
            msg, bc = "Needle withdrawing", (110, 80, 30)
        else:
            msg, bc = "Robot aligning the needle with the planned path", (110, 80, 30)
        bx0, bx1 = int(0.285 * w), int(0.715 * w)
        bar(img, bx0, int(44 * k_), bx1, int(74 * k_), bc, 0.85)
        (tw_, _), _ = cv2.getTextSize(msg, FONT, 0.5 * k_ + 0.1, 1)
        txt(img, msg, ((bx0 + bx1 - tw_) // 2, int(64 * k_)), 0.5 * k_ + 0.1)
        legend(img, w - int(262 * k_), int(46 * k_) + int(0.21 * w) + int(58 * k_), k_)
        # needle's-eye inset (top-right)
        if vol is not None and state != "ALIGN":
            isz = int(0.21 * w)
            inset = needle_eye(self.d.site_xpos[tip_id].copy(), isz)
            ix, iy = w - isz - int(12 * k_), int(46 * k_)
            img[iy:iy + isz, ix:ix + isz] = inset
            bar(img, ix, iy + isz, ix + isz, iy + isz + int(50 * k_), (40, 30, 20), 0.78)
            txt(img, "NEEDLE'S-EYE (real MRI)", (ix + 5, iy + isz + int(18 * k_)), 0.40 * k_ + 0.1)
            txt(img, f"depth {depth:5.1f}   to target {dist:5.1f} mm", (ix + 5, iy + isz + int(40 * k_)), 0.46 * k_ + 0.1, (120, 255, 150))
        return img


# ------------------------------------------------------------------ the film
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True) if False else out.parent.mkdir(parents=True, exist_ok=True)
if a.stills:
    Path(a.stills).mkdir(parents=True, exist_ok=True)
wr = None if a.live else cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W_, H_))
t0 = time.time()


def title_card(lines, secs, fade_in=True):
    if a.live:
        return
    n = int(secs * FPS)
    for k in range(n):
        img = np.zeros((H_, W_, 3), np.uint8)
        img[:] = (45, 35, 25)
        bar(img, 0, int(H_ * 0.40), W_, int(H_ * 0.405), TEAL, 1.0)
        for i, (s, scl, col) in enumerate(lines):
            (tw, _), _ = cv2.getTextSize(s, FONT, H_ / 1000 * scl, 1)
            txt(img, s, ((W_ - tw) // 2, int(H_ * (0.32 + 0.11 * i)) if i < 1 else int(H_ * (0.50 + 0.07 * (i - 1)))),
                H_ / 1000 * scl, col, 1)
        al = min(1.0, k / (0.6 * FPS)) * min(1.0, (n - k) / (0.5 * FPS)) if fade_in else 1.0
        wr.write((img * al).astype(np.uint8))


title_card([("7-DOF Surgical Robotic Assistant", 1.6, (255, 255, 255)),
            ("Physiological tremor filtering for sub-0.5 mm needle placement", 0.85, (235, 230, 140)),
            ("MuJoCo physics  |  damped least-squares IK  |  plan-relative low-pass filter", 0.62, (210, 210, 210)),
            ("Group 10  -  Soumya Singh (E071)  and  Harshal Khandekar (E033)", 0.70, (255, 255, 255)),
            ("MDRIIA 702CO0E012  -  NMIMS MPSTME", 0.58, (190, 190, 190))], 3.2)


def orbit(az, dist, el_deg):
    """Camera position on a sphere around the target (az in radians, 0 = +x, pi/2 = +y = in front of the patient's face)."""
    el = np.radians(el_deg)
    return target + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def _arc(a0, a1, u):
    d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi          # shortest way round
    return a0 + d * u


AZ_SIDE = float(np.arctan2(sgn[1], sgn[0]))              # the side the cut-away opens towards
AZ_FACE = float(np.arctan2(0.9, 0.7))                    # front-right of the patient: face, body and robot all in view
AZ_END = _arc(AZ_FACE, AZ_SIDE, 0.45)                    # the run ends only part-way round (calm, no big swings)
KEY_START = (AZ_FACE, 0.72, 18.0)                        # wide: patient + robot + needle (intro ends here, run starts here)
KEY_END = (AZ_END, 0.46, 24.0)                           # slow dolly-in over the run


def _key(k0, k1, u):
    u = smooth(u)
    return orbit(_arc(k0[0], k1[0], u), (1 - u) * k0[1] + u * k1[1], (1 - u) * k0[2] + u * k1[2])


def cam_intro(t):
    T = target
    k0 = (T + np.array([1.9, 2.3, 0.85]), T + np.array([-0.30, 0.0, -0.30]))
    k1 = (T + np.array([0.95, 1.15, 0.30]), T + np.array([-0.15, 0.0, -0.20]))
    k2 = (orbit(*KEY_START), T)
    if t < 6.0:
        u = smooth(t / 6.0); return lerp(k0[0], k1[0], u), lerp(k0[1], k1[1], u)
    u = smooth((t - 6.0) / (INTRO_T - 6.0))
    return lerp(k1[0], k2[0], u), lerp(k1[1], k2[1], u)


def cam_run(t):
    """One slow dolly-in over the whole 12 s run: the head, the arm and the needle stay in view."""
    return _key(KEY_START, KEY_END, t / 12.0), target


def intro_state(t):
    """Intro animation: robot rolls in, camera closes on the patient (opaque), THEN the head turns see-through,
    THEN the planned path, entry and target appear."""
    u = smooth((t - 0.4) / 3.6)
    off = -1.7 * (1 - u)
    m.body_pos[link0] = base0 + np.array([off, 0, 0])
    m.geom_pos[cart_ids] = cart0 + np.array([off, 0, 0])
    set_skin(1.0 if t < 8.0 else 1.0 + (SKIN_RUN - 1.0) * smooth((t - 8.0) / 3.0))
    pa = smooth((t - 11.0) / 1.8)
    m.mat_rgba[mat_path, 3] = ALPHA0[mat_path] * pa
    for mid in (mat_ent, mat_tgt):
        m.mat_rgba[mid, 3] = pa


def hud_image(cond, t, lat, dist, depth, lat_hist, trail):
    """Separate HUD window for the live mode (same panels as in the film)."""
    Wd, Hd = 1000, 470
    img = np.full((Hd, Wd, 3), (30, 25, 20), np.uint8)
    state = ("ALIGN" if t < sc.T_ALIGN else "APPROACH" if t < sc.T_ALIGN + sc.T_APPROACH else
             "INSERT" if t < sc.T_ALIGN + sc.T_APPROACH + sc.T_INSERT else
             "HOLD" if t < sc.T_ALIGN + sc.T_APPROACH + sc.T_INSERT + sc.T_HOLD else "RETRACT")
    label = "WITHOUT assistance (tremor + pseudo-inverse IK)" if cond == "BASE" else "WITH our assistant (filter + damped IK)"
    txt(img, f"{a.case} | {a.scenario} | {label} | {state} | t = {t:4.1f} s", (10, 28), 0.6)
    pts = [(10 + int(480 * i / max(1, lat_hist.maxlen - 1)), 200 - int(130 * min(v, 200.0) / 200.0)) for i, v in enumerate(lat_hist)]
    cv2.rectangle(img, (10, 50), (490, 205), (60, 50, 40), 1)
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, np.int32)], False, (80, 200, 255), 2, cv2.LINE_AA)
    txt(img, "deviation from planned line (um)", (16, 70), 0.5)
    txt(img, f"{lat:4.0f} um" if lat < 3000 else "approaching", (380, 70), 0.55, (80, 200, 255))
    xy = trail[-1] if len(trail) else (0.0, 0.0)
    img[225:225 + 230, 10:10 + 230] = micro_view(230, list(trail), xy)
    txt(img, "tip in target plane (orange = 0.5 mm)", (10, 462), 0.42)
    if vol is not None and state != "ALIGN":
        img[225:225 + 230, 260:260 + 230] = needle_eye(d0.site_xpos[tip_id].copy(), 230)
        txt(img, "NEEDLE'S-EYE (real MRI)", (500, 250), 0.55)
        txt(img, f"depth {depth:5.1f} mm", (500, 285), 0.6, (120, 255, 150))
        txt(img, f"to target {dist:5.1f} mm", (500, 320), 0.6, (120, 255, 150))
    return img


if a.live:
    import mujoco.viewer
    d0 = mujoco.MjData(m)
    d0.qpos[:7] = sc.HOME
    d0.ctrl[:7] = sc.HOME
    use_hud = True
    try:
        cv2.namedWindow("HUD: tracking and needle's-eye", cv2.WINDOW_AUTOSIZE)
    except Exception as e:
        print("HUD window unavailable:", e)
        use_hud = False
    print("What we compare: the SAME robot, plan and surgeon-hand tremor in both runs.\n"
          "  RUN 1 (BASE): no assistance - the tremor goes straight into the needle.\n"
          "  RUN 2 (PROP): our assistant filters the tremor out before it reaches the needle.")
    print(f"MuJoCo window: intro ({INTRO_T * a.slow:.0f} s) then BASE and PROP in slow motion x{a.slow}. "
          "The HUD window shows the live trace, tip view and needle's-eye. Close the MuJoCo window to stop.")
    with mujoco.viewer.launch_passive(m, d0, show_left_ui=False, show_right_ui=False) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        t_start = time.time()
        while v.is_running() and not a.no_intro:
            t = (time.time() - t_start) / a.slow
            if t > INTRO_T:
                break
            intro_state(t)
            mujoco.mj_forward(m, d0)
            look(*cam_intro(t), c=v.cam)
            v.sync()
            time.sleep(0.01)
        m.body_pos[link0] = base0
        m.geom_pos[cart_ids] = cart0
        set_skin(SKIN_RUN)
        for cond in a.conds.split(","):
            if not v.is_running():
                break
            mujoco.mj_resetData(m, d0)
            lat_hist = deque(maxlen=190)
            trail = deque(maxlen=125)

            class LiveSink:
                n = 0

                def sync(self_):
                    if not v.is_running():
                        raise SystemExit(0)
                    look(*cam_run(d0.time), c=v.cam)
                    v.sync()
                    self_.n += 1
                    if use_hud and self_.n % 2 == 0:
                        tip = d0.site_xpos[tip_id].copy()
                        rel = tip - target
                        lat = np.linalg.norm(rel - (rel @ ins) * ins) * 1e6
                        lat_hist.append(lat)
                        trail.append((float(rel @ _E1) * 1e3, float(rel @ _E2) * 1e3))
                        depth = max(0.0, float((tip - entry) @ ins)) * 1e3
                        cv2.imshow("HUD: tracking and needle's-eye", hud_image(cond, d0.time, lat, np.linalg.norm(rel) * 1e3, depth, lat_hist, trail))
                        cv2.waitKey(1)
                    time.sleep(0.016 * a.slow)

            r = sc.run_trial(m, cond, a.seed, a.amp, 1.0, viewer=LiveSink(), data=d0, plan=plan)["summary"]
            print(f"{cond}: hold RMS {r['hold_rms_mm'] * 1000:.0f} um  hold max {r['hold_max_mm'] * 1000:.0f} um  "
                  f"tremor rejection {r['tremor_rejection_dB']:.1f} dB", flush=True)
            time.sleep(1.5)
        print("Finished. Close the MuJoCo window (the view is free: drag to look around).", flush=True)
        while v.is_running():
            v.sync()
            time.sleep(0.05)
    cv2.destroyAllWindows()
    raise SystemExit(0)


if not a.no_intro:
    d0 = mujoco.MjData(m)
    d0.qpos[:7] = sc.HOME
    d0.ctrl[:7] = sc.HOME
    n = int(INTRO_T / FRAME_DT)
    for k in range(n):
        t = k * FRAME_DT
        intro_state(t)
        mujoco.mj_forward(m, d0)
        look(*cam_intro(t))
        renderer.update_scene(d0, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        if k < int(0.6 / FRAME_DT):
            img = (img * (k / (0.6 / FRAME_DT))).astype(np.uint8)
        if t >= 12.0:
            _pj = proj_from_scene()
            tag(img, _pj, entry, "ENTRY", (60, 220, 255), (-70, -45))
            tag(img, _pj, target, "TARGET", (110, 255, 130), (60, 50))
        if t < 5.0:
            caption(img, "The assistant robot is brought to the patient", "illustrative operating-room scene, simulated in MuJoCo")
        elif t < 8.0:
            caption(img, "The patient is positioned and the head is fixed", "the patient model is illustrative (not a real person)")
        elif t < 11.0:
            caption(img, "Patient anatomy comes from a real MRI scan", f"BraTS {a.case}: tumour regions from the supplied expert annotation (not a diagnosis)")
        else:
            caption(img, "The planner computes the needle path to the tumour", f"{a.scenario}: {scn['status']}  |  reachability, joint limits and singularity margin checked")
        if t >= 12.0:
            legend(img, W_ - int(262 * H_ / 720), int(46 * H_ / 720), H_ / 720)
        wr.write(img)
        if k % max(1, n // 10) == 0:
            print(f"  intro {100 * k // n}%  ({time.time() - t0:.0f}s)", flush=True)
        if a.stills and k % int(2 / FRAME_DT) == 0:
            cv2.imwrite(str(Path(a.stills) / f"intro_{t:05.2f}.png"), img)
    m.body_pos[link0] = base0
    m.geom_pos[cart_ids] = cart0
    print(f"intro done  {time.time() - t0:.0f}s", flush=True)

EXPLAIN = [("What are we comparing?", 1.5, (255, 255, 255)),
           ("The SAME robot, plan and surgeon-hand tremor are used in both runs", 0.78, (235, 230, 140)),
           (f"Tremor model: 8-12 Hz shake, {a.amp:.0f} um peak per axis, plus slow drift (the hand is modelled, not a real person)", 0.60, (200, 200, 200)),
           ("RUN 1   No assistance: the tremor goes straight into the needle", 0.80, (255, 255, 255)),
           ("RUN 2   Our assistant: the tremor is filtered out before it reaches the needle", 0.80, (255, 255, 255)),
           ("Goal: stay within 0.5 mm of the target (orange circle)", 0.70, (150, 255, 170))]
results = {}
set_skin(SKIN_RUN)
for cond in a.conds.split(","):
    d = mujoco.MjData(m)
    if cond == "BASE":
        main, sub = "RUN 1 - NO assistance: the hand tremor goes straight to the needle", "same plan, same tremor as run 2  |  plain pseudo-inverse IK, unfiltered command"
        title_card(EXPLAIN, 9.0)
    else:
        main, sub = "RUN 2 - WITH our assistant: the tremor is filtered out", "same plan, same tremor as run 1  |  3.5 Hz low-pass + damped least-squares IK"
    sink = Sink(wr, d, cond, cam_run, main, sub)
    r = sc.run_trial(m, cond, a.seed, a.amp, 1.0, viewer=sink, data=d, plan=plan)["summary"]
    results[cond] = r
    img, t, lat, dist, depth = sink.last
    res = (f"result: hold RMS {r['hold_rms_mm'] * 1000:.0f} um   hold max {r['hold_max_mm'] * 1000:.0f} um   "
           f"tremor rejection {r['tremor_rejection_dB']:.1f} dB")
    for _ in range(int(2.2 / FRAME_DT)):
        wr.write(sink.compose(img.copy(), t, lat, dist, depth, final=res))
    print(f"{cond}: {res}  [{time.time() - t0:.0f}s]", flush=True)

# ------------------------------------------------------------------ results card (read from the experiment CSV)
csv = sc.ROOT / "analytics" / "surgical_real_trials.csv"
lines = [("Result", 1.5, (255, 255, 255))]
if csv.exists():
    df = pd.read_csv(csv)
    g = df.groupby("cond")
    n = int(g.size().get("PROP", 0))
    hb, hp = g.hold_rms_mm.mean()["BASE"], g.hold_rms_mm.mean()["PROP"]
    rb, rp = g.tremor_rejection_dB.mean()["BASE"], g.tremor_rejection_dB.mean()["PROP"]
    ok = float((df.hold_max_mm < 0.5).mean())
    lines += [(f"N = {n} paired runs per condition, randomised tremor (MuJoCo, 500 Hz)", 0.78, (235, 230, 140)),
              (f"Hold error (RMS):  {hb * 1000:.0f} um  ->  {hp * 1000:.0f} um   ({100 * (hb - hp) / hb:.0f}% lower)", 0.88, (255, 255, 255)),
              (f"Tremor rejection:  {rb:.1f} dB  ->  {rp:.1f} dB", 0.88, (255, 255, 255)),
              (f"All runs under 0.5 mm placement error: {100 * ok:.0f}% of {len(df)}", 0.78, (255, 255, 255)),
              ("Simulation only. Anatomy illustrative (BraTS). Not a clinical device.", 0.62, (190, 190, 190))]
title_card(lines, 5.0)
if wr is not None:
    wr.release()
print(f"saved {out}  ({time.time() - t0:.0f}s)")
if not a.no_play:
    import os
    import subprocess
    import sys
    print("opening the film ...")
    try:
        os.startfile(str(out))                                    # Windows: opens it in your default player
    except AttributeError:
        try:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(out)])
        except Exception as e:
            print("could not open it automatically:", e)