"""Closed-loop 7-DOF needle positioning under simulated hand tremor (MuJoCo).

Group 10 - MDRIIA capstone.  One trial = ALIGN -> APPROACH -> INSERT -> HOLD -> RETRACT.
The tip must follow a planned straight path to the lesion target while a simulated
surgeon-hand disturbance (7-12 Hz multi-sine tremor + slow drift) is added to the command.

Conditions
  NT       no tremor, DLS-IK                      (robot's own tracking floor)
  BASE     tremor, raw command, plain pseudoinverse IK, no null-space   (baseline)
  PROP     tremor, plan-relative low-pass filter, adaptive-damped DLS-IK + null-space (proposed)
  DLSONLY  tremor, raw command, DLS-IK + null-space                     (ablation)
  LPFRAW   tremor, causal low-pass on the raw command, DLS-IK           (ablation: lag cost)

Everything logged comes from the MuJoCo state.  Nothing here is a hard-coded result.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "surgical_7dof_robot.xml"

# ---------------------------------------------------------------- parameters
DT_SIM = 0.001            # MJCF timestep (s)
CTRL_EVERY = 2            # control + logging at 500 Hz
DT_C = DT_SIM * CTRL_EVERY

HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
Q0_NOISE = 0.02           # rad, random initial-joint noise (initial-condition randomisation)

# planned approach for this case (offline reachability search, planner v0)
APPROACH_DIR = np.array([0.4433991790861954, 0.32203696543100496, 0.8364743635531682])
PATH_LEN = 0.06401645277336002      # m, target -> brain surface along APPROACH_DIR
STANDOFF = 0.06                     # m, needle tip stops this far outside the entry point

# phase durations (s)
T_ALIGN, T_APPROACH, T_INSERT, T_HOLD, T_RETRACT = 1.5, 1.5, 5.0, 2.0, 2.0

# outer Cartesian loop and IK
KP = 100.0                # 1/s   position loop (PI on measured tip error)
KI = 30.0                 # 1/s^2
KO = 5.0                  # 1/s   needle-axis alignment
I_CLAMP = 5e-4            # m  anti-windup
V_JOINT_MAX = 2.0         # rad/s joint speed limit (same for every condition)
LAMBDA0 = 0.08            # max damping of the adaptive DLS law
W_DAMP = 0.05             # damping starts below this manipulability
W_ABORT = 0.015           # FSM aborts below this manipulability
K_NULL = 2.0              # 1/s null-space posture gain

# tremor + tissue
FILTER_FC = 3.5           # Hz  low-pass cut-off of the proposed assistance
FILTER_ORDER = 2
DRIFT_UM = 40.0           # slow involuntary drift (not removable by the tremor filter)
TISSUE_K = 15.0           # N/m  ASSUMPTION - axial tissue stiffness (vary in sensitivity analysis)
TISSUE_C = 3.0            # N s/m ASSUMPTION - axial tissue damping

STATES = ["ALIGN", "APPROACH", "INSERT", "HOLD", "RETRACT", "ABORT"]
S_ALIGN, S_APPROACH, S_INSERT, S_HOLD, S_RETRACT, S_ABORT = range(6)

CONDITIONS = {
    #          tremor  filter  ik       null
    "NT":      (False, "none", "dls",   True),
    "BASE":    (True,  "none", "pinv",  False),
    "PROP":    (True,  "plan", "dls",   True),
    "DLSONLY": (True,  "none", "dls",   True),
    "LPFRAW":  (True,  "raw",  "dls",   True),
}


# ---------------------------------------------------------------- helpers
def load_model():
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def load_plan():
    meta = json.loads((ROOT / "meshes_v2" / "export_meta.json").read_text())
    target = np.array(meta["target_m"])
    direction = APPROACH_DIR / np.linalg.norm(APPROACH_DIR)
    entry = target + direction * PATH_LEN
    standoff = entry + direction * STANDOFF
    return dict(target=target, entry=entry, standoff=standoff, ins=-direction)


def min_jerk(tau):
    tau = np.clip(tau, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    return s, ds


class TremorModel:
    """Hand disturbance: three sinusoids per axis in 7-12 Hz (peak per-axis amplitude <= amp)
    plus a slow drift.  Fully determined by (seed, amp_um)."""

    def __init__(self, seed, amp_um, drift_um=DRIFT_UM):
        rng = np.random.default_rng(seed)
        self.f = rng.uniform(7.0, 12.0, size=(3, 3))
        self.ph = rng.uniform(0, 2 * np.pi, size=(3, 3))
        self.w = np.array([0.5, 0.3, 0.2])
        self.amp = amp_um * 1e-6
        self.fd = rng.uniform(0.3, 1.5, size=3)
        self.phd = rng.uniform(0, 2 * np.pi, size=3)
        self.ad = drift_um * 1e-6 * rng.uniform(0.5, 1.0, size=3)

    def __call__(self, t):
        tremor = self.amp * (np.sin(2 * np.pi * self.f * t + self.ph) @ self.w)
        drift = self.ad * np.sin(2 * np.pi * self.fd * t + self.phd)
        return tremor + drift, tremor


class StreamingLowpass:
    """Causal Butterworth low-pass applied sample by sample (500 Hz)."""

    def __init__(self, fc=FILTER_FC, order=FILTER_ORDER, fs=1.0 / DT_C):
        self.sos = butter(order, fc, btype="low", fs=fs, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2, 3))

    def reset(self, x0):
        """Start in steady state at constant input x0 (no start-up transient)."""
        self.zi = sosfilt_zi(self.sos)[:, :, None] * np.asarray(x0)[None, None, :]

    def __call__(self, x):
        y, self.zi = sosfilt(self.sos, x[None, :], axis=0, zi=self.zi)
        return y[0]


class NeedleRobot:
    """Kinematics helpers: Jacobian, damped least-squares IK, pose solver."""

    def __init__(self, model):
        self.m = model
        self.ds = mujoco.MjData(model)           # scratch data for kinematics
        self.tip = model.site("p_needle_tip").id
        self.lo, self.hi = model.jnt_range[:7].T

    def jac(self, q):
        ds = self.ds
        ds.qpos[:7] = q
        mujoco.mj_kinematics(self.m, ds)
        mujoco.mj_comPos(self.m, ds)
        J = np.zeros((6, self.m.nv))
        mujoco.mj_jacSite(self.m, ds, J[:3], J[3:], self.tip)
        pos = ds.site_xpos[self.tip].copy()
        axis = ds.site_xmat[self.tip].reshape(3, 3)[:, 2].copy()
        return J[:, :7], pos, axis

    @staticmethod
    def manipulability(J):
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def solve_pose(self, p, axis, q_seed, iters=400):
        q = q_seed.copy()
        for _ in range(iters):
            J, pos, ax = self.jac(q)
            e = np.r_[p - pos, np.cross(ax, axis)]
            if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-4:
                return True, q
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + 0.5 * dq, self.lo, self.hi)
        return False, q


def ik_step(J, dx6, q, q_nom, mode, use_null, w):
    """One resolved-rate step. mode 'pinv' = undamped, 'dls' = adaptive damping (dossier eq.)."""
    if mode == "pinv":
        lam = 1e-6
    else:
        lam = LAMBDA0 * (1.0 - w / W_DAMP) if w < W_DAMP else 1e-3
    JJt = J @ J.T + (lam**2) * np.eye(6)
    Jp = J.T @ np.linalg.inv(JJt)
    dq = Jp @ dx6
    if use_null:
        dq += (np.eye(7) - Jp @ J) @ (K_NULL * DT_C * (q_nom - q))
    return dq, lam


# ---------------------------------------------------------------- one trial
def run_trial(model, cond, seed, amp_um, k_scale=1.0, viewer=None, realtime=False, record=True, data=None):
    tremor_on, filt, ik_mode, use_null = CONDITIONS[cond]
    plan = load_plan()
    robot = NeedleRobot(model)
    m = model
    d = data if data is not None else mujoco.MjData(m)
    tip = robot.tip
    tip_body = int(m.site_bodyid[tip])
    ctrl_lo, ctrl_hi = m.actuator_ctrlrange[:7].T
    hand = TremorModel(seed, amp_um)
    rng = np.random.default_rng(seed + 10_000)
    lp = StreamingLowpass()

    # --- planning gate: both end poses must be reachable
    ok_s, q_s = robot.solve_pose(plan["standoff"], plan["ins"], HOME)
    ok_t, q_t = robot.solve_pose(plan["target"], plan["ins"], q_s)
    if not (ok_s and ok_t):
        return dict(summary=dict(cond=cond, seed=seed, plan_ok=0, aborted=1), telemetry=None)

    q0 = np.clip(HOME + rng.normal(0, Q0_NOISE, 7), robot.lo, robot.hi)
    d.qpos[:7] = q0
    d.ctrl[:7] = q0
    mujoco.mj_forward(m, d)
    q_cmd = q0.copy()

    t_end = [T_ALIGN, T_ALIGN + T_APPROACH, T_ALIGN + T_APPROACH + T_INSERT]
    t_end.append(t_end[-1] + T_HOLD)
    t_end.append(t_end[-1] + T_RETRACT)
    n_steps = int(round(t_end[-1] / DT_SIM)) + 1
    t_ins0, t_hold1 = t_end[1], t_end[3]

    state = S_ALIGN
    e_int = np.zeros(3)
    aborted = False
    events = []
    rows = []
    v_tip = np.zeros(6)
    F_N = 0.0
    w = cond_num = np.nan

    for step in range(n_steps):
        t = step * DT_SIM
        # ------- 500 Hz controller
        if step % CTRL_EVERY == 0:
            new_state = state
            if state != S_ABORT:
                if t >= t_end[3]:
                    new_state = S_RETRACT
                elif t >= t_end[2]:
                    new_state = S_HOLD
                elif t >= t_end[1]:
                    new_state = S_INSERT
                elif t >= t_end[0]:
                    new_state = S_APPROACH
            if new_state != state:
                events.append((t, "ENTER_" + STATES[new_state]))
                state = new_state
                e_int[:] = 0.0

            mujoco.mj_kinematics(m, d)
            p_meas = d.site_xpos[tip].copy()
            ax_meas = d.site_xmat[tip].reshape(3, 3)[:, 2].copy()

            # reference path
            if state == S_ALIGN:
                s, ds_ = min_jerk(t / T_ALIGN)
                q_cmd = q0 + s * (q_s - q0)
                p_ref = p_meas.copy()
                v_ref = np.zeros(3)
            elif state == S_ABORT:
                p_ref = abort_pos
                v_ref = np.zeros(3)
            else:
                if state == S_APPROACH:
                    a, b, tau = plan["standoff"], plan["entry"], (t - t_end[0]) / T_APPROACH
                    dur = T_APPROACH
                elif state == S_INSERT:
                    a, b, tau = plan["entry"], plan["target"], (t - t_end[1]) / T_INSERT
                    dur = T_INSERT
                elif state == S_HOLD:
                    a, b, tau, dur = plan["target"], plan["target"], 1.0, T_HOLD
                else:
                    a, b, tau = plan["target"], plan["standoff"], (t - t_end[3]) / T_RETRACT
                    dur = T_RETRACT
                s, ds_ = min_jerk(tau)
                p_ref = a + (b - a) * s
                v_ref = (b - a) * ds_ / dur

            # hand disturbance and assistance
            active = tremor_on and state in (S_INSERT, S_HOLD)
            dev, trem_only = hand(t) if active else (np.zeros(3), np.zeros(3))
            p_raw = p_ref + dev
            if filt == "plan":
                if active:
                    p_des = p_ref + lp(dev)
                else:
                    lp.reset(np.zeros(3))
                    p_des = p_ref
            elif filt == "raw":
                if active:
                    p_des = lp(p_raw)
                else:
                    lp.reset(p_raw)
                    p_des = p_raw
            else:
                p_des = p_raw

            J, _, _ = robot.jac(d.qpos[:7])
            w = robot.manipulability(J)
            sv = np.linalg.svd(J, compute_uv=False)
            cond_num = float(sv[0] / max(sv[-1], 1e-12))

            if state != S_ALIGN and state != S_ABORT:
                if w < W_ABORT and ik_mode == "dls":
                    state = S_ABORT
                    abort_pos = p_meas.copy()
                    aborted = True
                    events.append((t, "ABORT_LOW_MANIPULABILITY"))
                else:
                    e = p_des - p_meas
                    e_int = np.clip(e_int + e * DT_C, -I_CLAMP, I_CLAMP)
                    dx = v_ref * DT_C + KP * DT_C * e + KI * DT_C * e_int
                    e_o = np.cross(ax_meas, plan["ins"])
                    dx6 = np.r_[dx, KO * DT_C * e_o]
                    dq, lam = ik_step(J, dx6, d.qpos[:7], q_s, ik_mode, use_null, w)
                    lim = V_JOINT_MAX * DT_C
                    dq = np.clip(dq, -lim, lim)
                    q_cmd = np.clip(q_cmd + dq, robot.lo, robot.hi)
            d.ctrl[:7] = np.clip(q_cmd, ctrl_lo, ctrl_hi)

            if record:
                rows.append((t, state, *p_ref, *p_meas, *(p_raw - p_ref), *(p_des - p_ref),
                             np.linalg.norm(p_meas - p_ref), w, cond_num, F_N, *d.qpos[:7],
                             *d.qfrc_actuator[:7]))

        # ------- tissue resistance on the needle (axial spring-damper beyond the entry point)
        d.qfrc_applied[:] = 0.0
        if state in (S_INSERT, S_HOLD, S_RETRACT, S_ABORT):
            p_tip = d.site_xpos[tip]
            depth = float(np.dot(p_tip - plan["entry"], plan["ins"]))
            if depth > 0.0:
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, tip, v_tip, 0)
                v_ax = float(np.dot(v_tip[3:], plan["ins"]))
                F_N = k_scale * TISSUE_K * depth + TISSUE_C * v_ax
                F_N = max(F_N, 0.0)
                mujoco.mj_applyFT(m, d, -F_N * plan["ins"], np.zeros(3), p_tip, tip_body, d.qfrc_applied)
            else:
                F_N = 0.0
        mujoco.mj_step(m, d)
        if viewer is not None and step % 16 == 0:
            viewer.sync()
            if realtime:
                import time
                time.sleep(0.016)

    return dict(summary=_summarise(cond, seed, amp_um, k_scale, rows, t_ins0, t_hold1, t_end,
                                   aborted, plan), telemetry=(rows, events))


TELEMETRY_COLUMNS = (["t", "fsm"] + [f"ref_{a}" for a in "xyz"] + [f"tip_{a}" for a in "xyz"]
                     + [f"hand_{a}" for a in "xyz"] + [f"cmd_{a}" for a in "xyz"]
                     + ["err_norm", "w", "cond", "force_N"]
                     + [f"q{i}" for i in range(1, 8)] + [f"tau{i}" for i in range(1, 8)])


def _band(x, lo=7.0, hi=12.0, fs=1.0 / DT_C):
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _summarise(cond, seed, amp_um, k_scale, rows, t_ins0, t_hold1, t_end, aborted, plan):
    A = np.array(rows)
    t = A[:, 0]
    ref, tip = A[:, 2:5], A[:, 5:8]
    hand = A[:, 8:11]
    win = (t >= t_ins0) & (t <= t_hold1)
    hold = (t >= t_end[2]) & (t < t_end[3])
    err = np.linalg.norm(tip - ref, axis=1)
    tau = A[:, 25:32]
    rms = lambda v: float(np.sqrt(np.mean(v**2)))
    out = dict(cond=cond, seed=seed, tremor_amp_um=amp_um if CONDITIONS[cond][0] else 0.0,
               k_tissue_scale=k_scale, plan_ok=1, aborted=int(aborted))
    out["rmse_track_mm"] = rms(err[win]) * 1e3
    end_ins = np.argmin(np.abs(t - t_end[2]))
    out["place_err_end_insert_mm"] = float(np.linalg.norm(tip[end_ins] - plan["target"])) * 1e3
    out["hold_rms_mm"] = rms(err[hold]) * 1e3
    out["hold_max_mm"] = float(err[hold].max()) * 1e3
    out["pass_0p5mm_hold"] = int(out["hold_max_mm"] < 0.5)
    if CONDITIONS[cond][0]:
        bi = rms(_band(hand[win])) + 1e-12
        bo = rms(_band((tip - ref)[win])) + 1e-12
        out["tremor_rejection_dB"] = 20 * np.log10(bi / bo)
    else:
        out["tremor_rejection_dB"] = np.nan
    out["control_effort"] = float(np.sum(tau[win] ** 2) * DT_C)
    out["max_cond"] = float(np.nanmax(A[win, 16]))
    out["min_w"] = float(np.nanmin(A[win, 15]))
    out["max_force_N"] = float(np.nanmax(A[win, 17]))
    out["max_joint_speed"] = float(np.max(np.abs(np.diff(A[win, 18:25], axis=0))) / DT_C)
    return out