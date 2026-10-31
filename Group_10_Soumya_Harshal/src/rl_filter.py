"""rl_filter.py - pieces shared by rl_train.py and rl_eval.py: a learned, ADAPTIVE tremor filter.

PROBLEM (why RL, honestly): our plan-relative filter (condition PROP) knows the intended needle path, so it can remove the
tremor without lag. A real hand-held / doctor-driven system does NOT know the intent: it only sees the noisy hand signal
r = intent + tremor + drift. A fixed low-pass (LPFRAW) then trades tremor rejection against lag: while the needle is moving
the lag is large (millimetres), while it is holding the lag is zero and we could filter much harder.
An RL policy learns to SCHEDULE the filter (how much of each cut-off to use) from what it can observe:
    bank of 5 fixed 2nd-order Butterworth low-passes (1.5, 2.5, 4, 7, 14 Hz) + the unfiltered signal  ->  output = softmax-weighted mix
    observation (15): residuals r - bank_i, two velocity estimates, tremor envelope, previous weights
    action (6): mixing logits, re-decided every 20 ms.   reward: -(output - intent)^2 (intent is known only in simulation).
Nothing here sees the plan at run time. Policy = small MLP (PPO), exported to numpy so the 500 Hz controller stays fast."""
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, sosfilt, sosfilt_zi

FS = 500.0                                   # controller rate (Hz) = 1 / DT_C
BANK_FC = [1.5, 2.5, 4.0, 7.0, 14.0, None]                # None = pass the raw signal through (no filtering)
SOS = [butter(2, fc, btype="low", fs=FS, output="sos") for fc in BANK_FC if fc is not None]
NB = len(BANK_FC)
NF = len(SOS)
BLOCK = 10                                   # samples per policy decision (20 ms)
LAG = 20                                     # samples for the velocity estimate (40 ms)
SCALE_X = 1e-4                               # 100 um
SCALE_V = 1e-2                               # 10 mm/s
ENV_A = 1.0 - np.exp(-1.0 / (0.1 * FS))      # 100 ms envelope
OBS_DIM = NB + 3 + NB
EP_S = 12.0
N_EP = int(EP_S * FS)
TEMP = 3.0                                   # softmax sharpness


def softmax_w(a):
    a = np.clip(np.asarray(a, float), -1.0, 1.0) * TEMP
    e = np.exp(a - a.max())
    return e / e.sum()


# ------------------------------------------------------------------ features: vectorised (training) and streaming (controller)
def bank_outputs(r):
    """All five filter outputs for a whole signal (same start-up as the controller: steady state at r[0])."""
    cols = [sosfilt(s, r, zi=sosfilt_zi(s) * r[0])[0] for s in SOS] + [np.asarray(r, float)]
    return np.stack(cols, 1)


def features_vec(r, Y):
    n = len(r)
    resid = (r[:, None] - Y) / SCALE_X
    idx = np.maximum(np.arange(n) - LAG, 0)
    va = (Y[:, 1] - Y[idx, 1]) / (LAG / FS) / SCALE_V
    vb = (Y[:, 2] - Y[idx, 2]) / (LAG / FS) / SCALE_V
    env = lfilter([ENV_A], [1.0, ENV_A - 1.0], np.abs(r - Y[:, 2]) / SCALE_X)
    return np.clip(np.column_stack([resid, va, vb, env]), -10, 10).astype(np.float32)


class Axis:
    """Streaming version for ONE axis (called once per 500 Hz control tick)."""

    def __init__(self, x0):
        self.zi = [sosfilt_zi(s) * x0 for s in SOS]
        self.b1 = deque([x0] * (LAG + 1), maxlen=LAG + 1)
        self.b2 = deque([x0] * (LAG + 1), maxlen=LAG + 1)
        self.env = 0.0
        self.w = np.full(NB, 1.0 / NB)
        self.wcur = self.w.copy()
        self.k = 0

    def feed(self, x):
        y = np.empty(NB)
        for i in range(NF):
            o, self.zi[i] = sosfilt(SOS[i], np.array([x]), zi=self.zi[i])
            y[i] = o[0]
        y[NF] = x
        self.b1.append(y[1])
        self.b2.append(y[2])
        va = (self.b1[-1] - self.b1[0]) / (LAG / FS) / SCALE_V
        vb = (self.b2[-1] - self.b2[0]) / (LAG / FS) / SCALE_V
        self.env += ENV_A * (abs(x - y[2]) / SCALE_X - self.env)
        f = np.clip(np.r_[(x - y) / SCALE_X, va, vb, self.env], -10, 10).astype(np.float32)
        return y, f


class NumpyPolicy:
    def __init__(self, path):
        z = np.load(path)
        self.W0, self.b0, self.W1, self.b1, self.Wa, self.ba = (z[k] for k in ("W0", "b0", "W1", "b1", "Wa", "ba"))

    def __call__(self, obs):
        h = np.tanh(self.W0 @ obs + self.b0)
        h = np.tanh(self.W1 @ h + self.b1)
        return np.clip(self.Wa @ h + self.ba, -1.0, 1.0)


def rule_action(f):
    """Hand-written baseline (no learning): 1.5 Hz filter when the needle is still, raw signal when it moves."""
    s = float(np.clip((abs(f[NB + 1]) - 0.2) / 0.8, 0.0, 1.0))
    a = np.full(NB, -1.0)
    a[0], a[NB - 1] = 1.0 - 2.0 * s, 2.0 * s - 1.0
    return a


class AdaptiveFilter:
    """Drop-in for surgical_tremor_controller.StreamingLowpass (3-vector, reset(x0) / __call__(x)).
    mode: 'rl' (needs policy), 'rule', or ('fixed', i) -> bank filter i."""

    def __init__(self, mode="rl", policy=None, fixed=None):
        self.mode, self.policy, self.fixed = mode, policy, fixed
        self.ax = None

    def reset(self, x0):
        self.ax = [Axis(float(v)) for v in np.asarray(x0)]

    def __call__(self, x):
        if self.ax is None:
            self.reset(x)
        out = np.empty(3)
        for j, A in enumerate(self.ax):
            y, f = A.feed(float(x[j]))
            if self.mode == "fixed":
                out[j] = y[self.fixed]
                continue
            if A.k % BLOCK == 0:
                obs = np.r_[f, A.w]
                a = self.policy(obs) if self.mode == "rl" else rule_action(f)
                A.w = softmax_w(a)
                A.wcur = A.w
            A.k += 1
            out[j] = float(y @ A.wcur)
        return out


# ------------------------------------------------------------------ synthetic training episodes
def _minjerk(tau):
    tau = np.clip(tau, 0, 1)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def intent_signal(rng, n=N_EP):
    """What the hand INTENDS (one axis, metres): trial-like insertion/hold or random move/hold sequences."""
    x = np.zeros(n)
    if rng.random() < 0.4:                                   # shaped like our needle insertion trial
        D = rng.uniform(0.010, 0.060) * rng.uniform(0.2, 1.0) * rng.choice([-1, 1])
        i0 = int(rng.uniform(0.0, 1.0) * FS)
        i1 = i0 + int(rng.uniform(3.0, 6.0) * FS)
        i2 = i1 + int(rng.uniform(1.0, 3.0) * FS)
        i3 = min(n, i2 + int(rng.uniform(1.5, 3.0) * FS))
        x[i0:i1] = D * _minjerk((np.arange(i0, i1) - i0) / max(1, i1 - i0))
        x[i1:i2] = D
        if i3 > i2:
            x[i2:i3] = D * (1 - _minjerk((np.arange(i2, i3) - i2) / max(1, i3 - i2)))
        return x
    pos, i = 0.0, 0
    while i < n:
        if rng.random() < 0.45:                              # hold
            j = min(n, i + int(rng.uniform(0.5, 3.0) * FS))
            x[i:j] = pos
        else:                                                # move
            dur = int(rng.uniform(0.8, 6.0) * FS)
            j = min(n, i + dur)
            tgt = pos + rng.uniform(-0.040, 0.040)
            x[i:j] = pos + (tgt - pos) * _minjerk((np.arange(i, j) - i) / max(1, dur))
            pos = tgt if j - i == dur else x[j - 1]
            if j < n:
                x[j:] = pos
        i = j
    return x


def tremor_signal(rng, amp_um, n=N_EP, drift_um=40.0):
    """Same structure as TremorModel (3 sinusoids 7-12 Hz weighted 0.5/0.3/0.2 + slow drift), one axis."""
    t = np.arange(n) / FS
    f, ph = rng.uniform(7.0, 12.0, 3), rng.uniform(0, 2 * np.pi, 3)
    tre = amp_um * 1e-6 * (np.sin(2 * np.pi * f[None] * t[:, None] + ph) @ np.array([0.5, 0.3, 0.2]))
    fd, phd, ad = rng.uniform(0.3, 1.5), rng.uniform(0, 2 * np.pi), drift_um * 1e-6 * rng.uniform(0.5, 1.0)
    return tre + ad * np.sin(2 * np.pi * fd * t + phd)


def make_episode(rng, dr=False):
    """dr=True: domain randomisation = steady hand ... enhanced tremor, no drift ... measured drift (see run_tiers.py)."""
    intent = intent_signal(rng)
    if dr:
        amp, drift = rng.uniform(20, 250), rng.uniform(0, 300)
    else:
        amp, drift = rng.uniform(50, 250), 40.0
    r = intent + tremor_signal(rng, amp, drift_um=drift)
    Y = bank_outputs(r)
    return intent, r, Y, features_vec(r, Y)