"""tremor_baselines.py - the standard tremor filters a reviewer expects, as drop-in replacements for StreamingLowpass.
All three see ONLY the noisy hand signal (they do not know the plan), exactly like the fixed filter and the RL filter.
  KALMAN  constant-velocity intent + damped 10 Hz oscillator for the tremor (steady-state Kalman gain from the Riccati equation)
          10 Hz = centre of the 8-12 Hz physiological band (Elble 2005; Ladha 2023). Output = estimated intended position.
  WFLC    weighted-frequency Fourier linear combiner (Riviere, Rader & Thakor 1998): one adaptive oscillator that locks on the tremor.
  BMFLC   band-limited multiple Fourier linear combiner (Veluvolu & Ang 2011): a fixed bank of 6-14 Hz oscillators, LMS weights.
WFLC and BMFLC estimate the tremor itself; the output is  hand signal - estimated tremor.  Both are fed the hand signal after a
4 Hz high-pass so slow voluntary motion does not leak into the oscillator weights.  Work is done in units of 100 um.
Tuning (step sizes, noise levels) is done on held-out simulated hands by compare_methods.py --tune, never on the MuJoCo test seeds."""
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import butter, sosfilt, sosfilt_zi

FS = 500.0
DT = 1.0 / FS
SX = 1e-4
_HP = butter(2, 4.0, btype="high", fs=FS, output="sos")


class _HighPass:
    def __init__(self, x0):
        self.zi = sosfilt_zi(_HP) * x0

    def __call__(self, x):
        y, self.zi = sosfilt(_HP, np.array([x]), zi=self.zi)
        return y[0]


class _BMFLC:
    def __init__(self, x0, mu=0.05, fmin=6.0, fmax=14.0, df=1.0):
        self.f = np.arange(fmin, fmax + 1e-9, df)
        self.om = 2 * np.pi * self.f * DT
        self.w = np.zeros(2 * len(self.f))
        self.k, self.mu, self.hp = 0, mu, _HighPass(x0)

    def __call__(self, x):
        s = self.hp(x) / SX
        ph = self.om * self.k
        self.k += 1
        v = np.concatenate([np.sin(ph), np.cos(ph)])
        est = float(self.w @ v)
        self.w += self.mu * (s - est) * v / len(self.f)             # normalised LMS (v.v = number of frequencies)
        return x - est * SX


class _WFLC:
    def __init__(self, x0, mu=0.05, mu0=3e-4, f0=9.0):
        self.w, self.mu, self.mu0 = np.zeros(2), mu, mu0
        self.om, self.ph, self.hp = 2 * np.pi * f0 * DT, 0.0, _HighPass(x0)
        self.lo, self.hi = 2 * np.pi * 6.0 * DT, 2 * np.pi * 14.0 * DT

    def __call__(self, x):
        s = self.hp(x) / SX
        self.ph += self.om
        v = np.array([np.sin(self.ph), np.cos(self.ph)])
        est = float(self.w @ v)
        e = s - est
        self.om = float(np.clip(self.om + 2 * self.mu0 * e * (self.w[0] * v[1] - self.w[1] * v[0]), self.lo, self.hi))
        self.w += self.mu * e * v
        return x - est * SX


def kalman_model(q_i=100.0, q_t=0.01, r=0.1, f_t=10.0, rho=0.998):
    w = 2 * np.pi * f_t * DT
    F = np.zeros((4, 4))
    F[:2, :2] = [[1, DT], [0, 1]]
    F[2:, 2:] = rho * np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])
    H = np.array([1.0, 0, 1.0, 0])
    Q = np.zeros((4, 4))
    Q[:2, :2] = q_i * np.array([[DT**4 / 4, DT**3 / 2], [DT**3 / 2, DT**2]])
    Q[2:, 2:] = q_t * np.eye(2)
    P = solve_discrete_are(F.T, H.reshape(4, 1), Q, np.array([[r]]))
    K = P @ H / (H @ P @ H + r)
    return F, H, K


class _Kalman:
    def __init__(self, x0, q_i=100.0, q_t=0.01, r=0.1):
        self.F, self.H, self.K = kalman_model(q_i, q_t, r)
        self.xh, self.x0 = np.zeros(4), x0

    def __call__(self, x):
        z = (x - self.x0) / SX
        xp = self.F @ self.xh
        self.xh = xp + self.K * (z - self.H @ xp)
        return self.x0 + self.xh[0] * SX


KINDS = {"kalman": _Kalman, "wflc": _WFLC, "bmflc": _BMFLC}


class BaselineFilter:
    """Drop-in for surgical_tremor_controller.StreamingLowpass: reset(x0) / __call__(x) with 3-vectors."""

    def __init__(self, kind, **params):
        self.kind, self.params, self.ax = kind, params, None

    def reset(self, x0):
        self.ax = [KINDS[self.kind](float(v), **self.params) for v in np.asarray(x0)]

    def __call__(self, x):
        if self.ax is None:
            self.reset(x)
        return np.array([a(float(v)) for a, v in zip(self.ax, x)])