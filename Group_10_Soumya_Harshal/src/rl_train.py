"""Train the adaptive tremor filter with PPO (reinforcement learning) - plain NumPy, nothing to install.
    python src\\rl_train.py                 (about 3-5 min)        options:  --iters 250   --quick (10 s test)
PPO = proximal policy optimisation: actor (Gaussian policy) + critic (value function), GAE advantages, clipped objective,
Adam optimiser. The networks are 2 hidden layers x 64 tanh units; gradients are written out by hand (see mlp_* below).
Writes analytics/rl/policy.npz (the learned actor, used by rl_eval.py), analytics/rl/train_curve.csv and
analytics/rl/train_summary.csv (held-out simulated hands: filter output vs the INTENDED path)."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

import rl_filter as rf

ROOT = Path(__file__).resolve().parent.parent
H = 64


# ------------------------------------------------------------------ tiny MLP with hand-written backprop
def mlp_init(rng, n_in, n_out, out_scale):
    def lin(i, o, s=1.0):
        W = rng.normal(0, 1, (o, i)) * np.sqrt(2.0 / i) * s
        return W, np.zeros(o)
    W0, b0 = lin(n_in, H)
    W1, b1 = lin(H, H)
    W2, b2 = lin(H, n_out, out_scale)
    return [W0, b0, W1, b1, W2, b2]


def mlp_fwd(P, x):
    h1 = np.tanh(x @ P[0].T + P[1])
    h2 = np.tanh(h1 @ P[2].T + P[3])
    return h2 @ P[4].T + P[5], (x, h1, h2)


def mlp_bwd(P, cache, dout):
    x, h1, h2 = cache
    g4 = dout.T @ h2
    g5 = dout.sum(0)
    d2 = (dout @ P[4]) * (1 - h2**2)
    g2 = d2.T @ h1
    g3 = d2.sum(0)
    d1 = (d2 @ P[2]) * (1 - h1**2)
    g0 = d1.T @ x
    g1 = d1.sum(0)
    return [g0, g1, g2, g3, g4, g5]


class Adam:
    def __init__(self, params, lr):
        self.p, self.lr = params, lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads, max_norm=0.5):
        n = np.sqrt(sum(float((g**2).sum()) for g in grads))
        if n > max_norm:
            grads = [g * max_norm / n for g in grads]
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh, vh = self.m[i] / (1 - 0.9**self.t), self.v[i] / (1 - 0.999**self.t)
            self.p[i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


# ------------------------------------------------------------------ environment (E hands in parallel)
TRACK_W = 0.0   # extra weight on error while the intended hand is moving (0 = original reward)


class VecEnv:
    """Each episode = 12 s of (intent + tremor). Every 20 ms the agent chooses the filter-mix weights.
    reward = -(output - intent)^2 in units of (100 um)^2, minus a small penalty for jumpy weights."""

    def __init__(self, E, seed, dr=False):
        self.E, self.rng, self.dr = E, np.random.default_rng(seed), dr

    def reset(self):
        eps = [rf.make_episode(self.rng, self.dr) for _ in range(self.E)]
        self.intent = np.stack([e[0] for e in eps])
        self.Y = np.stack([e[2] for e in eps])
        self.F = np.stack([e[3] for e in eps])
        self.k = 0
        self.w = np.full((self.E, rf.NB), 1.0 / rf.NB)
        return self.obs()

    def obs(self):
        return np.concatenate([self.F[:, self.k * rf.BLOCK], self.w], 1).astype(np.float64)

    def step(self, A):
        a = np.clip(A, -1, 1) * rf.TEMP
        e = np.exp(a - a.max(1, keepdims=True))
        W = e / e.sum(1, keepdims=True)
        sl = slice(self.k * rf.BLOCK, (self.k + 1) * rf.BLOCK)
        out = np.einsum("ebn,en->eb", self.Y[:, sl], W)
        err = (out - self.intent[:, sl]) / rf.SCALE_X
        wgt = 1.0
        if TRACK_W > 0:
            seg = self.intent[:, sl]
            moving = (np.abs(seg[:, -1] - seg[:, 0]) / (rf.BLOCK / rf.FS) > 1e-3).astype(float)
            wgt = 1.0 + TRACK_W * moving[:, None]
        rew = -(wgt * np.clip(err**2, 0, 25.0)).mean(1) - 0.02 * np.abs(W - self.w).sum(1)
        self.w = W
        self.k += 1
        return self.obs() if (self.k + 1) * rf.BLOCK <= rf.N_EP else None, rew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--track-w", type=float, default=0.0, help="extra reward weight on error while the hand is moving")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dr", action="store_true", help="train on domain-randomised hands: tremor 20-250 um, drift 0-300 um")
    ap.add_argument("--paper", action="store_true", help="PPO paper MuJoCo settings (Schulman 2017, Table 3)")
    ap.add_argument("--out", default=str(ROOT / "analytics" / "rl"))
    a = ap.parse_args()
    global TRACK_W
    TRACK_W = a.track_w
    if a.quick:
        a.iters = 6
    rng = np.random.default_rng(a.seed)
    env = VecEnv(a.envs, 50_000 + a.seed, a.dr)                       # training hands: seeds 50000+ (the MuJoCo test uses 1000+)
    pi = mlp_init(rng, rf.OBS_DIM, rf.NB, 0.01)
    vf = mlp_init(rng, rf.OBS_DIM, 1, 1.0)
    log_std = np.full(rf.NB, -1.0)
    if a.paper:      # Schulman et al. 2017, Table 3 (MuJoCo): Adam 3e-4, 10 epochs, gamma 0.99, GAE lambda 0.95, clip 0.2
        opt_pi, opt_vf = Adam(pi + [log_std], 3e-4), Adam(vf, 3e-4)
        GAMMA, LAM, CLIP, EPOCHS, MB = 0.99, 0.95, 0.2, 10, 4
    else:            # our first settings
        opt_pi, opt_vf = Adam(pi + [log_std], 5e-4), Adam(vf, 1e-3)
        GAMMA, LAM, CLIP, EPOCHS, MB = 0.95, 0.9, 0.2, 6, 4
    T = rf.N_EP // rf.BLOCK
    curve = []
    t0 = time.time()
    print(f"training PPO: {a.iters} iterations x {a.envs} simulated hands x {T} decisions "
          f"= {a.iters * a.envs * T:,} steps", flush=True)
    for it in range(a.iters):
        O = np.zeros((T, a.envs, rf.OBS_DIM))
        A_ = np.zeros((T, a.envs, rf.NB))
        LP = np.zeros((T, a.envs))
        V = np.zeros((T, a.envs))
        R = np.zeros((T, a.envs))
        obs = env.reset()
        sd = np.exp(log_std)
        for t in range(T):
            mu, _ = mlp_fwd(pi, obs)
            act = mu + sd * rng.normal(size=mu.shape)
            LP[t] = (-0.5 * (((act - mu) / sd) ** 2).sum(1) - log_std.sum() - 0.5 * rf.NB * np.log(2 * np.pi))
            V[t] = mlp_fwd(vf, obs)[0][:, 0]
            O[t], A_[t] = obs, act
            nxt, R[t] = env.step(act)
            obs = nxt if nxt is not None else obs
        adv = np.zeros_like(R)
        last = 0.0
        for t in reversed(range(T)):                              # GAE (the episode simply ends after T decisions)
            nv = V[t + 1] if t + 1 < T else 0.0
            delta = R[t] + GAMMA * nv - V[t]
            last = delta + GAMMA * LAM * last
            adv[t] = last
        ret = adv + V
        N = T * a.envs
        Of, Af, LPf, advf, retf = (x.reshape(N, -1) if x.ndim > 2 else x.reshape(N) for x in (O, A_, LP, adv, ret))
        advf = (advf - advf.mean()) / (advf.std() + 1e-8)
        for ep in range(EPOCHS):
            idx = rng.permutation(N)
            for mb in np.array_split(idx, MB):
                x, act, lp_old, ad, rt = Of[mb], Af[mb], LPf[mb], advf[mb], retf[mb]
                sd = np.exp(log_std)
                mu, cache = mlp_fwd(pi, x)
                lp = -0.5 * (((act - mu) / sd) ** 2).sum(1) - log_std.sum() - 0.5 * rf.NB * np.log(2 * np.pi)
                ratio = np.exp(lp - lp_old)
                active = ~(((ratio > 1 + CLIP) & (ad > 0)) | ((ratio < 1 - CLIP) & (ad < 0)))
                coef = -(ad * ratio * active) / len(mb)                # d loss / d logp
                d_mu = coef[:, None] * (act - mu) / sd**2
                d_ls = (coef[:, None] * (((act - mu) / sd) ** 2 - 1.0)).sum(0)
                g = mlp_bwd(pi, cache, d_mu)
                opt_pi.p[-1] = log_std
                opt_pi.step(g + [d_ls])
                log_std = opt_pi.p[-1]
                log_std = np.clip(log_std, -2.5, 0.0)
                opt_pi.p[-1] = log_std
                v, vc = mlp_fwd(vf, x)
                gv = mlp_bwd(vf, vc, (v[:, 0] - rt)[:, None] / len(mb))
                opt_vf.step(gv, max_norm=1.0)
        ep_ret = float(R.sum(0).mean())
        curve.append((it, (it + 1) * a.envs * T, ep_ret, float(np.exp(log_std).mean())))
        if it % 5 == 0 or it == a.iters - 1:
            print(f"  iter {it:3d}  steps {(it + 1) * a.envs * T:8d}  mean episode reward {ep_ret:8.1f}  "
                  f"policy noise {np.exp(log_std).mean():.2f}  ({time.time() - t0:.0f}s)", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "policy.npz", W0=pi[0], b0=pi[1], W1=pi[2], b1=pi[3], Wa=pi[4], ba=pi[5])
    with open(out / "train_curve.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iteration", "env_steps", "mean_episode_reward", "policy_noise"])
        wr.writerows(curve)
    print("saved", out / "policy.npz")

    # ---- held-out simulated hands (NOT the MuJoCo trials: that is rl_eval.py)
    policy = rf.NumpyPolicy(out / "policy.npz")
    rg = np.random.default_rng(777_000)
    fname = lambda fc: "no filter (raw)" if fc is None else f"fixed {fc} Hz"
    names = [fname(fc) for fc in rf.BANK_FC] + ["speed rule (no learning)", "RL (PPO)"]
    errs = {n: [] for n in names}
    hold_err = {n: [] for n in names}
    for _ in range(60):
        intent, r, Y, F = rf.make_episode(rg)
        outs = {fname(fc): Y[:, i].copy() for i, fc in enumerate(rf.BANK_FC)}
        for key, nm in (("rule", "speed rule (no learning)"), ("rl", "RL (PPO)")):
            o = np.zeros(rf.N_EP)
            w = np.full(rf.NB, 1.0 / rf.NB)
            for b in range(rf.N_EP // rf.BLOCK):
                f = F[b * rf.BLOCK]
                act = rf.rule_action(f) if key == "rule" else policy(np.r_[f, w])
                w = rf.softmax_w(act)
                sl = slice(b * rf.BLOCK, (b + 1) * rf.BLOCK)
                o[sl] = Y[sl] @ w
            outs[nm] = o
        still = np.abs(np.gradient(intent, 1.0 / rf.FS)) < 1e-4          # samples where the intended path is not moving
        for n in names:
            e = outs[n] - intent
            errs[n].append(np.sqrt(np.mean(e**2)) * 1e6)
            hold_err[n].append(np.sqrt(np.mean(e[still] ** 2)) * 1e6 if still.sum() > 100 else np.nan)
    print("\nHELD-OUT SIMULATED HANDS (60 episodes): error between filter output and the INTENDED path, um RMS (lower = better)")
    print(f"  {'method':28s} {'all samples':>12s} {'while holding still':>20s}")
    with open(out / "train_summary.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["method", "rms_error_um_all", "rms_error_um_holding"])
        for n in names:
            m_all, m_hold = float(np.mean(errs[n])), float(np.nanmean(hold_err[n]))
            print(f"  {n:28s} {m_all:12.1f} {m_hold:20.1f}")
            wr.writerow([n, round(m_all, 2), round(m_hold, 2)])


if __name__ == "__main__":
    main()