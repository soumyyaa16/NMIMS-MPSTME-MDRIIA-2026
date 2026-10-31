"""Watch a controller drive live in MuJoCo's interactive viewer (real time, episodes repeat until you close it).

    python scripts/live_viewer.py                                  # capstone controller (mission FSM + dynamic window) through a mixed street
    python scripts/live_viewer.py --controller ppo --family crowded  # PPO through a crowd
    python scripts/live_viewer.py --controller pure_pursuit --no-shield  # the unprotected baseline, to show a failure
    python scripts/live_viewer.py --controller dwa --family mixed --seed 5010
    python scripts/live_viewer.py --rich                           # collidable street furniture (a window per episode)

Mouse: left-drag rotates, right-drag pans, wheel zooms, space pauses. Needs a display; no GPU beyond OpenGL.
The top-left text of the window shows the mission FSM state (DISPATCH, ROAD_CRUISE, CURB_SIDEWALK, ARRIVE, AED_HANDOVER), speed and
payload shock for the capstone controller (``fsm_dwa``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import aedrover.capstone.fsm  # noqa: E402,F401  (registers fsm_dwa)
from aedrover.analysis.experiments import controller_spec  # noqa: E402
from aedrover.control.safety_filter import SafetyFilter, SafetyParams  # noqa: E402
from aedrover.nav.base import make_controller  # noqa: E402
from aedrover.sim.env import AEDRoverEnv  # noqa: E402
from aedrover.sim.vehicle_mjcf import VehicleParams  # noqa: E402
from aedrover.viz.live import LiveScene, LiveState  # noqa: E402
from aedrover.viz.overlays import OverlayConfig  # noqa: E402

MAX_STEPS_PER_FRAME = 8          # control steps (50 Hz) simulated between two drawn frames when the display is the bottleneck


def run_episode(a, env, ctrl, shield, v, seed: int, scene=None) -> bool:
    """One episode in the viewer ``v``; False when the window was closed.

    The simulation follows the wall clock: every drawn frame first steps the physics until its clock has caught up
    (at most ``MAX_STEPS_PER_FRAME`` control steps), then draws the state it reached. A slow display therefore costs frames,
    not simulated speed; only if the physics itself cannot keep up does the real-time factor in the overlay drop below 1.
    """
    print(f"episode: {a.controller}, {a.family}, seed {seed}" + (f", {len(env.furniture)} pieces of furniture" if a.rich else ""), flush=True)
    t0 = time.perf_counter()
    term = trunc = False
    info: dict = {}
    obs = env.obs
    while v.is_running():
        wall = (time.perf_counter() - t0) * a.speed
        n = 0
        while not (term or trunc) and env._t < wall and n < MAX_STEPS_PER_FRAME:
            obs = env.obs
            vv, dl = ctrl.act(obs)
            if shield is not None:
                vv, dl = shield(obs, vv, dl)
            _, _, term, trunc, info = env.step(np.array([vv, dl]))
            n += 1
        if n == MAX_STEPS_PER_FRAME and env._t < wall - 0.25:        # the physics is the bottleneck: do not run away
            t0 += (wall - env._t - 0.25) / a.speed
        rtf = env._t / max(wall, 1e-6)
        state = ctrl.state.value if hasattr(ctrl, "fsm") else ""
        if scene is not None and info:                  # the dressed model mirrors the live physics state (street, people, rover)
            with v.lock():
                scene.update(LiveState(t=env._t, dt=env.dt, lidar=np.asarray(env.obs["lidar"]), shock_g=info["shock_g"]))
        follow(v, env, scene)
        v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     (f"FSM: {state}\n" if state else "") + f"{a.controller}  t = {env._t:5.1f} s  (x{min(rtf, 9.9):.1f} real time)\n"
                     f"speed {obs['vx']:.2f} m/s",
                     f"payload shock {info.get('shock_g', 0.0):.2f} g (budget 3 g)"))
        v.sync()
        if n == 0:
            time.sleep(0.002)
        if term or trunc:
            ep = info["episode"]
            if ep["outcome"] == "goal" and hasattr(ctrl, "fsm"):
                ctrl.fsm.begin_handover(env._t)
                v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "FSM: AED_HANDOVER\nAED delivered, pads applied by bystander", ""))
                v.sync()
            print(f"  {ep['outcome']} in {ep['time_s']:.1f} s, peak payload shock {ep['peak_shock_g']:.2f} g", flush=True)
            time.sleep(1.0)
            return True
    return False

def track(v, env) -> None:
    """Start behind the rover looking along the street. ``follow`` moves the look-at point with it every frame; the mouse
    still rotates (left drag) and zooms (wheel) around the rover."""
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance, v.cam.elevation, v.cam.azimuth = 4.5, -20.0, 0.0
    follow(v, env)


def follow(v, env, scene=None) -> None:
    p = (scene.scene.data if scene is not None else env.world.data).xpos[env.world.b_chassis]
    v.cam.lookat[:] = (p[0], p[1], p[2] + 0.1)


def dressed_scene(env, shadows: bool = False) -> LiveScene:
    """The street, people and rover drawn by MuJoCo itself: a visual copy of the physics model that mirrors its live state.

    Nothing is simulated on it; every frame the physics state is copied across and MuJoCo's own viewer rasterises it.
    """
    scene = LiveScene(env, size=(640, 360), backend="classic", overlays=OverlayConfig(ribbon=False, rollouts=False))
    m = scene.scene.model
    m.vis.headlight.ambient[:] = 0.45                       # the look is tuned for a PBR renderer; the viewer needs fill light
    m.vis.headlight.diffuse[:] = 0.6
    if not shadows:                                         # shadows are the most expensive part of drawing this scene
        m.light_castshadow[:] = 0
    return scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="fsm_dwa", choices=("fsm_dwa", "ppo", "mppi", "dwa", "apf", "pure_pursuit"))
    ap.add_argument("--family", default="mixed")
    ap.add_argument("--seed", type=int, default=5021)
    ap.add_argument("--episodes", type=int, default=0, help="stop after this many episodes (0 = keep going with the next seed until you close the window); use 1 when recording")
    ap.add_argument("--vehicle", default="course", choices=("course", "nominal", "optimized"),
                    help="course = the starter pack's values (default); optimized = this repository's co-designed chassis")
    ap.add_argument("--cruise", type=float, default=3.5, help="road cruise speed [m/s]; the course guide specifies 3.5")
    ap.add_argument("--plain", action="store_true", help="the raw physics scene (flat ground, capsule people) instead of the dressed street")
    ap.add_argument("--shadows", action="store_true", help="draw shadows (prettier, slower; off by default)")
    ap.add_argument("--no-shield", action="store_true", help="disable the safety filter (shows the unprotected failure)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--rich", action="store_true",
                    help="collidable street furniture (a separate mode); the world is rebuilt per episode, so each gets its own window")
    a = ap.parse_args()

    extra = {"path": str(ROOT / "models" / "ppo_selected")} if a.controller == "ppo" else (
        {"nthread": 8} if a.controller == "mppi" else {})
    if a.controller == "fsm_dwa":                       # cruise speed from the course guide: 3.5 m/s, curb speed 1.2 m/s (FSMParams)
        # the settings of the evaluated "proposed" arm (capstone.monte_carlo._PROPOSED): kerb friction belief 0.35, 50 ms reaction time
        name, kw = "fsm_dwa", {"v_cruise": a.cruise, "fsm_kwargs": {"v_cruise": a.cruise, "mu_belief": 0.35}, "safety_kwargs": {"t_react": 0.05}}
    else:
        name, kw = controller_spec(a.controller, a.cruise, **extra)
    env = AEDRoverEnv(obs_mode="dict", veh=VehicleParams.by_name(a.vehicle), v_max=max(3.5, a.cruise), rich=a.rich)
    print(f"vehicle '{a.vehicle}': sprung {env.veh0.sprung_mass:.0f} kg, unsprung/wheel {env.veh0.unsprung_mass:.2f} kg, "
          f"k_s {env.veh0.susp_k:.0f} N/m, c_s {env.veh0.susp_c:.0f} N s/m, tyre mu {env.veh0.friction}, cruise {a.cruise} m/s", flush=True)
    ctrl, shield = make_controller(name, **kw), (None if a.no_shield else SafetyFilter(SafetyParams(t_react=0.05) if a.controller == "fsm_dwa" else SafetyParams()))
    seed = a.seed
    if not a.plain and not a.rich:
        # the dressed model is built per episode, and a viewer is bound to one model: one window per episode
        while True:
            env.reset(seed=seed, options={"family": a.family})
            ctrl.reset(env)
            if shield is not None:
                shield.reset()
            scene = dressed_scene(env, a.shadows)
            with mujoco.viewer.launch_passive(scene.scene.model, scene.scene.data) as v:
                v.opt.geomgroup[:] = 1
                track(v, env)
                ok = run_episode(a, env, ctrl, shield, v, seed, scene)
            scene.close()
            if not ok or a.episodes == 1:
                return
            seed += 1
            a.episodes = max(a.episodes - 1, 0) if a.episodes else 0
    if a.rich:
        # a rich environment compiles a new model on every reset, and a viewer is bound to one model: one window per episode
        while True:
            env.reset(seed=seed, options={"family": a.family})
            ctrl.reset(env)
            if shield is not None:
                shield.reset()
            with mujoco.viewer.launch_passive(env.world.model, env.world.data) as v:
                track(v, env)
                if not run_episode(a, env, ctrl, shield, v, seed) or a.episodes == 1:
                    return
            seed += 1
            a.episodes = max(a.episodes - 1, 0) if a.episodes else 0
    m, d = env.world.model, env.world.data
    with mujoco.viewer.launch_passive(m, d) as v:
        track(v, env)
        while v.is_running():
            env.reset(seed=seed, options={"family": a.family})
            ctrl.reset(env)
            if shield is not None:
                shield.reset()
            if not run_episode(a, env, ctrl, shield, v, seed) or a.episodes == 1:
                break
            seed += 1
            a.episodes = max(a.episodes - 1, 0) if a.episodes else 0


if __name__ == "__main__":
    main()
