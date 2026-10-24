"""Scenario planner: computes, for every lesion target x approach zone, whether the plan is
REACHABLE / FLAGGED / BLOCKED.  Nothing is hard-coded: outcomes come from ray-casts on the brain mesh,
inverse kinematics, manipulability along the path and MuJoCo geometry distance.
    python src/plan_scenarios.py            ->  analytics/scenarios.json, analytics/scenario_outcomes.csv
Lesion targets are 'lesion from supplied annotation' (BraTS labels), not a diagnosis."""
import json
from collections import Counter

import mujoco
import numpy as np
import pandas as pd
import trimesh
from scipy import ndimage as ndi

import surgical_tremor_controller as sc

MESH_DIR = sc.ROOT / "meshes_v2"
STANDOFF = sc.STANDOFF
MAX_PATH = 0.100          # m  longest acceptable straight needle path in tissue
W_OK, W_ABORT = 0.03, sc.W_ABORT      # manipulability: >=W_OK fine, W_ABORT..W_OK flagged, <W_ABORT blocked
CLEAR_BLOCK, CLEAR_FLAG = 0.010, 0.030  # m  arm-to-brain-hull clearance (excluding the needle)
LIM_BLOCK, LIM_FLAG = 0.10, 0.25        # rad  smallest distance of any joint from its limit along the path
MIN_LESION_R = 0.003      # m  inscribed radius below which the target is flagged
N_DIR = 240               # candidate approach directions (Fibonacci sphere)
ZONE_HALF_ANGLE = np.deg2rad(50)
ZONES = {"superior": [0, 0, 1], "anterior": [0, 1, 0], "posterior": [0, -1, 0],
         "left": [-1, 0, 0], "right": [1, 0, 0], "inferior": [0, 0, -1]}
LESIONS = {"enhancing": "tumor", "necrotic": "necrotic", "edema": "edema"}


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def lesion_targets(pitch=0.001):
    """Deepest interior point of each lesion region, from the exported label meshes.
    Enhancing tumour is a shell around the necrotic core and the edema surrounds both, so the region is the
    filled mesh minus the filled inner meshes (a plain fill would close the cavity)."""
    fills = {n: trimesh.load(MESH_DIR / f"{n}.obj", process=False).voxelized(pitch).fill()
             for n in ("tumor", "necrotic", "edema")}
    allp = np.vstack([v.points for v in fills.values()])
    lo = allp.min(0) - pitch
    shape = np.ceil((allp.max(0) + pitch - lo) / pitch).astype(int) + 1
    M = {}
    for n, v in fills.items():
        a = np.zeros(shape, bool)
        a[tuple(np.rint((v.points - lo) / pitch).astype(int).T)] = True
        M[n] = a
    regions = {"enhancing": M["tumor"] & ~M["necrotic"], "necrotic": M["necrotic"],
               "edema": M["edema"] & ~M["tumor"] & ~M["necrotic"]}
    out = {}
    for name, mask in regions.items():
        dist = ndi.distance_transform_edt(mask) * pitch
        idx = np.unravel_index(np.argmax(dist), dist.shape)
        out[name] = (lo + np.array(idx) * pitch, float(dist.max()), float(mask.sum() * pitch**3 * 1e6))
    return out


def first_exit(brain, p, dirs):
    """Distance from p to the first brain-surface crossing along each direction (inf if none)."""
    out = np.full(len(dirs), np.inf)
    for s in range(0, len(dirs), 40):
        d = dirs[s:s + 40]
        o = np.repeat(p[None], len(d), 0)
        loc, ray, _ = brain.ray.intersects_location(o, d, multiple_hits=True)
        for k in range(len(d)):
            h = loc[ray == k]
            if len(h):
                out[s + k] = np.min(np.linalg.norm(h - p, axis=1))
    return out


class Checker:
    def __init__(self, model):
        self.m = model
        self.robot = sc.NeedleRobot(model)
        self.brain_g = model.geom("brain_geom").id
        root = model.body("p_link0").id
        bodies = {root}
        for b in range(model.nbody):
            par = b
            while par != 0:
                if par == root:
                    bodies.add(b); break
                par = model.body_parentid[par]
        needle = model.geom("p_needle_shaft").id
        self.geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies and g != needle
                      and (model.geom_group[g] == 3 or model.geom_contype[g] == 0 and model.geom_group[g] == 0)]
        self.d = mujoco.MjData(model)

    def _margin(self, q):
        return float(np.min(np.minimum(q - self.robot.lo, self.robot.hi - q)))

    def clearance(self, q):
        self.d.qpos[:7] = q
        mujoco.mj_forward(self.m, self.d)
        best = 1.0
        for g in self.geoms:
            dist = mujoco.mj_geomDistance(self.m, self.d, g, self.brain_g, 0.3, None)
            best = min(best, dist)
        return best

    def evaluate(self, target, dirn, path_len):
        ins = -dirn
        entry = target + dirn * path_len
        standoff = entry + dirn * STANDOFF
        ok_s, q_s = self.robot.solve_pose(standoff, ins, sc.HOME)
        if not ok_s:
            return dict(status="BLOCKED", reason="IK_unreachable_standoff")
        q, ws = q_s, []
        margin = self._margin(q_s)
        for f in np.linspace(0, 1, 7):
            p = standoff + (target - standoff) * f
            ok, q = self.robot.solve_pose(p, ins, q)
            if not ok:
                return dict(status="BLOCKED", reason="IK_unreachable_path")
            J, _, _ = self.robot.jac(q)
            ws.append(self.robot.manipulability(J))
            margin = min(margin, self._margin(q))
        wmin = float(min(ws))
        clr = min(self.clearance(q_s), self.clearance(q))
        res = dict(min_w=wmin, clearance_mm=clr * 1e3, limit_margin_rad=margin)
        if margin < LIM_BLOCK:
            return dict(status="BLOCKED", reason="joint_limit", **res)
        if wmin < W_ABORT:
            return dict(status="BLOCKED", reason="singular_path", **res)
        if clr < CLEAR_BLOCK:
            return dict(status="BLOCKED", reason="arm_brain_collision", **res)
        flags = []
        if wmin < W_OK: flags.append("low_manipulability")
        if clr < CLEAR_FLAG: flags.append("low_clearance")
        if margin < LIM_FLAG: flags.append("near_joint_limit")
        return dict(status="FLAGGED" if flags else "REACHABLE", reason="+".join(flags) or "ok", **res)


def main():
    model = sc.load_model()
    chk = Checker(model)
    brain = trimesh.load(MESH_DIR / "brain.obj", process=False)
    meta = json.loads((MESH_DIR / "export_meta.json").read_text())
    dirs_all = fib_sphere(N_DIR)
    rows, scenarios = [], []
    targets = lesion_targets()
    for lname in LESIONS:
        target, rad, vol_cm3 = targets[lname]
        inside_brain = bool(brain.contains(target[None])[0])
        extra = {}
        if lname == "enhancing":
            extra["target_vs_export_meta_mm"] = float(np.linalg.norm(target - np.array(meta["target_m"])) * 1e3)
        lens = first_exit(brain, target, dirs_all)
        print(f"\n{lname}: target (mm) {np.round(target*1e3,1)}  inscribed radius {rad*1e3:.1f} mm"
              f"  volume {vol_cm3:.1f} cm3  inside brain: {inside_brain}  {extra}", flush=True)
        for zone, axis in ZONES.items():
            axis = np.array(axis, float)
            cand = [i for i in range(N_DIR) if np.arccos(np.clip(dirs_all[i] @ axis, -1, 1)) <= ZONE_HALF_ANGLE
                    and np.isfinite(lens[i]) and lens[i] <= MAX_PATH]
            n_far = int(sum(np.arccos(np.clip(dirs_all[i] @ axis, -1, 1)) <= ZONE_HALF_ANGLE for i in range(N_DIR))
                        - len(cand))
            evals = []
            for i in sorted(cand, key=lambda i: lens[i]):
                e = chk.evaluate(target, dirs_all[i], lens[i])
                e.update(dir=dirs_all[i].tolist(), path_len_m=float(lens[i]))
                evals.append(e)
            rank = {"REACHABLE": 0, "FLAGGED": 1, "BLOCKED": 2}
            if evals:
                best = sorted(evals, key=lambda e: (rank[e["status"]], e["path_len_m"]))[0]
                reasons = Counter(e["reason"] for e in evals)
            else:
                best = dict(status="BLOCKED", reason="no_path_within_%dmm" % (MAX_PATH * 1e3))
                reasons = Counter()
            if best["status"] != "BLOCKED" and rad < MIN_LESION_R:
                best["status"], best["reason"] = "FLAGGED", best["reason"] + "+small_lesion"
            row = dict(lesion=lname, zone=zone, status=best["status"], reason=best["reason"],
                       n_candidates=len(cand), n_too_long=n_far, inscribed_radius_mm=rad * 1e3,
                       pct_candidates_reachable=100.0 * np.mean([e["status"] == "REACHABLE" for e in evals]) if evals else 0.0,
                       pct_candidates_blocked=100.0 * np.mean([e["status"] == "BLOCKED" for e in evals]) if evals else 100.0,
                       target_mm=json.dumps(np.round(target * 1e3, 1).tolist()),
                       path_len_mm=best.get("path_len_m", np.nan) * 1e3 if "path_len_m" in best else np.nan,
                       min_w=best.get("min_w", np.nan), clearance_mm=best.get("clearance_mm", np.nan),
                       limit_margin_rad=best.get("limit_margin_rad", np.nan),
                       block_reasons=json.dumps(dict(reasons)))
            rows.append(row)
            print(f"  {zone:9s} {row['status']:9s} {row['reason']:28s} cand {len(cand):3d}  "
                  f"path {row['path_len_mm']:.0f} mm  w {row['min_w']:.3f}  clr {row['clearance_mm']:.0f} mm  "
                  f"ok {row['pct_candidates_reachable']:.0f}%  lim {row['limit_margin_rad']:.2f} rad", flush=True)
            if best["status"] != "BLOCKED":
                scenarios.append(dict(id=f"{lname}_{zone}", lesion=lname, zone=zone, status=best["status"],
                                      target_m=target.tolist(), approach_dir=best["dir"],
                                      path_len_m=best["path_len_m"], min_w=best["min_w"],
                                      clearance_mm=best["clearance_mm"], limit_margin_rad=best["limit_margin_rad"]))
    pd.DataFrame(rows).to_csv(sc.ROOT / "analytics" / "scenario_outcomes.csv", index=False)
    (sc.ROOT / "analytics" / "scenarios.json").write_text(json.dumps(scenarios, indent=2))
    df = pd.DataFrame(rows)
    print("\n", df.pivot(index="lesion", columns="zone", values="status").to_string())
    print("\nsaved analytics/scenario_outcomes.csv, analytics/scenarios.json")


if __name__ == "__main__":
    main()