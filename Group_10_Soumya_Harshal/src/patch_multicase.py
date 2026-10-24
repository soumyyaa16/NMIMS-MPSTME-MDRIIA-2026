"""One-off: adds multi-case support (--case) to the controller, planner and scenario runner.
    python src/patch_multicase.py"""
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
PATCHES = {
"surgical_tremor_controller.py": [
(r'''
# ---------------------------------------------------------------- helpers
def load_model():
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))

''',
r'''
# ---------------------------------------------------------------- helpers
CASES_DIR = ROOT / "cases"


def case_meshdir(case=None):
    """Mesh folder of a case (None = the original patient in meshes_v2)."""
    return ROOT / "meshes_v2" if case is None else CASES_DIR / case / "meshes"


def load_model(case=None):
    """Load the scene.  With case='c160' the lesion/brain meshes and the target marker come from cases/c160."""
    if case is None:
        return mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    import re
    meshdir = case_meshdir(case).resolve()
    meta = json.loads((meshdir / "export_meta.json").read_text())
    xml = MODEL_PATH.read_text()
    assert 'meshdir="../meshes_v2"' in xml, "scene XML has an unexpected meshdir"
    xml = xml.replace('meshdir="../meshes_v2"', f'meshdir="{meshdir.as_posix()}"')
    tx, ty, tz = meta["target_m"]
    xml = re.sub(r'(<site name="target_site" pos=")[^"]*(")', rf"\g<1>{tx:.7f} {ty:.7f} {tz:.7f}\g<2>", xml)
    for name in ("tumor", "necrotic", "edema"):          # a label may be absent in a case
        if not (meshdir / f"{name}.obj").exists():
            xml = "\n".join(l for l in xml.split("\n") if f'file="{name}.obj"' not in l and f'mesh="{name}"' not in l)
    tmp = MODEL_PATH.parent / f"_case_{case}.xml"
    tmp.write_text(xml)
    return mujoco.MjModel.from_xml_path(str(tmp))

'''),
],
"plan_scenarios.py": [
(r'''REACHABLE / FLAGGED / BLOCKED.  Nothing is hard-coded: outcomes come from ray-casts on the brain mesh,
inverse kinematics, manipulability along the path and MuJoCo geometry distance.
    python src/plan_scenarios.py            ->  analytics/scenarios.json, analytics/scenario_outcomes.csv
Lesion targets are 'lesion from supplied annotation' (BraTS labels), not a diagnosis."""
import json
from collections import Counter''',
r'''REACHABLE / FLAGGED / BLOCKED.  Nothing is hard-coded: outcomes come from ray-casts on the brain mesh,
inverse kinematics, manipulability along the path and MuJoCo geometry distance.
    python src/plan_scenarios.py                 ->  analytics/scenarios.json, analytics/scenario_outcomes.csv
    python src/plan_scenarios.py --case c160     ->  analytics/cases/c160/scenarios.json, ...
Lesion targets are 'lesion from supplied annotation' (BraTS labels), not a diagnosis."""
import argparse
import json
from collections import Counter'''),
(r'''    filled mesh minus the filled inner meshes (a plain fill would close the cavity)."""
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
        idx = np.unravel_index(np.argmax(dist), dist.shape)''',
r'''    filled mesh minus the filled inner meshes (a plain fill would close the cavity)."""
    fills = {n: trimesh.load(MESH_DIR / f"{n}.obj", process=False).voxelized(pitch).fill()
             for n in ("tumor", "necrotic", "edema") if (MESH_DIR / f"{n}.obj").exists()}
    allp = np.vstack([v.points for v in fills.values()])
    lo = allp.min(0) - pitch
    shape = np.ceil((allp.max(0) + pitch - lo) / pitch).astype(int) + 1
    M = {}
    for n, v in fills.items():
        a = np.zeros(shape, bool)
        a[tuple(np.rint((v.points - lo) / pitch).astype(int).T)] = True
        M[n] = a
    none = np.zeros(shape, bool)
    T_, N_, E_ = M.get("tumor", none), M.get("necrotic", none), M.get("edema", none)
    regions = {"enhancing": T_ & ~N_, "necrotic": N_, "edema": E_ & ~T_ & ~N_}
    out = {}
    for name, mask in regions.items():
        if mask.sum() < 50:
            continue
        dist = ndi.distance_transform_edt(mask) * pitch
        idx = np.unravel_index(np.argmax(dist), dist.shape)'''),
(r'''

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
        inside_brain = bool(brain.contains(target[None])[0])''',
r'''

def main(case=None):
    global MESH_DIR
    MESH_DIR = sc.case_meshdir(case)
    out_dir = sc.ROOT / "analytics" if case is None else sc.ROOT / "analytics" / "cases" / case
    out_dir.mkdir(parents=True, exist_ok=True)
    model = sc.load_model(case)
    chk = Checker(model)
    brain = trimesh.load(MESH_DIR / "brain.obj", process=False)
    meta = json.loads((MESH_DIR / "export_meta.json").read_text())
    dirs_all = fib_sphere(N_DIR)
    rows, scenarios = [], []
    targets = lesion_targets()
    for lname in LESIONS:
        if lname not in targets:
            print(f"\n{lname}: absent in this case - skipped")
            continue
        target, rad, vol_cm3 = targets[lname]
        inside_brain = bool(brain.contains(target[None])[0])'''),
(r'''            if best["status"] != "BLOCKED" and rad < MIN_LESION_R:
                best["status"], best["reason"] = "FLAGGED", best["reason"] + "+small_lesion"
            row = dict(lesion=lname, zone=zone, status=best["status"], reason=best["reason"],
                       n_candidates=len(cand), n_too_long=n_far, inscribed_radius_mm=rad * 1e3,
                       pct_candidates_reachable=100.0 * np.mean([e["status"] == "REACHABLE" for e in evals]) if evals else 0.0,''',
r'''            if best["status"] != "BLOCKED" and rad < MIN_LESION_R:
                best["status"], best["reason"] = "FLAGGED", best["reason"] + "+small_lesion"
            row = dict(case=case or "orig", lesion=lname, zone=zone, status=best["status"], reason=best["reason"],
                       n_candidates=len(cand), n_too_long=n_far, inscribed_radius_mm=rad * 1e3,
                       pct_candidates_reachable=100.0 * np.mean([e["status"] == "REACHABLE" for e in evals]) if evals else 0.0,'''),
(r'''                  f"ok {row['pct_candidates_reachable']:.0f}%  lim {row['limit_margin_rad']:.2f} rad", flush=True)
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
''',
r'''                  f"ok {row['pct_candidates_reachable']:.0f}%  lim {row['limit_margin_rad']:.2f} rad", flush=True)
            if best["status"] != "BLOCKED":
                scenarios.append(dict(id=f"{lname}_{zone}", case=case, lesion=lname, zone=zone, status=best["status"],
                                      target_m=target.tolist(), approach_dir=best["dir"],
                                      path_len_m=best["path_len_m"], min_w=best["min_w"],
                                      clearance_mm=best["clearance_mm"], limit_margin_rad=best["limit_margin_rad"]))
    pd.DataFrame(rows).to_csv(out_dir / "scenario_outcomes.csv", index=False)
    (out_dir / "scenarios.json").write_text(json.dumps(scenarios, indent=2))
    df = pd.DataFrame(rows)
    print("\n", df.pivot(index="lesion", columns="zone", values="status").to_string())
    print(f"\nsaved {out_dir}/scenario_outcomes.csv and scenarios.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="case id from src/import_case.py, e.g. c160 (default: original patient)")
    main(ap.parse_args().case)
'''),
],
"run_scenarios.py": [
(r'''"""Run the tremor experiment on every planned scenario.
    python src/run_scenarios.py [--n 5] [--ids enhancing_superior,edema_anterior]
Needs analytics/scenarios.json from plan_scenarios.py.  Writes analytics/scenario_trials.csv."""
import argparse
import json''',
r'''"""Run the tremor experiment on every planned scenario.
    python src/run_scenarios.py [--n 5] [--ids enhancing_superior,edema_anterior]
Needs scenarios.json from plan_scenarios.py.  Writes scenario_trials.csv next to it.
    python src/run_scenarios.py --case c160 --n 3        (analytics/cases/c160/...)"""
import argparse
import json'''),
(r'''ap.add_argument("--ids", default="", help="comma list of scenario ids (default: all)")
ap.add_argument("--seed0", type=int, default=2000)
a = ap.parse_args()

scns = json.loads((sc.ROOT / "analytics" / "scenarios.json").read_text())
if a.ids:
    keep = a.ids.split(",")
    scns = [s for s in scns if s["id"] in keep]
model = sc.load_model()
rows, t0 = [], time.time()
for s in scns:''',
r'''ap.add_argument("--ids", default="", help="comma list of scenario ids (default: all)")
ap.add_argument("--seed0", type=int, default=2000)
ap.add_argument("--case", default=None, help="case id (default: original patient)")
a = ap.parse_args()

adir = sc.ROOT / "analytics" if a.case is None else sc.ROOT / "analytics" / "cases" / a.case
scns = json.loads((adir / "scenarios.json").read_text())
if a.ids:
    keep = a.ids.split(",")
    scns = [s for s in scns if s["id"] in keep]
model = sc.load_model(a.case)
rows, t0 = [], time.time()
for s in scns:'''),
(r'''        for c in ("NT", "BASE", "PROP"):
            r = sc.run_trial(model, c, seed, amp, ks, plan=plan)["summary"]
            r.update(scenario=s["id"], planner_status=s["status"], path_len_mm=s["path_len_m"] * 1e3,
                     plan_min_w=s["min_w"])
            rows.append(r)
    print(f"{s['id']:22s} done  elapsed {time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
df.to_csv(sc.ROOT / "analytics" / "scenario_trials.csv", index=False)
pd.set_option("display.width", 220)
g = df.pivot_table(index="scenario", columns="cond",''',
r'''        for c in ("NT", "BASE", "PROP"):
            r = sc.run_trial(model, c, seed, amp, ks, plan=plan)["summary"]
            r.update(case=a.case or "orig", scenario=s["id"], planner_status=s["status"], path_len_mm=s["path_len_m"] * 1e3,
                     plan_min_w=s["min_w"])
            rows.append(r)
    print(f"{s['id']:22s} done  elapsed {time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
df.to_csv(adir / "scenario_trials.csv", index=False)
pd.set_option("display.width", 220)
g = df.pivot_table(index="scenario", columns="cond",'''),
],
}
for fname, edits in PATCHES.items():
    p = HERE / fname
    s = p.read_text()
    if "case_meshdir" in s or "--case" in s:
        print(fname, "already patched, skipped")
        continue
    for old, new in edits:
        assert s.count(old) == 1, (fname, old[:60])
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", fname)