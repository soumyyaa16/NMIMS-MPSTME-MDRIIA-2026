"""One-off: lets run_trial() accept a plan, so scenarios can be run.  python src/patch_controller.py"""
import pathlib

p = pathlib.Path(__file__).resolve().parent / "surgical_tremor_controller.py"
s = p.read_text()
if "plan_from_scenario" in s:
    raise SystemExit("already patched")
NEW_FN = '''def plan_from_scenario(scn):
    """Build a plan dict from one entry of analytics/scenarios.json (produced by plan_scenarios.py)."""
    target = np.array(scn["target_m"])
    direction = np.array(scn["approach_dir"])
    direction = direction / np.linalg.norm(direction)
    entry = target + direction * scn["path_len_m"]
    return dict(target=target, entry=entry, standoff=entry + direction * STANDOFF, ins=-direction)


def min_jerk(tau):'''
for old, new in [("def min_jerk(tau):", NEW_FN),
                 ("record=True, data=None):", "record=True, data=None, plan=None):"),
                 ("    plan = load_plan()\n    robot = NeedleRobot(model)",
                  "    plan = plan if plan is not None else load_plan()\n    robot = NeedleRobot(model)")]:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
p.write_text(s)
print("patched", p)