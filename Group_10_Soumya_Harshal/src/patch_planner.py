"""One-off: adds the joint-limit gate to plan_scenarios.py.  python src/patch_planner.py"""
import pathlib

p = pathlib.Path(__file__).resolve().parent / "plan_scenarios.py"
s = p.read_text()
if "LIM_BLOCK" in s:
    raise SystemExit("already patched")
EDITS = [
('''CLEAR_BLOCK, CLEAR_FLAG = 0.010, 0.030  # m  arm-to-brain-hull clearance (excluding the needle)''',
'''CLEAR_BLOCK, CLEAR_FLAG = 0.010, 0.030  # m  arm-to-brain-hull clearance (excluding the needle)
LIM_BLOCK, LIM_FLAG = 0.10, 0.25        # rad  smallest distance of any joint from its limit along the path'''),
('''        q, ws = q_s, []
        for f in np.linspace(0, 1, 7):''','''        q, ws = q_s, []
        margin = self._margin(q_s)
        for f in np.linspace(0, 1, 7):'''),
('''            J, _, _ = self.robot.jac(q)
            ws.append(self.robot.manipulability(J))''','''            J, _, _ = self.robot.jac(q)
            ws.append(self.robot.manipulability(J))
            margin = min(margin, self._margin(q))'''),
('''        res = dict(min_w=wmin, clearance_mm=clr * 1e3)''','''        res = dict(min_w=wmin, clearance_mm=clr * 1e3, limit_margin_rad=margin)
        if margin < LIM_BLOCK:
            return dict(status="BLOCKED", reason="joint_limit", **res)'''),
('''        if clr < CLEAR_FLAG: flags.append("low_clearance")''','''        if clr < CLEAR_FLAG: flags.append("low_clearance")
        if margin < LIM_FLAG: flags.append("near_joint_limit")'''),
('''    def clearance(self, q):''','''    def _margin(self, q):
        return float(np.min(np.minimum(q - self.robot.lo, self.robot.hi - q)))

    def clearance(self, q):'''),
('''min_w=best.get("min_w", np.nan), clearance_mm=best.get("clearance_mm", np.nan),''','''min_w=best.get("min_w", np.nan), clearance_mm=best.get("clearance_mm", np.nan),
                       limit_margin_rad=best.get("limit_margin_rad", np.nan),'''),
('''                                      clearance_mm=best["clearance_mm"]))''','''                                      clearance_mm=best["clearance_mm"], limit_margin_rad=best["limit_margin_rad"]))'''),
('''path {row['path_len_mm']:.0f} mm  w {row['min_w']:.3f}  clr {row['clearance_mm']:.0f} mm"''','''path {row['path_len_mm']:.0f} mm  w {row['min_w']:.3f}  clr {row['clearance_mm']:.0f} mm  "
                  f"ok {row['pct_candidates_reachable']:.0f}%  lim {row['limit_margin_rad']:.2f} rad"'''),
]
for old, new in EDITS:
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
p.write_text(s)
print("patched", p)