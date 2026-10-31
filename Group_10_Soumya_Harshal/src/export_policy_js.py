"""Export the trained RL filter policies to the browser (docs/site/policy.js) for the Tremor Lab page.
Run from the repo root:  python src\\export_policy_js.py"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
out = {}
for s in range(1, 11):
    f = ROOT / "analytics" / f"rl_dr_s{s}" / "policy.npz"
    if not f.exists():
        print("missing", f)
        continue
    z = np.load(f)
    out[str(s)] = {k: np.round(z[k].astype(float), 5).tolist() for k in ("W0", "b0", "W1", "b1", "Wa", "ba")}
dest = ROOT / "docs" / "site" / "policy.js"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text("window.POLICIES=" + json.dumps(out, separators=(",", ":")) + ";\n", encoding="utf-8")
print(f"wrote {dest} with {len(out)} policies ({dest.stat().st_size / 1e6:.2f} MB)")
