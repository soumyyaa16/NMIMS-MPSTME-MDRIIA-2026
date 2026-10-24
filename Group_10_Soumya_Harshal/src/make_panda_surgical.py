"""Builds models/franka_emika_panda/panda_surgical.xml from Menagerie's panda_nohand.xml.
Changes (declare as 'Modified' in dossier 2.2):
  1. needle holder + needle shaft + needle_tip site on the 'attachment' body
  2. gravcomp="1" on link0..link7 (models the Franka's built-in gravity compensation)
"""
import re
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "models" / "franka_emika_panda"
src = (D / "panda_nohand.xml").read_text(encoding="utf-8")

# 1. needle on the attachment body
pat = re.compile(r'(<body name="attachment"[^>]*>\s*<site name="attachment_site"/>)(\s*</body>)')
needle = (
    '\n                      <geom name="needle_holder" type="cylinder" size="0.012 0.02" pos="0 0 0.02"'
    ' rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>'
    '\n                      <geom name="needle_shaft" type="capsule" fromto="0 0 0.04 0 0 0.19" size="0.0008"'
    ' rgba="0.85 0.87 0.9 1" contype="0" conaffinity="0"/>'
    '\n                      <site name="needle_tip" pos="0 0 0.19" size="0.0015" rgba="1 0.1 0.1 1"/>'
)
out, n1 = pat.subn(lambda m: m.group(1) + needle + m.group(2), src)
assert n1 == 1, "attachment body not found exactly once"

# 2. gravity compensation on the 8 link bodies
out, n2 = re.subn(r'<body name="(link[0-7])"', r'<body name="\1" gravcomp="1"', out)
assert n2 == 8, f"expected 8 link bodies, found {n2}"

(D / "panda_surgical.xml").write_text(out, encoding="utf-8")
print("wrote panda_surgical.xml | needle edits:", n1, "| gravcomp bodies:", n2)