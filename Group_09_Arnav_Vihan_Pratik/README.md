# Group 09 M20 quadruped capstone

This is the self-contained assigned group folder. Run commands **from this folder**. The supplied M20 has four articulated legs, twelve leg motors, four foot-wheel motors and seventeen original meshes. It rolls through an industrial yard, avoids three alternating barriers and performs paired-leg transfers over two 8 cm terraces.

|Member|Roll|Confirmed SAP|
|-|-|-|
|Arnav Saurabh Surve|E064|70362400067|
|Vihan Shripad Joshi|E070|70362400086|
|Pratik Mangesh Gaikwad|E073|70362300032|

## Project Showcase \& Video Demonstration

Published LinkedIn Showcase:https://lnkd.in/p/dkFQcPcj

## Run

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\\\\run\\\_demo.ps1 -Setup
powershell -NoProfile -ExecutionPolicy Bypass -File .\\\\run\\\_demo.ps1 -Live
```

Open a second PowerShell in this folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\\\\run\\\_demo.ps1 -Business
```

Python 3.12 is required; use `-Setup -Python C:\\\\path\\\\to\\\\python.exe` if the py launcher is unavailable. The launcher verifies the supplied version. The live viewport shows actual FSM, decisions, contacts and climb counters. The business dashboard is interactive and explicitly labels assumed inputs.

## Verified final study

60 independent simulations, 30 per policy, ten at each of three roughness bounds. Proposed: 30/30 accepted, two climbs and zero harmful contacts each. Reactive baseline: 0/30 accepted; all timeouts with contacts. Raw scientific telemetry is **actual 500 Hz**, 4,495,972 samples in 254 CSV parts. 54 tests passed in a separate group-only copy, and full seed-42 qpos/qvel/ctrl/targets/time replay arrays matched bitwise.

The baseline lacks mapping and paired climbing. These results demonstrate tested integration, not superiority over a strong planner or expert operator. Ideal sensing/exact pose, rigid low terraces and uncalibrated business assumptions remain explicit limits.

```powershell
.\\\\.venv\\\\Scripts\\\\python.exe -m pytest -q
.\\\\.venv\\\\Scripts\\\\python.exe analytics\\\\analyze\\\_capstone.py
.\\\\.venv\\\\Scripts\\\\python.exe analytics\\\\plot\\\_capstone.py
.\\\\.venv\\\\Scripts\\\\python.exe analytics\\\\telemetry\\\_business\\\_bridge.py
```

To execute fresh physics: `python analytics/run\\\_capstone\\\_benchmark.py --workers 3 --out analytics/capstone\\\_rerun` using the local environment. Never overwrite evidence to replace a failure.

## Documents

* [Required four-page dossier](docs/MDRIIA_Group_09_Technical_Dossier.pdf) and [editable original-field Word form](docs/MDRIIA_Group_09_Technical_Dossier.docx).
* [Detailed ten-page run guide](docs/M20_Complete_Step_by_Step_Run_Guide.pdf), also supplied in DOCX and Markdown.
* [Extended technical report](docs/M20_Extended_Technical_Report.pdf).
* [Four-page IEEE research manuscript](docs/paper/Group09_HazardousTerrain_IEEE.pdf) with editable TeX.
* [Verification and discrepancy resolution](docs/CAPSTONE_VERIFICATION_REPORT.md).
* [Showcase recording plan](docs/SHOWCASE_RECORDING_GUIDE.md).

All 212 faculty Word controls remain; 146 technical fields are completed and 66 personal fields are preserved for students. Recordings/webcams, genuine commits, peer review, personal declarations/signatures, public LinkedIn URL and each student's Teams Turn in remain human actions. No contribution history or public posting is fabricated.

## Evidence layout

`M20/` contains the canonical runtime and assets; `src/` entry points; `models/` portable MJCF aliases; `analytics/capstone\\\_500hz/` the final design, raw CSVs, source snapshot and inference; `analytics/capstone\\\_checks/pilot\\\_42/` the default full-course check. The separate `telemetry.csv`/`states.npz` visualization channel is 50 Hz and is not the scientific logger.

Historical `benchmark/`, `obstacle\\\_course/`, `climbing\\\_course/` and documents explicitly marked historical concern different protocols/source versions. Do not pool them with the final independent study. The GitHub subset omits historical evidence and replay videos/binaries; the full offline archive retains them. Source provenance and original templates remain identifiable.

AI assistance was substantial. Every member must review, understand, contribute and personally attest their own work.

