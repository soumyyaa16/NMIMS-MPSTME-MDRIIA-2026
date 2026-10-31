# Expo and viva runbook (Saturday showcase)

No slides: you present the running simulation, the business graphs and a webcam picture-in-picture of every member.
Numbers below are from the committed run (`analytics/`); re-read `docs/CAPSTONE_RESULTS.md` if you re-run anything.

## Before you leave home

```bash
git clone <your fork>            # a fresh clone, as the rubric asks
cd shock-aware-aed-delivery
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev,viz]"      # or, in the course folder: pip install -r requirements.txt
python src/test_env.py           # (course folder) every line [OK]
python -m pytest -m "not slow" -n auto
```

## Live demo, five minutes

| Step | Command | What to point at |
|---|---|---|
| 1. The robot and its FSM | `python scripts/live_viewer.py --family mixed --seed 71001` (MuJoCo's own viewer, live physics, dressed street; in the course folder `python src/live_viewer.py ...`). Optional cinematic window: `python scripts/live_cinematic.py --controller fsm_dwa --family mixed --seed 71001`; `--plain` on the viewer shows the raw physics scene | the state changes DISPATCH, ROAD_CRUISE, CURB_SIDEWALK, ARRIVE, AED_HANDOVER (top-left); the speed drops in CURB_SIDEWALK |
| 2. One transition and its line of code | open `src/aedrover/capstone/fsm.py`, `MissionFSM.update` | `in_kerb_zone(x)` fires CURB_SIDEWALK; `x_goal - x <= arrive_m` fires ARRIVE |
| 3. A failure that is handled | `python scripts/live_viewer.py --controller pure_pursuit --no-shield --family mixed --seed 71001` (collides), then the step-1 command on the same seed (reaches the goal) | same street and seed: the unprotected baseline collides, the safety filter and planner avoid it |
| 4. 500 Hz evidence | `analytics/telemetry/representative_run_500hz.csv` and `docs/figures/capstone_trajectory.png` | one row every 0.002 s, FSM state in every row |
| 5. Statistics | `docs/figures/capstone_ttfs_boxplot.png`, `analytics/capstone_statistics.csv` | Welch t, Cohen's d, interval; success rates beside it |
| 6. Business | `docs/figures/business_sensitivity.png`, `business_queue.png`, `business_fmea.png` | break-even, queue delay, RPN before and after |

## What the results say (be ready to say it plainly)

* Time to first shock, successful runs in the three protocol conditions: baseline about 13.8 min, proposed about 10.5 min,
  Welch p < 0.001, d about 1.2. But the baseline reaches the goal in only 68 of 150 runs, the proposed controller in 116, so
  the success rates matter as much as the means.
* The proposed controller tracks the centreline worse (RMSE about 0.8 m against 0.02 m) and uses more control effort: it
  leaves the line to avoid pedestrians and obstacles, which the baseline does not do. Lower RMSE is not better if the robot
  collides.
* Payload shock: at the guide's 3.5 m/s cruise the naive controller exceeds 3 g in 28 % of runs, the proposed controller in 3.6 %
  (worst 3.44 g), so the 3 g budget is still a recorded DEVIATION. Vehicle values are the course guide's (`VehicleParams.course`).
* The pod beats the median ambulance only through parallel dispatch: it pays off in the slow tail of ambulance responses,
  at short range (the README places break-even near half a kilometre).
* The labour-reallocation case alone does not pay back (kappa far above 1). The reason to deploy is the health benefit,
  reported as QALY per ALS vehicle-year of cost. Say this before anyone asks.

## Likely viva questions, with where the answer is

| Question | Answer and evidence |
|---|---|
| Why that `solref` and friction? What if you halve them? | tyre `solref` 0.03 s, ratio 1.0 gives about 200 kN/m (196 kN/m validated, `docs/VALIDATION.md`); friction 1.0 sliding. Halving the time constant is the experiment to run live: edit `tyre_solref` in `sim/vehicle_mjcf.py`, re-run `experiments/01_validate_suspension.py`, read the new stiffness. Do not quote an outcome you have not run |
| Show one FSM transition and the line that triggers it | `MissionFSM.update` in `capstone/fsm.py`, and `transitions` (time, from, to, reason) |
| Which single input moves payback most, and where does it come from? | `analytics/roi_tornado.csv`; the incidence of arrests has no verified national value, it is an ASSUMPTION and the sensitivity graph's x-axis |
| Why Welch's t-test? What does d mean? | unequal variances and group sizes (68 vs 116 successes); d is the difference in pooled standard deviations, about 0.8 is a large effect |
| What did AI get wrong and how did you prove it? | your own account; evidence candidates in `docs/AI_AUDIT_CANDIDATES.md` (Larsen shock coefficient 0.011 not 0.046; Schierbeck numbers; mismatched paper descriptions; hard-coded kappa) |
| Why is the business case negative? | `docs/CAPSTONE.md`, "What the numbers say": one hub sees under one cardiac-arrest call a day, so the saved ambulance hours are far below a pod's running cost |
| Which standards deviations did you accept? | `docs/STANDARDS_COMPLIANCE.md`: 1.4 m/s pedestrian-zone cap, clearance below 0.5 m, 3 g budget exceeded in 3.6 % of runs |

## Submission checklist (each member)

1. Fill the fields only you can answer in the dossier: authorship (2.2), LinkedIn URL and video facts, AI audit, peer
   reviews, Section 6, sign-off. Do not edit the file in Google Docs.
2. Export the PDF from Word (File, Export), commit the `.docx` and `.pdf` to `Group_03_Kashish_Vaishnavi/docs/`.
3. Post the 3 to 4 minute video natively on LinkedIn (live simulation, business graph, every member on webcam).
4. Open the MS Teams assignment, attach the PDF and press Turn in yourself.
5. Work on your own feature branch, at least four conventional commits each, open a pull request into `main`.
