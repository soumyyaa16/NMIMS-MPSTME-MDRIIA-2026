# AI audit: evidence for dossier section 5.1

Section 5.1 asks each group for **three cases where an AI tool gave wrong robotics or economic data**, how the group
caught each one and what was actually true. **Part B** (below) holds the three cases written into the dossier: errors made by
the AI assistant (Claude) while building this repository, each caught by checking. **Part A** lists discrepancies found in the
course starter pack; the starter pack is not an AI tool the team used, so they are supporting evidence about verifying sources,
not dossier cases. Replace any detail with your own account if it differs from what you experienced.

## Part A: discrepancies in the course starter pack

| # | What the starter material said | What the primary source or the code shows | Error type (dossier list) | How to verify it yourself |
|---|---|---|---|---|
| 1 | Survival model `0.67 - 0.023 t_CPR - 0.046 t_defib` attributed to Larsen et al. (1993) | The published model has three terms: 0.023 (CPR), **0.011** (first shock), 0.021 (ACLS). `0.046` is not a coefficient of this model (`docs/CLINICAL_MODEL.md`; tests pin the published values in `tests/test_survival.py`) | Wrong number or statistic | Read the abstract of Larsen 1993 (PMID 8214853); run `python -m pytest tests/test_survival.py` |
| 2 | Schierbeck et al. (2023): "211 suspected OHCAs", median time saving 1 min 52 s | Secondary coverage and the hosted paper give 55 suspected-arrest deliveries, AED first in 37 (67 %), median lead 3 min 14 s. The title also differs: the paper is a prospective observational study of real-life suspected arrests in Sweden | Real source, wrong details (author, year, venue) | Open doi 10.1016/S2589-7500(23)00161-9 and compare the abstract with `docs/LITERATURE_REVIEW_AND_FOUNDATIONAL_PAPERS.md` of the starter pack |
| 3 | Manuscript blueprint: "Larsen et al. formulated robust collision avoidance algorithms", "Tripathi et al. evaluated multi-agent coordination", "Naess et al. developed formulations for mechanical damping" | The three papers are a cardiac-arrest survival model, an in-hospital arrest circadian study and a retrospective study of ambulance response times. None concerns robot control | Real source, wrong details | Read the three titles and abstracts; compare with Section II of the blueprint |
| 4 | Starter economics script: `kappa = 0.095`, `payback_horizon_months = (0.35 / (1 - kappa)) * 12`, ambulance speed 7.5 km/h typed in as constants | These are inputs typed in as outputs. Computed from stated inputs the labour-lens kappa is far above 1 (`analytics/roi_results.json`); the course rules call hard-coded results "not analysis" | Wrong economics / cost logic | Run `python analytics/aed_delivery_roi.py` and compare; read the rubric's integrity safeguard 2 |
| 5 | Starter MJCF and controller: payload shock in `step_simulation` is `2.68 + np.random.normal(0, 0.08)` at kerbs | That is a generated target, not physics; real payload acceleration comes from the MuJoCo accelerometer on the isolator (`aed_shock_sensor` / `payload_acc`) | Other | Read `aed_navigation_controller.py` of the starter pack; compare with `analytics/aed_delivery_real_trials.csv` column `peak_payload_g` |

If you used an AI tool for any part of this repository's code or text, log it in dossier 5.1 and in the share-of-writing
question in Section 6 even if it is not one of the three error cases.

## Part B: cases from building this repository (AI tool: Claude)

These three are events of this repository's own history, written into dossier section 5.1 as AI-01 to AI-03. Unlike the table above, whose
errors come from the course starter pack (not from an AI tool the team used), these are errors made by the AI assistant while building the
project and caught by checking.

| # | What the AI did | What was true | How it was caught | Error type |
|---|---|---|---|---|
| AI-01 | Built the controller on the course vehicle (30 kN/m tyre) with a kerb-speed table measured for a 196 kN/m tyre | The course tyre needs its own table; measured with `experiments/02_curb_traversability.py --design course` | Traced why runs stalled at the kerb lip; payload shock above 3 g fell from 7.6 % to 3.6 % of runs | Wrong physics / equation |
| AI-02 | First health model compared the pod with the median ambulance only, so the survival gain was exactly zero | The pod is dispatched with the ambulance: expected gain over the whole response-time distribution is +2.3 points | Noticed the exact zero, recomputed with the Naess distribution and the Larsen model | Wrong economics / cost logic |
| AI-03 | First dossier box-plot showed only the lane-splitting condition while the text quoted pooled means | Pooled protocol conditions, n = 68 vs 116, as in `capstone_statistics.csv` | Checked each figure against the statistics table before the video | Wrong number or statistic |

Use them in the viva only as events you can explain: ask what the symptom was, what check exposed it, and what the corrected value is.