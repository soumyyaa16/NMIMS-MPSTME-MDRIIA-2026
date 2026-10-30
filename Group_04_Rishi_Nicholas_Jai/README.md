# MDRIIA Group 04: Multi-Arm Robotic Gripper for Non-Cooperative Space Debris Capture in LEO
**Course:** Modern Day Robotics and Its Industrial Applications (MDRIIA - Course Code: 702CO0E012)  
**Academic Term:** Academic Year 2026–2027 | Semester V (B.Tech CSBS)  
**Institution:** SVKM's NMIMS MPSTME, Mumbai  

---

## 1. Authorized Research Title & Problem Statement

> "How can a multi-arm robotic gripper mechanism simulated in MuJoCo utilize impedance contact control to synchronize with and capture tumbling non-cooperative orbital debris in LEO while mitigating collision impulse and momentum transfer?"

### Core Engineering Focus
* **Physics & Kinematics:** Google DeepMind MuJoCo multi-body dynamic physics simulation (`models/space_debris_gripper.xml`).
* **Autonomous Control:** Closed-loop Python control architecture with student implementation boundaries (`src/debris_capture_controller.py`).
* **Technoeconomic Evaluation:** Dimensionless CSBS operational economics, cost per operation, and return on investment model (`analytics/constellation_economics.py`).
* **Empirical Validation:** Reproducible benchmark trials across 60 paired Monte Carlo runs and statistical hypothesis testing (`analytics/real_trials_summary.csv`, `analytics/impedance_real_trials.csv`, and `analytics/rigid_pd_real_trials.csv`).

---

> [!NOTE]
> ### The Devil's Advocate: Reality Check & Theoretical Roast
> *"Modeling viscoelastic impedance contact control in microgravity, where your biggest mathematical triumph will be transferring just enough rotational impulse during capture to convert a dead 2-ton Soviet booster into an unstoppable 10,000 RPM orbital buzzsaw."*

---

## 2. Student Engineering Team Matrix

| Roll No | SAP ID | Student Name | Technical Role | Assigned Files | Core Viva Defense Area |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `E034` | `70362400032` | **Rishi Vinod Koli** | Lead Orbital Dynamics & MuJoCo Multi-Body Physics Architect | `models/space_debris_gripper.xml`<br>`src/debris_capture_controller.py` | Zero-gravity multi-body spacecraft dynamics, Generalized Jacobian Matrix (GJM), and momentum transfer during contact. |
| `E035` | `70362400029` | **Nicholas Lewis** | Impedance Contact Control & Robotic Kinematics Engineer | `analytics/generate_paper_figures.py`<br>`analytics/run_trials.py` | Cartesian impedance force control, tumbling satellite spin matching, and post-contact detumbling damping. |
| `E036` | `70362400014` | **Jai Maini** | CSBS Commercial Space Economics & Satellite De-Orbiting Business Analyst | `analytics/constellation_economics.py`<br>`models/space_debris_gripper.xml` | LEO orbital slot preservation economics, Kessler syndrome collision risk reduction, and multi-mission ADR amortization models. |


### Student Engineering Commendation & Acknowledgments
SVKM's NMIMS MPSTME conveys sincere appreciation and heartfelt gratitude to **Rishi Vinod Koli (E034)**, **Nicholas Lewis (E035)**, and **Jai Maini (E036)** for their disciplined commitment, late-night debugging, and technical craftsmanship throughout Semester VI. Your rigorous work in DeepMind MuJoCo physics modeling, closed-loop telemetry instrumentation, and Computer Science & Business Systems (CSBS) technoeconomic modeling exemplifies the highest standards of undergraduate engineering inquiry.

> *"Scientists discover the world that exists; engineers create the world that never was."*  
> — **Theodore von Kármán**

> *"There is no substitute for hard work. Genius is one percent inspiration and ninety-nine percent perspiration."*  
> — **Thomas A. Edison**

---

## 3. Foundational Literature Benchmarks (Strict 2 Seminal : 4 Recent Ratio)

The research project is theoretically anchored on six foundational peer-reviewed publications validated against global bibliographic databases.

| # | Author (Year) | Type | Paper Title | Venue / Indexing | Active DOI Link | Primary Extracted Formulation | Student Lead |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Yan et al. (2020)** | Seminal | *Multi-objective configuration optimization for coordinated capture of dual-arm space robot* | Acta Astronautica | [https://doi.org/10.1016/j.actaastro.2019.11.002](https://doi.org/10.1016/j.actaastro.2019.11.002) | `Generalized Jacobian $J_g = J_m - J_b I_b^{-1...` | Rishi Vinod Koli (E034) |
| 2 | **Rybus et al. (2022)** | Recent | *Optimal collision-free path planning of a free-floating space robot using splines* | Acta Astronautica | [https://doi.org/10.1016/j.actaastro.2021.10.012](https://doi.org/10.1016/j.actaastro.2021.10.012) | `Momentum conservation $I_s \omega_0 + \sum I_...` | Rishi Vinod Koli (E034) & Nicholas Lewis (E035) |
| 3 | **Han et al. (2020)** | Recent | *Combined spacecraft stabilization control after multiple impacts during the capture of non-cooperative targets* | Acta Astronautica | [https://doi.org/10.1016/j.actaastro.2020.05.035](https://doi.org/10.1016/j.actaastro.2020.05.035) | `Contact impulse equation $I_{\text{imp}} = \i...` | Nicholas Lewis (E035) |
| 4 | **Wang et al. (2021)** | Recent | *A strategy to decelerate and capture a spinning object by a dual-arm space robot* | Aerospace Science and Technology | [https://doi.org/10.1016/j.ast.2021.106682](https://doi.org/10.1016/j.ast.2021.106682) | `Spin velocity matching $\lim_{t \to t_c} (\om...` | Nicholas Lewis (E035) & Jai Maini (E036) |
| 5 | **Tao et al. (2021)** | Recent | *Impedance-Sliding Mode Control With Force Constraints for Space Robots Capturing Non-Cooperative Targets* | IEEE Access | [https://doi.org/10.1109/ACCESS.2021.3129835](https://doi.org/10.1109/ACCESS.2021.3129835) | `Target impedance dynamic $M_d \ddot{e} + D_d ...` | Jai Maini (E036) & Nicholas Lewis (E035) |
| 6 | **Luo et al. (2017)** | Seminal | *A review of uncertainty propagation in orbital mechanics* | Progress in Aerospace Sciences | [https://doi.org/10.1016/j.paerosci.2016.12.002](https://doi.org/10.1016/j.paerosci.2016.12.002) | `Clohessy-Wiltshire (CW) relative equations: $...` | Rishi Vinod Koli (E034) & Jai Maini (E036) |

For the exhaustive literature analysis, mathematical derivations, and viva defense questions, refer to:
* [`docs/LITERATURE_REVIEW_AND_FOUNDATIONAL_PAPERS.md`](docs/LITERATURE_REVIEW_AND_FOUNDATIONAL_PAPERS.md)
* [`docs/RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md`](docs/RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md)

---

## 4. Project Demonstration & Academic Showcase (LinkedIn)

[![Watch Video Demonstration on LinkedIn](docs/figures/video_poster.png)](https://lnkd.in/p/d9U73k7b)

* **Video Demonstration:** [Watch 60-Second Walkthrough on LinkedIn](https://lnkd.in/p/d9U73k7b) *(Click thumbnail above to open LinkedIn post)*
* **Student Presenters:** **Rishi Vinod Koli (E034)**, **Nicholas Lewis (E035)**, **Jai Maini (E036)**
* **Academic Institutional Tags:** SVKM's NMIMS MPSTME | Academic Directorate | Industry 4.0 Robotics
* **Submission Protocol:** Record a 60–90 second demonstration of your MuJoCo simulation and telemetry. Publish on LinkedIn tagging MPSTME, Dean, and Course Faculty. Insert your live post URL in `docs/TEAM_ROSTER.json` under `"linkedin_url"`, and submit a pull request to update this project dossier and the cohort dashboard.

---

## 5. Repository Directory Architecture

```
.
|-- README.md                                          <- Front-page research charter, student roster & literature
|-- RESEARCH_AND_IMPLEMENTATION_GUIDE.md               <- Comprehensive technical engineering guide
|-- docs/
|   |-- TEAM_ROSTER.json                               <- Machine-readable team configuration
|   |-- LITERATURE_REVIEW_AND_FOUNDATIONAL_PAPERS.md   <- Exhaustive literature dossier (6 verified papers)
|   |-- RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md         <- 4-page IEEE conference manuscript blueprint
|   `-- figures/
|       |-- figure1_system_architecture.png            <- 300 DPI system architecture diagram
|       |-- figure2_kinematic_telemetry.png            <- 300 DPI kinematics and simulation telemetry
|       `-- figure3_comparative_performance.png        <- 300 DPI comparative benchmark visualization
|-- models/
|   |-- .gitkeep
|   `-- space_debris_gripper.xml                                    <- MuJoCo MJCF simulation model
|-- src/
|   |-- test_env.py                                    <- Physics validation & environment tester
|   `-- debris_capture_controller.py                                      <- Autonomous control loop (# TODO student boundaries)
`-- analytics/
    |-- .gitkeep
    |-- constellation_economics.py                                      <- CSBS dimensionless technoeconomic model & cost per operation visualizer
    |-- generate_paper_figures.py                                      <- Standalone 300 DPI publication figure generator
    |-- run_trials.py                                                  <- Monte Carlo trial runner for impedance vs rigid baseline
    |-- real_trials_summary.csv                                        <- Empirical benchmark dataset across N=60 paired runs
    |-- impedance_real_trials.csv                                      <- Empirical trials for proposed compliant impedance controller (N=30)
    |-- rigid_pd_real_trials.csv                                       <- Empirical trials for baseline rigid PD controller (N=30)
    `-- runs/                                                          <- High-rate 500 Hz timeseries telemetry runs (force, torque, kinematics)
```

---

## 5. Pedagogical Boundaries: Guidance vs Student Ownership

To ensure academic rigor and authentic student learning, this repository enforces strict boundaries between scaffolding and student deliverables:

* **Provided by Course Scaffolding:**
  * Validated physical model architecture in MuJoCo MJCF (`models/space_debris_gripper.xml`).
  * Verification toolchain and smoke test script (`src/test_env.py`).
  * Literature foundation, mathematical formulations, and conference paper blueprint (`docs/`).
  * Dimensionless technoeconomic framework (`analytics/constellation_economics.py`).
* **Student Technical Deliverables (Required for Evaluation):**
  * Implement and tune the control algorithm in `src/debris_capture_controller.py` (filling all marked `# TODO [Student Roll / Name]` blocks).
  * Run physics simulation trials to collect and expand empirical data in `analytics/real_trials_summary.csv` (`impedance_real_trials.csv` and `rigid_pd_real_trials.csv`).
  * Re-run `analytics/generate_paper_figures.py` to regenerate publication figures with live experimental telemetry.
  * Author the final 4-page IEEE conference paper in `docs/RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md` and defend individual Git commits during the oral viva.

---

## 6. Sprint 0 Onboarding & Physics Environment Verification

Verify your local Python and MuJoCo simulation environment:

```powershell
# Step 1: Clone repository and navigate to group directory
git clone <repository_url>
cd <repository_root>/Group_04_Rishi_Nicholas_Jai

# Step 2: Checkout your individual feature branch
# Example for lead student:
git checkout -b feat/e034-lead-orbital-dynamic

# Step 3: Run environment smoke test
python src/test_env.py

# Step 4: Verify 300 DPI publication figures
python analytics/generate_paper_figures.py
```
