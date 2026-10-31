# MDRIIA Group 05: Collaborative Dual-UAV Campus Perimeter Patrol and Surveillance
**Course:** Modern Day Robotics and Its Industrial Applications (MDRIIA - Course Code: 702CO0E012)  
**Academic Term:** Academic Year 2026–2027 | Semester VI (B.Tech CSBS)  
**Institution:** SVKM's NMIMS MPSTME, Mumbai  

---

## 1. Authorized Research Title & Problem Statement

> "To what extent can a collaborative dual-UAV surveillance system simulated in MuJoCo optimize campus perimeter patrol cycle time and OpenCV human detection latency compared to static security guard patrols?"

### Core Engineering Focus
* **Physics & Kinematics:** Google DeepMind MuJoCo multi-body dynamic physics simulation (`models/campus_perimeter_patrol.xml`).
* **Autonomous Control:** Closed-loop Python control architecture with student implementation boundaries (`src/aerial_patrol_swarm.py`).
* **Technoeconomic Evaluation:** Dimensionless CSBS operational economics and return on investment model (`analytics/campus_security_economics.py`).
* **Empirical Validation:** Reproducible benchmark trials and statistical hypothesis testing (`analytics/campus_patrol_real_trials.csv`).

---

> [!NOTE]
> ### The Devil's Advocate: Reality Check & Theoretical Roast
> *"Architecting a cooperative multi-agent aerial consensus protocol to detect campus perimeter breaches, completely ignoring that 15-knot coastal sea breezes, campus WiFi dead zones, and a flock of territorial pigeons will immediately turn your decentralized swarm consensus into a synchronized double-tree landing."*

---

## 2. Student Engineering Team Matrix

| Roll No | SAP ID | Student Name | Technical Role | Assigned Git Branch | Core Viva Defense Area |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `E043` | `70362400030` | **Muaaz Mohammed Iqbal Shaikh** | Lead UAV Flight Dynamics, Path Planner & Coordinated Fleet Architect | `feat/e043-lead-uav-flight-dyna` | Dual-quadrotor 6-DOF aerodynamics, thrust-to-weight modeling, waypoint coverage optimization, and inter-UAV collision avoidance. |
| `E051` | `70362400067` | **Aditya Rajkumar** | Computer Vision, OpenCV Human Detection & Tracking Specialist | `feat/e051-computer-vision-open` | Downward aerial OpenCV human detection pipeline, bounding box latency, false alarm mitigation, and target handoff tracking. |
| `E075` | `70362400003` | **Zaid Rezaur Rahman** | Restricted-Zone Geo-Fencing & Intrusion Telemetry Lead | `feat/e075-restricted-zone-geo-` | Ray-casting point-in-polygon geo-fencing, simulated GPS/IMU noise injection, and real-time perimeter breach logging. |
| `E077` | `70362400080` | **Soumya Subhankar Ranasingh** | CSBS Campus Security Operations & OpEx Payback Analyst | `feat/e077-csbs-campus-security` | Guard labor substitution modeling, patrol cycle acceleration (from 45 to 8.5 min), and multi-year drone fleet OpEx payback analysis. |


### Student Engineering Commendation & Acknowledgments
SVKM's NMIMS MPSTME conveys sincere appreciation and heartfelt gratitude to **Muaaz Mohammed Iqbal Shaikh (E043)**, **Aditya Rajkumar (E051)**, **Zaid Rezaur Rahman (E075)**, and **Soumya Subhankar Ranasingh (E077)** for their disciplined commitment, late-night debugging, and technical craftsmanship throughout Semester VI. Your rigorous work in DeepMind MuJoCo physics modeling, closed-loop telemetry instrumentation, and Computer Science & Business Systems (CSBS) technoeconomic modeling exemplifies the highest standards of undergraduate engineering inquiry.

> *"Scientists discover the world that exists; engineers create the world that never was."*  
> — **Theodore von Kármán**

> *"There is no substitute for hard work. Genius is one percent inspiration and ninety-nine percent perspiration."*  
> — **Thomas A. Edison**

---

## 3. Foundational Literature Benchmarks (Strict 2 Seminal : 4 Recent Ratio)

The research project is theoretically anchored on six foundational peer-reviewed publications validated against global bibliographic databases.

| # | Author (Year) | Type | Paper Title | Venue / Indexing | Active DOI Link | Primary Extracted Formulation | Student Lead |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Guerrero-Bonilla et al. (2021)** | Recent | *Perimeter Surveillance Based on Set-Invariance* | IEEE Robotics and Automation Letters | [https://doi.org/10.1109/LRA.2020.3028055](https://doi.org/10.1109/LRA.2020.3028055) | `Control Barrier Function $h(x) \ge 0$; barrie...` | Muaaz Mohammed Iqbal Shaikh (E043) & Zaid Rezaur Rahman (E075) |
| 2 | **Javaid et al. (2023)** | Recent | *Communication and Control in Collaborative UAVs: Recent Advances and Future Trends* | IEEE Transactions on Intelligent Transportation Systems | [https://doi.org/10.1109/TITS.2023.3248841](https://doi.org/10.1109/TITS.2023.3248841) | `Network consensus equation $\dot{x}_i = -\sum...` | Muaaz Mohammed Iqbal Shaikh (E043) |
| 3 | **Wu et al. (2024)** | Recent | *Multi-UAV Collaborative Dynamic Task Allocation Method Based on ISOM and Attention Mechanism* | IEEE Transactions on Vehicular Technology | [https://doi.org/10.1109/TVT.2023.3341878](https://doi.org/10.1109/TVT.2023.3341878) | `Task allocation utility $U = \sum_{i} (w_1 T_...` | Soumya Subhankar Ranasingh (E077) & Muaaz Shaikh (E043) |
| 4 | **Cabreira et al. (2019)** | Seminal | *Survey on Coverage Path Planning with Unmanned Aerial Vehicles* | Drones | [https://doi.org/10.3390/drones3010004](https://doi.org/10.3390/drones3010004) | `Coverage time $T_{\text{cov}} = \frac{A}{w_{\...` | Zaid Rezaur Rahman (E075) & Aditya Rajkumar (E051) |
| 5 | **Mittal et al. (2020)** | Recent | *Deep learning-based object detection in low-altitude UAV datasets: A survey* | Image and Vision Computing | [https://doi.org/10.1016/j.imavis.2020.104046](https://doi.org/10.1016/j.imavis.2020.104046) | `Mean Average Precision $\text{mAP} = \frac{1}...` | Aditya Rajkumar (E051) |
| 6 | **Agmon et al. (2008)** | Seminal | *Multi-robot perimeter patrol in adversarial settings* | IEEE International Conference on Robotics and Automation (ICRA) | [https://doi.org/10.1109/ROBOT.2008.4543563](https://doi.org/10.1109/ROBOT.2008.4543563) | `Maximum patrol time lag $T_{\text{lag}} = \ma...` | Muaaz Shaikh (E043) & Soumya Subhankar Ranasingh (E077) |

For the exhaustive literature analysis, mathematical derivations, and viva defense questions, refer to:
* [`docs/LITERATURE_REVIEW_AND_FOUNDATIONAL_PAPERS.md`](docs/LITERATURE_REVIEW_AND_FOUNDATIONAL_PAPERS.md)
* [`docs/RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md`](docs/RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md)

---

## 4. Project Demonstration & Academic Showcase (LinkedIn)

[![Watch Video Demonstration on LinkedIn](docs/figures/video_poster.png)](https://lnkd.in/p/d6ZwW_uJ)

* **Video Demonstration:** [Watch 60-Second Walkthrough on LinkedIn](https://lnkd.in/p/d6ZwW_uJ) *(Click thumbnail above to open LinkedIn post)*
* **Student Presenters:** **Muaaz Mohammed Iqbal Shaikh (E043)**, **Aditya Rajkumar (E051)**, **Zaid Rezaur Rahman (E075)**, **Soumya Subhankar Ranasingh (E077)**
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
|   `-- campus_perimeter_patrol.xml                                    <- MuJoCo MJCF simulation model
|-- src/
|   |-- test_env.py                                    <- Physics validation & environment tester
|   `-- aerial_patrol_swarm.py                                      <- Autonomous control loop (# TODO student boundaries)
`-- analytics/
    |-- .gitkeep
    |-- campus_security_economics.py                                      <- CSBS dimensionless technoeconomic model
    |-- generate_paper_figures.py                      <- Standalone 300 DPI publication figure generator
    `-- SIMULATED_campus_patrol_benchmark.csv                                    <- SIMULATED starter targets (np.random), NOT results
```

---

## 5. Pedagogical Boundaries: Guidance vs Student Ownership

To ensure academic rigor and authentic student learning, this repository enforces strict boundaries between scaffolding and student deliverables:

* **Provided by Course Scaffolding:**
  * Validated physical model architecture in MuJoCo MJCF (`models/campus_perimeter_patrol.xml`).
  * Verification toolchain and smoke test script (`src/test_env.py`).
  * Literature foundation, mathematical formulations, and conference paper blueprint (`docs/`).
  * Dimensionless technoeconomic framework (`analytics/campus_security_economics.py`).
* **Student Technical Deliverables (Required for Evaluation):**
  * Implement and tune the control algorithm in `src/aerial_patrol_swarm.py` (filling all marked `# TODO [Student Roll / Name]` blocks).
  * Run physics simulation trials to collect and expand empirical data in `analytics/campus_patrol_real_trials.csv`.
  * Re-run `analytics/generate_paper_figures.py` to regenerate publication figures with live experimental telemetry.
  * Author the final 4-page IEEE conference paper in `docs/RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md` and defend individual Git commits during the oral viva.

---

## 6. Sprint 0 Onboarding & Physics Environment Verification

Verify your local Python and MuJoCo simulation environment:

```powershell
# Step 1: Clone repository and navigate to group directory
git clone <repository_url>
cd <repository_root>/Group_05_Muaaz_Aditya_Zaid_Soumya

# Step 2: Checkout your individual feature branch
# Example for lead student:
git checkout -b feat/e043-lead-uav-flight-dyna

# Step 3: Run environment smoke test
python src/test_env.py

# Step 4: Verify 300 DPI publication figures
python analytics/generate_paper_figures.py
```
