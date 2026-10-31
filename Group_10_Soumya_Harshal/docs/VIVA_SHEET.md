# Viva sheet (Group 10)

**One line:** a joint controller + tremor-filter design study of a 7-DOF arm in MuJoCo, plus a capacity-based business model. Simulation only, not clinical.

1. **Why is tremor only about 4% of the error budget?** Whole-system DBS radial error is about 1.06 mm (Giridharan 2022). Our simulated tremor hold error is 0.041-0.045 mm, so 3.9-4.2%. The rest is imaging, registration, brain shift and other sources we do not model.
2. **What does the learned filter actually buy?** At KP=50 (tuned) about 18% lower hold error on enhanced tremor (typical hand 10-14%); at KP=100 about 38-40%. In absolute terms 0.005-0.008 mm, 0.5-0.8% of the budget. We make no outcome or cost claim.
3. **Why did the benefit shrink from 38% to 18%?** A high-gain loop passes more tremor, so the filter has more to remove. Controller bandwidth does most of the rejection. We report both and say so; this is why the paper is a joint-design study.
4. **Does RL beat Kalman?** On hold error, yes in every tier with paired CIs. On tracking error it is worse in the tremor tiers. Baselines were tuned on held-out hands but several sit at their grid edge, so they may be under-tuned.
5. **What is the real floor?** Slow drift: about 0.28 mm hold error at measured-scale drift, and no tremor filter removes it. It needs imaging feedback.
6. **Where does 91 min come from?** Owens 2026 medians: 220 min (frame + C-arm) minus 129 min (robot + intra-op CT). It is a robot+CT workflow comparison, not our arm.
7. **Does the robot save money?** No. Robot case cost is about 10% higher (capital excluded, not significant, p = 0.374). The value is capacity: 32.4 OR-hours per year at 21.3 cases per year.
8. **Does it pay back?** P = 2.52 at today's volume. Break-even is about 54 cases per year at margin 0.5, or margin 0.89 at today's volume. The extra-case margin is an assumption, so we sweep it.
9. **Weakest numbers in the model?** Robot price (reseller blog), the CapEx ratio built from Childers' cost per minute, margin, and device life. All are labelled and swept.
10. **Feedback delay?** At KP=100 the loop is stable up to about 2 ms and diverges from 6 ms; at KP=50 it holds to about 12 ms (single-seed probe, to confirm). Real latency must be measured before any hardware claim.
11. **Is the tremor real?** No. It is a model (3 sinusoids at 7-12 Hz plus slow drift) injected on the reference. No recorded tremor data and no hardware, so no clinical claim.
12. **Is this the first RL tremor work?** No. Noorani & Naiem 2024 is prior art; we claim a joint controller-and-filter design study in simulation.
13. **What would a journal need?** Recorded tremor data, hardware latency, and registration error. Workshop or conference level is the honest target.
