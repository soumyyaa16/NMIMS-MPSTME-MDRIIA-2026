# Convenience targets. On Windows use the equivalent commands from the README (PowerShell).
PY ?= python

.PHONY: install test lint validate kerbs codesign benchmark-smoke benchmark train-ppo demo report capstone

install:
	$(PY) -m pip install -e ".[dev,viz,rl]"

test:
	$(PY) -m pytest -m "not slow" -n auto

lint:
	$(PY) -m ruff check src tests scripts experiments

validate:            # analytic checks of the MuJoCo model (suspension, rolling, stability)
	$(PY) experiments/01_validate_suspension.py

kerbs:               # kerb traversability map (RQ1)
	$(PY) experiments/02_curb_traversability.py --preset standard --design optimized

codesign:            # mechanical co-design search (RQ1), about 10 minutes on 28 cores
	$(PY) experiments/06_mech_codesign.py

benchmark-smoke:     # a few minutes
	$(PY) experiments/03_controller_benchmark.py --n 10 --tag smoke

benchmark:           # the full study (see docs/REPRODUCE.md for the exact commands and run times)
	$(PY) experiments/03_controller_benchmark.py --n 100 --tag standard --controllers pure_pursuit apf dwa mppi ppo

train-ppo:
	$(PY) -m aedrover.learning.train_ppo --steps 12000000 --n-envs 24 --out checkpoints/ppo_shielded

demo:
	$(PY) scripts/render_demo.py --controller mppi --family mixed --seed 1002 --name mixed_mppi

report:
	$(PY) scripts/make_report.py

capstone:            # course capstone evidence: 750 episodes at 500 Hz, business model, standards, generated docs
	$(PY) experiments/09_capstone_monte_carlo.py --n 50
	$(PY) analytics/aed_delivery_roi.py
	$(PY) analytics/standards_check.py
	$(PY) scripts/make_capstone_doc.py

