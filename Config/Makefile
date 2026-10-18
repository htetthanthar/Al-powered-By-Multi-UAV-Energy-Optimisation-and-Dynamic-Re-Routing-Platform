.PHONY: help install data train test benchmark serve dashboard sim clean docker-up docker-down

PYTHON = python
PIP = pip

help:
	@echo "=========================================================================="
	@echo " UAV DYNAMIC ROUTE OPTIMIZATION & SCHEDULING ENGINE - MAKEFILE"
	@echo "=========================================================================="
	@echo "  make install     : Install all Python dependencies"
	@echo "  make data        : Generate synthetic telemetry & weather datasets"
	@echo "  make train       : Train Weather LSTM, Energy MLP, and RL Policy Network"
	@echo "  make test        : Run pytest suite on all operational constraints"
	@echo "  make benchmark   : Run 100-trial statistical ablation study"
	@echo "  make figures     : Export publication-grade evaluation charts"
	@echo "  make sim         : Run the multi-UAV virtual airspace simulator"
	@echo "  make serve       : Start the FastAPI real-time telemetry backend"
	@echo "  make dashboard   : Launch the Streamlit operator mission control UI"
	@echo "  make run-all     : Execute the entire data -> train -> test -> benchmark flow"
	@echo "  make docker-up   : Build and start multi-container services via Docker"
	@echo "  make clean       : Remove temporary cache, checkpoints, and bytecode"
	@echo "=========================================================================="

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) generate_datasets.py

train:
	$(PYTHON) train_weather.py
	$(PYTHON) train_energy.py
	$(PYTHON) train_rl_router.py

test:
	pytest test_pipeline.py -v

benchmark:
	$(PYTHON) ablation_benchmark.py

figures:
	$(PYTHON) generate_evaluation_figures.py

sim:
	$(PYTHON) simulate_fleet_mission.py

serve:
	$(PYTHON) server.py

dashboard:
	streamlit run dashboard.py

run-all: data train test benchmark figures sim

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf data/*.duckdb
	@echo "[+] Cleanup complete."