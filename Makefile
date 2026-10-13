SHELL := /bin/bash

SIM_COMPOSE := docker compose -f docker-compose.sim.yml

.PHONY: sim-up sim-down sim-logs sim-shell sim-start-xarm7 sim-ps setup venv install

setup: venv install

venv:
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip

install:
	. .venv/bin/activate && pip install -r requirements.txt

sim-up:
	$(SIM_COMPOSE) up -d uf_sim

sim-down:
	$(SIM_COMPOSE) down

sim-logs:
	$(SIM_COMPOSE) logs -f uf_sim

sim-shell:
	docker exec -it uf_software /bin/bash

sim-start-xarm7:
	docker exec -it uf_software /xarm_scripts/xarm_start.sh 7 7

sim-ps:
	docker ps --filter name=uf_software
