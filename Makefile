.PHONY: help venv install run test artifacts evaluate docker-build docker-run compose-build compose-up compose-down compose-logs ci

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
PYTEST := $(VENV)/bin/pytest

IMAGE_NAME ?= recsys-streamlit
PORT ?= 8501

help:
	@printf "Доступные команды:\n"
	@printf "  make venv           - создать виртуальное окружение\n"
	@printf "  make install        - установить зависимости в .venv\n"
	@printf "  make run            - запустить Streamlit локально\n"
	@printf "  make test           - запустить pytest\n"
	@printf "  make artifacts      - собрать TF-IDF артефакты\n"
	@printf "  make evaluate       - пересчитать offline-метрики\n"
	@printf "  make docker-build   - собрать Docker-образ\n"
	@printf "  make docker-run     - запустить Docker-образ локально\n"
	@printf "  make compose-build  - собрать сервис через docker compose\n"
	@printf "  make compose-up     - поднять сервис через docker compose\n"
	@printf "  make compose-down   - остановить docker compose\n"
	@printf "  make compose-logs   - смотреть логи docker compose\n"
	@printf "  make ci             - локальный прогон compileall + pytest\n"

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(STREAMLIT) run app/Home.py

test:
	$(PYTEST)

artifacts:
	$(PYTHON) scripts/build_artifacts.py

evaluate:
	$(PYTHON) scripts/evaluate.py

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p $(PORT):8501 $(IMAGE_NAME)

compose-build:
	docker compose build

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f app

ci:
	python3 -m compileall app src tests scripts
	$(PYTEST)
