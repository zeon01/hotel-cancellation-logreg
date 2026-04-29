.PHONY: help setup data features train tune evaluate report lint test clean all

help:
	@echo "Targets:"
	@echo "  setup     - uv sync + pre-commit install"
	@echo "  data      - download raw + run preprocessing"
	@echo "  features  - feature engineering"
	@echo "  train     - fit primary model"
	@echo "  tune      - hyperparameter search (CV over C)"
	@echo "  evaluate  - metrics + calibration + diagnostics"
	@echo "  report    - render figures into reports/figures"
	@echo "  lint      - ruff check + ruff format --check"
	@echo "  test      - pytest -q"
	@echo "  clean     - remove interim/processed/figures"
	@echo "  all       - data -> features -> train -> evaluate -> report"

setup:
	uv sync
	uv run pre-commit install

data:
	uv run python scripts/download_data.py
	uv run python -c "from cancellation_logreg.data import build_clean_frame; from cancellation_logreg.preprocess import write_processed; write_processed()"

features:
	uv run python -c "from cancellation_logreg.features import build_feature_frame; build_feature_frame()"

train:
	uv run python -m cancellation_logreg.modeling.train

tune:
	uv run python -m cancellation_logreg.modeling.tune

evaluate:
	uv run python -m cancellation_logreg.modeling.evaluate

report:
	uv run python scripts/make_dataset.py --report

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

clean:
	rm -rf data/interim/* data/processed/* reports/figures/* docs/figures/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +

all: data features train evaluate report
