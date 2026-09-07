.PHONY: all install data check-data ingest views test extracts clean distclean

VENV   := .venv
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

# Full pipeline: environment -> data -> database -> contract tests.
all: ingest

# --- environment -----------------------------------------------------------
install: $(VENV)/.installed
$(VENV)/.installed: pyproject.toml
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"
	touch $@

# --- data ------------------------------------------------------------------
# Fetches the eight source tables from the commit pinned in src/schema.py.
data: install
	$(PYTHON) -m src.fetch_data

# Verifies the sources are present and correctly sized without downloading.
# Named separately so CI can assert the precondition and fail with a useful
# message instead of a stack trace from the loader.
check-data: install
	$(PYTHON) -m src.fetch_data --check

# --- pipeline --------------------------------------------------------------
# `ingest` depends on check-data, and runs the contract suite and view
# creation immediately afterwards. Ordering is guaranteed via recursive $(MAKE):
# a database that fails its contract must never be treated as done, and views
# must not be created against a database that hasn't passed the contract.
ingest: check-data
	$(PYTHON) -m src.ingest
	@echo
	@$(MAKE) --no-print-directory test
	@echo
	@$(MAKE) --no-print-directory views

# Creates the five cleaning-layer views defined in sql/01_cleaning_views.sql.
# Run independently with `make views` if you edit the SQL without touching the
# raw tables (views are cheap to recreate; no re-ingestion needed).
views: install
	$(PYTHON) -m src.create_views

test: install
	$(PYTEST) tests/

extracts: install
	$(PYTHON) -m src.export_extracts

# --- cleaning --------------------------------------------------------------
clean:
	rm -f data/processed/strata.duckdb data/processed/strata.duckdb.wal

distclean: clean
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
