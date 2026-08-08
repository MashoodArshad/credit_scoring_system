# ============================================================
# Credit Scoring System — developer convenience commands
# Usage: make <target>
# ============================================================
.PHONY: install dev lint format typecheck test clean

install:        ## Install core runtime dependencies
	pip install -r requirements.txt

dev:            ## Install core + development dependencies
	pip install -r requirements-dev.txt

lint:           ## Lint src/ and tests/ with ruff
	ruff check src tests

format:         ## Auto-format src/ and tests/ with ruff
	ruff format src tests

typecheck:      ## Static type-check src/ with mypy
	mypy src

test:           ## Run the pytest suite
	pytest

clean:          ## Remove caches and temp build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
