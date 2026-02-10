# Use: make run | make test | make lint-format  (do NOT run ./Makefile as a script)
# Lint and format with Ruff (use uv so Ruff from pyproject is used)
.PHONY: lint format lint-format run test
lint:
	uv run ruff check --fix .
format:
	uv run ruff format .
lint-format: lint format

# Run full pipeline (data → nested CV → save models → permutation test)
run:
	uv run python run_pipeline.py

# Run tests (imports + optional model load if models/ exists)
test:
	uv run pytest tests/ -v
