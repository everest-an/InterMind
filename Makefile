.PHONY: install serve test lint typecheck clean

install:
	pip install -e ".[dev]"

serve:
	uvicorn latentcore.app:create_app --factory --reload --host 0.0.0.0 --port 8000

test:
	pytest --cov=latentcore --cov-report=term-missing -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/latentcore/

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
