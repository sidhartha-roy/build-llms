.PHONY: install install-dev install-pytorch clean test format lint help

help:
	@echo "Available targets:"
	@echo "  install         - Install project dependencies"
	@echo "  install-dev     - Install project with dev dependencies"
	@echo "  install-pytorch - Install PyTorch nightly (CUDA 13.0)"
	@echo "  clean          - Remove build artifacts and cache"
	@echo "  test           - Run tests with pytest"
	@echo "  format         - Format code with black"
	@echo "  lint           - Lint code with ruff"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-pytorch:
	pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

install-all: install-pytorch install-dev

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	pytest

format:
	black .

lint:
	ruff check .
