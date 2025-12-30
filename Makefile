# =============================================================================
# RAG Data Assistant - Makefile
# Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)
# =============================================================================

.PHONY: help setup install test lint run demo docker-up docker-down clean

# Default target
help:
	@echo "RAG Data Assistant - Available Commands:"
	@echo ""
	@echo "  make setup      - Set up development environment"
	@echo "  make install    - Install Python dependencies"
	@echo "  make test       - Run all tests"
	@echo "  make lint       - Run linters"
	@echo "  make run        - Run API server locally"
	@echo "  make demo       - Start demo with Docker Compose"
	@echo "  make docker-up  - Start all Docker services"
	@echo "  make docker-down - Stop all Docker services"
	@echo "  make clean      - Clean up generated files"
	@echo ""

# Setup development environment
setup:
	@echo "Setting up development environment..."
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	cp .env.example .env
	@echo "Setup complete! Edit .env with your API keys."

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Run unit tests only
test-unit:
	pytest tests/unit/ -v

# Run integration tests
test-integration:
	pytest tests/integration/ -v

# Run linters
lint:
	black src/ tests/ --check
	isort src/ tests/ --check
	flake8 src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Run API server locally
run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

# Start demo with Docker
demo:
	docker-compose up -d
	@echo ""
	@echo "Services started! Access:"
	@echo "  - API:        http://localhost:8080"
	@echo "  - API Docs:   http://localhost:8080/docs"
	@echo "  - UI:         http://localhost:8501"
	@echo "  - Grafana:    http://localhost:3000"
	@echo ""

# Start Docker services
docker-up:
	docker-compose up -d

# Stop Docker services
docker-down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Rebuild Docker images
docker-rebuild:
	docker-compose build --no-cache

# Ingest sample documents
ingest-samples:
	python scripts/ingest_sample_docs.py

# Generate sample data
generate-data:
	python scripts/generate_sample_data.py

# Run benchmarks
benchmark:
	python scripts/benchmark_performance.py

# Evaluate RAG quality
evaluate:
	python scripts/evaluate_rag_quality.py

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf chroma_data/ 2>/dev/null || true
	rm -rf logs/ 2>/dev/null || true
	@echo "Cleaned up generated files."

# Deep clean (including Docker volumes)
clean-all: clean
	docker-compose down -v
	@echo "All data volumes removed."
