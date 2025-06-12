.PHONY: help install setup up down clean benchmark benchmark-quick benchmark-extensive test status logs

# Professional PostgreSQL UUIDv7 Benchmark Suite Makefile
# Provides convenient commands for running comprehensive benchmarks

help:
	@echo "Professional PostgreSQL UUIDv7 Benchmark Suite"
	@echo "================================================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install          - Install Python dependencies and setup environment"
	@echo "  make setup            - Complete environment setup (install + containers)"
	@echo ""
	@echo "Container Management:"
	@echo "  make up               - Start PostgreSQL containers"
	@echo "  make down             - Stop PostgreSQL containers"
	@echo "  make status           - Show container status"
	@echo ""
	@echo "Benchmark Commands:"
	@echo "  make benchmark        - Run standard benchmark (50K iterations, 5 runs)"
	@echo "  make benchmark-quick  - Run quick benchmark (5K iterations, 3 runs)"
	@echo "  make benchmark-extensive - Run extensive benchmark (100K iterations, 10 runs)"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make test             - Test database connections and functions"
	@echo "  make logs             - Show PostgreSQL container logs"
	@echo "  make clean            - Clean up containers, volumes, and results"
	@echo ""
	@echo "Professional Features:"
	@echo "  • High-precision timing (nanosecond accuracy)"
	@echo "  • Statistical significance testing"
	@echo "  • Comprehensive reporting and data export"
	@echo "  • PostgreSQL 17 vs 18 comparison"

install:
	@echo "Installing Python dependencies..."
	python3 -m venv venv || true
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "✅ Python environment ready"

setup: install up
	@echo "Complete environment setup finished"
	@echo "Ready to run benchmarks!"

up:
	@echo "Starting PostgreSQL containers..."
	docker-compose up -d
	@echo "Waiting for databases to initialize..."
	@sleep 15
	@echo "✅ PostgreSQL containers ready"

down:
	@echo "Stopping PostgreSQL containers..."
	docker-compose down

status:
	@echo "Container Status:"
	@docker-compose ps
	@echo ""
	@echo "PostgreSQL 17 Status:"
	@docker exec postgres17-uuidv7 pg_isready -U postgres 2>/dev/null && echo "✅ Ready" || echo "❌ Not Ready"
	@echo "PostgreSQL 18 Status:"
	@docker exec postgres18-uuidv7 pg_isready -U postgres 2>/dev/null && echo "✅ Ready" || echo "❌ Not Ready"

test:
	@echo "Testing database connections..."
	@./run_professional_benchmark.sh quick 2>/dev/null | grep -E "(✅|❌|Testing)" || echo "Run 'make up' first"

benchmark:
	@echo "Running standard professional benchmark..."
	@echo "This will take 15-30 minutes for high-precision results"
	./run_professional_benchmark.sh standard

benchmark-quick:
	@echo "Running quick benchmark for testing..."
	@echo "This will take 3-5 minutes"
	./run_professional_benchmark.sh quick

benchmark-extensive:
	@echo "Running extensive benchmark for maximum precision..."
	@echo "This will take 45-90 minutes for highest accuracy"
	./run_professional_benchmark.sh extensive

logs:
	@echo "PostgreSQL 17 Logs:"
	@echo "==================="
	@docker-compose logs postgres17 | tail -20
	@echo ""
	@echo "PostgreSQL 18 Logs:"
	@echo "==================="
	@docker-compose logs postgres18 | tail -20

clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose down -v
	@echo "Cleaning up result files..."
	rm -rf results/
	rm -f *.json *.png *.md
	@echo "✅ Cleanup completed"

# Advanced targets for development
.PHONY: dev-install dev-test dev-profile

dev-install: install
	./venv/bin/pip install pytest pytest-benchmark black flake8 mypy

dev-test:
	@echo "Running development tests..."
	./venv/bin/python -m pytest tests/ -v

dev-profile:
	@echo "Running benchmark with profiling..."
	./venv/bin/python -m memory_profiler professional_benchmark.py --quick