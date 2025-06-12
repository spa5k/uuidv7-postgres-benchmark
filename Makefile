.PHONY: help up down benchmark clean install

help:
	@echo "Available commands:"
	@echo "  make install   - Install Python dependencies"
	@echo "  make up        - Start PostgreSQL containers"
	@echo "  make down      - Stop PostgreSQL containers"
	@echo "  make benchmark - Run the benchmark"
	@echo "  make clean     - Clean up containers and volumes"

install:
	pip install -r requirements.txt

up:
	docker-compose up -d
	@echo "Waiting for databases to start..."
	@sleep 10

down:
	docker-compose down

benchmark: up
	python benchmark.py

clean:
	docker-compose down -v
	rm -f benchmark_results.json benchmark_results.png detailed_analysis.png BENCHMARK_REPORT.md