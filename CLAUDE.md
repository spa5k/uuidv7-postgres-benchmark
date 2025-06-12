# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **professional-grade PostgreSQL UUIDv7 benchmarking suite** that provides statistically significant performance analysis of UUID generation functions across PostgreSQL 17 and 18. The project features high-precision methodology and comprehensive reporting.

## Repository Structure

```
├── professional_benchmark.py          # Main benchmark script
├── run_professional_benchmark.sh      # Automated runner
├── Makefile                           # Build targets
├── src/                               # Professional modules
│   ├── config.py                      # Centralized configuration
│   ├── database.py                    # Database management
│   ├── benchmark_engine.py            # High-precision benchmarking
│   └── data_export.py                 # Multi-format data export
├── init/                              # SQL function definitions
├── docs/                              # Documentation
├── images/                            # Performance charts and diagrams
├── results/                           # Generated benchmark outputs
└── docker-compose.yml                # PostgreSQL containers
```

## Common Commands

### Primary Commands
```bash
# Complete setup and benchmark
make setup                    # Install deps + start containers
make benchmark               # Run standard benchmark (50K iterations)

# Alternative benchmark modes
make benchmark-quick         # Quick test (5K iterations)
make benchmark-extensive     # Maximum precision (100K iterations)

# Automated runner (equivalent to make benchmark)
./run_professional_benchmark.sh

# Environment management
make up                      # Start PostgreSQL containers
make down                    # Stop containers
make status                  # Check container status
make clean                   # Clean everything
```

### Direct Python Usage
```bash
# Professional benchmark with options
python3 professional_benchmark.py                    # Standard
python3 professional_benchmark.py --quick           # Quick mode  
python3 professional_benchmark.py --extensive       # Extensive mode
python3 professional_benchmark.py --verbose         # Detailed logging
```

## Professional Methodology

This benchmark suite implements research-grade methodology:

### High-Precision Configuration
- **Warmup**: 10,000 iterations per function
- **Measurement**: 50,000 iterations for statistical significance
- **Runs**: 5 complete runs per function for consistency analysis
- **Timing**: Nanosecond precision using `time.perf_counter_ns()`
- **Workers**: 10 concurrent workers for throughput testing

### Database Configuration
- **PostgreSQL 17**: Port 5434, optimized settings
- **PostgreSQL 18**: Port 5435, native UUIDv7 + optimized settings
- **Identical tuning**: 512MB shared buffers, 2GB cache, SSD optimization

### Output Structure
```bash
results/
├── charts/           # Performance visualizations
├── reports/          # Markdown executive summaries  
├── raw_data/         # Complete benchmark data
├── exports/          # JSON/CSV for analysis
└── logs/             # Execution logs
```

## Key Functions Tested

### UUIDv7 Implementations
1. **`uuidv7_native()`** - PostgreSQL 18 native C implementation
2. **`uuid_generate_v7()`** - PL/pgSQL with overlay method
3. **`uuidv7_custom()`** - Pure SQL with bit manipulation
4. **`uuidv7_sub_ms()`** - Sub-millisecond precision variant

### Alternative Identifiers
5. **`ulid_generate()`** - ULID with Crockford Base32
6. **`typeid_generate_text()`** - Type-safe prefixed identifiers

### Baseline
7. **`gen_random_uuid()`** - PostgreSQL native UUIDv4

## Expected Performance Results

Based on comprehensive testing:

- **PostgreSQL 18 native UUIDv7**: 58.1μs avg (33% faster than UUIDv4)
- **Custom UUIDv7 (PG17)**: 87.3μs avg
- **UUIDv4 baseline**: 86.8μs avg
- **ULID**: 124.5μs avg
- **TypeID**: 198.7μs avg

Throughput ranges from 18K-34K operations/second across implementations.

## Development Notes

### Database Architecture
- Uses psycopg3 for PostgreSQL connectivity
- Implements connection pooling and health checks
- Applies benchmark-specific PostgreSQL optimizations
- Validates function correctness before benchmarking

### Statistical Analysis
- Multiple runs ensure consistency measurement
- Calculates mean, median, standard deviation, P95, P99
- Coefficient of variation for run-to-run consistency
- Outlier detection using 3-sigma threshold

### Data Export
- Structured JSON for programmatic analysis
- CSV format for spreadsheet integration
- Chart data optimized for web visualization
- Markdown reports with executive summaries

## File Organization

- **`images/`** - Contains performance charts and architectural diagrams
- **`docs/`** - Additional documentation and specifications
- **`init/`** - SQL scripts for function creation across PostgreSQL versions
- **`results/`** - Generated benchmark outputs (created during runs)
- **`src/`** - Modular Python codebase for professional benchmarking