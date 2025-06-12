# UUIDv7 PostgreSQL Benchmark

A comprehensive benchmark comparing three different UUIDv7 implementations in PostgreSQL across versions 17 and 18 beta.

## Overview

This project benchmarks three UUIDv7 generating functions:

1. **uuid_generate_v7()** - PL/pgSQL implementation using overlay and bit manipulation
2. **uuidv7()** - SQL-based implementation with similar approach
3. **uuidv7_sub_ms()** - Sub-millisecond precision version with fractional timestamp

## Features Tested

- **Single-threaded performance** - Average generation time for individual UUIDs
- **Concurrent performance** - Throughput under multi-threaded load
- **Uniqueness** - Collision testing with 1 million UUIDs
- **Time ordering** - Verifies UUIDs maintain chronological order
- **Performance stability** - Consistency of generation times

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/spa5k/uuidv7-postgres-benchmark.git
cd uuidv7-postgres-benchmark
```

2. Start PostgreSQL containers:

```bash
docker-compose up -d
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Run the benchmark:

```bash
python benchmark.py
```

## Results

The benchmark generates:

- `benchmark_results.json` - Raw benchmark data
- `benchmark_results.png` - Performance comparison charts
- `detailed_analysis.png` - Detailed performance analysis
- `BENCHMARK_REPORT.md` - Markdown formatted report

## Docker Setup

The project uses Docker Compose to run:

- PostgreSQL 17 (port 5432)
- PostgreSQL 18 Beta 1 (port 5433)

Both instances are configured with:

- Database: `uuidv7_benchmark`
- User: `postgres`
- Password: `postgres`

## Function Implementations

### Function 1: uuid_generate_v7()

- Uses PL/pgSQL
- Overlays timestamp on random UUID
- Sets version bits for UUIDv7

### Function 2: uuidv7()

- Pure SQL implementation
- Similar approach to Function 1
- Slightly different syntax

### Function 3: uuidv7_sub_ms()

- Includes sub-millisecond precision
- Uses fractional part of timestamp
- Method 3 from UUIDv7 specification

## Benchmark Metrics

1. **Generation Time**

   - Average, median, min, max
   - 95th and 99th percentiles
   - Standard deviation

2. **Throughput**

   - UUIDs generated per second
   - Multi-threaded performance

3. **Correctness**
   - Collision rate
   - Time ordering accuracy

## License

MIT
