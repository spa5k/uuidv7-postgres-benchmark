# Benchmark System Specifications

## Reproducible Environment Setup

This document defines the exact system specifications used for UUIDv7 PostgreSQL benchmarking to ensure reproducible results.

### Container Resource Allocation

Each PostgreSQL container is allocated identical resources:

| Resource | Allocation | Justification |
|----------|------------|---------------|
| **Memory Limit** | 2GB | Sufficient for PostgreSQL + buffers without host memory pressure |
| **Memory Reservation** | 1GB | Guaranteed minimum memory allocation |
| **CPU Limit** | 2.0 cores | Allows PostgreSQL to utilize multiple cores for parallel operations |
| **CPU Reservation** | 1.0 core | Guaranteed minimum CPU allocation |

### PostgreSQL Configuration

Identical configuration applied to both PostgreSQL 17 and 18 beta containers:

| Setting | Value | Purpose |
|---------|-------|---------|
| `shared_buffers` | 256MB | 25% of container memory for data caching |
| `effective_cache_size` | 1GB | 50% of container memory for query planner |
| `work_mem` | 4MB | Per-operation memory for sorts/joins |
| `maintenance_work_mem` | 64MB | Memory for maintenance operations |
| `wal_buffers` | 16MB | Write-ahead log buffer size |
| `max_wal_size` | 4GB | Maximum WAL size before checkpoint |
| `min_wal_size` | 1GB | Minimum WAL size to keep |
| `max_connections` | 200 | Maximum concurrent connections |
| `random_page_cost` | 1.1 | SSD-optimized random I/O cost |
| `effective_io_concurrency` | 200 | Expected concurrent I/O operations |
| `checkpoint_completion_target` | 0.9 | Spread checkpoints over 90% of interval |

### Database Environment

| Component | Configuration |
|-----------|---------------|
| **PostgreSQL 17** | Official postgres:17 Docker image |
| **PostgreSQL 18** | Official postgres:18beta1 Docker image |
| **Database Name** | `benchmark` |
| **User** | `postgres` |
| **Port Mapping** | PG17: 5434, PG18: 5435 |
| **Data Checksums** | Enabled via `--data-checksums` |
| **Logging** | Disabled for performance (`log_statement=none`) |

### Benchmark Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Single-threaded Iterations** | 5,000 | Individual function calls for latency measurement |
| **Concurrent Workers** | 5 | Parallel processes for throughput testing |
| **Concurrent Iterations per Worker** | 500 | Operations per worker (2,500 total) |
| **Warmup Iterations** | 10 | Initial calls to prime caches |
| **Collision Testing** | 50,000 | IDs generated to verify uniqueness |

### Test Environment Requirements

#### Host System Minimum Requirements
- **Docker Engine**: 20.10+
- **Available Memory**: 6GB+ (4GB for containers + 2GB host overhead)
- **Available CPU**: 4+ cores recommended
- **Disk Space**: 10GB+ for Docker images and data
- **Network**: Docker bridge networking

#### Python Dependencies
```bash
pip install psycopg[binary] matplotlib seaborn pandas
```

### Functions Under Test

| Implementation | Function Name | PostgreSQL Versions | Description |
|----------------|---------------|-------------------|-------------|
| **Native UUIDv7** | `uuidv7_native()` | 18 only | PostgreSQL 18 native C implementation |
| **PL/pgSQL UUIDv7** | `uuid_generate_v7()` | 17, 18 | Overlay method implementation |
| **Pure SQL UUIDv7** | `uuidv7_custom()` | 17, 18 | Pure SQL bit manipulation |
| **Sub-ms UUIDv7** | `uuidv7_sub_ms()` | 17, 18 | Sub-millisecond precision variant |
| **UUIDv4 Baseline** | `gen_random_uuid()` | 17, 18 | PostgreSQL native random UUID |
| **ULID** | `ulid_generate()` | 17, 18 | Base32-encoded time-ordered ID |
| **TypeID** | `typeid_generate_text()` | 17, 18 | Type-safe prefixed identifiers |

### Performance Metrics Collected

| Metric | Unit | Description |
|--------|------|-------------|
| **Average Latency** | microseconds (μs) | Mean function execution time |
| **Median Latency** | microseconds (μs) | 50th percentile execution time |
| **P95 Latency** | microseconds (μs) | 95th percentile execution time |
| **P99 Latency** | microseconds (μs) | 99th percentile execution time |
| **Concurrent Throughput** | IDs/second | Operations per second with 5 workers |
| **Storage Size** | bytes | Text representation size |
| **Collision Rate** | percentage | Duplicate IDs in 50,000 generation test |
| **Time Ordering Accuracy** | percentage | Correct chronological ordering |

### Reproducibility Notes

1. **Container Startup Order**: PostgreSQL 17 and 18 start simultaneously
2. **Initialization Wait**: 10-second delay after `pg_isready` confirms readiness
3. **Function Warmup**: 10 iterations before timing measurements
4. **Measurement Precision**: `time.perf_counter()` for microsecond accuracy
5. **Random Seed**: Not fixed - represents real-world usage
6. **System Load**: Benchmarks should run on idle systems for consistency

### Running the Benchmark

```bash
# Clone repository
git clone <repository-url>
cd uuidv7-postgres-benchmark

# Ensure Docker is running
docker --version

# Install Python dependencies
pip install psycopg[binary] matplotlib seaborn pandas

# Run complete benchmark suite
./run_benchmark.sh
```

### Expected Results Format

The benchmark generates several output files:

1. **enhanced_benchmark_results.json** - Complete raw results
2. **benchmark_data/performance_summary.json** - Key metrics summary
3. **benchmark_data/chart_data.json** - Data formatted for charts
4. **enhanced_postgresql_benchmark.png** - Performance visualizations

### Hardware Baseline

These specifications were tested on:
- **CPU**: Modern x86_64 processor (2+ GHz recommended)
- **RAM**: 16GB+ host memory
- **Storage**: SSD recommended for consistent I/O performance
- **OS**: macOS/Linux with Docker Desktop or Docker CE

Results may vary on different hardware configurations, but relative performance between implementations should remain consistent.