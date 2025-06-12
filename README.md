# PostgreSQL UUIDv7 Benchmark

Comprehensive benchmark suite for UUIDv7, ULID, and TypeID implementations in PostgreSQL, including PostgreSQL 18's native UUIDv7 support.

## Overview

This project provides comprehensive benchmarking for time-ordered identifier generation in PostgreSQL, featuring:

- **PostgreSQL 18 Native UUIDv7**: Benchmarks the new native uuidv7() function with C-level performance
- **Custom UUIDv7 implementations**: PL/pgSQL, Pure SQL, and Sub-millisecond precision variants
- **Alternative ID types**: ULID and TypeID for comprehensive comparison
- **Real-world testing**: Actual database operations across PostgreSQL 17 and 18

## Features

- **PostgreSQL 18 Native UUIDv7**: Benchmarks the new native uuidv7() function with C-level performance
- **Multiple UUIDv7 implementations**: PL/pgSQL, Pure SQL, and Sub-millisecond precision variants
- **Alternative ID types**: ULID and TypeID for comparison
- **Comprehensive metrics**: Single-threaded performance, concurrent throughput, collision detection
- **Time ordering analysis**: Accuracy of chronological sorting with monotonicity guarantees
- **Visual reporting**: Enhanced charts showing PostgreSQL 18 native performance
- **Version comparison**: Side-by-side PostgreSQL 17 vs 18 benchmarks
- **Docker support**: Easy setup with PostgreSQL 17 and 18 beta
- **Real benchmarks**: Uses actual database operations, not synthetic tests

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone https://github.com/spa5k/uuidv7-postgres-benchmark.git
   cd uuidv7-postgres-benchmark
   ```

2. **Start PostgreSQL containers**:
   ```bash
   docker-compose up -d
   ```

3. **Install Python dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run enhanced benchmarks** (includes PostgreSQL 18 native UUIDv7):
   ```bash
   python benchmark_pg18_native.py
   ```

## Benchmark Data and Visualizations

The benchmark suite automatically generates structured data files for integration with external applications:

### Generated Data Files

After running the benchmark, the following JSON files are created in the `benchmark_data/` directory:

- **`chart_data.json`**: Optimized data for chart libraries (Recharts, Chart.js, etc.)
  - Bar chart data: Performance comparison across implementations
  - Area chart data: Throughput analysis with concurrent vs single-threaded
  - Radar chart data: Normalized performance scores

- **`performance_summary.json`**: Key metrics summary
  - Average, median, P95, and P99 response times
  - Throughput measurements (IDs/second)
  - Storage requirements by implementation
  - PostgreSQL version comparison

- **`detailed_results.json`**: Complete benchmark results
  - Raw timing data for all iterations
  - Statistical analysis details
  - Version-specific performance breakdowns

### Visualization Assets

The benchmark also generates visual reports:

- **`enhanced_postgresql_benchmark.png`**: Comprehensive performance charts
- **Performance comparison tables**: Markdown-formatted results for documentation

### Usage in External Projects

The JSON data files are designed for easy integration:

```javascript
// Example: Using chart_data.json in a React component
import chartData from './benchmark_data/chart_data.json';

const PerformanceChart = () => {
  const data = chartData.bar_chart;
  // Use with Recharts, Chart.js, etc.
};
```

```python
# Example: Python analysis
import json
with open('benchmark_data/performance_summary.json') as f:
    data = json.load(f)
    performance = data['performance_summary']
```

5. **Or run standard benchmarks**:
   ```bash
   python benchmark_extended.py
   ```

## Benchmark Scripts

- `benchmark_pg18_native.py`: **Enhanced benchmark with PostgreSQL 18 native UUIDv7 support**
- `benchmark_extended.py`: Main benchmark script with all ID types
- `benchmark_simplified.py`: Lightweight version for quick testing
- `benchmark.py`: Original UUIDv7-only benchmark

## Docker Setup

The project uses Docker Compose to run:

- **PostgreSQL 17** (port 5434) - Stable release
- **PostgreSQL 18 Beta** (port 5435) - With native UUIDv7 support

Both instances are configured with:
- Database: `benchmark`
- User: `postgres` 
- Password: `postgres`

## Expected Results

Based on comprehensive testing with 5,000 iterations:

### PostgreSQL 18 Enhanced Results

#### Single-threaded Performance

| Implementation           | Avg Time (μs) | Performance vs UUIDv4 |
| ------------------------ | ------------- | --------------------- |
| **Native uuidv7() (PG18)** | **58.1**      | **33% faster** 🚀     |
| **UUIDv7 (PL/pgSQL)**    | 72.3          | 16% faster            |
| **ULID**                 | 79.9          | 7% faster             |
| **UUIDv7 (Pure SQL)**    | 82.3          | 5% faster             |
| **UUIDv4 (Baseline)**    | 86.3          | Baseline              |
| **TypeID**               | 86.5          | 0% (equivalent)       |
| **UUIDv7 (Sub-ms)**      | 90.6          | 5% slower             |

#### Concurrent Throughput (5 Workers)

| Implementation            | IDs/second | Performance vs UUIDv4 |
| ------------------------- | ---------- | --------------------- |
| **Native uuidv7() (PG18)** | **34,127**  | **16% higher** 🚀      |
| **UUIDv4 (Baseline)**     | 29,492     | Baseline              |
| **ULID**                  | 26,523     | 10% lower             |
| **TypeID**                | 25,775     | 13% lower             |
| **UUIDv7 (Pure SQL)**     | 25,085     | 15% lower             |
| **UUIDv7 (Sub-ms)**       | 21,658     | 27% lower             |
| **UUIDv7 (PL/pgSQL)**     | 18,126     | 39% lower             |

### Key Findings

🎯 **PostgreSQL 18's native uuidv7() breaks the performance trade-off** - it delivers both time ordering AND superior performance compared to UUIDv4.

📊 **Performance Leadership**: Native implementation outperforms all alternatives in both single-threaded and concurrent scenarios.

⚡ **C-level Performance**: The native implementation leverages PostgreSQL's core C code for maximum efficiency.

🔄 **Monotonicity Guarantee**: PostgreSQL 18 ensures ordering within database sessions using 12-bit sub-millisecond precision.

## PostgreSQL 18 Native UUIDv7 Features

PostgreSQL 18 introduces comprehensive native UUIDv7 support with RFC 9562 compliance:

### Core Functions
- **`uuidv7()`**: Generate UUIDv7 with current timestamp
- **`uuidv7(timestamp)`**: Generate UUIDv7 for specific time
- **`uuid_extract_timestamp(uuid)`**: Extract timestamp from any UUIDv7
- **`uuid_extract_version(uuid)`**: Get UUID version number
- **`uuidv4()`**: Alias for `gen_random_uuid()`

### Technical Advantages
- **C-level implementation**: Maximum performance through PostgreSQL core code
- **12-bit sub-millisecond precision**: Uses rand_a field for timestamp fraction
- **Monotonicity guarantee**: Ensures ordering within same database session
- **RFC 9562 compliance**: Follows latest UUID standard (published May 2024)
- **Method 3 implementation**: "Replace LeftmostRandom Bits with Increased Clock Precision"

### Performance Benefits
- **33% faster** than UUIDv4 in single-threaded scenarios
- **16% higher throughput** than UUIDv4 in concurrent workloads
- **Best time ordering accuracy** (99.97%) with guaranteed monotonicity
- **Zero performance trade-offs** - time ordering AND superior speed

## Understanding the Results

### PostgreSQL 18 Native UUIDv7 Performance

1. **C-level implementation**: Direct integration with PostgreSQL core eliminates function call overhead
2. **Optimized timestamp handling**: Native code handles time extraction more efficiently than SQL
3. **Sub-millisecond precision**: 12-bit timestamp fraction provides better ordering without performance cost
4. **Session-level monotonicity**: Guaranteed ordering within database sessions prevents collisions
5. **RFC 9562 compliance**: Modern standard optimized for performance and ordering

### Why Native Implementation Breaks Trade-offs

- **No SQL parsing overhead**: Direct C function calls
- **Optimized memory allocation**: Integrated with PostgreSQL's memory management
- **Better CPU cache utilization**: Native code optimizations
- **Reduced context switching**: No PL/pgSQL interpreter overhead

### Custom Implementation Performance

- **PL/pgSQL overhead**: Custom functions show 38% concurrent throughput reduction vs UUIDv4
- **Time ordering cost**: 10-39% performance impact for time-ordered identifiers
- **Still excellent performance**: All implementations maintain >18K IDs/second throughput

## Implementation Architectures

### UUIDv7 Implementations Tested

1. **uuid_generate_v7()** - PL/pgSQL implementation using overlay method
2. **uuidv7_custom()** - Pure SQL implementation with bit manipulation  
3. **uuidv7_sub_ms()** - Sub-millisecond precision with fractional timestamp
4. **uuidv7_native()** - PostgreSQL 18 native C implementation (wrapper for benchmarking)

### Alternative ID Types

5. **ulid_generate()** - ULID with Crockford Base32 encoding
6. **typeid_generate_text()** - TypeID with type prefixes for safety

### Baseline Comparison

7. **gen_random_uuid()** - PostgreSQL native UUIDv4 for performance baseline

## Benchmark Metrics

1. **Generation Performance**
   - Average, median, min, max generation time
   - Standard deviation and percentiles
   - Single-threaded vs concurrent throughput

2. **Correctness & Ordering**
   - Collision detection across 50,000+ IDs
   - Time ordering accuracy percentage
   - Monotonicity validation

3. **Storage Analysis**
   - Binary vs text storage requirements
   - Compression characteristics
   - Index performance implications

## Output Files

The benchmark generates:
- `enhanced_benchmark_results.json` - Complete benchmark data with PostgreSQL 18 results
- `enhanced_postgresql_benchmark.png` - Performance visualization charts
- `benchmark_results.json` - Standard benchmark data
- Console output with detailed performance analysis

## Migration to PostgreSQL 18

When PostgreSQL 18 becomes available (late 2025), migration is straightforward:

```sql
-- Replace custom functions with native calls
ALTER TABLE your_table 
ALTER COLUMN id SET DEFAULT uuidv7();  -- PostgreSQL 18+

-- Extract timestamps from existing UUIDs
SELECT id, uuid_extract_timestamp(id) 
FROM your_table 
WHERE uuid_extract_version(id) = 7;

-- Generate historical UUIDs
SELECT uuidv7('2024-01-01 00:00:00'::timestamp);
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Priority areas:
- PostgreSQL 18 beta testing and validation
- Performance optimizations for custom implementations
- Additional ID format comparisons
- Enhanced visualization and reporting

## License

MIT License - see LICENSE file for details.

## Related Resources

- [Blog Post: Complete Guide to UUIDv7, ULID, and TypeID in PostgreSQL](https://www.saybackend.com/blog/uuidv7-postgres-comparison)
- [IETF RFC 9562: UUIDv7 Specification](https://datatracker.ietf.org/doc/rfc9562/)
- [PostgreSQL 18 Release Notes](https://www.postgresql.org/docs/18/release-18.html)