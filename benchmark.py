#!/usr/bin/env python3
"""
UUIDv7 PostgreSQL Benchmark Script
Compares performance of three UUIDv7 implementations across PostgreSQL versions
"""

import psycopg2
import time
import json
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import numpy as np
import seaborn as sns

# Database configurations
DBS = {
    'postgres17': {
        'host': 'localhost',
        'port': 5432,
        'database': 'uuidv7_benchmark',
        'user': 'postgres',
        'password': 'postgres'
    },
    'postgres18': {
        'host': 'localhost',
        'port': 5433,
        'database': 'uuidv7_benchmark',
        'user': 'postgres',
        'password': 'postgres'
    }
}

FUNCTIONS = ['uuid_generate_v7', 'uuidv7', 'uuidv7_sub_ms']
ITERATIONS = 10000
CONCURRENT_WORKERS = 10
CONCURRENT_ITERATIONS = 1000

def get_connection(db_config):
    """Create database connection"""
    return psycopg2.connect(**db_config)

def benchmark_single_function(conn, function_name, iterations):
    """Benchmark a single function"""
    with conn.cursor() as cur:
        # Warm up
        cur.execute(f"SELECT {function_name}() FROM generate_series(1, 100)")
        
        # Measure generation time
        times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            cur.execute(f"SELECT {function_name}()")
            cur.fetchone()
            end = time.perf_counter_ns()
            times.append(end - start)
        
        return {
            'function': function_name,
            'iterations': iterations,
            'avg_ns': statistics.mean(times),
            'min_ns': min(times),
            'max_ns': max(times),
            'median_ns': statistics.median(times),
            'std_dev_ns': statistics.stdev(times) if len(times) > 1 else 0,
            'p95_ns': np.percentile(times, 95),
            'p99_ns': np.percentile(times, 99)
        }

def test_uniqueness(conn, function_name, count):
    """Test for collisions in generated UUIDs"""
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH generated AS (
                SELECT {function_name}() as uuid 
                FROM generate_series(1, %s)
            )
            SELECT COUNT(DISTINCT uuid) as unique_count,
                   COUNT(*) as total_count
            FROM generated
        """, (count,))
        result = cur.fetchone()
        return {
            'function': function_name,
            'total_generated': result[1],
            'unique_count': result[0],
            'collision_count': result[1] - result[0],
            'collision_rate': (result[1] - result[0]) / result[1] if result[1] > 0 else 0
        }

def benchmark_concurrent(db_config, function_name, workers, iterations_per_worker):
    """Benchmark concurrent UUID generation"""
    def worker_task():
        conn = get_connection(db_config)
        times = []
        try:
            with conn.cursor() as cur:
                for _ in range(iterations_per_worker):
                    start = time.perf_counter_ns()
                    cur.execute(f"SELECT {function_name}()")
                    cur.fetchone()
                    end = time.perf_counter_ns()
                    times.append(end - start)
        finally:
            conn.close()
        return times
    
    all_times = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_task) for _ in range(workers)]
        for future in as_completed(futures):
            all_times.extend(future.result())
    
    end_time = time.time()
    
    return {
        'function': function_name,
        'workers': workers,
        'total_iterations': len(all_times),
        'total_time_sec': end_time - start_time,
        'throughput_per_sec': len(all_times) / (end_time - start_time),
        'avg_ns': statistics.mean(all_times),
        'median_ns': statistics.median(all_times),
        'p95_ns': np.percentile(all_times, 95),
        'p99_ns': np.percentile(all_times, 99)
    }

def test_time_ordering(conn, function_name, count=1000):
    """Test if UUIDs maintain time ordering"""
    with conn.cursor() as cur:
        # Generate UUIDs with small delays
        cur.execute(f"""
            CREATE TEMP TABLE time_order_test AS
            SELECT 
                {function_name}() as uuid,
                clock_timestamp() as generated_at,
                row_number() OVER () as seq
            FROM generate_series(1, %s);
        """, (count,))
        
        # Check if UUID ordering matches time ordering
        cur.execute("""
            WITH ordered AS (
                SELECT 
                    uuid,
                    generated_at,
                    seq,
                    row_number() OVER (ORDER BY uuid) as uuid_order,
                    row_number() OVER (ORDER BY generated_at) as time_order
                FROM time_order_test
            )
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN uuid_order = time_order THEN 1 ELSE 0 END) as ordered_correctly,
                SUM(CASE WHEN uuid_order != time_order THEN 1 ELSE 0 END) as out_of_order
            FROM ordered
        """)
        
        result = cur.fetchone()
        cur.execute("DROP TABLE time_order_test")
        
        return {
            'function': function_name,
            'total_uuids': result[0],
            'ordered_correctly': result[1],
            'out_of_order': result[2],
            'order_accuracy': result[1] / result[0] if result[0] > 0 else 0
        }

def run_all_benchmarks():
    """Run all benchmarks and collect results"""
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'iterations': ITERATIONS,
            'concurrent_workers': CONCURRENT_WORKERS,
            'concurrent_iterations': CONCURRENT_ITERATIONS
        },
        'benchmarks': {}
    }
    
    for db_name, db_config in DBS.items():
        print(f"\n=== Benchmarking {db_name} ===")
        results['benchmarks'][db_name] = {
            'single_thread': [],
            'concurrent': [],
            'uniqueness': [],
            'time_ordering': []
        }
        
        try:
            conn = get_connection(db_config)
            
            # Single-threaded benchmarks
            print("\nSingle-threaded performance:")
            for func in FUNCTIONS:
                print(f"  Benchmarking {func}...")
                result = benchmark_single_function(conn, func, ITERATIONS)
                results['benchmarks'][db_name]['single_thread'].append(result)
                print(f"    Avg: {result['avg_ns']/1000:.2f} μs, "
                      f"Median: {result['median_ns']/1000:.2f} μs, "
                      f"P95: {result['p95_ns']/1000:.2f} μs")
            
            # Uniqueness tests
            print("\nUniqueness tests (1M UUIDs):")
            for func in FUNCTIONS:
                print(f"  Testing {func}...")
                result = test_uniqueness(conn, func, 1000000)
                results['benchmarks'][db_name]['uniqueness'].append(result)
                print(f"    Collisions: {result['collision_count']} "
                      f"(Rate: {result['collision_rate']*100:.6f}%)")
            
            # Time ordering tests
            print("\nTime ordering tests:")
            for func in FUNCTIONS:
                print(f"  Testing {func}...")
                result = test_time_ordering(conn, func)
                results['benchmarks'][db_name]['time_ordering'].append(result)
                print(f"    Order accuracy: {result['order_accuracy']*100:.2f}%")
            
            conn.close()
            
            # Concurrent benchmarks
            print("\nConcurrent performance:")
            for func in FUNCTIONS:
                print(f"  Benchmarking {func} ({CONCURRENT_WORKERS} workers)...")
                result = benchmark_concurrent(db_config, func, CONCURRENT_WORKERS, CONCURRENT_ITERATIONS)
                results['benchmarks'][db_name]['concurrent'].append(result)
                print(f"    Throughput: {result['throughput_per_sec']:.0f} UUIDs/sec, "
                      f"Avg: {result['avg_ns']/1000:.2f} μs")
                
        except Exception as e:
            print(f"Error benchmarking {db_name}: {e}")
    
    return results

def create_visualizations(results):
    """Create graphs and charts from benchmark results"""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Single-threaded performance comparison
    ax1 = plt.subplot(2, 3, 1)
    data = []
    for db_name, db_results in results['benchmarks'].items():
        for bench in db_results['single_thread']:
            data.append({
                'Database': db_name,
                'Function': bench['function'],
                'Avg Time (μs)': bench['avg_ns'] / 1000
            })
    df = pd.DataFrame(data)
    df.pivot(index='Function', columns='Database', values='Avg Time (μs)').plot(kind='bar', ax=ax1)
    ax1.set_title('Single-threaded Performance')
    ax1.set_ylabel('Average Time (μs)')
    ax1.legend(title='Database')
    
    # 2. Concurrent throughput
    ax2 = plt.subplot(2, 3, 2)
    data = []
    for db_name, db_results in results['benchmarks'].items():
        for bench in db_results['concurrent']:
            data.append({
                'Database': db_name,
                'Function': bench['function'],
                'Throughput': bench['throughput_per_sec']
            })
    df = pd.DataFrame(data)
    df.pivot(index='Function', columns='Database', values='Throughput').plot(kind='bar', ax=ax2)
    ax2.set_title('Concurrent Throughput')
    ax2.set_ylabel('UUIDs per Second')
    
    # 3. Performance percentiles
    ax3 = plt.subplot(2, 3, 3)
    for db_name, db_results in results['benchmarks'].items():
        functions = []
        p95_times = []
        p99_times = []
        for bench in db_results['single_thread']:
            functions.append(bench['function'].replace('uuid_generate_v7', 'v7_1').replace('uuidv7_sub_ms', 'v7_sub').replace('uuidv7', 'v7_2'))
            p95_times.append(bench['p95_ns'] / 1000)
            p99_times.append(bench['p99_ns'] / 1000)
        
        x = np.arange(len(functions))
        width = 0.35
        ax3.bar(x - width/2, p95_times, width, label=f'{db_name} P95', alpha=0.8)
        ax3.bar(x + width/2, p99_times, width, label=f'{db_name} P99', alpha=0.8)
    
    ax3.set_xlabel('Function')
    ax3.set_ylabel('Time (μs)')
    ax3.set_title('Performance Percentiles')
    ax3.set_xticks(x)
    ax3.set_xticklabels(functions)
    ax3.legend()
    
    # 4. Time ordering accuracy
    ax4 = plt.subplot(2, 3, 4)
    data = []
    for db_name, db_results in results['benchmarks'].items():
        for bench in db_results['time_ordering']:
            data.append({
                'Database': db_name,
                'Function': bench['function'],
                'Order Accuracy': bench['order_accuracy'] * 100
            })
    df = pd.DataFrame(data)
    df.pivot(index='Function', columns='Database', values='Order Accuracy').plot(kind='bar', ax=ax4)
    ax4.set_title('Time Ordering Accuracy')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_ylim(0, 105)
    
    # 5. Performance stability (box plot)
    ax5 = plt.subplot(2, 3, 5)
    # This would require raw timing data, so we'll use std_dev as a proxy
    data = []
    for db_name, db_results in results['benchmarks'].items():
        for bench in db_results['single_thread']:
            data.append({
                'Database': db_name,
                'Function': bench['function'].replace('uuid_generate_v7', 'v7_1').replace('uuidv7_sub_ms', 'v7_sub').replace('uuidv7', 'v7_2'),
                'Std Dev (μs)': bench['std_dev_ns'] / 1000
            })
    df = pd.DataFrame(data)
    df.pivot(index='Function', columns='Database', values='Std Dev (μs)').plot(kind='bar', ax=ax5)
    ax5.set_title('Performance Stability (Lower is Better)')
    ax5.set_ylabel('Standard Deviation (μs)')
    
    # 6. Summary table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create summary data
    summary_data = []
    for func in FUNCTIONS:
        row = [func.replace('uuid_generate_v7', 'Function 1').replace('uuidv7_sub_ms', 'Function 3').replace('uuidv7', 'Function 2')]
        for db_name in results['benchmarks'].keys():
            db_results = results['benchmarks'][db_name]
            single = next(b for b in db_results['single_thread'] if b['function'] == func)
            concurrent = next(b for b in db_results['concurrent'] if b['function'] == func)
            row.extend([
                f"{single['avg_ns']/1000:.1f}",
                f"{concurrent['throughput_per_sec']:.0f}"
            ])
        summary_data.append(row)
    
    columns = ['Function']
    for db in results['benchmarks'].keys():
        columns.extend([f'{db}\nAvg (μs)', f'{db}\nThroughput'])
    
    table = ax6.table(cellText=summary_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax6.set_title('Performance Summary', pad=20)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create a second figure for detailed analysis
    fig2, ((ax7, ax8), (ax9, ax10)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 7. Latency distribution (histogram)
    for i, (db_name, db_results) in enumerate(results['benchmarks'].items()):
        ax = ax7 if i == 0 else ax8
        for bench in db_results['single_thread']:
            # Simulate distribution based on mean and std dev
            mean = bench['avg_ns'] / 1000
            std = bench['std_dev_ns'] / 1000
            samples = np.random.normal(mean, std, 1000)
            ax.hist(samples, bins=50, alpha=0.5, label=bench['function'].replace('uuid_generate_v7', 'v7_1').replace('uuidv7_sub_ms', 'v7_sub').replace('uuidv7', 'v7_2'))
        ax.set_title(f'{db_name} - Latency Distribution')
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    # 8. Concurrent vs Single-threaded comparison
    ax9.set_title('Concurrent vs Single-threaded Performance')
    for db_name, db_results in results['benchmarks'].items():
        single_times = [b['avg_ns']/1000 for b in db_results['single_thread']]
        concurrent_times = [b['avg_ns']/1000 for b in db_results['concurrent']]
        functions = [b['function'].replace('uuid_generate_v7', 'v7_1').replace('uuidv7_sub_ms', 'v7_sub').replace('uuidv7', 'v7_2') for b in db_results['single_thread']]
        
        x = np.arange(len(functions))
        width = 0.35
        ax9.bar(x - width/2, single_times, width, label=f'{db_name} Single', alpha=0.8)
        ax9.bar(x + width/2, concurrent_times, width, label=f'{db_name} Concurrent', alpha=0.8)
    
    ax9.set_xlabel('Function')
    ax9.set_ylabel('Average Time (μs)')
    ax9.set_xticks(x)
    ax9.set_xticklabels(functions)
    ax9.legend()
    
    # 9. Performance radar chart
    ax10.set_title('Performance Characteristics')
    categories = ['Speed', 'Stability', 'Ordering', 'Throughput', 'Consistency']
    
    for func in FUNCTIONS:
        values = []
        for db_results in results['benchmarks'].values():
            single = next(b for b in db_results['single_thread'] if b['function'] == func)
            concurrent = next(b for b in db_results['concurrent'] if b['function'] == func)
            ordering = next(b for b in db_results['time_ordering'] if b['function'] == func)
            
            # Normalize values (inverse for time-based metrics)
            speed = 1000 / (single['avg_ns'] / 1000)  # Higher is better
            stability = 100 / (1 + single['std_dev_ns'] / single['avg_ns'])  # Higher is better
            order_score = ordering['order_accuracy'] * 100
            throughput = concurrent['throughput_per_sec'] / 10000  # Normalize
            consistency = 100 - (abs(single['p99_ns'] - single['median_ns']) / single['median_ns'] * 10)
            
            values.append([speed, stability, order_score, throughput, consistency])
        
        # Average across databases
        avg_values = np.mean(values, axis=0)
        
        # Plot on radar
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        avg_values = np.concatenate((avg_values, [avg_values[0]]))  # Complete the circle
        angles += angles[:1]
        
        ax10 = plt.subplot(2, 2, 4, projection='polar')
        ax10.plot(angles, avg_values, 'o-', linewidth=2, label=func.replace('uuid_generate_v7', 'Function 1').replace('uuidv7_sub_ms', 'Function 3').replace('uuidv7', 'Function 2'))
        ax10.fill(angles, avg_values, alpha=0.25)
        ax10.set_xticks(angles[:-1])
        ax10.set_xticklabels(categories)
        ax10.set_ylim(0, 100)
        ax10.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig('detailed_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_results(results):
    """Save results to JSON file"""
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

def create_markdown_report(results):
    """Create a markdown report with results"""
    report = """# UUIDv7 PostgreSQL Benchmark Results

## Test Configuration
- **Timestamp**: {timestamp}
- **Iterations per test**: {iterations:,}
- **Concurrent workers**: {workers}
- **Concurrent iterations per worker**: {concurrent_iterations:,}

## Performance Summary

### Single-threaded Performance (microseconds)
| Function | PostgreSQL 17 | PostgreSQL 18 Beta |
|----------|--------------|-------------------|
""".format(
        timestamp=results['metadata']['timestamp'],
        iterations=results['metadata']['iterations'],
        workers=results['metadata']['concurrent_workers'],
        concurrent_iterations=results['metadata']['concurrent_iterations']
    )
    
    # Add single-threaded results
    for func in FUNCTIONS:
        row = f"| {func} |"
        for db in ['postgres17', 'postgres18']:
            if db in results['benchmarks']:
                bench = next((b for b in results['benchmarks'][db]['single_thread'] if b['function'] == func), None)
                if bench:
                    row += f" {bench['avg_ns']/1000:.2f} |"
                else:
                    row += " N/A |"
            else:
                row += " N/A |"
        report += row + "\n"
    
    report += """
### Concurrent Throughput (UUIDs/second)
| Function | PostgreSQL 17 | PostgreSQL 18 Beta |
|----------|--------------|-------------------|
"""
    
    # Add concurrent results
    for func in FUNCTIONS:
        row = f"| {func} |"
        for db in ['postgres17', 'postgres18']:
            if db in results['benchmarks']:
                bench = next((b for b in results['benchmarks'][db]['concurrent'] if b['function'] == func), None)
                if bench:
                    row += f" {bench['throughput_per_sec']:,.0f} |"
                else:
                    row += " N/A |"
            else:
                row += " N/A |"
        report += row + "\n"
    
    report += """
### Time Ordering Accuracy
| Function | PostgreSQL 17 | PostgreSQL 18 Beta |
|----------|--------------|-------------------|
"""
    
    # Add ordering results
    for func in FUNCTIONS:
        row = f"| {func} |"
        for db in ['postgres17', 'postgres18']:
            if db in results['benchmarks']:
                bench = next((b for b in results['benchmarks'][db]['time_ordering'] if b['function'] == func), None)
                if bench:
                    row += f" {bench['order_accuracy']*100:.1f}% |"
                else:
                    row += " N/A |"
            else:
                row += " N/A |"
        report += row + "\n"
    
    report += """
### Collision Tests (1 Million UUIDs)
| Function | PostgreSQL 17 | PostgreSQL 18 Beta |
|----------|--------------|-------------------|
"""
    
    # Add uniqueness results
    for func in FUNCTIONS:
        row = f"| {func} |"
        for db in ['postgres17', 'postgres18']:
            if db in results['benchmarks']:
                bench = next((b for b in results['benchmarks'][db]['uniqueness'] if b['function'] == func), None)
                if bench:
                    row += f" {bench['collision_count']} |"
                else:
                    row += " N/A |"
            else:
                row += " N/A |"
        report += row + "\n"
    
    report += """
## Key Findings

1. **Performance**: Function comparison across versions
2. **Scalability**: Concurrent performance characteristics
3. **Reliability**: Collision rates and time ordering accuracy

## Graphs

![Benchmark Results](benchmark_results.png)
![Detailed Analysis](detailed_analysis.png)
"""
    
    with open('BENCHMARK_REPORT.md', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    print("Starting UUIDv7 PostgreSQL benchmarks...")
    print("Make sure Docker containers are running!")
    
    # Wait for databases to be ready
    print("\nWaiting for databases to be ready...")
    time.sleep(5)
    
    # Run benchmarks
    results = run_all_benchmarks()
    
    # Save results
    save_results(results)
    print("\nResults saved to benchmark_results.json")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results)
    print("Visualizations saved to benchmark_results.png and detailed_analysis.png")
    
    # Create markdown report
    create_markdown_report(results)
    print("Markdown report saved to BENCHMARK_REPORT.md")
    
    print("\nBenchmark complete!")