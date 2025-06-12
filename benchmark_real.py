#!/usr/bin/env python3
"""
Real PostgreSQL UUIDv7 Benchmark - No simulations, only actual database calls
"""

import time
import statistics
import json
import psycopg
from datetime import datetime
import concurrent.futures
import threading
from typing import Dict, List, Any

# Database configurations
PG17_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'database': 'benchmark',
    'user': 'postgres',
    'password': 'postgres'
}

PG18_CONFIG = {
    'host': 'localhost',
    'port': 5435,
    'database': 'benchmark',
    'user': 'postgres',
    'password': 'postgres'
}

def wait_for_db(config, timeout=60):
    """Wait for database to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with psycopg.connect(**config) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception:
            time.sleep(2)
    return False

def check_pg_version_and_uuidv7(config):
    """Check PostgreSQL version and native UUIDv7 support"""
    try:
        with psycopg.connect(**config) as conn:
            with conn.cursor() as cur:
                # Get version
                cur.execute("SELECT current_setting('server_version_num')::INTEGER")
                version_num = cur.fetchone()[0]
                major_version = version_num // 10000
                
                # Check native uuidv7() support
                has_native_uuidv7 = False
                if version_num >= 180000:
                    try:
                        cur.execute("SELECT uuidv7()")
                        has_native_uuidv7 = True
                    except Exception:
                        pass
                
                return major_version, has_native_uuidv7
    except Exception as e:
        print(f"Error checking database: {e}")
        return None, False

def benchmark_function_real(config, function_call, iterations=5000):
    """Benchmark a function with real database calls"""
    times = []
    ids = []
    errors = 0
    
    try:
        with psycopg.connect(**config) as conn:
            with conn.cursor() as cur:
                # Warmup
                for _ in range(10):
                    try:
                        cur.execute(function_call)
                        cur.fetchone()
                    except Exception:
                        pass
                
                # Real benchmark
                for _ in range(iterations):
                    start_time = time.perf_counter()
                    try:
                        cur.execute(function_call)
                        result = cur.fetchone()[0]
                        end_time = time.perf_counter()
                        
                        elapsed_us = (end_time - start_time) * 1_000_000
                        times.append(elapsed_us)
                        ids.append(str(result))
                    except Exception as e:
                        errors += 1
                        if errors > 100:  # Stop if too many errors
                            break
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return None
    
    if not times:
        return None
    
    return {
        'function': function_call,
        'iterations': len(times),
        'avg_time_us': statistics.mean(times),
        'median_time_us': statistics.median(times),
        'min_time_us': min(times),
        'max_time_us': max(times),
        'p95_time_us': sorted(times)[int(len(times) * 0.95)],
        'p99_time_us': sorted(times)[int(len(times) * 0.99)],
        'std_dev_us': statistics.stdev(times) if len(times) > 1 else 0,
        'errors': errors,
        'unique_ids': len(set(ids)),
        'total_ids': len(ids),
        'collision_rate': (len(ids) - len(set(ids))) / len(ids) if ids else 0
    }

def benchmark_concurrent_real(config, function_call, workers=5, iterations_per_worker=500):
    """Benchmark concurrent performance with real database calls"""
    def worker_func():
        worker_times = []
        worker_ids = []
        errors = 0
        
        try:
            with psycopg.connect(**config) as conn:
                with conn.cursor() as cur:
                    for _ in range(iterations_per_worker):
                        start_time = time.perf_counter()
                        try:
                            cur.execute(function_call)
                            result = cur.fetchone()[0]
                            end_time = time.perf_counter()
                            
                            elapsed_us = (end_time - start_time) * 1_000_000
                            worker_times.append(elapsed_us)
                            worker_ids.append(str(result))
                        except Exception:
                            errors += 1
                            if errors > 50:
                                break
        except Exception:
            pass
        
        return {
            'times': worker_times,
            'ids': worker_ids,
            'errors': errors
        }
    
    # Run concurrent workers
    overall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_func) for _ in range(workers)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    overall_end = time.perf_counter()
    
    # Aggregate results
    all_times = []
    all_ids = []
    total_errors = 0
    
    for result in results:
        all_times.extend(result['times'])
        all_ids.extend(result['ids'])
        total_errors += result['errors']
    
    total_duration = overall_end - overall_start
    successful_ops = len(all_times)
    throughput = successful_ops / total_duration if total_duration > 0 else 0
    
    return {
        'function': function_call,
        'workers': workers,
        'total_operations': successful_ops,
        'total_duration_sec': total_duration,
        'throughput_per_sec': throughput,
        'avg_time_us': statistics.mean(all_times) if all_times else 0,
        'median_time_us': statistics.median(all_times) if all_times else 0,
        'p95_time_us': sorted(all_times)[int(len(all_times) * 0.95)] if all_times else 0,
        'errors': total_errors,
        'collision_rate': (len(all_ids) - len(set(all_ids))) / len(all_ids) if all_ids else 0
    }

def main():
    print("=== REAL PostgreSQL UUIDv7 Benchmark ===")
    print("No simulations - only actual database calls")
    print()
    
    # Define functions to test
    functions_to_test = {
        'pg17': [
            ('SELECT gen_random_uuid()', 'UUIDv4 (baseline)'),
            ('SELECT uuid_generate_v7()', 'UUIDv7 (PL/pgSQL)'),
            ('SELECT uuidv7_custom()', 'UUIDv7 (Pure SQL)'),
            ('SELECT uuidv7_sub_ms()', 'UUIDv7 (Sub-ms)'),
            ('SELECT ulid_generate()', 'ULID'),
            ('SELECT typeid_generate_text(\'test\')', 'TypeID')
        ],
        'pg18': [
            ('SELECT gen_random_uuid()', 'UUIDv4 (baseline)'),
            ('SELECT uuidv7()', 'Native UUIDv7 (PG18)'),  # Native function
            ('SELECT uuid_generate_v7()', 'UUIDv7 (PL/pgSQL)'),
            ('SELECT uuidv7_custom()', 'UUIDv7 (Pure SQL)'),
            ('SELECT uuidv7_sub_ms()', 'UUIDv7 (Sub-ms)'),
            ('SELECT ulid_generate()', 'ULID'),
            ('SELECT typeid_generate_text(\'test\')', 'TypeID')
        ]
    }
    
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'real_database_calls',
            'single_thread_iterations': 5000,
            'concurrent_workers': 5,
            'concurrent_iterations_per_worker': 500
        },
        'postgresql_17': {},
        'postgresql_18': {}
    }
    
    # Test PostgreSQL 17
    print("Testing PostgreSQL 17...")
    if wait_for_db(PG17_CONFIG):
        version, has_native = check_pg_version_and_uuidv7(PG17_CONFIG)
        print(f"  Version: PostgreSQL {version}")
        print(f"  Native UUIDv7: {has_native}")
        
        pg17_results = {'single_thread': {}, 'concurrent': {}}
        
        for sql, name in functions_to_test['pg17']:
            print(f"  Testing {name}...")
            
            # Single-threaded test
            single_result = benchmark_function_real(PG17_CONFIG, sql)
            if single_result:
                pg17_results['single_thread'][name] = single_result
                print(f"    Single: {single_result['avg_time_us']:.1f} μs avg")
            
            # Concurrent test
            concurrent_result = benchmark_concurrent_real(PG17_CONFIG, sql)
            if concurrent_result:
                pg17_results['concurrent'][name] = concurrent_result
                print(f"    Concurrent: {concurrent_result['throughput_per_sec']:.0f} ops/sec")
        
        results['postgresql_17'] = pg17_results
    else:
        print("  ❌ PostgreSQL 17 not available")
    
    print()
    
    # Test PostgreSQL 18
    print("Testing PostgreSQL 18...")
    if wait_for_db(PG18_CONFIG):
        version, has_native = check_pg_version_and_uuidv7(PG18_CONFIG)
        print(f"  Version: PostgreSQL {version}")
        print(f"  Native UUIDv7: {has_native}")
        
        pg18_results = {'single_thread': {}, 'concurrent': {}}
        
        for sql, name in functions_to_test['pg18']:
            print(f"  Testing {name}...")
            
            # Single-threaded test
            single_result = benchmark_function_real(PG18_CONFIG, sql)
            if single_result:
                pg18_results['single_thread'][name] = single_result
                print(f"    Single: {single_result['avg_time_us']:.1f} μs avg")
            
            # Concurrent test
            concurrent_result = benchmark_concurrent_real(PG18_CONFIG, sql)
            if concurrent_result:
                pg18_results['concurrent'][name] = concurrent_result
                print(f"    Concurrent: {concurrent_result['throughput_per_sec']:.0f} ops/sec")
        
        results['postgresql_18'] = pg18_results
    else:
        print("  ❌ PostgreSQL 18 not available")
    
    # Save results
    with open('real_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Real benchmark results saved to: real_benchmark_results.json")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for pg_version in ['postgresql_17', 'postgresql_18']:
        if pg_version in results and results[pg_version]:
            print(f"\n{pg_version.upper()}:")
            single_results = results[pg_version].get('single_thread', {})
            for name, data in single_results.items():
                print(f"  {name}: {data['avg_time_us']:.1f} μs")

if __name__ == "__main__":
    main()