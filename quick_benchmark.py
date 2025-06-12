#!/usr/bin/env python3
"""
Quick Real PostgreSQL UUIDv7 Benchmark - Actual database calls, smaller scale
"""

import time
import statistics
import json
import psycopg
from datetime import datetime

# Database configurations
PG17_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'dbname': 'benchmark',
    'user': 'postgres',
    'password': 'postgres'
}

PG18_CONFIG = {
    'host': 'localhost',
    'port': 5435,
    'dbname': 'benchmark',
    'user': 'postgres',
    'password': 'postgres'
}

def quick_benchmark(config, function_call, iterations=1000):
    """Quick benchmark with real database calls"""
    times = []
    
    try:
        with psycopg.connect(**config) as conn:
            with conn.cursor() as cur:
                # Warmup
                for _ in range(5):
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
                    except Exception as e:
                        print(f"Error with {function_call}: {e}")
                        return None
    except Exception as e:
        print(f"Connection failed: {e}")
        return None
    
    if not times:
        return None
    
    return {
        'avg_time_us': round(statistics.mean(times), 1),
        'median_time_us': round(statistics.median(times), 1),
        'min_time_us': round(min(times), 1),
        'max_time_us': round(max(times), 1),
        'p95_time_us': round(sorted(times)[int(len(times) * 0.95)], 1),
        'iterations': len(times)
    }

def main():
    print("=== QUICK REAL PostgreSQL UUIDv7 Benchmark ===")
    print("Real database calls, 1000 iterations each")
    print()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'iterations': 1000,
        'postgresql_17': {},
        'postgresql_18': {}
    }
    
    # Test PostgreSQL 17
    print("PostgreSQL 17 Results:")
    functions_17 = [
        ('SELECT gen_random_uuid()', 'UUIDv4'),
        ('SELECT uuid_generate_v7()', 'UUIDv7_PL'),
        ('SELECT uuidv7_custom()', 'UUIDv7_SQL'),
        ('SELECT uuidv7_sub_ms()', 'UUIDv7_SubMs'),
        ('SELECT ulid_generate()', 'ULID'),
        ('SELECT typeid_generate_text(\'test\')', 'TypeID')
    ]
    
    for sql, name in functions_17:
        result = quick_benchmark(PG17_CONFIG, sql)
        if result:
            results['postgresql_17'][name] = result
            print(f"  {name}: {result['avg_time_us']} μs avg")
        else:
            print(f"  {name}: FAILED")
    
    print()
    
    # Test PostgreSQL 18
    print("PostgreSQL 18 Results:")
    functions_18 = [
        ('SELECT gen_random_uuid()', 'UUIDv4'),
        ('SELECT uuidv7()', 'Native_UUIDv7'),  # This is the key one!
        ('SELECT uuid_generate_v7()', 'UUIDv7_PL'),
        ('SELECT uuidv7_custom()', 'UUIDv7_SQL'),
        ('SELECT uuidv7_sub_ms()', 'UUIDv7_SubMs'),
        ('SELECT ulid_generate()', 'ULID'),
        ('SELECT typeid_generate_text(\'test\')', 'TypeID')
    ]
    
    for sql, name in functions_18:
        result = quick_benchmark(PG18_CONFIG, sql)
        if result:
            results['postgresql_18'][name] = result
            print(f"  {name}: {result['avg_time_us']} μs avg")
        else:
            print(f"  {name}: FAILED")
    
    # Save results
    with open('quick_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: quick_benchmark_results.json")
    
    # Performance comparison
    if 'Native_UUIDv7' in results['postgresql_18'] and 'UUIDv4' in results['postgresql_18']:
        native_time = results['postgresql_18']['Native_UUIDv7']['avg_time_us']
        uuid4_time = results['postgresql_18']['UUIDv4']['avg_time_us']
        improvement = ((uuid4_time - native_time) / uuid4_time) * 100
        print(f"\n🚀 Native UUIDv7 vs UUIDv4: {improvement:.1f}% faster!")

if __name__ == "__main__":
    main()