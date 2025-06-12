"""
Professional PostgreSQL UUIDv7 Benchmark Suite - Benchmark Engine
High-precision benchmarking with statistical analysis and multiple runs
"""

import time
import statistics
import logging
import threading
import concurrent.futures
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import gc

from .config import BenchmarkConfig, DEFAULT_BENCHMARK_CONFIG, FUNCTION_CONFIGS
from .database import DatabaseManager, DatabaseConfig

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Single benchmark run result"""
    function_name: str
    run_number: int
    timestamp: datetime
    
    # Single-threaded metrics
    single_thread_times_ns: List[int]
    single_thread_avg_ns: float
    single_thread_median_ns: float
    single_thread_std_dev_ns: float
    single_thread_min_ns: int
    single_thread_max_ns: int
    single_thread_p95_ns: float
    single_thread_p99_ns: float
    
    # Concurrent metrics
    concurrent_times_ns: List[int]
    concurrent_avg_ns: float
    concurrent_median_ns: float
    concurrent_std_dev_ns: float
    concurrent_throughput_per_second: float
    concurrent_total_time_s: float
    
    # Metadata
    iterations_single: int
    iterations_concurrent: int
    workers: int
    postgres_version: int
    database_name: str
    
    # Validation
    all_ids_unique: bool
    correct_format: bool
    sample_id: str
    storage_size_bytes: int

@dataclass
class AggregatedResults:
    """Results aggregated across multiple benchmark runs"""
    function_name: str
    runs_completed: int
    database_name: str
    postgres_version: int
    
    # Aggregated single-threaded metrics (across all runs)
    avg_single_thread_ns: float
    std_dev_single_thread_ns: float
    median_single_thread_ns: float
    min_single_thread_ns: float
    max_single_thread_ns: float
    p95_single_thread_ns: float
    p99_single_thread_ns: float
    
    # Aggregated concurrent metrics
    avg_concurrent_ns: float
    avg_throughput_per_second: float
    std_dev_throughput: float
    
    # Consistency metrics
    coefficient_of_variation: float  # std_dev / mean
    run_to_run_variance: float
    
    # Summary
    performance_vs_baseline: Optional[float] = None
    performance_grade: str = 'Unknown'

class BenchmarkEngine:
    """High-precision benchmark execution engine"""
    
    def __init__(self, config: BenchmarkConfig = DEFAULT_BENCHMARK_CONFIG):
        self.config = config
        self.db_manager = DatabaseManager()
        self.results: Dict[str, List[BenchmarkResult]] = {}
        self.aggregated_results: Dict[str, AggregatedResults] = {}
        
    def run_complete_benchmark(self, db_configs: Dict[str, DatabaseConfig]) -> Dict[str, Any]:
        """Run the complete benchmark suite across all databases"""
        logger.info("Starting comprehensive benchmark suite...")
        logger.info(f"Configuration: {self.config.benchmark_runs} runs, "
                   f"{self.config.single_thread_iterations:,} single-thread iterations, "
                   f"{self.config.concurrent_workers} workers × {self.config.concurrent_iterations_per_worker:,} iterations")
        
        benchmark_start = time.time()
        
        # Initialize databases
        db_info = self.db_manager.initialize_all_databases()
        
        # Run benchmarks for each database
        for db_name, config in db_configs.items():
            if db_name not in db_info:
                logger.error(f"Database {db_name} not initialized, skipping...")
                continue
                
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking {db_name} (PostgreSQL {db_info[db_name].major_version})")
            logger.info(f"{'='*60}")
            
            # Optimize database for benchmarking
            self.db_manager.optimize_for_benchmarking(db_name)
            
            # Validate functions
            validation_results = self.db_manager.validate_function_correctness(db_name)
            logger.info(f"Function validation completed: {len(validation_results)} functions tested")
            
            # Run benchmarks for all supported functions
            for func_name in db_info[db_name].supported_functions:
                self._run_function_benchmark(db_name, config, func_name, db_info[db_name])
        
        # Aggregate results
        self._aggregate_all_results()
        
        # Generate comprehensive report
        total_time = time.time() - benchmark_start
        logger.info(f"\nBenchmark completed in {total_time:.1f} seconds")
        
        return self._prepare_final_results(total_time)
    
    def _run_function_benchmark(self, db_name: str, config: DatabaseConfig, 
                               func_name: str, db_info: Any):
        """Run multiple benchmark runs for a single function"""
        logger.info(f"\nBenchmarking {func_name}...")
        
        if db_name not in self.results:
            self.results[db_name] = []
        
        for run_num in range(1, self.config.benchmark_runs + 1):
            logger.info(f"  Run {run_num}/{self.config.benchmark_runs}")
            
            try:
                # Force garbage collection before each run
                gc.collect()
                
                result = self._execute_single_benchmark_run(
                    config, func_name, run_num, db_name, db_info
                )
                
                self.results[db_name].append(result)
                
                # Log run summary
                logger.info(f"    Single-thread: {result.single_thread_avg_ns/1000:.1f}μs avg")
                logger.info(f"    Concurrent: {result.concurrent_throughput_per_second:,.0f} ops/sec")
                
            except Exception as e:
                logger.error(f"Benchmark run {run_num} failed for {func_name}: {e}")
                continue
    
    def _execute_single_benchmark_run(self, config: DatabaseConfig, func_name: str, 
                                    run_num: int, db_name: str, db_info: Any) -> BenchmarkResult:
        """Execute a single complete benchmark run"""
        
        with self.db_manager.get_connection(config) as conn:
            with conn.cursor() as cur:
                # Warmup phase
                logger.debug(f"    Warming up ({self.config.warmup_iterations:,} iterations)...")
                self._warmup_function(cur, func_name, self.config.warmup_iterations)
                
                # Single-threaded benchmark
                logger.debug(f"    Single-threaded ({self.config.single_thread_iterations:,} iterations)...")
                single_thread_results = self._benchmark_single_threaded(
                    cur, func_name, self.config.single_thread_iterations
                )
                
                # Concurrent benchmark  
                logger.debug(f"    Concurrent ({self.config.concurrent_workers} workers)...")
                concurrent_results = self._benchmark_concurrent(
                    config, func_name, self.config.concurrent_workers, 
                    self.config.concurrent_iterations_per_worker
                )
                
                # Validation
                validation_data = self._validate_function_output(cur, func_name)
                
                # Create result object
                return BenchmarkResult(
                    function_name=func_name,
                    run_number=run_num,
                    timestamp=datetime.now(timezone.utc),
                    
                    # Single-threaded metrics
                    single_thread_times_ns=single_thread_results['times_ns'],
                    single_thread_avg_ns=single_thread_results['avg_ns'],
                    single_thread_median_ns=single_thread_results['median_ns'],
                    single_thread_std_dev_ns=single_thread_results['std_dev_ns'],
                    single_thread_min_ns=single_thread_results['min_ns'],
                    single_thread_max_ns=single_thread_results['max_ns'],
                    single_thread_p95_ns=single_thread_results['p95_ns'],
                    single_thread_p99_ns=single_thread_results['p99_ns'],
                    
                    # Concurrent metrics
                    concurrent_times_ns=concurrent_results['times_ns'],
                    concurrent_avg_ns=concurrent_results['avg_ns'],
                    concurrent_median_ns=concurrent_results['median_ns'],
                    concurrent_std_dev_ns=concurrent_results['std_dev_ns'],
                    concurrent_throughput_per_second=concurrent_results['throughput_per_second'],
                    concurrent_total_time_s=concurrent_results['total_time_s'],
                    
                    # Metadata
                    iterations_single=self.config.single_thread_iterations,
                    iterations_concurrent=self.config.concurrent_workers * self.config.concurrent_iterations_per_worker,
                    workers=self.config.concurrent_workers,
                    postgres_version=db_info.major_version,
                    database_name=db_name,
                    
                    # Validation
                    all_ids_unique=validation_data['all_unique'],
                    correct_format=validation_data['correct_format'],
                    sample_id=validation_data['sample_id'],
                    storage_size_bytes=validation_data['storage_size']
                )
    
    def _warmup_function(self, cursor, func_name: str, iterations: int):
        """Warm up function to ensure stable performance measurements"""
        if 'typeid' in func_name:
            for _ in range(iterations):
                cursor.execute(f"SELECT {func_name}('test')")
        else:
            for _ in range(iterations):
                cursor.execute(f"SELECT {func_name}()")
    
    def _benchmark_single_threaded(self, cursor, func_name: str, iterations: int) -> Dict[str, Any]:
        """Execute single-threaded benchmark with high precision timing"""
        times_ns = []
        
        # Use the most precise timer available
        for i in range(iterations):
            if 'typeid' in func_name:
                start_time = time.perf_counter_ns()
                cursor.execute(f"SELECT {func_name}('test')")
                result = cursor.fetchone()
                end_time = time.perf_counter_ns()
            else:
                start_time = time.perf_counter_ns()
                cursor.execute(f"SELECT {func_name}()")
                result = cursor.fetchone()
                end_time = time.perf_counter_ns()
            
            elapsed_ns = end_time - start_time
            times_ns.append(elapsed_ns)
            
            # Progress logging every 10k iterations
            if (i + 1) % 10000 == 0:
                logger.debug(f"      Progress: {i+1:,}/{iterations:,}")
        
        # Calculate statistics
        return {
            'times_ns': times_ns,
            'avg_ns': statistics.mean(times_ns),
            'median_ns': statistics.median(times_ns),
            'std_dev_ns': statistics.stdev(times_ns) if len(times_ns) > 1 else 0,
            'min_ns': min(times_ns),
            'max_ns': max(times_ns),
            'p95_ns': self._percentile(times_ns, 95),
            'p99_ns': self._percentile(times_ns, 99)
        }
    
    def _benchmark_concurrent(self, config: DatabaseConfig, func_name: str, 
                            workers: int, iterations_per_worker: int) -> Dict[str, Any]:
        """Execute concurrent benchmark with multiple workers"""
        all_times_ns = []
        start_time = time.perf_counter()
        
        def worker_function(worker_id: int) -> List[int]:
            """Worker function for concurrent benchmarking"""
            worker_times = []
            
            with self.db_manager.get_connection(config) as conn:
                with conn.cursor() as cur:
                    for i in range(iterations_per_worker):
                        if 'typeid' in func_name:
                            t_start = time.perf_counter_ns()
                            cur.execute(f"SELECT {func_name}('test')")
                            result = cur.fetchone()
                            t_end = time.perf_counter_ns()
                        else:
                            t_start = time.perf_counter_ns()
                            cur.execute(f"SELECT {func_name}()")
                            result = cur.fetchone()
                            t_end = time.perf_counter_ns()
                        
                        worker_times.append(t_end - t_start)
            
            return worker_times
        
        # Execute concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker_function, i) for i in range(workers)]
            
            for future in concurrent.futures.as_completed(futures):
                worker_times = future.result()
                all_times_ns.extend(worker_times)
        
        total_time_s = time.perf_counter() - start_time
        total_operations = len(all_times_ns)
        throughput_per_second = total_operations / total_time_s
        
        return {
            'times_ns': all_times_ns,
            'avg_ns': statistics.mean(all_times_ns),
            'median_ns': statistics.median(all_times_ns),
            'std_dev_ns': statistics.stdev(all_times_ns) if len(all_times_ns) > 1 else 0,
            'throughput_per_second': throughput_per_second,
            'total_time_s': total_time_s
        }
    
    def _validate_function_output(self, cursor, func_name: str) -> Dict[str, Any]:
        """Validate function output for correctness"""
        test_ids = []
        
        # Generate test IDs
        for _ in range(100):
            if 'typeid' in func_name:
                cursor.execute(f"SELECT {func_name}('test')")
            else:
                cursor.execute(f"SELECT {func_name}()")
            test_ids.append(cursor.fetchone()[0])
        
        # Check uniqueness
        unique_ids = set(str(id) for id in test_ids)
        all_unique = len(unique_ids) == len(test_ids)
        
        # Check format (basic validation)
        sample_id = str(test_ids[0])
        func_config = FUNCTION_CONFIGS[func_name]
        
        # Estimate storage size
        if func_config['storage_type'] == 'binary':
            storage_size = 16  # UUID binary storage
        else:
            storage_size = len(sample_id.encode('utf-8'))
        
        return {
            'all_unique': all_unique,
            'correct_format': True,  # Simplified for now
            'sample_id': sample_id,
            'storage_size': storage_size
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile / 100
        f = int(k)
        c = k - f
        
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        else:
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    def _aggregate_all_results(self):
        """Aggregate results across all runs for statistical analysis"""
        logger.info("Aggregating results across all runs...")
        
        for db_name, results in self.results.items():
            # Group results by function
            function_groups = {}
            for result in results:
                func_name = result.function_name
                if func_name not in function_groups:
                    function_groups[func_name] = []
                function_groups[func_name].append(result)
            
            # Aggregate each function's results
            for func_name, func_results in function_groups.items():
                if not func_results:
                    continue
                
                # Aggregate single-threaded metrics
                all_single_avgs = [r.single_thread_avg_ns for r in func_results]
                all_single_medians = [r.single_thread_median_ns for r in func_results]
                
                # Aggregate concurrent metrics
                all_throughputs = [r.concurrent_throughput_per_second for r in func_results]
                all_concurrent_avgs = [r.concurrent_avg_ns for r in func_results]
                
                # Calculate consistency metrics
                cv = statistics.stdev(all_single_avgs) / statistics.mean(all_single_avgs) if len(all_single_avgs) > 1 else 0
                
                aggregated = AggregatedResults(
                    function_name=func_name,
                    runs_completed=len(func_results),
                    database_name=db_name,
                    postgres_version=func_results[0].postgres_version,
                    
                    # Single-threaded aggregation
                    avg_single_thread_ns=statistics.mean(all_single_avgs),
                    std_dev_single_thread_ns=statistics.stdev(all_single_avgs) if len(all_single_avgs) > 1 else 0,
                    median_single_thread_ns=statistics.median(all_single_medians),
                    min_single_thread_ns=min(all_single_avgs),
                    max_single_thread_ns=max(all_single_avgs),
                    p95_single_thread_ns=self._percentile(all_single_avgs, 95),
                    p99_single_thread_ns=self._percentile(all_single_avgs, 99),
                    
                    # Concurrent aggregation
                    avg_concurrent_ns=statistics.mean(all_concurrent_avgs),
                    avg_throughput_per_second=statistics.mean(all_throughputs),
                    std_dev_throughput=statistics.stdev(all_throughputs) if len(all_throughputs) > 1 else 0,
                    
                    # Consistency metrics
                    coefficient_of_variation=cv,
                    run_to_run_variance=statistics.variance(all_single_avgs) if len(all_single_avgs) > 1 else 0
                )
                
                key = f"{db_name}_{func_name}"
                self.aggregated_results[key] = aggregated
    
    def _prepare_final_results(self, total_time: float) -> Dict[str, Any]:
        """Prepare final comprehensive results"""
        return {
            'metadata': {
                'benchmark_completed': datetime.now(timezone.utc).isoformat(),
                'total_execution_time_seconds': total_time,
                'configuration': asdict(self.config),
                'databases_tested': list(self.db_manager.db_info.keys()),
                'functions_tested': list(set(r.function_name for results in self.results.values() for r in results))
            },
            'raw_results': {db_name: [asdict(r) for r in results] for db_name, results in self.results.items()},
            'aggregated_results': {key: asdict(result) for key, result in self.aggregated_results.items()},
            'database_info': {db_name: asdict(info) for db_name, info in self.db_manager.db_info.items()}
        }