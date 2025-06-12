#!/usr/bin/env python3
"""
Professional PostgreSQL UUIDv7 Benchmark Suite
High-precision, statistically significant benchmark with comprehensive reporting

This is the main benchmark script that provides professional-grade performance
analysis of UUID generation functions across PostgreSQL 17 and 18.

Features:
- 10,000+ warmup iterations per function
- 50,000+ measurement iterations for statistical significance
- 5 complete benchmark runs per function
- Comprehensive statistical analysis
- Professional reporting and data export
- Optimized database configurations
"""

import sys
import logging
import argparse
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import (
    DATABASE_CONFIGS,
    DEFAULT_BENCHMARK_CONFIG,
    validate_config,
    load_env_overrides,
)
from src.database import DatabaseManager
from src.benchmark_engine import BenchmarkEngine
from src.data_export import DataExporter


def setup_logging(verbose: bool = False):
    """Configure professional logging"""
    level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    log_file = (
        Path("results") / "logs" / f'benchmark_{time.strftime("%Y%m%d_%H%M%S")}.log'
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Reduce noise from matplotlib
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def print_banner():
    """Print professional benchmark banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                Professional PostgreSQL UUIDv7 Benchmark Suite               ║
║                                                                              ║
║  High-Precision Performance Analysis with Statistical Significance Testing  ║
║                                                                              ║
║  • 10,000+ warmup iterations per function                                   ║
║  • 50,000+ measurement iterations for accuracy                              ║
║  • 5 complete benchmark runs for consistency analysis                       ║
║  • Professional reporting and data export                                   ║
║  • PostgreSQL 17 vs 18 comparison with native UUIDv7                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def validate_environment():
    """Validate that the environment is ready for benchmarking"""
    logger = logging.getLogger(__name__)

    # Check configuration
    if not validate_config():
        logger.error("Configuration validation failed")
        return False

    # Check if Docker containers are running
    db_manager = DatabaseManager()

    for db_name, config in DATABASE_CONFIGS.items():
        try:
            logger.info(f"Testing connection to {db_name}...")
            with db_manager.get_connection(config) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            logger.info(f"✅ {db_name} connection successful")
        except Exception as e:
            logger.error(f"❌ Failed to connect to {db_name}: {e}")
            logger.error(
                "Please ensure Docker containers are running: docker-compose up -d"
            )
            return False

    return True


def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(
        description="Professional PostgreSQL UUIDv7 Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python professional_benchmark.py                 # Full benchmark
  python professional_benchmark.py --quick         # Quick mode (fewer iterations)
  python professional_benchmark.py --extensive     # Extensive mode (more iterations)
  python professional_benchmark.py --verbose       # Detailed logging
  
Environment Variables:
  BENCHMARK_QUICK_MODE=1                           # Enable quick mode
  BENCHMARK_EXTENSIVE_MODE=1                       # Enable extensive mode
        """,
    )

    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (5K iterations, 3 runs)"
    )
    parser.add_argument(
        "--extensive",
        action="store_true",
        help="Extensive mode (100K iterations, 10 runs)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-charts", action="store_true", help="Skip chart generation"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Print banner
    print_banner()

    # Load environment overrides
    if args.quick:
        import os

        os.environ["BENCHMARK_QUICK_MODE"] = "1"
    elif args.extensive:
        import os

        os.environ["BENCHMARK_EXTENSIVE_MODE"] = "1"

    load_env_overrides()

    # Log configuration
    config = DEFAULT_BENCHMARK_CONFIG
    logger.info("Benchmark Configuration:")
    logger.info(f"  Warmup Iterations: {config.warmup_iterations:,}")
    logger.info(f"  Single-Thread Iterations: {config.single_thread_iterations:,}")
    logger.info(f"  Concurrent Workers: {config.concurrent_workers}")
    logger.info(
        f"  Concurrent Iterations per Worker: {config.concurrent_iterations_per_worker:,}"
    )
    logger.info(f"  Benchmark Runs: {config.benchmark_runs}")
    logger.info(f"  Output Directory: {args.output_dir}")

    try:
        # Validate environment
        logger.info("Validating environment...")
        if not validate_environment():
            logger.error("Environment validation failed. Exiting.")
            return 1

        # Initialize benchmark engine
        logger.info("Initializing benchmark engine...")
        benchmark_engine = BenchmarkEngine(config)

        # Run benchmark
        logger.info("Starting benchmark execution...")
        benchmark_start_time = time.time()

        results = benchmark_engine.run_complete_benchmark(DATABASE_CONFIGS)

        benchmark_duration = time.time() - benchmark_start_time
        logger.info(f"Benchmark completed in {benchmark_duration:.1f} seconds")

        # Export results
        logger.info("Exporting results...")
        exporter = DataExporter(args.output_dir)

        if args.no_charts:
            # Temporarily disable chart generation
            original_generate_charts = exporter._generate_charts
            exporter._generate_charts = lambda x: Path("charts_disabled")

        exported_files = exporter.export_complete_dataset(results)

        # Print summary
        print("\n" + "=" * 80)
        print("🎉 BENCHMARK COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        # Print key results
        if results["aggregated_results"]:
            print("\n📊 TOP PERFORMERS:")

            # Find best single-thread performer
            all_functions = []
            for key, agg_result in results["aggregated_results"].items():
                db_name, func_name = key.split("_", 1)
                all_functions.append(
                    {
                        "name": func_name,
                        "db": db_name,
                        "pg_version": agg_result["postgres_version"],
                        "avg_time_us": agg_result["avg_single_thread_ns"] / 1000,
                        "throughput": agg_result["avg_throughput_per_second"],
                    }
                )

            fastest = min(all_functions, key=lambda x: x["avg_time_us"])
            highest_throughput = max(all_functions, key=lambda x: x["throughput"])

            print(
                f"  🚀 Fastest Single-Thread: {fastest['name']} on PG{fastest['pg_version']} ({fastest['avg_time_us']:.1f}μs)"
            )
            print(
                f"  ⚡ Highest Throughput: {highest_throughput['name']} on PG{highest_throughput['pg_version']} ({highest_throughput['throughput']:,.0f} ops/sec)"
            )

        print(f"\n📁 FILES GENERATED:")
        for file_type, filepath in exported_files.items():
            if filepath != "charts_disabled":
                print(f"  {file_type}: {filepath}")

        print(f"\n📈 DETAILED RESULTS: {args.output_dir}/reports/")
        print(f"📊 CHARTS: {args.output_dir}/charts/")
        print(f"💾 RAW DATA: {args.output_dir}/raw_data/")
        print(f"📋 CSV EXPORT: {args.output_dir}/exports/")

        logger.info("Benchmark suite completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        print("\n⚠️  Benchmark interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        print(f"\n❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
