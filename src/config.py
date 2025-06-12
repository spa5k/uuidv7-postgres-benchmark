"""
Professional PostgreSQL UUIDv7 Benchmark Suite - Configuration
Centralized configuration for consistent benchmarking across all environments
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database connection configuration"""

    host: str = "localhost"
    user: str = "postgres"
    password: str = "postgres"
    database: str = "benchmark"
    port: int = 5432


@dataclass
class BenchmarkConfig:
    """Benchmark execution configuration"""

    # Professional benchmark parameters - high iteration counts for accuracy
    warmup_iterations: int = 10000  # 10k warmup runs
    single_thread_iterations: int = 50000  # 50k for high accuracy
    concurrent_workers: int = 10  # More workers for realistic load
    concurrent_iterations_per_worker: int = 5000  # 5k per worker
    benchmark_runs: int = 5  # Run each benchmark 5 times

    # Timeout and safety settings
    function_timeout_seconds: int = 30
    connection_timeout_seconds: int = 10
    max_memory_usage_mb: int = 1024  # 1GB limit

    # Output configuration
    detailed_logging: bool = True
    export_raw_data: bool = True
    generate_charts: bool = True


# Database configurations for PostgreSQL 17 and 18
DATABASE_CONFIGS: Dict[str, DatabaseConfig] = {
    "postgresql17": DatabaseConfig(
        host="localhost",
        port=5434,
        database="benchmark",
        user="postgres",
        password="postgres",
    ),
    "postgresql18": DatabaseConfig(
        host="localhost",
        port=5435,
        database="benchmark",
        user="postgres",
        password="postgres",
    ),
}

# Function configurations for comprehensive testing
FUNCTION_CONFIGS: Dict[str, Dict[str, Any]] = {
    # UUIDv7 Implementations
    "uuid_generate_v7": {
        "name": "UUIDv7 (PL/pgSQL)",
        "description": "PL/pgSQL implementation with overlay method",
        "category": "UUIDv7",
        "pg_versions": [17, 18],
        "storage_type": "binary",
        "expected_length": 36,
        "time_ordered": True,
    },
    "uuidv7_custom": {
        "name": "UUIDv7 (Pure SQL)",
        "description": "Pure SQL implementation with bit manipulation",
        "category": "UUIDv7",
        "pg_versions": [17, 18],
        "storage_type": "binary",
        "expected_length": 36,
        "time_ordered": True,
    },
    "uuidv7_sub_ms": {
        "name": "UUIDv7 (Sub-millisecond)",
        "description": "Sub-millisecond precision implementation",
        "category": "UUIDv7",
        "pg_versions": [17, 18],
        "storage_type": "binary",
        "expected_length": 36,
        "time_ordered": True,
    },
    "uuidv7_native": {
        "name": "Native uuidv7() (PG18)",
        "description": "PostgreSQL 18 native C-level implementation",
        "category": "UUIDv7_Native",
        "pg_versions": [18],
        "storage_type": "binary",
        "expected_length": 36,
        "time_ordered": True,
    },
    # Alternative ID Types
    "ulid_generate": {
        "name": "ULID",
        "description": "Universally Unique Lexicographically Sortable Identifier",
        "category": "Alternative",
        "pg_versions": [17, 18],
        "storage_type": "text",
        "expected_length": 26,
        "time_ordered": True,
    },
    "typeid_generate_text": {
        "name": "TypeID",
        "description": "Type-safe prefixed identifiers",
        "category": "Alternative",
        "pg_versions": [17, 18],
        "storage_type": "text",
        "expected_length": 30,  # Variable based on prefix
        "time_ordered": True,
    },
    # Baseline Comparison
    "gen_random_uuid": {
        "name": "UUIDv4 (Baseline)",
        "description": "PostgreSQL native random UUID generation",
        "category": "Baseline",
        "pg_versions": [17, 18],
        "storage_type": "binary",
        "expected_length": 36,
        "time_ordered": False,
    },
}

# Default benchmark configuration
DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig()

# Output directory structure
OUTPUT_DIRS = {
    "results": "results",
    "charts": "results/charts",
    "raw_data": "results/raw_data",
    "reports": "results/reports",
    "exports": "results/exports",
}

# File name templates
FILE_TEMPLATES = {
    "benchmark_results": "benchmark_results_{timestamp}.json",
    "detailed_results": "detailed_results_{timestamp}.json",
    "summary_report": "benchmark_report_{timestamp}.md",
    "chart_data": "chart_data_{timestamp}.json",
    "performance_summary": "performance_summary_{timestamp}.json",
}

# PostgreSQL optimization settings for benchmarking
POSTGRES_BENCHMARK_SETTINGS = {
    "shared_buffers": "512MB",
    "effective_cache_size": "2GB",
    "work_mem": "8MB",
    "maintenance_work_mem": "128MB",
    "wal_buffers": "32MB",
    "max_wal_size": "8GB",
    "min_wal_size": "2GB",
    "checkpoint_completion_target": "0.9",
    "random_page_cost": "1.1",
    "effective_io_concurrency": "200",
    "max_connections": "300",
    "log_statement": "none",
    "log_min_duration_statement": "-1",
    "track_activities": "off",
    "track_counts": "off",
}


def get_connection_string(config: DatabaseConfig) -> str:
    """Generate connection string from config"""
    return f"postgresql://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}"


def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        # Validate database configs
        for name, config in DATABASE_CONFIGS.items():
            if not all([config.host, config.user, config.database]):
                raise ValueError(f"Invalid database config for {name}")

        # Validate benchmark config
        if DEFAULT_BENCHMARK_CONFIG.warmup_iterations < 1000:
            raise ValueError("Warmup iterations too low for accurate results")

        if DEFAULT_BENCHMARK_CONFIG.single_thread_iterations < 10000:
            raise ValueError(
                "Single thread iterations too low for statistical significance"
            )

        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


# Environment-specific overrides
def load_env_overrides():
    """Load configuration overrides from environment variables"""
    if os.getenv("BENCHMARK_QUICK_MODE"):
        DEFAULT_BENCHMARK_CONFIG.warmup_iterations = 1000
        DEFAULT_BENCHMARK_CONFIG.single_thread_iterations = 5000
        DEFAULT_BENCHMARK_CONFIG.concurrent_iterations_per_worker = 1000
        DEFAULT_BENCHMARK_CONFIG.benchmark_runs = 3

    if os.getenv("BENCHMARK_EXTENSIVE_MODE"):
        DEFAULT_BENCHMARK_CONFIG.warmup_iterations = 25000
        DEFAULT_BENCHMARK_CONFIG.single_thread_iterations = 100000
        DEFAULT_BENCHMARK_CONFIG.concurrent_iterations_per_worker = 10000
        DEFAULT_BENCHMARK_CONFIG.benchmark_runs = 10


# Statistical significance thresholds
STATISTICAL_THRESHOLDS = {
    "min_samples": 1000,
    "confidence_level": 0.95,
    "significant_difference_percent": 5.0,
    "outlier_std_dev_threshold": 3.0,
}
