"""
Professional PostgreSQL UUIDv7 Benchmark Suite - Database Management
Handles all database connections, initialization, and function validation
"""

import time
import psycopg
import logging
from typing import Dict, List, Tuple, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass

from .config import DatabaseConfig, DATABASE_CONFIGS, FUNCTION_CONFIGS

logger = logging.getLogger(__name__)


@dataclass
class DatabaseInfo:
    """Database version and capability information"""

    version_string: str
    version_number: int
    major_version: int
    has_native_uuidv7: bool
    supported_functions: List[str]


class DatabaseManager:
    """Manages database connections and initialization"""

    def __init__(self):
        self.db_info: Dict[str, DatabaseInfo] = {}
        self.connection_pools: Dict[str, Any] = {}

    @contextmanager
    def get_connection(self, config: DatabaseConfig):
        """Get a database connection with proper error handling"""
        conn = None
        try:
            # Use conninfo string format for psycopg3 compatibility
            conninfo = f"host={config.host} port={config.port} dbname={config.database} user={config.user} password={config.password} connect_timeout=10"
            conn = psycopg.connect(conninfo)
            yield conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def check_database_info(self, config: DatabaseConfig) -> DatabaseInfo:
        """Get comprehensive database information and capabilities"""
        with self.get_connection(config) as conn:
            with conn.cursor() as cur:
                # Get version information
                cur.execute(
                    "SELECT version(), current_setting('server_version_num')::INTEGER"
                )
                version_string, version_number = cur.fetchone()
                major_version = version_number // 10000

                # Check native UUIDv7 support (PostgreSQL 18+)
                has_native_uuidv7 = False
                if version_number >= 180000:
                    try:
                        cur.execute("SELECT uuidv7()")
                        has_native_uuidv7 = True
                        logger.info(
                            f"Native uuidv7() support confirmed for PostgreSQL {major_version}"
                        )
                    except Exception:
                        logger.warning(
                            f"PostgreSQL {major_version} detected but native uuidv7() not available"
                        )

                # Test which functions are available
                supported_functions = self._test_function_availability(
                    cur, major_version
                )

                return DatabaseInfo(
                    version_string=version_string,
                    version_number=version_number,
                    major_version=major_version,
                    has_native_uuidv7=has_native_uuidv7,
                    supported_functions=supported_functions,
                )

    def _test_function_availability(self, cursor, pg_version: int) -> List[str]:
        """Test which benchmark functions are available"""
        available_functions = []

        for func_name, func_config in FUNCTION_CONFIGS.items():
            if pg_version not in func_config["pg_versions"]:
                continue

            try:
                # Test function with appropriate parameters
                if func_name == "uuidv7_native":
                    if pg_version >= 18:
                        cursor.execute("SELECT uuidv7_native()")
                    else:
                        continue
                elif "typeid" in func_name:
                    cursor.execute(f"SELECT {func_name}('test')")
                else:
                    cursor.execute(f"SELECT {func_name}()")

                available_functions.append(func_name)
                logger.debug(f"Function {func_name} available")

            except Exception as e:
                logger.warning(f"Function {func_name} not available: {e}")

        return available_functions

    def initialize_all_databases(self) -> Dict[str, DatabaseInfo]:
        """Initialize and validate all configured databases"""
        logger.info("Initializing databases...")

        for db_name, config in DATABASE_CONFIGS.items():
            try:
                logger.info(f"Checking {db_name} ({config.host}:{config.port})...")

                # Wait for database to be ready
                self._wait_for_database(config)

                # Get database info
                db_info = self.check_database_info(config)
                self.db_info[db_name] = db_info

                logger.info(
                    f"{db_name}: PostgreSQL {db_info.major_version} "
                    f"({len(db_info.supported_functions)} functions available)"
                )

                if db_info.has_native_uuidv7:
                    logger.info(f"{db_name}: Native uuidv7() support confirmed")

            except Exception as e:
                logger.error(f"Failed to initialize {db_name}: {e}")
                raise

        return self.db_info

    def _wait_for_database(self, config: DatabaseConfig, max_attempts: int = 30):
        """Wait for database to be ready with retries"""
        for attempt in range(max_attempts):
            try:
                with self.get_connection(config) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                logger.info(f"Database ready at {config.host}:{config.port}")
                return
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.info(
                        f"Waiting for database... (attempt {attempt + 1}/{max_attempts})"
                    )
                    time.sleep(2)
                else:
                    logger.error(
                        f"Database not ready after {max_attempts} attempts: {e}"
                    )
                    raise

    def validate_function_correctness(self, db_name: str) -> Dict[str, Any]:
        """Validate that all functions produce correct output"""
        config = DATABASE_CONFIGS[db_name]
        validation_results = {}

        with self.get_connection(config) as conn:
            with conn.cursor() as cur:
                db_info = self.db_info[db_name]

                for func_name in db_info.supported_functions:
                    func_config = FUNCTION_CONFIGS[func_name]

                    try:
                        # Generate test IDs
                        test_ids = []
                        for _ in range(100):  # Generate 100 test IDs
                            if "typeid" in func_name:
                                cur.execute(f"SELECT {func_name}('test')")
                            else:
                                cur.execute(f"SELECT {func_name}()")
                            test_ids.append(cur.fetchone()[0])

                        # Validate properties
                        validation = {
                            "function": func_name,
                            "samples_generated": len(test_ids),
                            "all_unique": len(set(str(id) for id in test_ids))
                            == len(test_ids),
                            "correct_format": self._validate_id_format(
                                test_ids[0], func_config
                            ),
                            "time_ordered": self._validate_time_ordering(
                                test_ids, func_config
                            )
                            if func_config["time_ordered"]
                            else None,
                            "sample_id": str(test_ids[0]),
                            "length_consistent": all(
                                len(str(id)) == len(str(test_ids[0]))
                                for id in test_ids[:10]
                            ),
                        }

                        validation_results[func_name] = validation

                        if not validation["all_unique"]:
                            logger.warning(f"{func_name}: Generated duplicate IDs!")
                        if not validation["correct_format"]:
                            logger.warning(f"{func_name}: Incorrect ID format!")

                    except Exception as e:
                        validation_results[func_name] = {
                            "function": func_name,
                            "error": str(e),
                            "valid": False,
                        }
                        logger.error(f"Validation failed for {func_name}: {e}")

        return validation_results

    def _validate_id_format(self, id_value: Any, func_config: Dict[str, Any]) -> bool:
        """Validate that ID matches expected format"""
        try:
            id_str = str(id_value)

            # UUID format validation
            if func_config["category"] in ["UUIDv7", "UUIDv7_Native", "Baseline"]:
                import uuid

                # Remove hyphens and check hex
                hex_part = id_str.replace("-", "")
                if len(hex_part) != 32:
                    return False
                int(hex_part, 16)  # Validate hex format
                return True

            # ULID format validation
            elif func_config["name"] == "ULID":
                if len(id_str) != 26:
                    return False
                # ULID uses Crockford Base32
                valid_chars = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
                return all(c in valid_chars for c in id_str)

            # TypeID format validation
            elif func_config["name"] == "TypeID":
                if "_" not in id_str:
                    return False
                prefix, suffix = id_str.split("_", 1)
                return len(suffix) == 26 and len(prefix) > 0

            return True

        except Exception:
            return False

    def _validate_time_ordering(
        self, ids: List[Any], func_config: Dict[str, Any]
    ) -> bool:
        """Validate that IDs are time-ordered (basic check)"""
        try:
            # For time-ordered IDs, later generated should generally sort later
            # This is a simplified check - real validation would need timestamp extraction
            id_strings = [str(id) for id in ids]

            # Check if first few IDs are in ascending order (allowing for some variation)
            ascending_count = 0
            for i in range(1, min(10, len(id_strings))):
                if id_strings[i] >= id_strings[i - 1]:
                    ascending_count += 1

            # Allow some variance due to rapid generation
            return ascending_count >= 6  # At least 60% should be in order

        except Exception:
            return False

    def optimize_for_benchmarking(self, db_name: str):
        """Apply PostgreSQL optimizations for accurate benchmarking"""
        config = DATABASE_CONFIGS[db_name]

        optimization_settings = [
            "SET work_mem = '8MB'",
            "SET maintenance_work_mem = '128MB'",
            "SET effective_cache_size = '2GB'",
            "SET random_page_cost = 1.1",
            "SET effective_io_concurrency = 200",
            "SET track_activities = off",
            "SET track_counts = off",
            "SET log_statement = 'none'",
            "SET log_min_duration_statement = -1",
        ]

        with self.get_connection(config) as conn:
            with conn.cursor() as cur:
                for setting in optimization_settings:
                    try:
                        cur.execute(setting)
                        logger.debug(f"Applied: {setting}")
                    except Exception as e:
                        logger.warning(f"Could not apply {setting}: {e}")

                conn.commit()

        logger.info(f"Applied benchmark optimizations to {db_name}")

    def get_system_stats(self, db_name: str) -> Dict[str, Any]:
        """Get system and database statistics"""
        config = DATABASE_CONFIGS[db_name]

        with self.get_connection(config) as conn:
            with conn.cursor() as cur:
                # Get various system stats
                stats = {}

                # Database size
                cur.execute("SELECT pg_database_size(current_database())")
                stats["database_size_bytes"] = cur.fetchone()[0]

                # Cache hit ratio
                cur.execute("""
                    SELECT round(
                        100.0 * sum(blks_hit) / (sum(blks_hit) + sum(blks_read)), 2
                    ) as cache_hit_ratio
                    FROM pg_stat_database 
                    WHERE datname = current_database()
                """)
                result = cur.fetchone()
                stats["cache_hit_ratio"] = result[0] if result[0] else 0

                # Current connections
                cur.execute("SELECT count(*) FROM pg_stat_activity")
                stats["active_connections"] = cur.fetchone()[0]

                # Memory settings
                cur.execute("SELECT current_setting('shared_buffers')")
                stats["shared_buffers"] = cur.fetchone()[0]

                cur.execute("SELECT current_setting('work_mem')")
                stats["work_mem"] = cur.fetchone()[0]

                return stats
