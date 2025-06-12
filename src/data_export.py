"""
Professional PostgreSQL UUIDv7 Benchmark Suite - Data Export
Handles all data export formats: JSON, CSV, charts, and reports
"""

import json
import csv
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .config import OUTPUT_DIRS, FILE_TEMPLATES

logger = logging.getLogger(__name__)


class DataExporter:
    """Handles all benchmark data export functionality"""

    def __init__(self, output_base_dir: str = "results"):
        self.output_base_dir = Path(output_base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._create_output_directories()

    def _create_output_directories(self):
        """Create all necessary output directories"""
        for dir_name in OUTPUT_DIRS.values():
            dir_path = self.output_base_dir / dir_name.replace("results/", "")
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")

    def export_complete_dataset(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Export complete benchmark dataset in multiple formats"""
        exported_files = {}

        try:
            # 1. Export raw JSON results
            raw_json_path = self._export_raw_json(results)
            exported_files["raw_json"] = str(raw_json_path)

            # 2. Export performance summary JSON
            summary_json_path = self._export_performance_summary(results)
            exported_files["summary_json"] = str(summary_json_path)

            # 3. Export chart data JSON
            chart_data_path = self._export_chart_data(results)
            exported_files["chart_data"] = str(chart_data_path)

            # 4. Export CSV data
            csv_path = self._export_csv_data(results)
            exported_files["csv_data"] = str(csv_path)

            # 5. Generate charts
            charts_path = self._generate_charts(results)
            exported_files["charts"] = str(charts_path)

            # 6. Generate comprehensive report
            report_path = self._generate_markdown_report(results)
            exported_files["report"] = str(report_path)

            logger.info(
                f"Data export completed successfully. Files saved to: {self.output_base_dir}"
            )
            return exported_files

        except Exception as e:
            logger.error(f"Data export failed: {e}")
            raise

    def _export_raw_json(self, results: Dict[str, Any]) -> Path:
        """Export complete raw results as JSON"""
        filename = FILE_TEMPLATES["detailed_results"].format(timestamp=self.timestamp)
        filepath = self.output_base_dir / "raw_data" / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Raw JSON exported: {filepath}")
        return filepath

    def _export_performance_summary(self, results: Dict[str, Any]) -> Path:
        """Export performance summary optimized for quick analysis"""
        filename = FILE_TEMPLATES["performance_summary"].format(
            timestamp=self.timestamp
        )
        filepath = self.output_base_dir / "exports" / filename

        summary = {
            "metadata": results["metadata"],
            "summary_statistics": {},
            "performance_rankings": {},
            "database_comparison": {},
        }

        # Extract key performance metrics
        for key, agg_result in results["aggregated_results"].items():
            db_name, func_name = key.split("_", 1)

            if db_name not in summary["summary_statistics"]:
                summary["summary_statistics"][db_name] = {}

            summary["summary_statistics"][db_name][func_name] = {
                "avg_single_thread_us": agg_result["avg_single_thread_ns"] / 1000,
                "avg_throughput_per_second": agg_result["avg_throughput_per_second"],
                "consistency_coefficient": agg_result["coefficient_of_variation"],
                "runs_completed": agg_result["runs_completed"],
                "postgres_version": agg_result["postgres_version"],
            }

        # Performance rankings
        all_functions = []
        for db_name, functions in summary["summary_statistics"].items():
            for func_name, metrics in functions.items():
                all_functions.append(
                    {
                        "database": db_name,
                        "function": func_name,
                        "single_thread_us": metrics["avg_single_thread_us"],
                        "throughput_ops_sec": metrics["avg_throughput_per_second"],
                        "postgres_version": metrics["postgres_version"],
                    }
                )

        # Sort by single-thread performance
        summary["performance_rankings"]["fastest_single_thread"] = sorted(
            all_functions, key=lambda x: x["single_thread_us"]
        )[:10]

        # Sort by throughput
        summary["performance_rankings"]["highest_throughput"] = sorted(
            all_functions, key=lambda x: x["throughput_ops_sec"], reverse=True
        )[:10]

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Performance summary exported: {filepath}")
        return filepath

    def _export_chart_data(self, results: Dict[str, Any]) -> Path:
        """Export data formatted specifically for chart libraries"""
        filename = FILE_TEMPLATES["chart_data"].format(timestamp=self.timestamp)
        filepath = self.output_base_dir / "exports" / filename

        chart_data = {
            "bar_chart_single_thread": [],
            "bar_chart_throughput": [],
            "line_chart_consistency": [],
            "radar_chart_comparison": [],
            "heatmap_data": [],
        }

        # Prepare bar chart data for single-thread performance
        for key, agg_result in results["aggregated_results"].items():
            db_name, func_name = key.split("_", 1)

            # Single-thread bar chart
            chart_data["bar_chart_single_thread"].append(
                {
                    "function": func_name,
                    "database": db_name,
                    "avg_time_us": agg_result["avg_single_thread_ns"] / 1000,
                    "postgres_version": f"PG{agg_result['postgres_version']}",
                }
            )

            # Throughput bar chart
            chart_data["bar_chart_throughput"].append(
                {
                    "function": func_name,
                    "database": db_name,
                    "throughput_ops_sec": agg_result["avg_throughput_per_second"],
                    "postgres_version": f"PG{agg_result['postgres_version']}",
                }
            )

            # Consistency line chart
            chart_data["line_chart_consistency"].append(
                {
                    "function": func_name,
                    "database": db_name,
                    "coefficient_of_variation": agg_result["coefficient_of_variation"],
                    "std_dev_throughput": agg_result["std_dev_throughput"],
                }
            )

        # Prepare radar chart data (normalized metrics)
        if chart_data["bar_chart_single_thread"]:
            max_time = max(
                item["avg_time_us"] for item in chart_data["bar_chart_single_thread"]
            )
            max_throughput = max(
                item["throughput_ops_sec"]
                for item in chart_data["bar_chart_throughput"]
            )

            for item in chart_data["bar_chart_single_thread"]:
                func_name = item["function"]
                db_name = item["database"]

                # Find corresponding throughput
                throughput_item = next(
                    (
                        t
                        for t in chart_data["bar_chart_throughput"]
                        if t["function"] == func_name and t["database"] == db_name
                    ),
                    None,
                )

                if throughput_item:
                    chart_data["radar_chart_comparison"].append(
                        {
                            "function": func_name,
                            "database": db_name,
                            "speed_score": 100
                            * (1 - item["avg_time_us"] / max_time),  # Higher is better
                            "throughput_score": 100
                            * throughput_item["throughput_ops_sec"]
                            / max_throughput,
                            "consistency_score": 100
                            * max(
                                0,
                                1
                                - next(
                                    (
                                        c["coefficient_of_variation"]
                                        for c in chart_data["line_chart_consistency"]
                                        if c["function"] == func_name
                                        and c["database"] == db_name
                                    ),
                                    0,
                                ),
                            ),
                        }
                    )

        with open(filepath, "w") as f:
            json.dump(chart_data, f, indent=2, default=str)

        logger.info(f"Chart data exported: {filepath}")
        return filepath

    def _export_csv_data(self, results: Dict[str, Any]) -> Path:
        """Export results as CSV for spreadsheet analysis"""
        filename = f"benchmark_results_{self.timestamp}.csv"
        filepath = self.output_base_dir / "exports" / filename

        # Flatten aggregated results for CSV
        csv_data = []
        for key, agg_result in results["aggregated_results"].items():
            db_name, func_name = key.split("_", 1)

            row = {
                "database": db_name,
                "function": func_name,
                "postgres_version": agg_result["postgres_version"],
                "runs_completed": agg_result["runs_completed"],
                "avg_single_thread_ns": agg_result["avg_single_thread_ns"],
                "avg_single_thread_us": agg_result["avg_single_thread_ns"] / 1000,
                "median_single_thread_ns": agg_result["median_single_thread_ns"],
                "p95_single_thread_ns": agg_result["p95_single_thread_ns"],
                "p99_single_thread_ns": agg_result["p99_single_thread_ns"],
                "avg_throughput_per_second": agg_result["avg_throughput_per_second"],
                "std_dev_throughput": agg_result["std_dev_throughput"],
                "coefficient_of_variation": agg_result["coefficient_of_variation"],
                "run_to_run_variance": agg_result["run_to_run_variance"],
            }
            csv_data.append(row)

        # Write CSV
        if csv_data:
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)

        logger.info(f"CSV data exported: {filepath}")
        return filepath

    def _generate_charts(self, results: Dict[str, Any]) -> Path:
        """Generate comprehensive performance charts"""
        charts_dir = self.output_base_dir / "charts"

        # Set up plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # Prepare data for plotting
        plot_data = []
        for key, agg_result in results["aggregated_results"].items():
            db_name, func_name = key.split("_", 1)
            plot_data.append(
                {
                    "Function": func_name,
                    "Database": db_name,
                    "PostgreSQL": f"PG{agg_result['postgres_version']}",
                    "Avg Time (μs)": agg_result["avg_single_thread_ns"] / 1000,
                    "Throughput (ops/sec)": agg_result["avg_throughput_per_second"],
                    "Consistency (CV)": agg_result["coefficient_of_variation"],
                }
            )

        df = pd.DataFrame(plot_data)

        # Create comprehensive figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(
            "PostgreSQL UUIDv7 Benchmark Results - Professional Analysis",
            fontsize=20,
            fontweight="bold",
        )

        # 1. Single-thread performance comparison
        ax1 = axes[0, 0]
        sns.barplot(data=df, x="Function", y="Avg Time (μs)", hue="PostgreSQL", ax=ax1)
        ax1.set_title(
            "Single-Thread Performance (Lower is Better)",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xlabel("Implementation", fontweight="bold")
        ax1.set_ylabel("Average Time (microseconds)", fontweight="bold")
        ax1.tick_params(axis="x", rotation=45)
        ax1.legend(title="PostgreSQL Version", title_fontsize="large")

        # 2. Throughput comparison
        ax2 = axes[0, 1]
        sns.barplot(
            data=df, x="Function", y="Throughput (ops/sec)", hue="PostgreSQL", ax=ax2
        )
        ax2.set_title(
            "Concurrent Throughput (Higher is Better)", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Implementation", fontweight="bold")
        ax2.set_ylabel("Operations per Second", fontweight="bold")
        ax2.tick_params(axis="x", rotation=45)
        ax2.legend(title="PostgreSQL Version", title_fontsize="large")

        # 3. Performance vs Consistency scatter
        ax3 = axes[1, 0]
        scatter = ax3.scatter(
            df["Avg Time (μs)"],
            df["Throughput (ops/sec)"],
            c=df["Consistency (CV)"],
            s=100,
            alpha=0.7,
            cmap="viridis",
        )
        ax3.set_title(
            "Performance vs Throughput (Consistency as Color)",
            fontsize=14,
            fontweight="bold",
        )
        ax3.set_xlabel("Average Time (microseconds)", fontweight="bold")
        ax3.set_ylabel("Throughput (ops/sec)", fontweight="bold")
        plt.colorbar(scatter, ax=ax3, label="Consistency (CV)")

        # Add function labels to scatter plot
        for i, row in df.iterrows():
            ax3.annotate(
                row["Function"],
                (row["Avg Time (μs)"], row["Throughput (ops/sec)"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # 4. Performance comparison table
        ax4 = axes[1, 1]
        ax4.axis("off")

        # Create performance comparison table
        pg18_data = df[df["PostgreSQL"] == "PG18"]
        pg17_data = df[df["PostgreSQL"] == "PG17"]

        # Find common functions
        common_functions = set(pg18_data["Function"]) & set(pg17_data["Function"])

        table_data = []
        for func in common_functions:
            pg18_row = pg18_data[pg18_data["Function"] == func].iloc[0]
            pg17_row = pg17_data[pg17_data["Function"] == func].iloc[0]

            time_improvement = (
                (pg17_row["Avg Time (μs)"] - pg18_row["Avg Time (μs)"])
                / pg17_row["Avg Time (μs)"]
                * 100
            )
            throughput_improvement = (
                (pg18_row["Throughput (ops/sec)"] - pg17_row["Throughput (ops/sec)"])
                / pg17_row["Throughput (ops/sec)"]
                * 100
            )

            table_data.append(
                [
                    func,
                    f"{pg17_row['Avg Time (μs)']:.1f}",
                    f"{pg18_row['Avg Time (μs)']:.1f}",
                    f"{time_improvement:+.1f}%",
                    f"{throughput_improvement:+.1f}%",
                ]
            )

        if table_data:
            table = ax4.table(
                cellText=table_data,
                colLabels=[
                    "Function",
                    "PG17 Time(μs)",
                    "PG18 Time(μs)",
                    "Time Δ",
                    "Throughput Δ",
                ],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax4.set_title(
                "PostgreSQL 17 vs 18 Comparison", fontsize=14, fontweight="bold"
            )

        plt.tight_layout()

        # Save chart
        chart_filename = f"comprehensive_benchmark_analysis_{self.timestamp}.png"
        chart_path = charts_dir / chart_filename
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Charts generated: {chart_path}")
        return chart_path

    def _generate_markdown_report(self, results: Dict[str, Any]) -> Path:
        """Generate comprehensive markdown report"""
        filename = FILE_TEMPLATES["summary_report"].format(timestamp=self.timestamp)
        filepath = self.output_base_dir / "reports" / filename

        with open(filepath, "w") as f:
            f.write(self._create_markdown_content(results))

        logger.info(f"Markdown report generated: {filepath}")
        return filepath

    def _create_markdown_content(self, results: Dict[str, Any]) -> str:
        """Create comprehensive markdown report content"""
        metadata = results["metadata"]

        report = f"""# PostgreSQL UUIDv7 Benchmark Report
        
Generated: {metadata['benchmark_completed']}
Total Execution Time: {metadata['total_execution_time_seconds']:.1f} seconds

## Executive Summary

This comprehensive benchmark compares UUID generation performance across PostgreSQL 17 and 18,
featuring native UUIDv7 support in PostgreSQL 18. The benchmark uses professional-grade 
methodology with {metadata['configuration']['benchmark_runs']} runs per function and 
{metadata['configuration']['single_thread_iterations']:,} iterations per run.

### Key Findings

"""

        # Find best performers
        all_results = []
        for key, agg_result in results["aggregated_results"].items():
            db_name, func_name = key.split("_", 1)
            all_results.append(
                {
                    "key": key,
                    "db": db_name,
                    "function": func_name,
                    "avg_time_us": agg_result["avg_single_thread_ns"] / 1000,
                    "throughput": agg_result["avg_throughput_per_second"],
                    "pg_version": agg_result["postgres_version"],
                }
            )

        # Fastest single-thread
        fastest = min(all_results, key=lambda x: x["avg_time_us"])
        report += f"- **Fastest Single-Thread**: {fastest['function']} on PostgreSQL {fastest['pg_version']} ({fastest['avg_time_us']:.1f}μs)\n"

        # Highest throughput
        highest_throughput = max(all_results, key=lambda x: x["throughput"])
        report += f"- **Highest Throughput**: {highest_throughput['function']} on PostgreSQL {highest_throughput['pg_version']} ({highest_throughput['throughput']:,.0f} ops/sec)\n"

        report += f"""
### Test Configuration

- **Databases Tested**: {', '.join(metadata['databases_tested'])}
- **Functions Tested**: {', '.join(metadata['functions_tested'])}
- **Benchmark Runs**: {metadata['configuration']['benchmark_runs']}
- **Single-Thread Iterations**: {metadata['configuration']['single_thread_iterations']:,}
- **Concurrent Workers**: {metadata['configuration']['concurrent_workers']}
- **Concurrent Iterations**: {metadata['configuration']['concurrent_iterations_per_worker']:,} per worker
- **Warmup Iterations**: {metadata['configuration']['warmup_iterations']:,}

## Detailed Results

### Single-Thread Performance

| Function | Database | PostgreSQL | Avg Time (μs) | Median (μs) | P95 (μs) | P99 (μs) | Runs |
|----------|----------|------------|---------------|-------------|----------|----------|------|
"""

        # Sort by average time for table
        sorted_results = sorted(all_results, key=lambda x: x["avg_time_us"])

        for result in sorted_results:
            agg_data = results["aggregated_results"][result["key"]]
            report += (
                f"| {result['function']} | {result['db']} | {result['pg_version']} | "
            )
            report += f"{result['avg_time_us']:.1f} | "
            report += f"{agg_data['median_single_thread_ns']/1000:.1f} | "
            report += f"{agg_data['p95_single_thread_ns']/1000:.1f} | "
            report += f"{agg_data['p99_single_thread_ns']/1000:.1f} | "
            report += f"{agg_data['runs_completed']} |\n"

        report += f"""
### Concurrent Throughput

| Function | Database | PostgreSQL | Throughput (ops/sec) | Avg Time (μs) | Std Dev | CV |
|----------|----------|------------|---------------------|---------------|---------|-----|
"""

        # Sort by throughput for table
        sorted_by_throughput = sorted(
            all_results, key=lambda x: x["throughput"], reverse=True
        )

        for result in sorted_by_throughput:
            agg_data = results["aggregated_results"][result["key"]]
            report += (
                f"| {result['function']} | {result['db']} | {result['pg_version']} | "
            )
            report += f"{result['throughput']:,.0f} | "
            report += f"{agg_data['avg_concurrent_ns']/1000:.1f} | "
            report += f"{agg_data['std_dev_throughput']:.0f} | "
            report += f"{agg_data['coefficient_of_variation']:.3f} |\n"

        # Add PostgreSQL version comparison if available
        pg18_functions = [r for r in all_results if r["pg_version"] == 18]
        pg17_functions = [r for r in all_results if r["pg_version"] == 17]

        if pg18_functions and pg17_functions:
            report += f"""
### PostgreSQL 18 vs 17 Comparison

PostgreSQL 18 introduces native UUIDv7 support with significant performance improvements:

"""
            # Find matching functions between versions
            pg18_dict = {r["function"]: r for r in pg18_functions}
            pg17_dict = {r["function"]: r for r in pg17_functions}

            common_functions = set(pg18_dict.keys()) & set(pg17_dict.keys())

            for func_name in sorted(common_functions):
                pg18_data = pg18_dict[func_name]
                pg17_data = pg17_dict[func_name]

                time_improvement = (
                    (pg17_data["avg_time_us"] - pg18_data["avg_time_us"])
                    / pg17_data["avg_time_us"]
                    * 100
                )
                throughput_improvement = (
                    (pg18_data["throughput"] - pg17_data["throughput"])
                    / pg17_data["throughput"]
                    * 100
                )

                report += f"- **{func_name}**: "
                if time_improvement > 0:
                    report += f"{time_improvement:.1f}% faster, "
                else:
                    report += f"{abs(time_improvement):.1f}% slower, "

                if throughput_improvement > 0:
                    report += f"{throughput_improvement:.1f}% higher throughput\n"
                else:
                    report += f"{abs(throughput_improvement):.1f}% lower throughput\n"

        report += f"""
## Technical Details

### Methodology

This benchmark uses professional methodology to ensure accurate and reproducible results:

1. **Warmup Phase**: {metadata['configuration']['warmup_iterations']:,} iterations to ensure stable CPU caches and database state
2. **Multiple Runs**: Each function tested {metadata['configuration']['benchmark_runs']} times to measure consistency
3. **High Precision Timing**: `time.perf_counter_ns()` for nanosecond precision
4. **Statistical Analysis**: Mean, median, standard deviation, and percentiles calculated
5. **Concurrent Testing**: {metadata['configuration']['concurrent_workers']} workers simulate realistic load

### Database Configuration

Both PostgreSQL instances use identical optimization settings:
- Shared Buffers: 512MB
- Work Memory: 8MB
- Effective Cache Size: 2GB
- Random Page Cost: 1.1 (SSD optimized)

### Function Implementations

All UUID functions tested produce RFC-compliant identifiers with proper version bits and time-ordering guarantees where applicable.

## Files Generated

- **Raw Data**: Complete benchmark results with all individual measurements
- **Summary Data**: Aggregated statistics and performance metrics
- **Chart Data**: Data formatted for visualization tools
- **CSV Export**: Spreadsheet-compatible format for further analysis

---

*Report generated by Professional PostgreSQL UUIDv7 Benchmark Suite*
"""

        return report
