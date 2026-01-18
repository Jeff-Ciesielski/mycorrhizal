#!/usr/bin/env python3
"""
Unified benchmark runner for Mycorrhizal libraries

Provides a convenient CLI for running benchmarks with various options:
- Run all benchmarks or specific libraries
- Generate comparison reports
- Export results to JSON
- Control benchmark iterations and timing
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display results"""
    print(f"\n{'=' * 70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('=' * 70)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nError running: {description}")
        return False

    return True


def run_all_benchmarks(args):
    """Run all benchmarks"""
    cmd = [
        "pytest",
        "src/mycorrhizal/benchmarks/",
        "-v",
        "--benchmark-only",
        "--benchmark-sort=name",
    ]

    if args.json_output:
        cmd.extend(["--benchmark-json", args.json_output])

    if args.histogram:
        cmd.append("--benchmark-histogram")

    return run_command(cmd, "All Benchmarks")


def run_library_benchmarks(args, library):
    """Run benchmarks for a specific library"""
    benchmark_files = {
        "hypha": "src/mycorrhizal/benchmarks/bench_hypha.py",
        "rhizomorph": "src/mycorrhizal/benchmarks/bench_rhizomorph.py",
        "septum": "src/mycorrhizal/benchmarks/bench_septum.py",
    }

    if library not in benchmark_files:
        print(f"Unknown library: {library}")
        print(f"Available libraries: {', '.join(benchmark_files.keys())}")
        return False

    cmd = [
        "pytest",
        benchmark_files[library],
        "-v",
        "--benchmark-only",
        f"--benchmark-sort={args.sort_by}",
    ]

    if args.json_output:
        cmd.extend(["--benchmark-json", args.json_output])

    if args.histogram:
        cmd.append("--benchmark-histogram")

    return run_command(cmd, f"{library.capitalize()} Benchmarks")


def run_comparison(args):
    """Run benchmarks with comparison against previous run"""
    cmd = [
        "pytest",
        "src/mycorrhizal/benchmarks/",
        "-v",
        "--benchmark-only",
        "--benchmark-compare",
    ]

    if args.baseline:
        cmd.extend(["--benchmark-compare-fail", f"min:{args.baseline}"])

    return run_command(cmd, "Benchmark Comparison")


def main():
    parser = argparse.ArgumentParser(
        description="Mycorrhizal Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python run_benchmarks.py

  # Run only Hypha benchmarks
  python run_benchmarks.py --library hypha

  # Run with JSON output
  python run_benchmarks.py --json-output results.json

  # Generate comparison report
  python run_benchmarks.py --compare

  # Run with histogram generation
  python run_benchmarks.py --histogram
        """
    )

    parser.add_argument(
        "--library",
        choices=["hypha", "rhizomorph", "septum"],
        help="Run benchmarks for specific library only"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare against previous benchmark run"
    )

    parser.add_argument(
        "--baseline",
        type=str,
        help="Baseline for comparison (e.g., 'min:5%%' for fail threshold)"
    )

    parser.add_argument(
        "--json-output",
        type=str,
        metavar="FILE",
        help="Export results to JSON file"
    )

    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Generate histogram plots in .benchmarks/ directory"
    )

    parser.add_argument(
        "--sort-by",
        type=str,
        default="name",
        choices=["name", "fullname", "stddev", "min", "max", "mean", "median"],
        help="Sort benchmark results by field (default: name)"
    )

    args = parser.parse_args()

    # Verify we're in the right directory
    if not Path("src/mycorrhizal/benchmarks").exists():
        print("Error: Must run from repository root directory")
        sys.exit(1)

    # Run appropriate benchmarks
    success = True

    if args.compare:
        success = run_comparison(args)
    elif args.library:
        success = run_library_benchmarks(args, args.library)
    else:
        success = run_all_benchmarks(args)

    if not success:
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Benchmark run completed successfully!")
    print("=" * 70)

    if args.json_output:
        print(f"\nResults saved to: {args.json_output}")

    if args.histogram:
        print("\nHistograms saved to: .benchmarks/")


if __name__ == "__main__":
    main()
