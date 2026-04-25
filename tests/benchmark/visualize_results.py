#!/usr/bin/env python3
"""
Visualize benchmark results.

Creates comparison charts for:
- Latency comparison
- Throughput comparison
- Accuracy comparison
"""

import json
import sys
from pathlib import Path
import argparse

def load_results(results_dir):
    """Load all benchmark results from directory."""
    results_dir = Path(results_dir)
    all_results = {}

    for json_file in results_dir.glob('benchmark_*.json'):
        with open(json_file) as f:
            data = json.load(f)
            device = data['device'].get('cpu', data['device']['machine'])
            # Clean up device name
            if 'Apple' in device or 'M4' in device:
                device = 'M4 Pro'
            elif 'arm' in device.lower():
                device = 'ARM Device'
            all_results[device] = data

    return all_results


def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)

    for device, data in all_results.items():
        print(f"\n{device}")
        print("-" * 60)
        print(f"{'Runtime':<20} {'Latency':<12} {'Throughput':<12} {'Accuracy':<10}")
        print("-" * 60)

        for r in data['results']:
            print(f"{r['runtime']:<20} "
                  f"{r['latency_ms']:.2f} ms     "
                  f"{r['throughput_ips']:.1f}/s     "
                  f"{r['accuracy_pct']:.2f}%")


def generate_markdown_report(all_results, output_file):
    """Generate a markdown report."""
    with open(output_file, 'w') as f:
        f.write("# TBN Benchmark Results\n\n")
        f.write(f"Generated: {Path(output_file).stat().st_mtime}\n\n")

        for device, data in all_results.items():
            f.write(f"## {device}\n\n")
            f.write("| Runtime | Latency (ms) | Throughput (img/s) | Accuracy (%) |\n")
            f.write("|---------|--------------|--------------------|--------------| " + "|\n".join(
                f"| {r['runtime']} | {r['latency_ms']:.2f} | {r['throughput_ips']:.1f} | {r['accuracy_pct']:.2f} |"
                for r in data['results']
            ) + "\n\n")

    print(f"\nReport saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--input', '-i', type=str, default='results',
                        help='Input directory with benchmark results')
    parser.add_argument('--output', '-o', type=str, default='benchmark_report.md',
                        help='Output markdown file')

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    results_dir = script_dir / args.input
    output_file = script_dir / args.output

    all_results = load_results(results_dir)

    if not all_results:
        print("No results found. Run benchmarks first.")
        return

    print_comparison_table(all_results)
    generate_markdown_report(all_results, output_file)


if __name__ == '__main__':
    main()
