#!/usr/bin/env python3
"""Generate benchmark plots from CSV results."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(csv_path: str) -> pd.DataFrame:
    """Load benchmark results from CSV."""
    df = pd.read_csv(csv_path)
    return df


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std for each configuration."""
    stats = df.groupby(['device', 'impl', 'matrix_type', 'm', 'n', 'k',
                        'mblk', 'nblk', 'kblk', 'mmk', 'nmk']).agg({
        'time_ms': ['mean', 'std', 'min', 'max'],
        'gflops': ['mean', 'std', 'min', 'max']
    }).reset_index()
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
    return stats


def plot_gflops_by_size(df: pd.DataFrame, output_dir: Path, device: str = None):
    """Plot GFLOPS vs matrix size for different implementations."""
    stats = compute_statistics(df)

    if device:
        stats = stats[stats['device'] == device]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: GFLOPS by size for random_dense
    ax1 = axes[0]
    for impl in stats['impl'].unique():
        data = stats[(stats['impl'] == impl) & (stats['matrix_type'] == 'random_dense')]
        if len(data) > 0:
            ax1.errorbar(data['m'], data['gflops_mean'], yerr=data['gflops_std'],
                        marker='o', label=impl, capsize=3)
    ax1.set_xlabel('Matrix Size (m=n=k)')
    ax1.set_ylabel('GFLOPS')
    ax1.set_title('Performance by Matrix Size (random_dense)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: GFLOPS by matrix type for largest size
    ax2 = axes[1]
    max_size = stats['m'].max()
    data_max = stats[stats['m'] == max_size]

    x = np.arange(len(data_max['matrix_type'].unique()))
    width = 0.2

    for i, impl in enumerate(sorted(data_max['impl'].unique())):
        impl_data = data_max[data_max['impl'] == impl].sort_values('matrix_type')
        ax2.bar(x + i * width, impl_data['gflops_mean'], width,
               label=impl, yerr=impl_data['gflops_std'], capsize=2)

    ax2.set_xlabel('Matrix Type')
    ax2.set_ylabel('GFLOPS')
    ax2.set_title(f'Performance by Matrix Type (size={max_size})')
    ax2.set_xticks(x + width * (len(data_max['impl'].unique()) - 1) / 2)
    ax2.set_xticklabels(sorted(data_max['matrix_type'].unique()), rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / f'gflops_comparison{"_" + device if device else ""}.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_time_distribution(df: pd.DataFrame, output_dir: Path, device: str = None):
    """Plot time distribution as violin/box plots."""
    if device:
        df = df[df['device'] == device]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter to largest size and random_dense for cleaner plot
    max_size = df['m'].max()
    data = df[(df['m'] == max_size) & (df['matrix_type'] == 'random_dense')]

    impls = sorted(data['impl'].unique())
    positions = np.arange(len(impls))

    box_data = [data[data['impl'] == impl]['time_ms'].values for impl in impls]

    bp = ax.boxplot(box_data, positions=positions, labels=impls, patch_artist=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(impls)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel('Implementation')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Time Distribution (size={max_size}, random_dense)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / f'time_distribution{"_" + device if device else ""}.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_device_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare performance across devices."""
    if len(df['device'].unique()) < 2:
        print("Only one device in data, skipping device comparison plot")
        return

    stats = compute_statistics(df)

    # Filter to random_dense and largest size per device
    max_sizes = stats.groupby('device')['m'].max()

    fig, ax = plt.subplots(figsize=(10, 6))

    devices = sorted(stats['device'].unique())
    x = np.arange(len(devices))
    width = 0.15

    impls = sorted(stats['impl'].unique())

    for i, impl in enumerate(impls):
        values = []
        errors = []
        for device in devices:
            max_size = max_sizes[device]
            data = stats[(stats['device'] == device) &
                        (stats['impl'] == impl) &
                        (stats['m'] == max_size) &
                        (stats['matrix_type'] == 'random_dense')]
            if len(data) > 0:
                values.append(data['gflops_mean'].values[0])
                errors.append(data['gflops_std'].values[0])
            else:
                values.append(0)
                errors.append(0)

        ax.bar(x + i * width, values, width, label=impl, yerr=errors, capsize=2)

    ax.set_xlabel('Device')
    ax.set_ylabel('GFLOPS')
    ax.set_title('Performance by Device (max size, random_dense)')
    ax.set_xticks(x + width * (len(impls) - 1) / 2)
    ax.set_xticklabels(devices, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'device_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark plots')
    parser.add_argument('csv', help='Path to benchmark results CSV')
    parser.add_argument('--output-dir', '-o', default='plots', help='Output directory for plots')
    parser.add_argument('--device', '-d', help='Filter to specific device')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading results from {args.csv}...")
    df = load_results(args.csv)
    print(f"Loaded {len(df)} rows")

    print("\nGenerating plots...")
    plot_gflops_by_size(df, output_dir, args.device)
    plot_time_distribution(df, output_dir, args.device)
    plot_device_comparison(df, output_dir)

    # Save statistics
    stats = compute_statistics(df)
    stats_path = output_dir / 'statistics.csv'
    stats.to_csv(stats_path, index=False)
    print(f"\nSaved statistics to {stats_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
