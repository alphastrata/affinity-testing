import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import argparse


def create_core_performance_heatmap(csv_path, output_path=None, title="Core Performance Heatmap"):
    """
    Creates a heatmap showing performance across different cores.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Benchmark data file not found at '{csv_path}'")
        print("Please run the benchmark first to generate the CSV file.")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Validate required columns
    required_cols = ["core_id", "ops_per_second", "unit"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: CSV file is missing required columns: {missing_cols}")
        return

    # Ensure ops_per_second is numeric
    df['ops_per_second'] = pd.to_numeric(df['ops_per_second'], errors='coerce')
    df = df.dropna(subset=['ops_per_second'])

    if df.empty:
        print("Error: No valid data found in CSV file after processing.")
        return

    # Sort by core_id to ensure proper ordering
    df_sorted = df.sort_values('core_id')

    # Create a matrix for the heatmap
    # For this we'll create a simplified view where each core is represented 
    num_cores = len(df_sorted)
    core_ids = df_sorted['core_id'].tolist()
    performances = df_sorted['ops_per_second'].tolist()

    # Create a square matrix for visualization (pad if needed)
    # In simple form we'll create a 1D heatmap that shows performance by core
    performance_matrix = np.array(performances).reshape(-1, 1)  # Shape: (num_cores, 1)
    
    # Create the heatmap
    plt.figure(figsize=(10, max(6, num_cores * 0.4)))  # Adjust height based on number of cores
    
    # Use a color map where hotter colors represent higher performance
    ax = sns.heatmap(
        performance_matrix,
        xticklabels=['Performance'],
        yticklabels=[f'Core {cid}' for cid in core_ids],
        annot=True,
        fmt='.2f',
        cmap='viridis',
        cbar_kws={'label': 'Performance (ops/sec)'},
        linewidths=0.5
    )
    
    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0)
    plt.title(title)
    plt.xlabel("")
    
    # Adjust layout
    plt.tight_layout()

    # Determine output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = f"{base_name}_heatmap.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")

    # Find and report the best performing core
    best_idx = df['ops_per_second'].idxmax()
    best_core_id = df.loc[best_idx, 'core_id']
    best_performance = df.loc[best_idx, 'ops_per_second']
    unit = df['unit'].dropna().iloc[0] if "unit" in df.columns else "units"
    print(f"Best performing core: Core {best_core_id} with {best_performance:.2f} {unit}")

    plt.show()


def create_aggregated_heatmap(csv_path, output_path=None, title="Aggregated Benchmark Heatmap"):
    """
    Creates a heatmap comparing performance across different benchmarks/types.
    This function assumes different benchmark types are in the same CSV with a 'benchmark_type' column.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Benchmark data file not found at '{csv_path}'")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Check if there's a benchmark_type column for aggregated view
    if 'benchmark_type' not in df.columns:
        print("Warning: 'benchmark_type' column not found. Creating basic heatmap by core only.")
        create_core_performance_heatmap(csv_path, output_path, title)
        return

    # Validate required columns
    required_cols = ["core_id", "ops_per_second", "benchmark_type", "unit"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: CSV file is missing required columns: {missing_cols}")
        print("Expected columns for aggregated heatmap: 'core_id', 'ops_per_second', 'benchmark_type', 'unit'")
        return

    # Ensure ops_per_second is numeric
    df['ops_per_second'] = pd.to_numeric(df['ops_per_second'], errors='coerce')
    df = df.dropna(subset=['ops_per_second'])

    if df.empty:
        print("Error: No valid data found in CSV file after processing.")
        return

    # Pivot the data to create a matrix with cores as rows and benchmark types as columns
    pivot_df = df.pivot(index='core_id', columns='benchmark_type', values='ops_per_second')

    # Create the heatmap
    plt.figure(figsize=(max(10, len(pivot_df.columns) * 1.2), max(6, len(pivot_df.index) * 0.4)))
    
    # Use a color map where hotter colors represent higher performance
    ax = sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        cbar_kws={'label': 'Performance'},
        linewidths=0.5,
        center=pivot_df.values.flatten().mean()  # Center colorbar on average
    )
    
    plt.title(title)
    plt.xlabel("Benchmark Type")
    plt.ylabel("Core ID")
    
    # Adjust layout
    plt.tight_layout()

    # Determine output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = f"{base_name}_aggregated_heatmap.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Aggregated heatmap saved to {output_path}")

    # Find and report the best performing combination
    best_idx = df['ops_per_second'].idxmax()
    best_core_id = df.loc[best_idx, 'core_id']
    best_benchmark_type = df.loc[best_idx, 'benchmark_type']
    best_performance = df.loc[best_idx, 'ops_per_second']
    unit = df['unit'].dropna().iloc[0] if "unit" in df.columns else "units"
    print(f"Best performing: {best_benchmark_type} on Core {best_core_id} with {best_performance:.2f} {unit}")

    plt.show()


def create_multi_dimensional_heatmap(csv_path, output_path=None, title="Multi-dimensional Performance Heatmap"):
    """
    Creates a more complex heatmap with multiple performance dimensions.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Benchmark data file not found at '{csv_path}'")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # For this we'll create a more complex heatmap if we have additional dimensions
    # like different measurements over time or different metrics
    
    if 'benchmark_type' in df.columns:
        # If we have benchmark types, use the aggregated heatmap function
        create_aggregated_heatmap(csv_path, output_path, title)
    else:
        # Otherwise, create the basic core performance heatmap
        create_core_performance_heatmap(csv_path, output_path, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create heatmaps from benchmark data")
    parser.add_argument("csv_file", help="Path to the CSV file containing benchmark results")
    parser.add_argument("--output", "-o", help="Output file path for the plot", default=None)
    parser.add_argument("--title", "-t", help="Title for the plot", 
                       default="Core Performance Heatmap")
    parser.add_argument("--type", "-T", choices=["core", "aggregated", "multi"], 
                       default="core", help="Type of heatmap to create")

    args = parser.parse_args()

    if args.type == "core":
        create_core_performance_heatmap(args.csv_file, args.output, args.title)
    elif args.type == "aggregated":
        create_aggregated_heatmap(args.csv_file, args.output, args.title)
    elif args.type == "multi":
        create_multi_dimensional_heatmap(args.csv_file, args.output, args.title)