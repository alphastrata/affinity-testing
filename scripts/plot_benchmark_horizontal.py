import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse


def plot_horizontal_bar_chart(csv_path, output_path=None, title="Performance by Core", 
                            xlabel="Performance", ylabel="Core ID"):
    """
    Reads and plots the benchmark data as a horizontal bar chart.
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
    df = df.dropna(subset=['ops_per_second'])  # Remove rows with invalid values

    if df.empty:
        print("Error: No valid data found in CSV file after processing.")
        return

    # Sort by performance for better visualization (ascending for horizontal bars)
    df_sorted = df.sort_values('ops_per_second', ascending=True)

    # Use the first unit found in the data if available
    unit = df_sorted['unit'].dropna().iloc[0] if "unit" in df_sorted.columns else "units"

    # Create horizontal bar chart
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a color gradient based on performance
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))

    # Create horizontal bar chart
    bars = ax.barh(range(len(df_sorted)), df_sorted['ops_per_second'], color=colors)

    # Add value labels on the bars
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['ops_per_second'] * 0.5, i, f"{row['ops_per_second']:.2f}",
                ha='center', va='center', fontweight='bold', color='white')

    # Customize the plot
    ax.set_xlabel(f"{xlabel} ({unit})")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([f"Core {cid}" for cid in df_sorted['core_id']])

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Determine output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = f"{base_name}_horizontal.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Horizontal bar chart saved to {output_path}")

    # Find and report the best performing core
    best_idx = df['ops_per_second'].idxmax()
    best_core_id = df.loc[best_idx, 'core_id']
    best_performance = df.loc[best_idx, 'ops_per_second']
    print(f"Best performing core: Core {best_core_id} with {best_performance:.2f} {unit}")

    plt.show()


def plot_aggregated_horizontal_bars(csv_path, output_path=None, title="Aggregated Benchmark Results"):
    """
    Creates a horizontal bar chart showing aggregated results from different benchmarks.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Benchmark data file not found at '{csv_path}'")
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

    # Sort by performance for better visualization
    df_sorted = df.sort_values('ops_per_second', ascending=True)

    # Create horizontal bar chart
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a color gradient based on performance
    colors = plt.cm.plasma(np.linspace(0, 1, len(df_sorted)))

    # Create horizontal bar chart
    bars = ax.barh(range(len(df_sorted)), df_sorted['ops_per_second'], color=colors)

    # Add value labels on the bars
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['ops_per_second'] * 0.5, i, f"{row['ops_per_second']:.2f}",
                ha='center', va='center', fontweight='bold', color='white')

    # Customize the plot
    unit = df_sorted['unit'].dropna().iloc[0] if "unit" in df_sorted.columns else "units"
    ax.set_xlabel(f"Performance ({unit})")
    ax.set_ylabel("Core ID")
    ax.set_title(title)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([f"Core {cid}" for cid in df_sorted['core_id']])

    # Adjust layout
    plt.tight_layout()

    # Determine output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = f"{base_name}_aggregated_horizontal.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Aggregated horizontal bar chart saved to {output_path}")

    # Find and report the best performing core
    best_idx = df['ops_per_second'].idxmax()
    best_core_id = df.loc[best_idx, 'core_id']
    best_performance = df.loc[best_idx, 'ops_per_second']
    print(f"Best performing core: Core {best_core_id} with {best_performance:.2f} {unit}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create horizontal bar charts from benchmark data")
    parser.add_argument("csv_file", help="Path to the CSV file containing benchmark results")
    parser.add_argument("--output", "-o", help="Output file path for the plot", default=None)
    parser.add_argument("--title", "-t", help="Title for the plot", default="Performance by Core")
    parser.add_argument("--type", "-T", choices=["individual", "aggregated"], 
                       default="individual", help="Type of chart to create")

    args = parser.parse_args()

    if args.type == "individual":
        plot_horizontal_bar_chart(args.csv_file, args.output, args.title)
    elif args.type == "aggregated":
        plot_aggregated_horizontal_bars(args.csv_file, args.output, args.title)