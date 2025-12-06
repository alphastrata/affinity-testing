import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import argparse


def plot_comprehensive_results(csv_path, output_path=None):
    """
    Creates comprehensive plots with both horizontal bar charts and heatmaps.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Benchmark data file not found at '{csv_path}'")
        print("Please run the benchmark first to generate the CSV file.")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Validate required columns
    required_cols = ["core_id", "ops_per_second", "unit", "validation_status"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: CSV file is missing some required columns: {missing_cols}")
        # Check for old format columns
        required_old_cols = ["core_id", "throughput", "unit"]
        if all(col in df.columns for col in required_old_cols):
            # Map old columns to new ones
            df.rename(columns={'throughput': 'ops_per_second'}, inplace=True)
            if 'validation_status' not in df.columns:
                df['validation_status'] = 'Valid'
        else:
            print(f"Error: CSV file is missing critical columns.")
            return

    # Ensure ops_per_second is numeric
    df['ops_per_second'] = pd.to_numeric(df['ops_per_second'], errors='coerce')
    df = df.dropna(subset=['ops_per_second'])

    if df.empty:
        print("Error: No valid data found in CSV file after processing.")
        return

    # Sort by performance for better visualization
    df_sorted = df.sort_values('ops_per_second', ascending=True)

    # Determine unit
    unit = df_sorted['unit'].dropna().iloc[0] if "unit" in df_sorted.columns else "units"

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Horizontal Bar Chart - Performance by Core
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
    bars = ax1.barh(range(len(df_sorted)), df_sorted['ops_per_second'], color=colors)
    
    # Add value labels on the bars
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax1.text(row['ops_per_second'] * 0.5, i, f"{row['ops_per_second']:.2f}",
                ha='center', va='center', fontweight='bold', color='white')
    
    ax1.set_xlabel(f'Performance ({unit})')
    ax1.set_ylabel('Core ID')
    ax1.set_title('Performance by Core (Horizontal Bar Chart)')
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels([f"Core {cid}" for cid in df_sorted['core_id']])

    # 2. Heatmap - Performance by Core
    performance_matrix = np.array(df_sorted['ops_per_second']).reshape(-1, 1)
    
    sns.heatmap(
        performance_matrix,
        xticklabels=['Performance'],
        yticklabels=[f'Core {cid}' for cid in df_sorted['core_id']],
        annot=True,
        fmt='.2f',
        cmap='viridis',
        cbar_kws={'label': f'Performance ({unit})'},
        ax=ax2,
        linewidths=0.5
    )
    ax2.set_title('Performance Heatmap by Core')

    # 3. Performance Distribution Histogram
    ax3.hist(df['ops_per_second'], bins=min(20, len(df)), color='skyblue', edgecolor='black')
    ax3.set_xlabel(f'Performance ({unit})')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Performance Distribution')
    
    # Add mean line
    mean_perf = df['ops_per_second'].mean()
    ax3.axvline(mean_perf, color='red', linestyle='--', label=f'Mean: {mean_perf:.2f}')
    ax3.legend()

    # 4. Performance vs Core ID Scatter Plot
    scatter = ax4.scatter(df['core_id'], df['ops_per_second'], 
                         c=df['ops_per_second'], cmap='viridis', s=100, alpha=0.7)
    ax4.set_xlabel('Core ID')
    ax4.set_ylabel(f'Performance ({unit})')
    ax4.set_title('Performance vs Core ID')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label(f'Performance ({unit})')

    # Adjust layout
    plt.tight_layout()

    # Determine output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = f"{base_name}_comprehensive.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive plot saved to {output_path}")

    # Find and report the best performing core
    best_idx = df['ops_per_second'].idxmax()
    best_core_id = df.loc[best_idx, 'core_id']
    best_performance = df.loc[best_idx, 'ops_per_second']
    print(f"Best performing core: Core {best_core_id} with {best_performance:.2f} {unit}")

    # Report any validation issues
    invalid_results = df[df['validation_status'].str.contains('Invalid|invalid', na=False)]
    if not invalid_results.empty:
        print(f"Warning: {len(invalid_results)} invalid results detected:")
        for _, row in invalid_results.iterrows():
            print(f"  - Core {row['core_id']}: {row['validation_status']}")

    plt.show()


def standardize_old_format_data(csv_path, output_path=None):
    """
    Converts old format CSV data to new standardized format.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at '{csv_path}'")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Check if this is old format data
    old_format_cols = ['core_id', 'throughput', 'unit']
    new_format_cols = ['core_id', 'ops_per_second', 'unit', 'validation_status', 'execution_time', 'work_completed']
    
    if all(col in df.columns for col in old_format_cols) and not all(col in df.columns for col in new_format_cols):
        print("Detected old format CSV. Converting to new standardized format...")
        
        # Rename columns
        df.rename(columns={'throughput': 'ops_per_second'}, inplace=True)
        
        # Add missing columns with default values
        if 'validation_status' not in df.columns:
            df['validation_status'] = 'Valid'
        if 'execution_time' not in df.columns:
            df['execution_time'] = 1.0  # Placeholder
        if 'work_completed' not in df.columns:
            df['work_completed'] = 0    # Placeholder
            
        # Ensure proper column order
        df = df[['core_id', 'ops_per_second', 'unit', 'validation_status', 'execution_time', 'work_completed']]
        
        # Create standardized output path if none provided
        if output_path is None:
            name_part = os.path.splitext(os.path.basename(csv_path))[0]
            output_path = f"{name_part}_standardized.csv"
        
        df.to_csv(output_path, index=False)
        print(f"Standardized data saved to {output_path}")
        return output_path
    
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create comprehensive plots from benchmark data")
    parser.add_argument("csv_file", help="Path to the CSV file containing benchmark results")
    parser.add_argument("--output", "-o", help="Output file path for the plot", default=None)

    args = parser.parse_args()

    # First, standardize the data if needed
    standardized_csv = standardize_old_format_data(args.csv_file)
    
    # Then create comprehensive plots
    plot_comprehensive_results(standardized_csv, args.output)