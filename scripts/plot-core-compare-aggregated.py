import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def aggregate_all_benchmarks():
    """
    Creates an aggregated bar chart comparing the total performance across all benchmark types per core.
    """
    # Define all the benchmark files and their names
    benchmark_files = [
        ("simple-math-stats.csv", "Simple Math", "blue"),
        ("division-math-stats.csv", "Division Math", "orange"), 
        ("simd-math-stats.csv", "SIMD Math", "green"),
        ("simd-math-arm-stats.csv", "NEON SIMD", "red"),
        ("avx512-math-stats.csv", "AVX512 Math", "purple"),
        ("avx512-math-arm-stats.csv", "NEON AVX512", "brown"),
        ("matrix-math-stats.csv", "Matrix Math", "pink"),
        ("neon-add-stats.csv", "NEON Add", "gray"),
        ("neon-advanced-stats.csv", "NEON Advanced", "olive"),
        ("neon-integer-stats.csv", "NEON Integer", "cyan"),
    ]
    
    # Dictionary to store data for each core
    core_data = {}
    available_files = []
    
    # Load data from each benchmark file
    for filename, name, color in benchmark_files:
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                df.columns = df.columns.str.strip()
                
                if "core_id" in df.columns and "throughput" in df.columns:
                    print(f"Loaded {name}: {len(df)} core entries")
                    available_files.append((filename, name, color))
                    
                    # Add data for each core in this benchmark
                    for _, row in df.iterrows():
                        core_id = int(row["core_id"])
                        throughput = float(row["throughput"])
                        
                        if core_id not in core_data:
                            core_data[core_id] = {}
                        core_data[core_id][name] = throughput
                else:
                    print(f"Warning: {filename} missing required columns")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"Warning: File {filename} not found")
    
    if not core_data:
        print("No benchmark data files found. Make sure to run benchmarks first.")
        print("Example commands to run benchmarks:")
        print("  ./target/release/simple-math --output simple-math-stats.csv")
        print("  ./target/release/division-math --output division-math-stats.csv")
        print("  ./target/release/matrix-math --output matrix-math-stats.csv")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(core_data, orient='index')
    df.index.name = 'core_id'
    df = df.reset_index()
    
    # Calculate aggregate score for each core (sum of all benchmark performances)
    # Only include columns that are benchmark results (not the core_id)
    benchmark_columns = [col for col in df.columns if col != 'core_id']
    
    # Fill NaN values with 0 (cores that weren't tested in certain benchmarks)
    df[benchmark_columns] = df[benchmark_columns].fillna(0)
    
    # Calculate aggregate score - sum of all benchmark performances for each core
    df['aggregate_score'] = df[benchmark_columns].sum(axis=1)
    
    # Sort by aggregate score for better visualization
    df_sorted = df.sort_values('aggregate_score', ascending=False)
    
    print(f"\nTop performing cores by aggregate score:")
    for _, row in df_sorted.head(5).iterrows():
        print(f"  Core {int(row['core_id'])}: {row['aggregate_score']:.2e}")
    
    # Create the plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create color mapping based on performance ranking
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
    
    bars = ax.bar(range(len(df_sorted)), df_sorted['aggregate_score'], color=colors)
    
    # Customize the plot
    ax.set_xlabel("Core ID (sorted by aggregate performance)", fontsize=12)
    ax.set_ylabel("Aggregate Performance Score", fontsize=12)
    ax.set_title("Aggregate Core Performance Across All Benchmark Types", fontsize=14, fontweight='bold')
    
    # Set x-axis labels to show core IDs
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels([f"{int(cid)}" for cid in df_sorted['core_id']], rotation=45)
    
    # Add values as text on top of bars (only for the top 10 to avoid clutter)
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        if i < 10:  # Only label top 10 for readability
            ax.text(i, row['aggregate_score'], f"{row['aggregate_score']:.1e}", 
                   ha='center', va='bottom', rotation=90, fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add unit explanation
    ax.text(0.02, 0.98, '1e9 = 1 billion, 1e6 = 1 million, etc.\n(Aggregate of all benchmark types)', 
             transform=ax.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "core-aggregate-performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nAggregate performance chart saved to: {output_path}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"- Total cores tested: {len(df_sorted)}")
    print(f"- Best performing core: Core {int(df_sorted.iloc[0]['core_id'])} with score {df_sorted.iloc[0]['aggregate_score']:.2e}")
    print(f"- Worst performing core: Core {int(df_sorted.iloc[-1]['core_id'])} with score {df_sorted.iloc[-1]['aggregate_score']:.2e}")
    print(f"- Average performance: {df_sorted['aggregate_score'].mean():.2e}")
    
    # Also create a detailed breakdown visualization
    create_detailed_breakdown(df, benchmark_columns, available_files)


def create_detailed_breakdown(df, benchmark_columns, available_files):
    """
    Creates a detailed breakdown visualization showing the contribution of each benchmark to the aggregate score.
    """
    if not benchmark_columns:
        return
        
    # Create a stacked bar chart showing contribution of each benchmark to the aggregate score
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Prepare data for stacked bar chart
    # Sort cores by aggregate performance
    df_sorted = df.sort_values('aggregate_score', ascending=False)
    
    # Create a mapping from benchmark name to color based on file information
    color_map = {}
    for _, name, color in available_files:
        color_map[name] = color
    
    # Use default colors if specific color not available
    default_colors = plt.cm.tab20(np.linspace(0, 1, len(benchmark_columns)))
    
    bottom = np.zeros(len(df_sorted))
    
    for i, benchmark in enumerate(benchmark_columns):
        values = df_sorted[benchmark].values
        color = color_map.get(benchmark, default_colors[i % len(default_colors)])
        ax.bar(range(len(df_sorted)), values, bottom=bottom, 
               label=benchmark, color=color, alpha=0.8)
        bottom += values
    
    ax.set_xlabel("Core ID (sorted by aggregate performance)", fontsize=12)
    ax.set_ylabel("Performance Score", fontsize=12)
    ax.set_title("Detailed Breakdown of Core Performance by Benchmark Type", fontsize=14, fontweight='bold')
    
    # Set x-axis labels to show core IDs
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels([f"{int(cid)}" for cid in df_sorted['core_id']], rotation=45)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "core-performance-breakdown.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed breakdown chart saved to: {output_path}")


if __name__ == "__main__":
    print("Creating aggregated core performance comparison...")
    aggregate_all_benchmarks()