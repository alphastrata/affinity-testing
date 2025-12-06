import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_all_neon_implementations():
    """
    Plots all NEON implementations for comparison across cores
    """
    files = [
        ("simd-math-arm-stats.csv", "Basic NEON Multiply", "blue"),
        ("neon-advanced-stats.csv", "Advanced NEON Multiply", "red"),
        ("neon-integer-stats.csv", "NEON Integer Operations", "green"),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[3, 1])

    all_data = {}
    for csv_file, label, color in files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()

            if "core_id" in df.columns and "throughput" in df.columns:
                all_data[label] = (df, color)
            else:
                print(f"Warning: {csv_file} missing required columns")
        else:
            print(f"Warning: File {csv_file} not found")

    if all_data:
        has_data = False
        all_core_ids = set()
        for label, (df, color) in all_data.items():
            all_core_ids.update(df["core_id"].values)

        # Sort core IDs for consistent plotting
        sorted_core_ids = sorted(list(all_core_ids))

        for label, (df, color) in all_data.items():
            # Ensure df is sorted by core_id for consistent plotting
            df_sorted = df.sort_values('core_id')
            ax1.plot(df_sorted["core_id"], df_sorted["throughput"], marker="o", linestyle="-",
                    label=label, color=color, linewidth=2, markersize=8)
            has_data = True

        if has_data:
            # Add unit explanation as text on the plot
            ax1.text(0.02, 0.98, f'1e9 = 1 billion, 1e6 = 1 million, etc.',
                     transform=ax1.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax1.set_xlabel("Core ID")
            ax1.set_ylabel("Throughput (ops/sec)")
            ax1.set_title("NEON SIMD Implementations Comparison per Core")
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(sorted_core_ids)
            ax1.legend(loc='upper right')

            # For the aggregated view, let's create an average of all implementations per core
            # and highlight the best performing combination
            combined_data = pd.DataFrame({'core_id': sorted_core_ids})
            for label, (df, color) in all_data.items():
                # Merge each implementation's data, filling missing values with 0
                temp = pd.merge(combined_data[['core_id']], df[['core_id', 'throughput']],
                               on='core_id', how='left')
                temp.columns = ['core_id', f'{label}_throughput']
                combined_data = pd.merge(combined_data, temp, on='core_id', how='left')

                # Fill NaN values with 0 where there's no data for a core
                combined_data[f'{label}_throughput'] = combined_data[f'{label}_throughput'].fillna(0)

            # Calculate an aggregate score per core (sum of all implementations)
            score_cols = [col for col in combined_data.columns if col.endswith('_throughput')]
            combined_data['aggregate_score'] = combined_data[score_cols].sum(axis=1)

            # Sort by aggregate score for the bar chart
            combined_data_sorted = combined_data.sort_values('aggregate_score', ascending=True)

            # Create a colormap based on performance
            colors = plt.cm.viridis(np.linspace(0, 1, len(combined_data_sorted)))

            ax2.barh(range(len(combined_data_sorted)), combined_data_sorted['aggregate_score'], color=colors)
            ax2.set_xlabel("Aggregate Performance Score")
            ax2.set_ylabel("Core ID")
            ax2.set_title("Aggregated Core Performance (All NEON Implementations)")
            ax2.set_yticks(range(len(combined_data_sorted)))
            ax2.set_yticklabels([f"Core {cid}" for cid in combined_data_sorted['core_id']])

            # Add values as text on bars
            for i, (idx, row) in enumerate(combined_data_sorted.iterrows()):
                ax2.text(row['aggregate_score'] * 0.5, i, f"{row['aggregate_score']:.2e}",
                        ha='center', va='center', fontweight='bold', color='white')

            plt.tight_layout()
            output_path = "neon-implementations-comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_path}")
        else:
            print("No valid data found to plot")
    else:
        print("No valid data files found to plot")


def plot_neon_operations_comparison():
    """
    Creates separate plots for different NEON operation types with enhanced features
    """
    files = {
        "neon-add-stats.csv": "NEON Float Add Operations",
        "neon-mixed-stats.csv": "NEON Mixed Operations",
        "neon-double-stats.csv": "NEON Double Precision",
        "neon-fma-stats.csv": "NEON Fused Multiply-Add"
    }

    for csv_file, title in files.items():
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()

            if "core_id" in df.columns and "throughput" in df.columns:
                # Find the best performing core
                best_core_idx = df["throughput"].idxmax()
                best_core_id = df.loc[best_core_idx, "core_id"]
                best_throughput = df.loc[best_core_idx, "throughput"]

                unit = df["unit"].dropna().iloc[0] if "unit" in df.columns else "ops/sec"

                plt.style.use("seaborn-v0_8-whitegrid")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[3, 1])

                # Main plot: Throughput per core
                ax1.plot(df["core_id"], df["throughput"], marker="o", linestyle="-",
                        linewidth=2, markersize=8, color="purple", label=f'Performance ({unit})')

                # Highlight the best core
                ax1.scatter(best_core_id, best_throughput, color='red', s=150, zorder=5,
                           label=f'Best Core {best_core_id}: {best_throughput:.2e} {unit}',
                           edgecolors='black', linewidth=2)

                # Improve y-axis formatting and add unit explanation
                ax1.set_xlabel("Core ID")
                ax1.set_ylabel(f"Throughput ({unit})")

                # Format scientific notation more clearly
                ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

                # Add unit explanation as text on the plot
                ax1.text(0.02, 0.98, f'1e9 = 1 billion, 1e6 = 1 million, etc.',
                         transform=ax1.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                ax1.set_title(title)
                ax1.grid(True, alpha=0.3)
                ax1.set_xticks(df["core_id"])
                ax1.legend(loc='upper right')

                # Second plot: Aggregated core performance as horizontal bar chart
                df_sorted = df.sort_values('throughput', ascending=True)  # Sort for better visualization

                # Create a colormap based on performance
                colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))

                ax2.barh(range(len(df_sorted)), df_sorted['throughput'], color=colors)
                ax2.set_xlabel(f"Throughput ({unit})")
                ax2.set_ylabel("Core ID")
                ax2.set_title("Aggregated Core Performance (Sorted)")
                ax2.set_yticks(range(len(df_sorted)))
                ax2.set_yticklabels([f"Core {cid}" for cid in df_sorted['core_id']])

                # Add values as text on bars
                for i, (idx, row) in enumerate(df_sorted.iterrows()):
                    ax2.text(row['throughput'] * 0.5, i, f"{row['throughput']:.2e}",
                            ha='center', va='center', fontweight='bold', color='white')

                plt.tight_layout()
                output_path = f"{csv_file.replace('-stats.csv', '')}-throughput.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"Plot saved to {output_path}")
                print(f"Best performing core: Core {best_core_id} with {best_throughput:.2e} {unit}")

                plt.close()  # Close the figure to free memory
            else:
                print(f"Warning: {csv_file} missing required columns")


if __name__ == "__main__":
    plot_all_neon_implementations()
    plot_neon_operations_comparison()