import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_cpu_usage(file_path):
    """
    Reads CPU usage data and plots the average load per core as a stacked bar chart.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'")
        return

    # Read the CSV data
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    required_cols = ['core_id', 'user_percent', 'system_percent']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file '{file_path}' is missing one or more required columns: {required_cols}")
        return

    # Calculate the average usage for each core
    avg_usage = df.groupby('core_id')[['user_percent', 'system_percent']].mean()

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    cores = avg_usage.index
    user_p = avg_usage['user_percent']
    system_p = avg_usage['system_percent']

    # Create stacked bar chart
    ax.bar(cores, user_p, label='User %')
    ax.bar(cores, system_p, bottom=user_p, label='System %')

    ax.set_xlabel('Core ID')
    ax.set_ylabel('Average CPU Usage (%)')
    ax.set_title('Average CPU Load per Core')
    ax.set_xticks(cores)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot
    output_path = 'cpu_usage_matplotlib.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot-many-on-one.py <path_to_csv>")
        sys.exit(1)

    plot_cpu_usage(sys.argv[1])
