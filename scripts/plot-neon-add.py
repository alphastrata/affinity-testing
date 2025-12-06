import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_FILE = "neon-add-stats.csv"


def plot_throughput(csv_path):
    """
    Reads and plots the throughput data from the NEON add benchmark CSV.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Benchmark data file not found at '{csv_path}'")
        print(
            "Please run the benchmark first: ./target/release/neon-add --output neon-add-stats.csv"
        )
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "core_id" not in df.columns or "throughput" not in df.columns:
        print(
            f"Error: CSV file '{csv_path}' is missing required 'core_id' or 'throughput' columns."
        )
        return

    unit = df["unit"].dropna().iloc[0] if "unit" in df.columns else "units"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(df["core_id"], df["throughput"], marker="o", linestyle="-", color="orange")
    ax.set_xlabel("Core ID")
    ax.set_ylabel(f"Throughput ({unit})")
    ax.set_title("NEON SIMD Float Addition Throughput per Core")
    ax.grid(True)
    ax.set_xticks(df["core_id"])

    output_path = "neon-add-throughput.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_throughput(CSV_FILE)
