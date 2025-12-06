#!/bin/bash
# This script builds and runs all benchmarks, then generates plots for their results.
set -e

echo "--- Building project in release mode... ---"
cargo build --release

# Detect architecture to run platform-appropriate benchmarks
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Run common benchmarks available on all architectures
echo -e "\n--- Running simple-math benchmark... ---"
./target/release/simple-math --output simple-math-stats.csv

echo -e "\n--- Running division-math benchmark... ---"
./target/release/division-math --output division-math-stats.csv

echo -e "\n--- Running matrix-math benchmark... ---"
./target/release/matrix-math --output matrix-math-stats.csv

echo -e "\n--- Running many-on-one pinning demonstration... ---"
echo "This will spawn a thread for each CPU core and pin them all to core 0."
./target/release/many-on-one --output many-on-one-stats.csv

# Run architecture-specific benchmarks
if [[ "$ARCH" == "x86_64" || "$ARCH" == "amd64" ]]; then
    echo -e "\n--- Running x86_64-specific benchmarks... ---"

    echo -e "\n--- Running simd-math (AVX2) benchmark... ---"
    if ./target/release/simd-math --output simd-math-stats.csv; then
        echo "AVX2 benchmark complete."
    else
        echo -e "--- Skipping AVX2 benchmark: Not supported on this CPU. ---"
        rm -f simd-math-stats.csv
    fi

    # Check for AVX512 support before running the benchmark
    echo -e "\n--- Running avx512-math (AVX512) benchmark... ---"
    if ./target/release/avx512-math --output avx512-math-stats.csv; then
        echo "AVX512 benchmark complete."
    else
        echo -e "--- Skipping AVX512 benchmark: Not supported on this CPU. ---"
        rm -f avx512-math-stats.csv
    fi

elif [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
    echo -e "\n--- Running ARM64-specific benchmarks... ---"

    echo -e "\n--- Running simd-math-arm (NEON) benchmark... ---"
    if ./target/release/simd-math-arm --output simd-math-arm-stats.csv; then
        echo "NEON basic benchmark complete."
    else
        echo -e "--- Skipping NEON basic benchmark: Not supported on this CPU. ---"
        rm -f simd-math-arm-stats.csv
    fi

    echo -e "\n--- Running avx512-math-arm (NEON) benchmark... ---"
    if ./target/release/avx512-math-arm --output avx512-math-arm-stats.csv; then
        echo "NEON advanced benchmark complete."
    else
        echo -e "--- Skipping NEON advanced benchmark: Not supported on this CPU. ---"
        rm -f avx512-math-arm-stats.csv
    fi

    # Run the new NEON implementations
    echo -e "\n--- Running neon-advanced benchmark... ---"
    if ./target/release/neon-advanced --output neon-advanced-stats.csv; then
        echo "NEON advanced multiply benchmark complete."
    else
        echo -e "--- Skipping NEON advanced multiply benchmark: Not supported on this CPU. ---"
        rm -f neon-advanced-stats.csv
    fi

    echo -e "\n--- Running neon-integer benchmark... ---"
    if ./target/release/neon-integer --output neon-integer-stats.csv; then
        echo "NEON integer benchmark complete."
    else
        echo -e "--- Skipping NEON integer benchmark: Not supported on this CPU. ---"
        rm -f neon-integer-stats.csv
    fi

    echo -e "\n--- Running neon-add benchmark... ---"
    if ./target/release/neon-add --output neon-add-stats.csv; then
        echo "NEON add benchmark complete."
    else
        echo -e "--- Skipping NEON add benchmark: Not supported on this CPU. ---"
        rm -f neon-add-stats.csv
    fi

else
    echo "Unknown architecture: $ARCH. Some benchmarks may not run correctly."
fi

echo -e "\n--- Generating plots... ---"
uv run scripts/plot-simple-math.py
uv run scripts/plot-division-math.py
uv run scripts/plot-matrix-math.py
uv run scripts/plot-many-on-one.py many-on-one-stats.csv

# Architecture-specific plotting
if [[ "$ARCH" == "x86_64" || "$ARCH" == "amd64" ]]; then
    if [ -f simd-math-stats.csv ]; then
        uv run scripts/plot-simd-math.py
    fi
    if [ -f avx512-math-stats.csv ]; then
        uv run scripts/plot-avx512-math.py
    fi
elif [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
    if [ -f simd-math-arm-stats.csv ]; then
        # Use a generic plotting script or create one for ARM if it doesn't exist
        if [ -f scripts/plot-simd-math.py ]; then
            cp scripts/plot-simd-math.py scripts/plot-simd-math-arm.py 2>/dev/null || true
        fi
        if [ -f scripts/plot-simd-math-arm.py ]; then
            uv run scripts/plot-simd-math-arm.py
        fi
    fi
    if [ -f avx512-math-arm-stats.csv ]; then
        # Use generic plotting script or create one if needed
        if [ -f scripts/plot-avx512-math.py ]; then
            # We'll copy and adapt the AVX512 plotter for ARM
            cp scripts/plot-avx512-math.py scripts/plot-avx512-math-arm.py 2>/dev/null || true
        fi
        if [ -f scripts/plot-avx512-math-arm.py ]; then
            uv run scripts/plot-avx512-math-arm.py
        fi
    fi

    # Run new NEON plotting scripts
    if [ -f neon-advanced-stats.csv ]; then
        uv run scripts/plot-neon-advanced.py
    fi
    if [ -f neon-integer-stats.csv ]; then
        uv run scripts/plot-neon-integer.py
    fi
    if [ -f neon-add-stats.csv ]; then
        # Create a simple plotting script for add operations if it doesn't exist
        if [ ! -f scripts/plot-neon-add.py ]; then
            cat > scripts/plot-neon-add.py << 'EOF'
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
EOF
        fi
        uv run scripts/plot-neon-add.py
    fi

    # Run comparison plotting
    uv run scripts/plot-neon-comparison.py
fi

echo -e "\n--- All tasks complete. ---"