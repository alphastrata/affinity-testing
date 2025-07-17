#!/bin/bash
# This script builds and runs all benchmarks, then generates plots for their results.
set -e

echo "--- Building project in release mode... ---"
cargo build --release

echo -e "\n--- Running simple-math benchmark... ---"
./target/release/simple-math --output simple-math-stats.csv

echo -e "\n--- Running division-math benchmark... ---"
./target/release/division-math --output division-math-stats.csv

echo -e "\n--- Running simd-math (AVX2) benchmark... ---"
./target/release/simd-math --output simd-math-stats.csv

# Check for AVX512 support before running the benchmark
echo -e "
--- Running simd-math (AVX512) benchmark... ---"
if ./target/release/avx512-math --output avx512-math-stats.csv; then
    echo "AVX512 benchmark complete."
else
    echo -e "--- Skipping AVX512 benchmark: Not supported on this CPU. ---"
    rm -f avx512-math-stats.csv
fi

echo -e "\n--- Running matrix-math benchmark... ---"
./target/release/matrix-math --output matrix-math-stats.csv

echo -e "
--- Running many-on-one pinning demonstration... ---"
echo "This will spawn a thread for each CPU core and pin them all to core 0."
./target/release/many-on-one --output many-on-one-stats.csv

echo -e "\n--- Generating plots... ---"
uv run scripts/plot-simple-math.py
uv run scripts/plot-division-math.py
uv run scripts/plot-simd-math.py
uv run scripts/plot-matrix-math.py
uv run scripts/plot-many-on-one.py many-on-one-stats.csv

if [ -f avx512-math-stats.csv ]; then
    uv run scripts/plot-avx512-math.py
fi

echo -e "\n--- All tasks complete. ---"