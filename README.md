# CPU Affinity experiments

This repo contains code testing CPU affinity, applications, implications and so on.

## Building

```bash
cargo build --release
```
or,
```sh
cargo run --bin <BINARY NAME> # run it without args to see the list

```
Which will give you a list of what's available.

## Binaries available.

*   `simple-math`: A benchmark for simple mathematical operations.
*   `simd-math`: A benchmark for SIMD (AVX2) accelerated mathematical operations.
*   `avx512-math`: A benchmark for SIMD (AVX512) accelerated mathematical operations.
*   `matrix-math`: A benchmark for matrix multiplication.
*   `division-math`: A benchmark for division-heavy mathematical operations.
*   `many-on-one`: A tool to demonstrate pinning multiple CPU-intensive threads to a single core.

Each binary can be run from the `affinity-testing` directory and accepts an `--output` flag to specify the path for the resulting CSV file.

### Usage:

**simple-math:**
```bash
./target/release/simple-math --output simple-math-stats.csv
```

**simd-math (AVX2):**
```bash
./target/release/simd-math --output simd-math-stats.csv
```

**simd-math (AVX512):**
```bash
# Note: This will only run on CPUs that support AVX512.
./target/release/avx512-math --output avx512-math-stats.csv
```

**matrix-math:**
```bash
./target/release/matrix-math --output matrix-math-stats.csv
```

**division-math:**
```bash
./target/release/division-math --output division-math-stats.csv
```

**many-on-one:**
```bash
# This will spawn a thread for each CPU core and pin them all to core 0 by default.
# Use htop or another system monitor to observe the effect.
./target/release/many-on-one --output many-on-one-stats.csv
# Or, specify the number of threads and the core to pin to:
./target/release/many-on-one --threads 4 --core-id 2 --output many-on-one-stats.csv
```

## Plotting Results

All(most all) of the bins have a corresponding plotting script to plot their output.

```bash
uv run scripts/plot-simple-math.py
uv run scripts/plot-simd-math.py
uv run scripts/plot-avx512-math.py
uv run scripts/plot-matrix-math.py
uv run scripts/plot-division-math.py
uv run scripts/plot-many-on-one.py many-on-one-stats.csv
```
