use argh::FromArgs;
use core_profiler::{profile_workload_on_all_cores, WorkloadResult};
use ndarray::Array;
use std::time::Instant;

const MATRIX_SIZE: usize = 256;
const OPERATIONS: u64 = 50;

#[derive(FromArgs)]
/// Profiles matrix multiplication throughput on all cores.
struct AppArgs {
    /// output file for statistics (e.g., matrix-math-stats.csv)
    #[argh(option, short = 'o')]
    output: String,
}

fn matrix_math_workload() -> (f64, String) {
    let start_time = Instant::now();
    let a = Array::from_elem((MATRIX_SIZE, MATRIX_SIZE), 1.01f32);
    let mut b = Array::from_elem((MATRIX_SIZE, MATRIX_SIZE), 1.02f32);

    for _ in 0..OPERATIONS {
        b = a.dot(&b);
    }

    let elapsed = start_time.elapsed();
    let flops =
        OPERATIONS as f64 * (2.0 * MATRIX_SIZE as f64 * MATRIX_SIZE as f64 * MATRIX_SIZE as f64);
    let gflops = flops / elapsed.as_secs_f64() / 1e9;

    // Prevent optimization
    if b.sum() == 0.0 {
        panic!("sum is zero, which should not happen.");
    }

    (gflops, "GFLOPs".to_string())
}

fn main() {
    let args: AppArgs = argh::from_env();
    let results = profile_workload_on_all_cores(matrix_math_workload);
    write_results(&args.output, &results).expect("Failed to write results");
    println!("\nProfiling complete. Results saved to {}", args.output);
}

fn write_results(path: &str, data: &[WorkloadResult]) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = csv::Writer::from_path(path)?;
    for record in data {
        writer.serialize(record)?;
    }
    writer.flush()?;
    Ok(())
}
