use argh::FromArgs;
use core_profiler::{profile_workload_on_all_cores, WorkloadResult};
use std::time::Instant;

const OPERATIONS: u64 = 500_000_000;

#[derive(FromArgs)]
/// Profiles simple floating-point division throughput on all cores.
struct AppArgs {
    /// output file for statistics (e.g., division-math-stats.csv)
    #[argh(option, short = 'o')]
    output: String,
}

fn division_math_workload() -> (f64, String) {
    let start_time = Instant::now();
    let mut x = 1.000001f64;

    for _ in 0..OPERATIONS {
        x /= 1.00000000005;
    }

    let elapsed = start_time.elapsed();
    let ops_per_second = OPERATIONS as f64 / elapsed.as_secs_f64();

    // Prevent optimization
    if x == 0.0 {
        println!("x is zero, which should not happen.");
    }

    (ops_per_second, "ops/sec".to_string())
}

fn main() {
    let args: AppArgs = argh::from_env();
    let results = profile_workload_on_all_cores(division_math_workload);
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
