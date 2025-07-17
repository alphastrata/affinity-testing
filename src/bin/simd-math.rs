use argh::FromArgs;
use core_profiler::{profile_workload_on_all_cores, WorkloadResult};
use std::arch::x86_64::*;
use std::time::Instant;

const SIMD_LOOPS: u64 = 250_000_000;
const OPERATIONS: u64 = SIMD_LOOPS * 8;

#[derive(FromArgs)]
/// Profiles SIMD (AVX2) math throughput on all cores.
struct AppArgs {
    /// output file for statistics (e.g., simd-math-stats.csv)
    #[argh(option, short = 'o')]
    output: String,
}

fn simd_math_workload() -> (f64, String) {
    let start_time = Instant::now();
    let mut a = unsafe { _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) };
    let b = unsafe { _mm256_set_ps(0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1) };

    for _ in 0..SIMD_LOOPS {
        a = unsafe { _mm256_mul_ps(a, b) };
    }

    let elapsed = start_time.elapsed();
    let ops_per_second = OPERATIONS as f64 / elapsed.as_secs_f64();

    // Prevent optimization
    let mut result = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(result.as_mut_ptr(), a) };
    if result[0] == 0.0 {
        println!("result is zero, which should not happen.");
    }

    (ops_per_second, "ops/sec".to_string())
}

fn main() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("AVX2 not supported on this CPU. Exiting.");
        return;
    }
    let args: AppArgs = argh::from_env();
    let results = profile_workload_on_all_cores(simd_math_workload);
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
