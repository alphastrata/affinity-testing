use argh::FromArgs;
use core_profiler::{profile_workload_on_all_cores, WorkloadResult};
use std::arch::x86_64::*;
use std::time::Instant;

// Each SIMD operation processes 16 floats. 100M loops * 16 = 1.6B float ops.
const SIMD_LOOPS: u64 = 100_000_000;
const OPERATIONS: u64 = SIMD_LOOPS * 16;

#[derive(FromArgs)]
/// Profiles SIMD (AVX512) math throughput on all cores.
struct AppArgs {
    /// output file for statistics (e.g., avx512-math-stats.csv)
    #[argh(option, short = 'o')]
    output: String,
}

// This function is marked as unsafe because it uses AVX512 intrinsics,
// which are only safe to call if the CPU supports them.
#[target_feature(enable = "avx512f")]
unsafe fn avx512_math_workload_inner() -> (f64, String) {
    let start_time = Instant::now();
    let mut a = _mm512_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    let b = _mm512_set_ps(0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1);

    for _ in 0..SIMD_LOOPS {
        a = _mm512_mul_ps(a, b);
    }

    let elapsed = start_time.elapsed();
    let ops_per_second = OPERATIONS as f64 / elapsed.as_secs_f64();

    // Prevent optimization
    let mut result = [0.0f32; 16];
    _mm512_storeu_ps(result.as_mut_ptr(), a);
    if result[0] == 0.0 {
        println!("result is zero, which should not happen.");
    }

    (ops_per_second, "ops/sec".to_string())
}

fn main() {
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("AVX512F not supported on this CPU. Exiting.");
        return;
    }
    let args: AppArgs = argh::from_env();
    // We call the unsafe inner function inside a closure, guarded by the feature detection above.
    let results = profile_workload_on_all_cores(|| unsafe { avx512_math_workload_inner() });
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
