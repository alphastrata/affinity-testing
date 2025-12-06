use argh::FromArgs;
use core_profiler::{WorkloadResult};

#[cfg(target_arch = "x86_64")]
const SIMD_LOOPS: u64 = 250_000_000;

#[cfg(target_arch = "x86_64")]
#[derive(FromArgs)]
/// Profiles SIMD (AVX2) math throughput on all cores.
struct AppArgs {
    /// output file for statistics (e.g., simd-math-stats.csv)
    #[argh(option, short = 'o')]
    output: String,
}

#[cfg(target_arch = "x86_64")]
fn simd_math_workload() -> (f64, String) {
    core_profiler::simd_x86::avx2_multiply_workload(SIMD_LOOPS)
}

#[cfg(not(target_arch = "x86_64"))]
fn simd_math_workload() -> (f64, String) {
    eprintln!("AVX2 SIMD workload is only supported on x86_64 architecture.");
    std::process::exit(1);
}

fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        use core_profiler::profile_workload_on_all_cores;
        let args: AppArgs = argh::from_env();
        let results = profile_workload_on_all_cores(simd_math_workload);
        write_results(&args.output, &results).expect("Failed to write results");
        println!("\nProfiling complete. Results saved to {}", args.output);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        eprintln!("AVX2 SIMD workload is only supported on x86_64 architecture.");
        std::process::exit(1);
    }
}

#[cfg(target_arch = "x86_64")]
fn write_results(path: &str, data: &[WorkloadResult]) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = csv::Writer::from_path(path)?;
    for record in data {
        writer.serialize(record)?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(not(target_arch = "x86_64"))]
fn write_results(_path: &str, _data: &[WorkloadResult]) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("AVX2 SIMD workload is only supported on x86_64 architecture.");
    std::process::exit(1);
}
