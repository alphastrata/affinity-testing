#[cfg(target_arch = "x86_64")]
use argh::FromArgs;

#[cfg(target_arch = "x86_64")]
use core_profiler::{WorkloadResult};

#[cfg(target_arch = "x86_64")]
// Each SIMD operation processes 16 floats. 100M loops * 16 = 1.6B float ops.
const SIMD_LOOPS: u64 = 100_000_000;

#[cfg(target_arch = "x86_64")]
#[derive(FromArgs)]
/// Profiles SIMD (AVX512) math throughput on all cores.
struct AppArgs {
    /// output file for statistics (e.g., avx512-math-stats.csv)
    #[argh(option, short = 'o')]
    output: String,
}

#[cfg(target_arch = "x86_64")]
fn avx512_math_workload() -> (f64, String) {
    core_profiler::simd_x86::avx512_multiply_workload(SIMD_LOOPS)
}

#[cfg(not(target_arch = "x86_64"))]
fn avx512_math_workload() -> (f64, String) {
    eprintln!("AVX512 workload is only supported on x86_64 architecture.");
    std::process::exit(1);
}

fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        use core_profiler::profile_workload_on_all_cores;
        let args: AppArgs = argh::from_env();
        let results = profile_workload_on_all_cores(avx512_math_workload);
        write_results(&args.output, &results).expect("Failed to write results");
        println!("\nProfiling complete. Results saved to {}", args.output);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        eprintln!("AVX512 workload is only supported on x86_64 architecture.");
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
fn write_results(_path: &str, _data: &[core_profiler::WorkloadResult]) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("AVX512 workload is only supported on x86_64 architecture.");
    std::process::exit(1);
}
