use argh::FromArgs;
use core_profiler::{WorkloadResult};

// Use 4 operations per SIMD loop for ARM NEON add operations (4 f32 lanes)
const SIMD_LOOPS: u64 = 500_000_000; // Increased loops to compensate for fewer lanes

#[derive(FromArgs)]
/// Profiles floating-point addition SIMD (NEON) throughput on all cores for ARM64.
struct AppArgs {
    /// output file for statistics (e.g., neon-add-stats.csv)
    #[argh(option, short = 'o')]
    output: String,
}

#[cfg(target_arch = "aarch64")]
fn neon_add_workload() -> (f64, String) {
    core_profiler::simd_arm::neon_add_workload(SIMD_LOOPS)
}

#[cfg(not(target_arch = "aarch64"))]
fn neon_add_workload() -> (f64, String) {
    eprintln!("NEON instructions are only available on ARM64 architecture.");
    std::process::exit(1);
}

fn main() {
    #[cfg(target_arch = "aarch64")]
    {
        use core_profiler::profile_workload_on_all_cores;
        let args: AppArgs = argh::from_env();
        let results = profile_workload_on_all_cores(neon_add_workload);
        write_results(&args.output, &results).expect("Failed to write results");
        println!("\nProfiling complete. Results saved to {}", args.output);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        eprintln!("This ARM NEON binary is only meant to be compiled for ARM64 architecture.");
        std::process::exit(1);
    }
}

fn write_results(path: &str, data: &[WorkloadResult]) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = csv::Writer::from_path(path)?;
    for record in data {
        writer.serialize(record)?;
    }
    writer.flush()?;
    Ok(())
}