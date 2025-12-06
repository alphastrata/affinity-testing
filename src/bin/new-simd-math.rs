#[cfg(target_arch = "x86_64")]
use argh::FromArgs;

#[cfg(target_arch = "x86_64")]
use core_profiler::{example_benchmarks::MathBenchmark, Benchmark, operation_counting::OperationType};

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("Math benchmark is only supported on x86_64 architecture.");
    std::process::exit(1);
}

#[cfg(target_arch = "x86_64")]
#[derive(FromArgs)]
/// Profiles math throughput on all cores using the new benchmarking framework.
struct AppArgs {
    /// output file for statistics (e.g., new-simd-math-stats.csv)
    #[argh(option, short = 'o')]
    output: String,
}

#[cfg(target_arch = "x86_64")]
fn main() {
    use core_profiler::{example_benchmarks::MathBenchmarkConfig, profile_workload_on_all_cores_new};
    
    let args: AppArgs = argh::from_env();
    
    // Create the benchmark using the new framework
    let benchmark = MathBenchmark::new("SIMD Math".to_string());
    let config = MathBenchmarkConfig {
        iterations: 10_000_000,
        operation_type: OperationType::FMA, // Fused Multiply-Add operations
    };

    // Run the benchmark on all cores using the new framework
    let results = profile_workload_on_all_cores_new(benchmark, &config);
    write_results(&args.output, &results).expect("Failed to write results");
    println!("\nProfiling complete. Results saved to {}", args.output);
    
    // Print summary statistics
    if !results.is_empty() {
        let total_ops: f64 = results.iter().map(|r| r.ops_per_second).sum();
        let avg_ops = total_ops / results.len() as f64;
        let max_result = results.iter().max_by(|a, b| a.ops_per_second.partial_cmp(&b.ops_per_second).unwrap()).unwrap();
        let min_result = results.iter().min_by(|a, b| a.ops_per_second.partial_cmp(&b.ops_per_second).unwrap()).unwrap();
        
        println!("Summary:");
        println!("  Average performance: {:.2} {}", avg_ops, max_result.unit);
        println!("  Best core ({}): {:.2} {}", max_result.core_id, max_result.ops_per_second, max_result.unit);
        println!("  Worst core ({}): {:.2} {}", min_result.core_id, min_result.ops_per_second, min_result.unit);
    }
}

#[cfg(target_arch = "x86_64")]
fn write_results(path: &str, data: &[core_profiler::BenchmarkResult]) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = csv::Writer::from_path(path)?;
    for record in data {
        writer.serialize(record)?;
    }
    writer.flush()?;
    Ok(())
}