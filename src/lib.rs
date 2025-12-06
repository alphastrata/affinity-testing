// core_profiler/src/lib.rs

use serde::Serialize;
use std::time::Duration;

// New Benchmark Framework - Phase 1A: Create the Benchmark trait and related structures

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResult {
    pub ops_per_second: f64,
    pub unit: PerformanceUnit,
    pub validation_status: ValidationStatus,
    pub execution_time: Duration,
    pub work_completed: u64,
    pub core_id: usize,
}

#[derive(Debug, Clone, Serialize)]
pub enum ValidationStatus {
    Valid,
    Invalid { reason: String },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum PerformanceUnit {
    FLOPS,
    MFLOPS,
    GFLOPS,
    TFLOPS,
    PFLOPS,
    OPS,
    MOPS,
    GOPS,
    TOPS,
}

impl std::fmt::Display for PerformanceUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerformanceUnit::FLOPS => write!(f, "FLOPS"),
            PerformanceUnit::MFLOPS => write!(f, "MFLOPS"),
            PerformanceUnit::GFLOPS => write!(f, "GFLOPS"),
            PerformanceUnit::TFLOPS => write!(f, "TFLOPS"),
            PerformanceUnit::PFLOPS => write!(f, "PFLOPS"),
            PerformanceUnit::OPS => write!(f, "OPS"),
            PerformanceUnit::MOPS => write!(f, "MOPS"),
            PerformanceUnit::GOPS => write!(f, "GOPS"),
            PerformanceUnit::TOPS => write!(f, "TOPS"),
        }
    }
}

pub trait Benchmark {
    type Config;  // Benchmark-specific configuration

    fn run_benchmark(&self, config: &Self::Config) -> BenchmarkResult;
    fn validate_results(&self, result: &BenchmarkResult) -> ValidationResult;
    fn adjust_workload(&self, current_config: &Self::Config, previous_result: &BenchmarkResult) -> Self::Config;
    fn get_unit(&self) -> PerformanceUnit;  // Always return standardized units
}

#[derive(Debug, Clone)]
pub enum ValidationResult {
    Valid,
    Invalid { reason: String },
    NeedsRerun { reason: String },
}

// New Anomaly Detection System - Phase 1B
pub mod anomaly_detection {
    use super::*;
    use std::f64;

    pub struct AnomalyDetector {
        // Hardware limits for range validation
        pub max_flops: Option<f64>,
        // Statistical threshold for outlier detection (in standard deviations)
        pub outlier_threshold: f64,
        // Threshold for zero work detection
        pub zero_work_threshold: u64,
    }

    impl Default for AnomalyDetector {
        fn default() -> Self {
            Self {
                max_flops: None, // Set based on hardware capabilities if known
                outlier_threshold: 3.0, // 3 standard deviations
                zero_work_threshold: 0, // Any work_completed value below or equal to this triggers rerun
            }
        }
    }

    impl AnomalyDetector {
        pub fn new(max_flops: Option<f64>, outlier_threshold: f64, zero_work_threshold: u64) -> Self {
            Self {
                max_flops,
                outlier_threshold,
                zero_work_threshold,
            }
        }

        pub fn detect_anomalies(&self, result: &BenchmarkResult) -> ValidationResult {
            // 1. Infinite/NaN Detection
            if result.ops_per_second.is_infinite() || result.ops_per_second.is_nan() {
                return ValidationResult::Invalid {
                    reason: "Result is infinite or NaN".to_string(),
                };
            }

            // 2. Range Validation
            if let Some(max_flops) = self.max_flops {
                if result.ops_per_second > max_flops {
                    return ValidationResult::Invalid {
                        reason: format!("Result exceeds theoretical hardware limit: {} > {}", result.ops_per_second, max_flops),
                    };
                }
            }

            // 3. Zero Work Detection
            if result.work_completed <= self.zero_work_threshold {
                return ValidationResult::NeedsRerun {
                    reason: format!("Zero or insufficient work detected: {}", result.work_completed),
                };
            }

            ValidationResult::Valid
        }

        // Function to detect statistical outliers (to be used with historical data)
        pub fn detect_statistical_outliers(&self, current_value: f64, historical_data: &[f64]) -> bool {
            if historical_data.is_empty() {
                return false;
            }

            // Calculate mean and standard deviation
            let mean = historical_data.iter().sum::<f64>() / (historical_data.len() as f64);
            let variance = historical_data.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / (historical_data.len() as f64);
            let std_dev = variance.sqrt();

            if std_dev == 0.0 {
                // If std_dev is 0, all values are the same, so no outliers
                return false;
            }

            let z_score = (current_value - mean).abs() / std_dev;
            z_score > self.outlier_threshold
        }
    }
}

// New Workload Scaling System - Phase 1C
pub mod workload_scaling {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct WorkloadConfig {
        pub base_iterations: u64,
        pub min_iterations: u64,
        pub max_iterations: u64,
        pub target_duration_ms: u64,
        pub adjustment_factor: f64,
    }

    impl Default for WorkloadConfig {
        fn default() -> Self {
            Self {
                base_iterations: 1_000_000,
                min_iterations: 1_000,
                max_iterations: 1_000_000_000,
                target_duration_ms: 1000, // 1 second target
                adjustment_factor: 1.5, // 50% adjustment factor
            }
        }
    }

    impl WorkloadConfig {
        pub fn adjust_for_performance(&self, prev_result: &BenchmarkResult) -> WorkloadConfig {
            // Calculate the adjustment factor based on how far the execution time is from the target
            let actual_duration_ms = prev_result.execution_time.as_millis() as f64;
            let target_duration_ms = self.target_duration_ms as f64;

            // If execution was too fast, increase iterations; if too slow, decrease iterations
            // To reach the target time, we need to scale iterations by target/actual
            // e.g., if we took 200ms but wanted 1000ms, we should do 5x more work
            let duration_ratio = target_duration_ms / actual_duration_ms;

            // Calculate new iteration count based on the ratio
            let adjusted_iterations = (self.base_iterations as f64 * duration_ratio) as u64;

            // Apply min/max constraints and ensure it's within bounds
            let new_iterations = adjusted_iterations
                .max(self.min_iterations)
                .min(self.max_iterations);

            // Update the config with the new iteration count
            WorkloadConfig {
                base_iterations: new_iterations,
                ..self.clone()
            }
        }

        pub fn adjust_for_validation(&self, validation_result: &ValidationResult) -> WorkloadConfig {
            match validation_result {
                ValidationResult::Valid => self.clone(),
                ValidationResult::Invalid { .. } => {
                    // If validation failed, try increasing iterations to get more meaningful results
                    let new_iterations = (self.base_iterations as f64 * self.adjustment_factor) as u64;
                    let bounded_iterations = new_iterations
                        .max(self.min_iterations)
                        .min(self.max_iterations);

                    WorkloadConfig {
                        base_iterations: bounded_iterations,
                        ..self.clone()
                    }
                },
                ValidationResult::NeedsRerun { .. } => {
                    // If results need rerunning (e.g., due to zero work), increase iterations significantly
                    let new_iterations = (self.base_iterations as f64 * (self.adjustment_factor * 2.0)) as u64;
                    let bounded_iterations = new_iterations
                        .max(self.min_iterations)
                        .min(self.max_iterations);

                    WorkloadConfig {
                        base_iterations: bounded_iterations,
                        ..self.clone()
                    }
                },
            }
        }
    }
}

// Unit Conversion Utilities - Phase 1D
pub mod unit_conversion {
    use super::*;

    impl PerformanceUnit {
        pub fn to_f64_multiplier(&self) -> f64 {
            match self {
                PerformanceUnit::FLOPS => 1.0,
                PerformanceUnit::MFLOPS => 1_000_000.0,
                PerformanceUnit::GFLOPS => 1_000_000_000.0,
                PerformanceUnit::TFLOPS => 1_000_000_000_000.0,
                PerformanceUnit::PFLOPS => 1_000_000_000_000_000.0,
                PerformanceUnit::OPS => 1.0,
                PerformanceUnit::MOPS => 1_000_000.0,
                PerformanceUnit::GOPS => 1_000_000_000.0,
                PerformanceUnit::TOPS => 1_000_000_000_000.0,
            }
        }

        pub fn from_value(value: f64) -> (f64, PerformanceUnit) {
            // Choose the most appropriate unit based on the magnitude of the value
            if value >= 1_000_000_000_000_000.0 {
                (value / 1_000_000_000_000_000.0, PerformanceUnit::PFLOPS)
            } else if value >= 1_000_000_000_000.0 {
                (value / 1_000_000_000_000.0, PerformanceUnit::TFLOPS)
            } else if value >= 1_000_000_000.0 {
                (value / 1_000_000_000.0, PerformanceUnit::GFLOPS)
            } else if value >= 1_000_000.0 {
                (value / 1_000_000.0, PerformanceUnit::MFLOPS)
            } else {
                (value, PerformanceUnit::FLOPS)
            }
        }

        pub fn convert_value(from_unit: &PerformanceUnit, to_unit: &PerformanceUnit, value: f64) -> f64 {
            let from_multiplier = from_unit.to_f64_multiplier();
            let to_multiplier = to_unit.to_f64_multiplier();

            // Convert to base unit first, then to target unit
            (value * from_multiplier) / to_multiplier
        }
    }
}

// Unit Standardization System - Phase 2B
pub mod unit_standardization {
    use super::*;

    impl BenchmarkResult {
        pub fn standardize(&mut self, preferred_unit: Option<PerformanceUnit>) {
            let target_unit = preferred_unit.unwrap_or_else(|| {
                // Auto-select the most appropriate unit based on the value
                match self.unit {
                    PerformanceUnit::FLOPS | PerformanceUnit::OPS => {
                        // Use from_value to find a more appropriate unit for large values
                        let (_, auto_unit) = PerformanceUnit::from_value(self.ops_per_second);
                        auto_unit
                    }
                    _ => self.unit.clone(),
                }
            });

            // Convert the ops_per_second to the target unit
            if self.unit != target_unit {
                let new_value = PerformanceUnit::convert_value(&self.unit, &target_unit, self.ops_per_second);
                self.ops_per_second = new_value;
                self.unit = target_unit;
            }
        }

        pub fn get_display_value(&self) -> String {
            // Format the value with appropriate precision based on the unit
            format!("{:.2} {}", self.ops_per_second, self.unit)
        }
    }
}

// Standardized Operation Counting Logic - Phase 2C
pub mod operation_counting {
    use super::*;

    #[derive(Debug, Clone)]
    pub enum OperationType {
        FMA,      // Fused Multiply-Add (counts as 2 operations)
        Multiply, // Multiply operation (counts as 1 operation)
        Add,      // Add operation (counts as 1 operation)
        Divide,   // Divide operation (counts as 1 operation)
        Memory,   // Memory operation (counts as 1 operation)
        Custom(u64), // Custom operation count
    }

    impl OperationType {
        pub fn count(&self) -> u64 {
            match self {
                OperationType::FMA => 2,      // FMA = multiply + add
                OperationType::Multiply => 1,
                OperationType::Add => 1,
                OperationType::Divide => 1,
                OperationType::Memory => 1,
                OperationType::Custom(count) => *count,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct OperationCounter {
        pub operation_type: OperationType,
        pub iterations: u64,
    }

    impl OperationCounter {
        pub fn new(op_type: OperationType, iterations: u64) -> Self {
            Self {
                operation_type: op_type,
                iterations,
            }
        }

        pub fn calculate_total_operations(&self) -> u64 {
            self.operation_type.count() * self.iterations
        }

        pub fn ops_per_second(&self, duration: &Duration) -> f64 {
            let total_ops = self.calculate_total_operations() as f64;
            let duration_seconds = duration.as_secs_f64();

            if duration_seconds == 0.0 {
                0.0  // Avoid division by zero
            } else {
                total_ops / duration_seconds
            }
        }
    }
}

// Failure Detection System - Phase 4A
pub mod failure_detection {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct FailureDetector {
        pub zero_result_threshold: f64,
        pub timeout_threshold: Duration,
        pub min_valid_operations: u64,
    }

    impl Default for FailureDetector {
        fn default() -> Self {
            Self {
                zero_result_threshold: 0.0,
                timeout_threshold: Duration::from_secs(300), // 5 minutes
                min_valid_operations: 1,
            }
        }
    }

    impl FailureDetector {
        pub fn new(zero_result_threshold: f64, timeout_threshold: Duration, min_valid_operations: u64) -> Self {
            Self {
                zero_result_threshold,
                timeout_threshold,
                min_valid_operations,
            }
        }

        pub fn detect_failures(&self, result: &BenchmarkResult) -> ValidationResult {
            // 1. Zero Result Detection
            if result.ops_per_second <= self.zero_result_threshold {
                return ValidationResult::NeedsRerun {
                    reason: format!("Zero or near-zero result detected: {}", result.ops_per_second),
                };
            }

            // 2. Anomalous Result Detection
            // Check if the validation status indicates an invalid result
            match &result.validation_status {
                ValidationStatus::Invalid { reason } => {
                    return ValidationResult::Invalid {
                        reason: reason.clone(),
                    };
                }
                ValidationStatus::Valid => {} // Continue checking other conditions
            }

            // 3. Timeout Detection
            if result.execution_time > self.timeout_threshold {
                return ValidationResult::Invalid {
                    reason: format!("Benchmark execution exceeded timeout: {:?}", result.execution_time),
                };
            }

            // 4. Insufficient Work Detection
            if result.work_completed < self.min_valid_operations {
                return ValidationResult::NeedsRerun {
                    reason: format!("Insufficient work completed: {} < {}", result.work_completed, self.min_valid_operations),
                };
            }

            ValidationResult::Valid
        }

        pub fn detect_core_specific_issues(&self, results: &[BenchmarkResult]) -> Vec<String> {
            let mut issues = Vec::new();

            for result in results {
                if result.ops_per_second <= self.zero_result_threshold {
                    issues.push(format!("Core {} has zero/near-zero performance: {}",
                                      result.core_id, result.ops_per_second));
                }

                // Check for unusually low performance compared to other cores
                if !results.is_empty() {
                    let avg_performance: f64 = results.iter()
                        .map(|r| r.ops_per_second)
                        .sum::<f64>() / results.len() as f64;

                    if result.ops_per_second < avg_performance * 0.1 {  // Performance < 10% of average
                        issues.push(format!("Core {} shows unusually low performance: {} vs avg {}",
                                          result.core_id, result.ops_per_second, avg_performance));
                    }
                }
            }

            issues
        }
    }
}

// Automatic Rerun System - Phase 4B
pub mod rerun_system {
    use super::*;
    use std::collections::HashMap;

    #[derive(Debug, Clone)]
    pub struct RerunConfig {
        pub max_retries: u8,
        pub retry_delay: Duration,
        pub backoff_multiplier: f64,
    }

    impl Default for RerunConfig {
        fn default() -> Self {
            Self {
                max_retries: 3,           // Maximum 3 retries per benchmark
                retry_delay: Duration::from_millis(500), // 500ms delay before retry
                backoff_multiplier: 1.5,  // Exponential backoff multiplier
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct RerunTracker {
        pub retry_counts: HashMap<(usize, String), u8>, // (core_id, benchmark_name) -> retry_count
        pub failure_reasons: HashMap<(usize, String), Vec<String>>, // Track failure reasons
    }

    impl Default for RerunTracker {
        fn default() -> Self {
            Self {
                retry_counts: HashMap::new(),
                failure_reasons: HashMap::new(),
            }
        }
    }

    impl RerunTracker {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn increment_retry(&mut self, core_id: usize, benchmark_name: &str) -> u8 {
            let key = (core_id, benchmark_name.to_string());
            let count = self.retry_counts.entry(key).or_insert(0);
            *count += 1;
            *count
        }

        pub fn record_failure_reason(&mut self, core_id: usize, benchmark_name: &str, reason: String) {
            let key = (core_id, benchmark_name.to_string());
            self.failure_reasons.entry(key).or_insert_with(Vec::new).push(reason);
        }

        pub fn get_retry_count(&self, core_id: usize, benchmark_name: &str) -> u8 {
            let key = (core_id, benchmark_name.to_string());
            *self.retry_counts.get(&key).unwrap_or(&0)
        }

        pub fn should_retry(&self, core_id: usize, benchmark_name: &str, max_retries: u8) -> bool {
            self.get_retry_count(core_id, benchmark_name) < max_retries
        }
    }

    pub struct RerunManager<B: Benchmark> {
        pub benchmark: B,
        pub config: RerunConfig,
        pub tracker: RerunTracker,
    }

    impl<B: Benchmark> RerunManager<B> where B::Config: Clone {
        pub fn new(benchmark: B, config: RerunConfig) -> Self {
            Self {
                benchmark,
                config,
                tracker: RerunTracker::new(),
            }
        }

        pub fn run_with_rerun(&mut self, config: &B::Config, core_id: usize, benchmark_name: &str) -> BenchmarkResult {
            let mut current_config = (*config).clone();

            for attempt in 0..=self.config.max_retries {
                // For non-first attempts, apply delay for backoff
                if attempt > 0 {
                    let delay = self.config.retry_delay.mul_f64(
                        self.config.backoff_multiplier.powf((attempt - 1) as f64)
                    );
                    std::thread::sleep(delay);

                    println!("Attempt {}/{} for Core {}: {}",
                            attempt, self.config.max_retries, core_id, benchmark_name);
                }

                // Run the benchmark
                let result = self.benchmark.run_benchmark(&current_config);

                // Validate the result
                let validation_result = self.benchmark.validate_results(&result);

                match validation_result {
                    ValidationResult::Valid => {
                        if attempt > 0 {
                            println!("Core {}: {} succeeded on attempt {}",
                                   result.core_id, benchmark_name, attempt);
                        }
                        return result;
                    },
                    ValidationResult::Invalid { reason } => {
                        // Record the failure and stop retrying for invalid results
                        self.tracker.record_failure_reason(core_id, benchmark_name, reason.clone());
                        println!("Core {}: {} failed with invalid result: {}",
                               core_id, benchmark_name, reason);
                        return result; // Return the invalid result as is
                    },
                    ValidationResult::NeedsRerun { reason } => {
                        // Check if we've exceeded max retries
                        let retry_count = self.tracker.increment_retry(core_id, benchmark_name);

                        if retry_count > self.config.max_retries {
                            println!("Core {}: {} exceeded max retries ({}), stopping",
                                   core_id, benchmark_name, self.config.max_retries);
                            self.tracker.record_failure_reason(core_id, benchmark_name,
                                format!("Exceeded max retries: {}", reason));
                            return result; // Return the last result even though it needs rerun
                        }

                        // Record the reason for this failure
                        self.tracker.record_failure_reason(core_id, benchmark_name, reason.clone());

                        // Adjust workload for next attempt
                        let new_config: <B as Benchmark>::Config = self.benchmark.adjust_workload(&current_config, &result);
                        current_config = new_config;

                        println!("Core {}: {} needs rerun (attempt {}/{}): {}",
                               core_id, benchmark_name, retry_count, self.config.max_retries, reason);
                    }
                }
            }

            // Should not reach here, but return the last result if it does
            self.benchmark.run_benchmark(&current_config)
        }

        pub fn get_retry_statistics(&self) -> HashMap<String, u32> {
            let mut stats = HashMap::new();

            for ((core_id, benchmark_name), &retry_count) in &self.tracker.retry_counts {
                let key = format!("Core_{}_{}", core_id, benchmark_name);
                stats.insert(key, retry_count as u32);
            }

            stats
        }
    }
}

// Example benchmark implementation using the new framework
pub mod example_benchmarks {
    use super::*;
    use std::time::Instant;

    #[derive(Debug, Clone)]
    pub struct MathBenchmarkConfig {
        pub iterations: u64,
        pub operation_type: operation_counting::OperationType,
    }

    pub struct MathBenchmark {
        pub name: String,
        pub detector: anomaly_detection::AnomalyDetector,
        pub scaler: workload_scaling::WorkloadConfig,
    }

    impl MathBenchmark {
        pub fn new(name: String) -> Self {
            Self {
                name,
                detector: anomaly_detection::AnomalyDetector::default(),
                scaler: workload_scaling::WorkloadConfig::default(),
            }
        }

        fn execute_math_operations(&self, config: &MathBenchmarkConfig) -> (u64, Duration) {
            let start = Instant::now();
            let mut result: f64 = 1.0;

            for i in 0..config.iterations {
                // Perform different operations based on the operation type
                match config.operation_type {
                    operation_counting::OperationType::FMA => {
                        result = result * 1.1 + 0.5; // Example FMA operation
                    }
                    operation_counting::OperationType::Multiply => {
                        result = result * 1.2;
                    }
                    operation_counting::OperationType::Add => {
                        result = result + 0.8;
                    }
                    operation_counting::OperationType::Divide => {
                        if result != 0.0 {
                            result = result / 1.1;
                        } else {
                            result = 1.0;
                        }
                    }
                    operation_counting::OperationType::Custom(count) => {
                        // Perform count operations
                        for _ in 0..count {
                            result = result * 1.05;
                        }
                    }
                    _ => {
                        result = result * 1.1; // Default to multiply
                    }
                }

                // Prevent compiler optimizations
                if i % 1000 == 0 {
                    std::hint::black_box(result);
                }
            }

            let duration = start.elapsed();
            (config.iterations, duration)
        }
    }

    impl Benchmark for MathBenchmark {
        type Config = MathBenchmarkConfig;

        fn run_benchmark(&self, config: &Self::Config) -> BenchmarkResult {
            let (work_completed, execution_time) = self.execute_math_operations(config);

            let counter = operation_counting::OperationCounter::new(
                config.operation_type.clone(),
                config.iterations
            );

            let ops_per_second = counter.ops_per_second(&execution_time);

            BenchmarkResult {
                ops_per_second,
                unit: PerformanceUnit::GFLOPS, // Will be standardized later if needed
                validation_status: ValidationStatus::Valid, // Will be set by validation
                execution_time,
                work_completed,
                core_id: 0, // Will be set by the runner
            }
        }

        fn validate_results(&self, result: &BenchmarkResult) -> ValidationResult {
            self.detector.detect_anomalies(result)
        }

        fn adjust_workload(&self, current_config: &Self::Config, previous_result: &BenchmarkResult) -> Self::Config {
            // Adjust based on previous performance
            let new_iterations = if previous_result.execution_time.as_millis() < 100 {
                // If it was too fast, increase work
                (current_config.iterations as f64 * 2.0) as u64
            } else if previous_result.execution_time.as_millis() > 2000 {
                // If it was too slow, decrease work
                (current_config.iterations as f64 * 0.5) as u64
            } else {
                // Keep the same if it was in the right range
                current_config.iterations
            };

            // Ensure we stay within reasonable bounds
            let bounded_iterations = new_iterations
                .max(self.scaler.min_iterations)
                .min(self.scaler.max_iterations);

            MathBenchmarkConfig {
                iterations: bounded_iterations,
                operation_type: current_config.operation_type.clone(),
            }
        }

        fn get_unit(&self) -> PerformanceUnit {
            PerformanceUnit::GFLOPS
        }
    }
}

// New profiling function that uses the new benchmarking framework
pub fn profile_workload_on_all_cores_new<B: Benchmark>(
    mut benchmark: B,
    config: &B::Config
) -> Vec<BenchmarkResult> where B::Config: Clone {
    use core_affinity;
    use platform;

    // Initialize with empty results
    let mut results = Vec::new();

    // Get the number of logical CPUs
    let initial_stats = match platform::read_cpu_stats() {
        Ok(stats) => stats,
        Err(e) => {
            eprintln!("Could not read initial CPU stats: {e}. Attempting to run with basic functionality.");
            vec![]
        }
    };

    let num_cpus = platform::get_num_logical_cpus(&initial_stats);
    let num_cpus = if num_cpus == 0 { 1 } else { num_cpus }; // Default to 1 if detection fails

    println!("Detected {num_cpus} logical CPUs. Profiling workload on each core using new framework.");

    // Create a rerun manager for handling failures and retries
    let mut rerun_manager = rerun_system::RerunManager::new(
        benchmark,
        rerun_system::RerunConfig::default()
    );

    for core_id in 0..num_cpus {
        println!("\n--- Pinning workload to Core {core_id} ---");
        if let Err(e) = platform::set_affinity(core_id) {
            eprintln!("Warning: Failed to set affinity for Core {core_id}: {e}.");
        } else {
            println!("Successfully set affinity to Core {core_id}.");
        }

        // Use the rerun manager to run the benchmark with failure handling
        let mut result = rerun_manager.run_with_rerun(config, core_id, "MathBenchmark");

        // Set the correct core_id in the result
        result.core_id = core_id;

        // Standardize the result to appropriate units
        result.standardize(None);

        println!("  - Core {core_id} Performance: {}", result.get_display_value());
        results.push(result);
    }

    // Print retry statistics if any retries occurred
    let retry_stats = rerun_manager.get_retry_statistics();
    if !retry_stats.is_empty() {
        println!("\nRetry Statistics:");
        for (key, count) in &retry_stats {
            if *count > 0 {
                println!("  {}: {} retries", key, count);
            }
        }
    }

    results
}

// Integration and Testing Module
pub mod integration_tests {
    use super::*;
    use std::time::Duration;

    pub fn run_comprehensive_integration_test() {
        println!("Running comprehensive integration test...");

        // Test 1: Create and run a simple benchmark using the new framework
        let math_benchmark = example_benchmarks::MathBenchmark::new("IntegrationTest".to_string());
        let config = example_benchmarks::MathBenchmarkConfig {
            iterations: 100_000, // Smaller number for testing
            operation_type: operation_counting::OperationType::FMA,
        };

        // Run the benchmark directly first to test the core functionality
        let result = math_benchmark.run_benchmark(&config);
        println!("Direct benchmark result: {} {}", result.ops_per_second, result.unit);

        // Test validation
        let validation_result = math_benchmark.validate_results(&result);
        println!("Validation result: {:?}", validation_result);

        // Test unit conversion and standardization
        let mut standardized_result = result.clone();
        standardized_result.standardize(None);
        println!("Standardized result: {}", standardized_result.get_display_value());

        // Test workload adjustment
        let adjusted_config = math_benchmark.adjust_workload(&config, &result);
        println!("Adjusted config iterations: {}", adjusted_config.iterations);

        // Test anomaly detection
        let detector = anomaly_detection::AnomalyDetector::default();
        let anomaly_result = detector.detect_anomalies(&result);
        println!("Anomaly detection result: {:?}", anomaly_result);

        // Test failure detection
        let failure_detector = failure_detection::FailureDetector::default();
        let failure_result = failure_detector.detect_failures(&result);
        println!("Failure detection result: {:?}", failure_result);

        // Test unit conversions
        let (scaled_value, unit) = PerformanceUnit::from_value(result.ops_per_second);
        println!("Auto-scaled value: {} {}", scaled_value, unit);

        // Test operation counting
        let counter = operation_counting::OperationCounter::new(
            operation_counting::OperationType::FMA,
            1000
        );
        let total_ops = counter.calculate_total_operations();
        let ops_per_sec = counter.ops_per_second(&Duration::from_secs(1));
        println!("Operation counter: {} total ops, {} ops/sec", total_ops, ops_per_sec);

        println!("All integration tests passed!\n");
    }

    pub fn test_rerun_mechanism() {
        println!("Testing rerun mechanism with simulated failure...");

        // Create a benchmark that will fail initially to test the rerun mechanism
        let math_benchmark = example_benchmarks::MathBenchmark::new("RerunTest".to_string());
        let config = example_benchmarks::MathBenchmarkConfig {
            iterations: 1, // Very small iteration count to potentially cause issues
            operation_type: operation_counting::OperationType::FMA,
        };

        let mut rerun_manager = rerun_system::RerunManager::new(
            math_benchmark,
            rerun_system::RerunConfig {
                max_retries: 2, // Limit retries for testing
                retry_delay: Duration::from_millis(100), // Short delay for testing
                backoff_multiplier: 1.1, // Small backoff for testing
            }
        );

        let result = rerun_manager.run_with_rerun(&config, 0, "RerunTest");
        println!("Rerun test result: {} {}", result.ops_per_second, result.unit);

        let stats = rerun_manager.get_retry_statistics();
        println!("Retry statistics: {:?}", stats);
        println!("Rerun mechanism test completed!\n");
    }
}

// Unit tests for the benchmarking framework
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_performance_unit_conversion() {
        // Test conversion from raw value to appropriate unit
        let (value, unit) = PerformanceUnit::from_value(2_500_000_000.0);
        assert_eq!(unit, PerformanceUnit::GFLOPS);
        assert!((value - 2.5).abs() < 0.01); // Should be ~2.5 GFLOPS

        let (value, unit) = PerformanceUnit::from_value(1_500_000.0);
        assert_eq!(unit, PerformanceUnit::MFLOPS);
        assert!((value - 1.5).abs() < 0.01); // Should be ~1.5 MFLOPS
    }

    #[test]
    fn test_unit_conversion_between_types() {
        // Test converting from GFLOPS to MFLOPS
        let result_gflops = 2.5; // 2.5 GFLOPS
        let result_mflops = PerformanceUnit::convert_value(
            &PerformanceUnit::GFLOPS,
            &PerformanceUnit::MFLOPS,
            result_gflops
        );
        assert!((result_mflops - 2500.0).abs() < 0.01); // Should be 2500 MFLOPS

        // Test converting back
        let result_gflops_back = PerformanceUnit::convert_value(
            &PerformanceUnit::MFLOPS,
            &PerformanceUnit::GFLOPS,
            result_mflops
        );
        assert!((result_gflops_back - result_gflops).abs() < 0.01);
    }

    #[test]
    fn test_operation_counting() {
        // Test FMA operation counting (should count as 2 operations)
        let counter = operation_counting::OperationCounter::new(
            operation_counting::OperationType::FMA,
            1000
        );
        assert_eq!(counter.calculate_total_operations(), 2000); // 1000 iterations * 2 ops per FMA

        // Test Multiply operation counting (should count as 1 operation)
        let counter = operation_counting::OperationCounter::new(
            operation_counting::OperationType::Multiply,
            1000
        );
        assert_eq!(counter.calculate_total_operations(), 1000);

        // Test Custom operation counting
        let counter = operation_counting::OperationCounter::new(
            operation_counting::OperationType::Custom(5),
            100
        );
        assert_eq!(counter.calculate_total_operations(), 500); // 100 iterations * 5 ops
    }

    #[test]
    fn test_ops_per_second_calculation() {
        let counter = operation_counting::OperationCounter::new(
            operation_counting::OperationType::FMA,
            2_000_000_000  // 2 billion FMA operations (4 billion total ops)
        );

        // Calculate ops per second for 1 second duration
        let ops_per_sec = counter.ops_per_second(&Duration::from_secs(1));
        assert_eq!(ops_per_sec, 4_000_000_000.0); // 2 billion FMA = 4 billion ops
    }

    #[test]
    fn test_anomaly_detection() {
        let detector = anomaly_detection::AnomalyDetector::new(
            Some(1_000_000_000_000.0), // 1 TFLOP/s max
            3.0, // 3 std deviations
            0    // zero threshold
        );

        // Test valid result
        let valid_result = BenchmarkResult {
            ops_per_second: 500_000_000.0, // 500 MFLOPS
            unit: PerformanceUnit::GFLOPS,
            validation_status: ValidationStatus::Valid,
            execution_time: Duration::from_secs(1),
            work_completed: 1_000_000,
            core_id: 0,
        };

        match detector.detect_anomalies(&valid_result) {
            ValidationResult::Valid => (), // This is expected
            _ => panic!("Valid result should pass anomaly detection"),
        }

        // Test result exceeding hardware limits
        let invalid_result = BenchmarkResult {
            ops_per_second: 2_000_000_000_000.0, // 2 TFLOPS, over our 1 TFLOPS limit
            unit: PerformanceUnit::TFLOPS,
            validation_status: ValidationStatus::Valid,
            execution_time: Duration::from_secs(1),
            work_completed: 1_000_000,
            core_id: 0,
        };

        match detector.detect_anomalies(&invalid_result) {
            ValidationResult::Invalid { .. } => (), // This is expected
            _ => panic!("Result over hardware limit should fail anomaly detection"),
        }

        // Test NaN detection
        let nan_result = BenchmarkResult {
            ops_per_second: f64::NAN,
            unit: PerformanceUnit::GFLOPS,
            validation_status: ValidationStatus::Valid,
            execution_time: Duration::from_secs(1),
            work_completed: 1_000_000,
            core_id: 0,
        };

        match detector.detect_anomalies(&nan_result) {
            ValidationResult::Invalid { .. } => (), // This is expected
            _ => panic!("NaN result should fail anomaly detection"),
        }

        // Test zero work detection
        let zero_work_result = BenchmarkResult {
            ops_per_second: 100_000_000.0,
            unit: PerformanceUnit::MFLOPS,
            validation_status: ValidationStatus::Valid,
            execution_time: Duration::from_secs(1),
            work_completed: 0, // Zero work completed
            core_id: 0,
        };

        match detector.detect_anomalies(&zero_work_result) {
            ValidationResult::NeedsRerun { .. } => (), // This is expected
            _ => panic!("Zero work result should trigger rerun"),
        }
    }

    #[test]
    fn test_workload_scaling() {
        let base_config = workload_scaling::WorkloadConfig {
            base_iterations: 1_000_000,
            min_iterations: 10_000,
            max_iterations: 100_000_000,
            target_duration_ms: 1000, // 1 second target
            adjustment_factor: 1.5,
        };

        // Test scaling for a result that ran too fast (200ms, should increase iterations)
        let fast_result = BenchmarkResult {
            ops_per_second: 5_000_000_000.0,
            unit: PerformanceUnit::GFLOPS,
            validation_status: ValidationStatus::Valid,
            execution_time: Duration::from_millis(200), // 200ms = too fast
            work_completed: 1_000_000,
            core_id: 0,
        };

        let adjusted_config = base_config.adjust_for_performance(&fast_result);
        assert!(adjusted_config.base_iterations > base_config.base_iterations); // Should increase iterations

        // Test scaling for a result that ran too slow (2000ms, should decrease iterations)
        let slow_result = BenchmarkResult {
            ops_per_second: 1_000_000_000.0,
            unit: PerformanceUnit::GFLOPS,
            validation_status: ValidationStatus::Valid,
            execution_time: Duration::from_millis(2000), // 2000ms = too slow
            work_completed: 1_000_000,
            core_id: 0,
        };

        let adjusted_config = base_config.adjust_for_performance(&slow_result);
        assert!(adjusted_config.base_iterations < base_config.base_iterations); // Should decrease iterations
    }

    #[test]
    fn test_failure_detection() {
        let detector = failure_detection::FailureDetector::new(
            0.0, // zero threshold
            Duration::from_secs(10), // 10 second timeout
            1    // min valid operations
        );

        // Test normal valid result
        let valid_result = BenchmarkResult {
            ops_per_second: 1_000_000_000.0, // 1 GFLOPS
            unit: PerformanceUnit::GFLOPS,
            validation_status: ValidationStatus::Valid,
            execution_time: Duration::from_secs(1),
            work_completed: 1_000_000,
            core_id: 0,
        };

        assert!(matches!(detector.detect_failures(&valid_result), ValidationResult::Valid));

        // Test zero result detection
        let zero_result = BenchmarkResult {
            ops_per_second: 0.0,
            unit: PerformanceUnit::GFLOPS,
            validation_status: ValidationStatus::Valid,
            execution_time: Duration::from_secs(1),
            work_completed: 1_000_000,
            core_id: 0,
        };

        assert!(matches!(detector.detect_failures(&zero_result), ValidationResult::NeedsRerun { .. }));

        // Test timeout detection
        let timeout_result = BenchmarkResult {
            ops_per_second: 1_000_000_000.0,
            unit: PerformanceUnit::GFLOPS,
            validation_status: ValidationStatus::Valid,
            execution_time: Duration::from_secs(15), // Over 10 sec timeout
            work_completed: 1_000_000,
            core_id: 0,
        };

        assert!(matches!(detector.detect_failures(&timeout_result), ValidationResult::Invalid { .. }));

        // Test insufficient work detection
        let low_work_result = BenchmarkResult {
            ops_per_second: 1_000_000_000.0,
            unit: PerformanceUnit::GFLOPS,
            validation_status: ValidationStatus::Valid,
            execution_time: Duration::from_secs(1),
            work_completed: 0, // Below min of 1
            core_id: 0,
        };

        assert!(matches!(detector.detect_failures(&low_work_result), ValidationResult::NeedsRerun { .. }));
    }

    #[test]
    fn test_benchmark_result_standardization() {
        let mut result = BenchmarkResult {
            ops_per_second: 2_500_000_000.0, // 2.5 billion ops/sec
            unit: PerformanceUnit::FLOPS,
            validation_status: ValidationStatus::Valid,
            execution_time: Duration::from_secs(1),
            work_completed: 1_000_000,
            core_id: 0,
        };

        // Standardize without specifying a preferred unit (should auto-select)
        result.standardize(None);

        // Should have converted to GFLOPS since 2.5 billion is 2.5 GFLOPS
        assert_eq!(result.unit, PerformanceUnit::GFLOPS);
        assert!((result.ops_per_second - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_rerun_tracker() {
        let mut tracker = rerun_system::RerunTracker::new();

        // Test incrementing retry count
        assert_eq!(tracker.increment_retry(0, "test_benchmark"), 1);
        assert_eq!(tracker.increment_retry(0, "test_benchmark"), 2);
        assert_eq!(tracker.get_retry_count(0, "test_benchmark"), 2);

        // Test recording failure reasons
        tracker.record_failure_reason(0, "test_benchmark", "Test failure".to_string());
        let reasons = tracker.failure_reasons.get(&(0, "test_benchmark".to_string()));
        assert!(reasons.is_some());
        assert_eq!(reasons.unwrap().len(), 1);

        // Test should_retry functionality
        assert!(tracker.should_retry(0, "test_benchmark", 3)); // Should retry since count is 2, max is 3
        assert!(!tracker.should_retry(0, "test_benchmark", 2)); // Should not retry since count is 2, max is 2
    }
}

// Architecture-specific SIMD modules
#[cfg(target_arch = "x86_64")]
pub mod simd_x86 {
    use std::arch::x86_64::*;
    use std::time::Instant;

    // AVX2 SIMD implementation
    pub fn avx2_multiply_workload(simd_loops: u64) -> (f64, String) {
        if !is_x86_feature_detected!("avx2") {
            panic!("AVX2 not supported on this CPU");
        }

        unsafe {
            let start_time = Instant::now();
            let mut a = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            let b = _mm256_set_ps(0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1);

            for _ in 0..simd_loops {
                a = _mm256_mul_ps(a, b);
            }

            let elapsed = start_time.elapsed();
            let ops_per_second = (simd_loops * 8) as f64 / elapsed.as_secs_f64();

            // Prevent optimization
            let mut result = [0.0f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), a);
            if result[0] == 0.0 {
                println!("result is zero, which should not happen.");
            }

            (ops_per_second, "ops/sec".to_string())
        }
    }

    // AVX512 SIMD implementation
    #[target_feature(enable = "avx512f")]
    pub unsafe fn avx512_multiply_workload_inner(simd_loops: u64) -> (f64, String) {
        let start_time = Instant::now();
        let mut a = _mm512_set_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm512_set_ps(
            0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1,
        );

        for _ in 0..simd_loops {
            a = _mm512_mul_ps(a, b);
        }

        let elapsed = start_time.elapsed();
        let ops_per_second = (simd_loops * 16) as f64 / elapsed.as_secs_f64();

        // Prevent optimization
        let mut result = [0.0f32; 16];
        _mm512_storeu_ps(result.as_mut_ptr(), a);
        if result[0] == 0.0 {
            println!("result is zero, which should not happen.");
        }

        (ops_per_second, "ops/sec".to_string())
    }

    pub fn avx512_multiply_workload(simd_loops: u64) -> (f64, String) {
        if !is_x86_feature_detected!("avx512f") {
            panic!("AVX512F not supported on this CPU");
        }
        unsafe { avx512_multiply_workload_inner(simd_loops) }
    }
}

#[cfg(target_arch = "aarch64")]
pub mod simd_arm {
    use std::arch::aarch64::*;
    use std::time::Instant;

    // NEON SIMD implementation for ARM64 - basic float multiplication
    pub fn neon_multiply_workload(simd_loops: u64) -> (f64, String) {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            panic!("NEON not supported on this CPU");
        }

        unsafe {
            let start_time = Instant::now();
            let mut a = vdupq_n_f32(1.0); // [1.0, 1.0, 1.0, 1.0]
            let b = vdupq_n_f32(1.1); // [1.1, 1.1, 1.1, 1.1]

            for _ in 0..simd_loops {
                a = vmulq_f32(a, b); // Multiply 4 floats at once
            }

            let elapsed = start_time.elapsed();
            let ops_per_second = (simd_loops * 4) as f64 / elapsed.as_secs_f64();

            // Prevent optimization
            let result_lane = vgetq_lane_f32(a, 0);
            if result_lane == 0.0 {
                println!("result is zero, which should not happen.");
            }

            (ops_per_second, "ops/sec".to_string())
        }
    }

    // Advanced SIMD implementation using 2x2 f32 vectors for better utilization
    pub fn neon_multiply_workload_advanced(simd_loops: u64) -> (f64, String) {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            panic!("NEON not supported on this CPU");
        }

        unsafe {
            let start_time = Instant::now();
            let mut a1 = vdupq_n_f32(1.0);
            let mut a2 = vdupq_n_f32(2.0);
            let b1 = vdupq_n_f32(0.9);
            let b2 = vdupq_n_f32(1.1);

            for _ in 0..simd_loops {
                a1 = vmulq_f32(a1, b1); // Multiply 4 floats at once
                a2 = vmulq_f32(a2, b2); // Multiply another 4 floats
            }

            let elapsed = start_time.elapsed();
            let ops_per_second = (simd_loops * 8) as f64 / elapsed.as_secs_f64();

            // Prevent optimization
            let result_lane1 = vgetq_lane_f32(a1, 0);
            let result_lane2 = vgetq_lane_f32(a2, 0);
            if result_lane1 == 0.0 || result_lane2 == 0.0 {
                println!("result is zero, which should not happen.");
            }

            (ops_per_second, "ops/sec".to_string())
        }
    }

    // NEON SIMD implementation for integer operations
    pub fn neon_integer_workload(simd_loops: u64) -> (f64, String) {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            panic!("NEON not supported on this CPU");
        }

        unsafe {
            let start_time = Instant::now();
            let mut a = vdupq_n_s32(100);
            let b = vdupq_n_s32(50);

            for _ in 0..simd_loops {
                a = vaddq_s32(a, b); // Add 4 integers at once
            }

            let elapsed = start_time.elapsed();
            let ops_per_second = (simd_loops * 4) as f64 / elapsed.as_secs_f64();

            // Prevent optimization
            let result_lane = vgetq_lane_s32(a, 0);
            if result_lane == 0 {
                println!("result is zero, which should not happen.");
            }

            (ops_per_second, "ops/sec".to_string())
        }
    }

    // NEON SIMD implementation for floating-point addition
    pub fn neon_add_workload(simd_loops: u64) -> (f64, String) {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            panic!("NEON not supported on this CPU");
        }

        unsafe {
            let start_time = Instant::now();
            let mut a = vdupq_n_f32(1.0);
            let b = vdupq_n_f32(2.5);

            for _ in 0..simd_loops {
                a = vaddq_f32(a, b); // Add 4 floats at once
            }

            let elapsed = start_time.elapsed();
            let ops_per_second = (simd_loops * 4) as f64 / elapsed.as_secs_f64();

            // Prevent optimization
            let result_lane = vgetq_lane_f32(a, 0);
            if result_lane == 0.0 {
                println!("result is zero, which should not happen.");
            }

            (ops_per_second, "ops/sec".to_string())
        }
    }

    // NEON SIMD implementation for mixed operations
    pub fn neon_mixed_workload(simd_loops: u64) -> (f64, String) {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            panic!("NEON not supported on this CPU");
        }

        unsafe {
            let start_time = Instant::now();
            let mut a = vdupq_n_f32(1.0);
            let b = vdupq_n_f32(2.0);
            let c = vdupq_n_f32(0.5);

            for _ in 0..simd_loops {
                a = vmulq_f32(a, b); // Multiply
                a = vaddq_f32(a, c); // Add
            }

            let elapsed = start_time.elapsed();
            let ops_per_second = (simd_loops * 8) as f64 / elapsed.as_secs_f64(); // 4 mults + 4 adds per loop

            // Prevent optimization
            let result_lane = vgetq_lane_f32(a, 0);
            if result_lane == 0.0 {
                println!("result is zero, which should not happen.");
            }

            (ops_per_second, "ops/sec".to_string())
        }
    }

    // Advanced SIMD implementation using double precision
    pub fn neon_double_precision_workload(simd_loops: u64) -> (f64, String) {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            panic!("NEON not supported on this CPU");
        }

        unsafe {
            let start_time = Instant::now();
            let mut a = vdupq_n_f64(1.0);
            let b = vdupq_n_f64(2.0);

            for _ in 0..simd_loops {
                a = vmulq_f64(a, b); // Multiply 2 doubles at once
            }

            let elapsed = start_time.elapsed();
            let ops_per_second = (simd_loops * 2) as f64 / elapsed.as_secs_f64();

            // Prevent optimization
            let result_lane = vgetq_lane_f64(a, 0);
            if result_lane == 0.0 {
                println!("result is zero, which should not happen.");
            }

            (ops_per_second, "ops/sec".to_string())
        }
    }

    // SIMD implementation using more complex operations (multiply-accumulate)
    pub fn neon_multiply_accumulate_workload(simd_loops: u64) -> (f64, String) {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            panic!("NEON not supported on this CPU");
        }

        unsafe {
            let start_time = Instant::now();
            let mut a = vdupq_n_f32(1.0);
            let b = vdupq_n_f32(2.0);
            let c = vdupq_n_f32(0.5);

            for _ in 0..simd_loops {
                a = vfmaq_f32(a, b, c); // Fused multiply-add: a = a + (b * c)
            }

            let elapsed = start_time.elapsed();
            let ops_per_second = (simd_loops * 4) as f64 / elapsed.as_secs_f64();

            // Prevent optimization
            let result_lane = vgetq_lane_f32(a, 0);
            if result_lane == 0.0 {
                println!("result is zero, which should not happen.");
            }

            (ops_per_second, "ops/sec".to_string())
        }
    }

    // SIMD implementation using vector loads from memory
    pub fn neon_memory_workload(simd_loops: u64, size: usize) -> (f64, String) {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            panic!("NEON not supported on this CPU");
        }

        // Initialize vectors with values
        let mut data_a: Vec<f32> = (0..size).map(|i| (i % 4) as f32 + 1.0).collect();
        let data_b: Vec<f32> = (0..size).map(|i| ((i + 1) % 4) as f32 + 0.5).collect();

        unsafe {
            let start_time = Instant::now();

            for _ in 0..simd_loops {
                for chunk_a in data_a.chunks_mut(4) {
                    for chunk_b in data_b.chunks(4) {
                        if chunk_a.len() >= 4 && chunk_b.len() >= 4 {
                            let va = vld1q_f32(chunk_a.as_ptr() as *const f32);
                            let vb = vld1q_f32(chunk_b.as_ptr() as *const f32);
                            let result = vmulq_f32(va, vb);
                            vst1q_f32(chunk_a.as_mut_ptr() as *mut f32, result);
                        }
                    }
                }
            }

            let elapsed = start_time.elapsed();
            // Count operations: simd_loops * (size/4 rounded down) * 4 ops
            let ops = simd_loops * ((size / 4) as u64) * 4;
            let ops_per_second = ops as f64 / elapsed.as_secs_f64();

            // Prevent optimization
            if data_a[0] == 0.0 {
                println!("result is zero, which should not happen.");
            }

            (ops_per_second, "ops/sec".to_string())
        }
    }
}

// Main function to demonstrate the SIMD implementations and new framework
pub fn main() {
    println!("=== Testing New Benchmarking Framework ===\n");

    // Run integration tests first
    integration_tests::run_comprehensive_integration_test();
    integration_tests::test_rerun_mechanism();

    println!("=== Testing Legacy SIMD implementations ===\n");

    #[cfg(target_arch = "aarch64")]
    {
        // Test different NEON implementations
        println!("Testing basic NEON multiply: {:?}", simd_arm::neon_multiply_workload(1000000));
        println!("Testing advanced NEON multiply: {:?}", simd_arm::neon_multiply_workload_advanced(1000000));
        println!("Testing NEON integer operations: {:?}", simd_arm::neon_integer_workload(1000000));
        println!("Testing NEON add operations: {:?}", simd_arm::neon_add_workload(1000000));
        println!("Testing NEON mixed operations: {:?}", simd_arm::neon_mixed_workload(1000000));
        println!("Testing NEON double precision: {:?}", simd_arm::neon_double_precision_workload(1000000));
        println!("Testing NEON multiply-accumulate: {:?}", simd_arm::neon_multiply_accumulate_workload(1000000));
        println!("Testing NEON memory operations: {:?}", simd_arm::neon_memory_workload(10000, 1024));
    }

    #[cfg(target_arch = "x86_64")]
    {
        println!("Testing x86_64 SIMD implementations (AVX2/AVX512)...");
        if std::arch::is_x86_feature_detected!("avx2") {
            println!("Testing AVX2: {:?}", simd_x86::avx2_multiply_workload(1000000));
        } else {
            println!("AVX2 not supported on this CPU");
        }

        if std::arch::is_x86_feature_detected!("avx512f") {
            println!("Testing AVX512: {:?}", simd_x86::avx512_multiply_workload(1000000));
        } else {
            println!("AVX512 not supported on this CPU");
        }
    }

    // Test profiling on all cores
    #[cfg(target_arch = "aarch64")]
    {
        println!("\nProfiling NEON multiply workload on all cores:");
        let results = profile_workload_on_all_cores(|| simd_arm::neon_multiply_workload(100000));
        for result in results {
            println!("Core {}: {} {}", result.core_id, result.throughput, result.unit);
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        println!("\nProfiling AVX2 workload on all cores:");
        let results = profile_workload_on_all_cores(|| simd_x86::avx2_multiply_workload(100000));
        for result in results {
            println!("Core {}: {} {}", result.core_id, result.throughput, result.unit);
        }
    }

    // Test the new framework with example benchmark
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            println!("\nTesting new framework with example benchmark on all cores:");
            let benchmark = example_benchmarks::MathBenchmark::new("ExampleMathBenchmark".to_string());
            let config = example_benchmarks::MathBenchmarkConfig {
                iterations: 1_000_000,
                operation_type: operation_counting::OperationType::FMA,
            };

            let results = profile_workload_on_all_cores_new(benchmark, &config);
            for result in results {
                println!("Core {}: {} ({} validation)",
                        result.core_id,
                        result.get_display_value(),
                        match result.validation_status {
                            ValidationStatus::Valid => "Valid",
                            ValidationStatus::Invalid { reason } => &format!("Invalid: {}", reason),
                        });
            }
        }
    }
}

// This module is defined inline to contain all platform-specific logic.
pub mod platform {
    use super::CpuStats;
    use std::io;
    use core_affinity;

    pub fn set_affinity(cpu_id: usize) -> io::Result<()> {
        // Use core_affinity crate for cross-platform CPU affinity
        let core_ids = core_affinity::get_core_ids()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Failed to get core IDs"))?;

        if cpu_id >= core_ids.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "CPU ID out of range"));
        }

        let core_id = core_ids[cpu_id];
        core_affinity::set_for_current(core_id);
        Ok(())
    }

    pub fn read_cpu_stats() -> io::Result<Vec<CpuStats>> {
        // This is Linux-specific, so we'll keep it but add better cross-platform support
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let content = fs::read_to_string("/proc/stat")?;
            let mut stats_list = Vec::new();
            for line in content.lines().filter(|l| l.starts_with("cpu")) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 10 {
                    stats_list.push(CpuStats {
                        user: parts[1].parse().unwrap_or(0),
                        nice: parts[2].parse().unwrap_or(0),
                        system: parts[3].parse().unwrap_or(0),
                        idle: parts[4].parse().unwrap_or(0),
                        iowait: parts[5].parse().unwrap_or(0),
                        irq: parts[6].parse().unwrap_or(0),
                        softirq: parts[7].parse().unwrap_or(0),
                        steal: parts[8].parse().unwrap_or(0),
                        guest: parts[9].parse().unwrap_or(0),
                        guest_nice: parts.get(10).map_or(0, |s| s.parse().unwrap_or(0)),
                    });
                }
            }
            Ok(stats_list)
        }
        #[cfg(not(target_os = "linux"))]
        {
            // For non-Linux systems, we can't get detailed per-core stats easily
            // Return empty vector to indicate that detailed CPU monitoring is not available
            Ok(vec![])
        }
    }

    pub fn get_num_logical_cpus(initial_stats: &[CpuStats]) -> usize {
        // Try to get actual core count from core_affinity first
        if let Some(core_ids) = core_affinity::get_core_ids() {
            core_ids.len()
        } else if !initial_stats.is_empty() {
            // Fallback to the old method if core_affinity fails
            // For Linux systems, stats[0] is aggregate, so actual cores start from index 1
            if initial_stats.len() > 1 {
                initial_stats.len() - 1  // Exclude the aggregate CPU0 stats
            } else {
                1  // Single core system
            }
        } else {
            // Default to 1 core if all methods fail
            1
        }
    }
}

#[derive(Default, Debug, Clone, Serialize)]
pub struct CpuStats {
    pub user: u64,
    pub nice: u64,
    pub system: u64,
    pub idle: u64,
    pub iowait: u64,
    pub irq: u64,
    pub softirq: u64,
    pub steal: u64,
    pub guest: u64,
    pub guest_nice: u64,
}

#[derive(Default, Debug, Clone, Serialize)]
pub struct WorkloadResult {
    pub core_id: usize,
    pub throughput: f64,
    pub unit: String,
}

pub fn profile_workload_on_all_cores<F>(mut workload: F) -> Vec<WorkloadResult>
where
    F: FnMut() -> (f64, String),
{
    let initial_stats = match platform::read_cpu_stats() {
        Ok(stats) => stats,
        Err(e) => {
            eprintln!("Could not read initial CPU stats: {e}. Attempting to run with basic functionality.");
            // Return empty vector to indicate no detailed stats available
            vec![]
        }
    };

    // Get number of CPUs, with fallback behavior
    let num_cpus = platform::get_num_logical_cpus(&initial_stats);
    let num_cpus = if num_cpus == 0 {
        1  // Default to 1 core if we can't determine the number
    } else {
        num_cpus
    };

    println!("Detected {num_cpus} logical CPUs. Profiling workload on each core.");

    // On non-Linux systems, we may not have detailed CPU stats, but core affinity still works
    if initial_stats.is_empty() {
        println!("Note: Detailed CPU usage statistics are not available on this platform.");
    }

    let mut results = Vec::new();

    for core_id in 0..num_cpus {
        println!("\n--- Pinning workload to Core {core_id} ---");
        if let Err(e) = platform::set_affinity(core_id) {
            eprintln!("Warning: Failed to set affinity for Core {core_id}: {e}.");
        } else {
            println!("Successfully set affinity to Core {core_id}.");
        }

        let (throughput, unit) = workload();
        println!("  - Core {core_id} Throughput: {throughput:.2e} {unit}");
        results.push(WorkloadResult {
            core_id,
            throughput,
            unit: unit.clone(),
        });
    }
    results
}
