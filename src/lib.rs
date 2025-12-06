// core_profiler/src/lib.rs

use serde::Serialize;

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

// Main function to demonstrate the SIMD implementations
pub fn main() {
    println!("Testing ARM64 SIMD implementations (NEON)...");

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
        println!("Profiling NEON multiply workload on all cores:");
        let results = profile_workload_on_all_cores(|| simd_arm::neon_multiply_workload(100000));
        for result in results {
            println!("Core {}: {} {}", result.core_id, result.throughput, result.unit);
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        println!("Profiling AVX2 workload on all cores:");
        let results = profile_workload_on_all_cores(|| simd_x86::avx2_multiply_workload(100000));
        for result in results {
            println!("Core {}: {} {}", result.core_id, result.throughput, result.unit);
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
