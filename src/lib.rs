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

    // NEON SIMD implementation for ARM64
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
}

// This module is defined inline to contain all platform-specific logic.
pub mod platform {
    use super::CpuStats;
    use std::io;

    // --- Linux Specific Implementations ---
    #[cfg(target_os = "linux")]
    pub mod os_specific {
        use super::super::CpuStats; // Access CpuStats from the root of the crate
        use libc::{cpu_set_t, sched_setaffinity, CPU_SET};
        use std::fs;
        use std::io;

        pub fn set_affinity(cpu_id: usize) -> io::Result<()> {
            unsafe {
                let mut cpuset: cpu_set_t = std::mem::zeroed();
                CPU_SET(cpu_id, &mut cpuset);
                if sched_setaffinity(0, std::mem::size_of::<cpu_set_t>(), &cpuset) == 0 {
                    Ok(())
                } else {
                    Err(io::Error::last_os_error())
                }
            }
        }

        pub fn read_cpu_stats() -> io::Result<Vec<CpuStats>> {
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
    }

    // --- Fallback for other OSes ---
    #[cfg(not(target_os = "linux"))]
    pub mod os_specific {
        use super::super::CpuStats;
        use std::io;
        pub fn set_affinity(_cpu_id: usize) -> io::Result<()> {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "CPU affinity not supported on this OS",
            ))
        }
        pub fn read_cpu_stats() -> io::Result<Vec<CpuStats>> {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "CPU stats not supported on this OS",
            ))
        }
    }

    // Common platform functions that call the OS-specific implementation
    pub fn set_affinity(cpu_id: usize) -> io::Result<()> {
        os_specific::set_affinity(cpu_id)
    }

    pub fn read_cpu_stats() -> io::Result<Vec<CpuStats>> {
        os_specific::read_cpu_stats()
    }

    pub fn get_num_logical_cpus(initial_stats: &[CpuStats]) -> usize {
        if initial_stats.is_empty() {
            0
        } else {
            initial_stats.len() - 1
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
            // Create a minimal stats vector to continue with at least 1 core
            vec![CpuStats::default()]
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
