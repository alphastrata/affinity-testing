// core_profiler/src/lib.rs

use serde::Serialize;

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
    let initial_stats = platform::read_cpu_stats().unwrap_or_else(|e| {
        eprintln!("Could not read initial CPU stats: {e}. Exiting.");
        std::process::exit(1);
    });

    let num_cpus = platform::get_num_logical_cpus(&initial_stats);
    if num_cpus == 0 {
        println!("Could not determine the number of logical CPUs. Running on current core only.");
        let (throughput, unit) = workload();
        return vec![WorkloadResult {
            core_id: 0,
            throughput,
            unit,
        }];
    }

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
