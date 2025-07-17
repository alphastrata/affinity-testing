use argh::FromArgs;
use core_profiler::platform::{self, get_num_logical_cpus, read_cpu_stats};
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

#[derive(FromArgs)]
/// A tool to demonstrate pinning multiple CPU-intensive threads to a single core.
struct AppArgs {
    /// number of threads to spawn
    #[argh(option, short = 't')]
    threads: Option<usize>,

    /// the core to pin all threads to
    #[argh(option, short = 'c', default = "0")]
    core_id: usize,

    /// duration to run the test in seconds
    #[argh(option, short = 'd', default = "10")]
    duration: u64,

    /// output file for cpu usage statistics
    #[argh(option, short = 'o')]
    output: Option<String>,
}

fn busy_work(duration: Duration) {
    let start = Instant::now();
    while start.elapsed() < duration {
        let mut _x = 0.0;
        for _ in 0..1_000_000 {
            _x += 1.0;
        }
    }
}

fn main() {
    let args: AppArgs = argh::from_env();
    let core_id_to_pin = args.core_id;
    let duration = Duration::from_secs(args.duration);

    let num_cpus = get_num_logical_cpus(&read_cpu_stats().expect("Could not read CPU stats"));

    let num_threads = args.threads.unwrap_or(num_cpus);

    println!(
        "Spawning {} threads and pinning them to core {}. Running for {} seconds.",
        num_threads,
        core_id_to_pin,
        duration.as_secs()
    );

    let barrier = Arc::new(Barrier::new(num_threads + 1));
    let mut handles = vec![];

    for i in 0..num_threads {
        let barrier = barrier.clone();
        let handle = thread::spawn(move || {
            if let Err(e) = platform::set_affinity(core_id_to_pin) {
                eprintln!("Warning: Failed to set affinity for thread {i}: {e}");
            }
            barrier.wait();
            busy_work(duration);
        });
        handles.push(handle);
    }

    let monitor_handle = if let Some(output_path) = args.output {
        let barrier = barrier.clone();
        let handle = thread::spawn(move || {
            let mut file = File::create(output_path).expect("Could not create output file");
            writeln!(
                file,
                "elapsed_ms,core_id,user_percent,system_percent,total_usage_percent"
            )
            .expect("Could not write header");

            barrier.wait();
            let start_time = Instant::now();
            let mut last_stats = read_cpu_stats().unwrap();

            while start_time.elapsed() < duration {
                thread::sleep(Duration::from_millis(100));
                let current_stats = read_cpu_stats().unwrap();

                for core_idx in 0..num_cpus {
                    // The first stat is the aggregate, so we skip it by adding 1.
                    let last_core_stats = &last_stats[core_idx + 1];
                    let current_core_stats = &current_stats[core_idx + 1];

                    let user_delta = current_core_stats.user.saturating_sub(last_core_stats.user);
                    let system_delta = current_core_stats
                        .system
                        .saturating_sub(last_core_stats.system);
                    let idle_delta = current_core_stats.idle.saturating_sub(last_core_stats.idle);

                    let total_delta = (current_core_stats.user
                        + current_core_stats.nice
                        + current_core_stats.system
                        + current_core_stats.idle
                        + current_core_stats.iowait
                        + current_core_stats.irq
                        + current_core_stats.softirq
                        + current_core_stats.steal)
                        - (last_core_stats.user
                            + last_core_stats.nice
                            + last_core_stats.system
                            + last_core_stats.idle
                            + last_core_stats.iowait
                            + last_core_stats.irq
                            + last_core_stats.softirq
                            + last_core_stats.steal);

                    let total_usage_percent = if total_delta > 0 {
                        100.0 * (1.0 - (idle_delta as f64 / total_delta as f64))
                    } else {
                        0.0
                    };

                    let user_percent = if total_delta > 0 {
                        100.0 * (user_delta as f64 / total_delta as f64)
                    } else {
                        0.0
                    };

                    let system_percent = if total_delta > 0 {
                        100.0 * (system_delta as f64 / total_delta as f64)
                    } else {
                        0.0
                    };

                    writeln!(
                        file,
                        "{},{},{:.2},{:.2},{:.2}",
                        start_time.elapsed().as_millis(),
                        core_idx,
                        user_percent,
                        system_percent,
                        total_usage_percent
                    )
                    .expect("Could not write to output file");
                }

                last_stats = current_stats;
            }
        });
        Some(handle)
    } else {
        barrier.wait();
        None
    };

    for handle in handles {
        handle.join().unwrap();
    }

    if let Some(handle) = monitor_handle {
        handle.join().unwrap();
    }

    println!("\nAll threads have completed their work.");
}
