// affinity-testing/build.rs

fn main() {
    // Only link `mach` on macOS.
    // The `_SC_NPROCESSORS_ONLN` constant is typically in `libSystem` which is linked by default.
    // `host_cpu_load_info` and `thread_policy_set` are in `libmach`.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=mach");
    }
}
