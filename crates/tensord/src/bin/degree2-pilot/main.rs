//! Development tool (not proof, PLAN decision 5): real stock executors on one GPU sharing GPU
//! weights through the machine's real `ResidentCustody` and seam. Measures what Degree 2 buys:
//! a replacement's first image with attached weights against a private fill, the bytes two
//! live executors hold (NVML total), the release after a revoke, and output equality.
//!
//! Linux only: elsewhere one executor's device memory is never handed to another.
#[cfg(target_os = "linux")]
mod pilot;

#[cfg(target_os = "linux")]
fn main() {
    pilot::main()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("degree2-pilot: Degree 2 hands GPU memory between processes, which only Linux does");
    std::process::exit(2);
}
