//! The real Memory over host pages, across real processes (Unix: the agents dial a lease socket).

#[cfg(unix)]
#[path = "processes/cases.rs"]
mod cases;

#[cfg(unix)]
fn main() {
    cases::main();
}

#[cfg(not(unix))]
fn main() {
    println!("processes: the agents dial a Unix lease socket; skipped here");
}
