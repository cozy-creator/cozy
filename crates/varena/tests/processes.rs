//! The real Memory, ledger, custody and RAM tier over host pages, across real processes (Unix:
//! the agents take their grants as descriptors).

#[cfg(unix)]
#[path = "processes/cases.rs"]
mod cases;

#[cfg(unix)]
fn main() {
    cases::main();
}

#[cfg(not(unix))]
fn main() {
    println!("processes: the agents take their grants as Unix descriptors; skipped here");
}
