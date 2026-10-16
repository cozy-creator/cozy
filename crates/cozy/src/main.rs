//! The `cozy` binary dispatches to its roles: CLI verbs, `calcifer` and `machine serve`.
use clap::Parser;

#[derive(Parser)]
#[command(
    name = "cozy",
    version,
    about = "Run models on your machines and rented GPUs"
)]
struct Cozy {}

fn main() {
    Cozy::parse();
}
