//! The `tensord` binary: the machine role for TensorD's and Calcifer's tests and an activated
//! update's service. `cozy machine serve` is the product's entry.
use clap::{Parser, Subcommand};
use std::process::ExitCode;
use tensord::{
    config::Flags,
    machine::grant::Lifetime,
    serve::{self, Dev},
};

#[derive(Parser)]
#[command(name = "tensord", version)]
struct Tensord {
    #[command(subcommand)]
    mode: Option<Mode>,
    #[command(flatten)]
    flags: Flags,
}

#[derive(Subcommand)]
enum Mode {
    /// The machine (the default; a rental unless --persistent).
    Run(Flags),
    /// The service an updated machine's stable parent starts.
    Service(Flags),
    /// What this binary is, as JSON.
    Version {
        #[arg(long)]
        json: bool,
    },
    /// A machine on a state directory with no grant.
    Serve(Dev),
}

fn main() -> ExitCode {
    let tensord = Tensord::parse();
    let lifetime = |flags: Flags| flags.lifetime().unwrap_or(Lifetime::Rental);
    let done = match tensord.mode {
        None => return serve::machine(lifetime(tensord.flags), false),
        Some(Mode::Run(flags)) => return serve::machine(lifetime(flags), false),
        Some(Mode::Service(flags)) => return serve::machine(lifetime(flags), true),
        Some(Mode::Version { .. }) => serve::version(),
        Some(Mode::Serve(dev)) => serve::dev(dev),
    };
    match done {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!("{error}");
            ExitCode::FAILURE
        }
    }
}
