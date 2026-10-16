//! `cozy machine`: this computer's machine, machines rented on any Hub, and other computers' machines
//! added by address. Every machine is reached through Calcifer.
use std::process::ExitCode;

use clap::{Subcommand, ValueEnum};
use cozy_proto::{error::Error, ids::Digest};

use crate::Context;

mod rent;

#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(subcommand)]
    verb: Verb,
}

// The verbs Calcifer's machine routes serve read their arguments once those routes land.
#[allow(dead_code)]
#[derive(Subcommand, Debug)]
enum Verb {
    /// Every machine: this computer's, rentals on every Hub, and added ones.
    List {
        /// Redraw the list live (a terminal only).
        #[arg(long)]
        watch: bool,
    },
    /// One machine and what it reports of itself.
    Show {
        /// A machine name or rental id.
        name: String,
    },
    /// Print a machine's log: its own, or the tensorfs or memory log it keeps.
    Logs {
        name: String,
        log: Option<Log>,
        /// Keep printing as it grows.
        #[arg(short, long)]
        follow: bool,
    },
    /// Rent a machine; with no SKU, list what is for rent.
    Rent(rent::Rent),
    /// End a rental (billing stops once its Hub confirms), or remove an added machine.
    End { name: String },
    /// Reset a rental's 15-minute idle clock once.
    Keepalive { name: String },
    /// Update a machine to its Hub's target software, as a run.
    Update { name: String },
    /// Add another computer's machine by its address and TLS fingerprint.
    Add {
        name: String,
        /// host:port
        address: String,
        /// The machine's TLS leaf, sha256:<hex> (`cozy machine show local` on that computer).
        #[arg(long)]
        fingerprint: Digest,
    },
    /// Make this computer a machine.
    Enable,
    /// Stop this computer's machine.
    Disable,
    /// On this computer: let the controller with this public key use this machine.
    Authorize { public_key: String },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum Log {
    Tensorfs,
    Memory,
}

pub async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    let verb = match args.verb {
        Verb::Rent(rent) => return rent::exec(ctx, rent).await,
        Verb::List { .. } => "machine list",
        Verb::Show { .. } => "machine show",
        Verb::Logs { .. } => "machine logs",
        Verb::End { .. } => "machine end",
        Verb::Keepalive { .. } => "machine keepalive",
        Verb::Update { .. } => "machine update",
        Verb::Add { .. } => "machine add",
        // This computer's machine role needs TensorD inside the binary.
        Verb::Enable => "machine enable",
        Verb::Disable => "machine disable",
        Verb::Authorize { .. } => "machine authorize",
    };
    Err(crate::not_ported(verb))
}

/// Money as people read it: `$3.19`.
fn usd(micros: i64) -> String {
    format!("${:.2}", micros as f64 / 1e6)
}
