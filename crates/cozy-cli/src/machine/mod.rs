//! `cozy machine`: this computer's machine, machines rented on any Hub, and other computers' machines
//! added by address. Every machine is reached through Calcifer.
use std::{process::ExitCode, time::Duration};

use clap::{Subcommand, ValueEnum};
use cozy_proto::{
    calcifer::{Machine, NewMachine},
    error::{Error, Kind},
    ids::Digest,
};

use crate::{Context, output};

mod end;
mod list;
mod rent;

pub(crate) use list::status_fields;

#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(subcommand)]
    verb: Verb,
}

// `update` and this computer's machine role read their arguments once they land.
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
    match args.verb {
        Verb::List { watch } => list::list(ctx, watch).await,
        Verb::Show { name } => list::show(ctx, &name).await,
        Verb::Logs { name, log, follow } => logs(ctx, &name, log, follow).await,
        Verb::Rent(rent) => rent::exec(ctx, rent).await,
        Verb::End { name } => end::end(ctx, &name).await,
        Verb::Keepalive { name } => end::keepalive(ctx, &name).await,
        Verb::Add {
            name,
            address,
            fingerprint,
        } => add(ctx, name, address, fingerprint).await,
        Verb::Update { .. } => Err(crate::not_ported("machine update")),
        // This computer's machine role needs TensorD inside the binary.
        Verb::Enable => Err(crate::not_ported("machine enable")),
        Verb::Disable => Err(crate::not_ported("machine disable")),
        Verb::Authorize { .. } => Err(crate::not_ported("machine authorize")),
    }
}

/// How often a command following a machine re-reads Calcifer, which answers from memory.
const FOLLOW: Duration = Duration::from_secs(1);

/// The route of one machine, by name, alias or rental id: `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`.
pub(crate) fn route(name: &str) -> Result<String, Error> {
    let mut bytes = name.bytes();
    let first = bytes.next().is_some_and(|b| b.is_ascii_alphanumeric());
    if !first
        || name.len() > 128
        || !bytes.all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        return Err(Error::new(
            Kind::Usage,
            "machine_name_invalid",
            format!("{name:?} is not a machine name or rental id"),
        )
        .next("cozy machine list"));
    }
    Ok(format!("/v1/machines/{name}"))
}

async fn machine(ctx: &Context, name: &str) -> Result<Machine, Error> {
    ctx.calcifer().await?.get(&route(name)?).await
}

async fn add(
    ctx: &Context,
    name: String,
    address: String,
    fingerprint: Digest,
) -> Result<ExitCode, Error> {
    route(&name)?;
    let ask = NewMachine::Add {
        name,
        address,
        fingerprint,
    };
    let added: Machine = ctx
        .calcifer()
        .await?
        .post("/v1/machines", &ask, None)
        .await?;
    if ctx.json {
        output::json(&added);
        return Ok(ExitCode::SUCCESS);
    }
    output::Record::new()
        .field("machine", &added.name)
        .field("address", added.address.as_deref().unwrap_or("—"))
        .field("state", list::state_cell(&added))
        .print(ctx);
    output::next(
        ctx,
        &[
            &format!("cozy run <org/package/function> --on {}", added.name),
            &format!("cozy machine show {}", added.name),
        ],
    );
    Ok(ExitCode::SUCCESS)
}

/// A machine's log; `None` is the machine's own (`cozy logs machine` is this computer's).
pub(crate) async fn logs(
    ctx: &Context,
    name: &str,
    log: Option<Log>,
    follow: bool,
) -> Result<ExitCode, Error> {
    // A machine serves its own log, and follows a log, once TensorD is inside the binary.
    let log = match (log, follow) {
        (None, _) => {
            return Err(crate::not_ported(
                "machine logs <name> (the machine's own log)",
            ));
        }
        (Some(_), true) => return Err(crate::not_ported("machine logs --follow")),
        (Some(Log::Tensorfs), false) => "tensorfs",
        (Some(Log::Memory), false) => "memory",
    };
    let text = ctx
        .calcifer()
        .await?
        .text(&format!("{}/logs/{log}", route(name)?))
        .await?;
    if ctx.json {
        output::json(&serde_json::json!({"machine": name, "log": log, "text": text}));
    } else if text.is_empty() {
        output::say(ctx, &format!("{name} has logged nothing in its {log} log"));
    } else {
        print!("{text}");
    }
    Ok(ExitCode::SUCCESS)
}

/// Money as people read it: `$3.19`.
fn usd(micros: i64) -> String {
    format!("${:.2}", micros as f64 / 1e6)
}

/// A duration as a clock reads: `41s`, `14m`, `1h12m`.
fn clock(ms: i64) -> String {
    let s = (ms / 1000).max(0);
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m", s / 60),
        _ => format!("{}h{:02}m", s / 3600, s % 3600 / 60),
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}
