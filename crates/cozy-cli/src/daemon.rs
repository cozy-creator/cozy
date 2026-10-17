//! `cozy up|down|status|logs`: Calcifer's lifecycle. Stopping Calcifer never stops work: machines
//! run what they accepted, and the next Calcifer resumes what it owed.
use std::{
    io::{self, Read as _, Seek as _, SeekFrom},
    path::Path,
    process::ExitCode,
    time::Duration,
};

use clap::ValueEnum;
use cozy_proto::{
    calcifer::{Holder, MachineState},
    error::{Error, Kind},
};
use serde_json::json;

use crate::{Context, calcifer_client::Client, output};

/// How often a followed log is read again: a sampling period.
const FOLLOW: Duration = Duration::from_millis(250);

#[derive(clap::Args, Debug)]
pub struct Up {}

pub async fn up(ctx: &Context, _args: Up) -> Result<ExitCode, Error> {
    let client = ctx.calcifer().await?;
    let holder = client.holder().expect("a connected client has a record");
    if ctx.json {
        output::json(&json!({ "calcifer": running(ctx, &holder) }));
    } else {
        calcifer_record(ctx, &holder).print(ctx);
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(clap::Args, Debug)]
pub struct Down {}

pub async fn down(ctx: &Context, _args: Down) -> Result<ExitCode, Error> {
    let Some(client) = Client::running(ctx).await? else {
        if ctx.json {
            output::json(&json!({ "calcifer": { "running": false }, "changed": false }));
        } else {
            output::Record::new()
                .field("calcifer", "not running")
                .print(ctx);
        }
        return Ok(ExitCode::SUCCESS);
    };
    let down = client.stop().await?;
    if ctx.json {
        output::json(
            &json!({ "calcifer": { "running": false }, "changed": true, "owed": down.owed }),
        );
        return Ok(ExitCode::SUCCESS);
    }
    let mut record = output::Record::new().field("calcifer", "stopped");
    if !down.owed.is_empty() {
        record = record
            .field("owed", down.owed.join(", "))
            .note("Machines keep running it; the next cozy command resumes it.");
    }
    record.print(ctx);
    Ok(ExitCode::SUCCESS)
}

/// Stops a running Calcifer and starts `program`'s in its place: `cozy update` after it replaced
/// the binary. `false`: none was running, and none is started.
pub(crate) async fn restart(ctx: &Context, program: &Path) -> Result<bool, Error> {
    let Some(client) = Client::running(ctx).await? else {
        return Ok(false);
    };
    client.stop().await?;
    Client::new(ctx, program.to_owned())?.ready().await?;
    Ok(true)
}

#[derive(clap::Args, Debug)]
pub struct Status {}

/// Calcifer, its machines and the runs not yet settled. Never starts Calcifer.
pub async fn status(ctx: &Context, _args: Status) -> Result<ExitCode, Error> {
    let Some(client) = Client::running(ctx).await? else {
        if ctx.json {
            output::json(&json!({ "calcifer": { "running": false } }));
        } else {
            output::Record::new()
                .field("calcifer", "not running")
                .print(ctx);
            output::next(ctx, &["cozy up"]);
        }
        return Ok(ExitCode::SUCCESS);
    };
    let holder = client.holder().expect("a running client has a record");
    // Calcifer's picture: it never waits on a Hub here.
    let mut machines = client.machines().await?.machines;
    machines.retain(|m| m.state != MachineState::Ended);
    let mut runs = vec![];
    for status in ["queued", "in_progress"] {
        runs.extend(client.runs(&[("status", status)]).await?.runs);
    }
    runs.sort_by_key(|run| run.number);
    if ctx.json {
        output::json(&json!({
            "calcifer": running(ctx, &holder), "machines": machines, "runs": runs,
        }));
        return Ok(ExitCode::SUCCESS);
    }
    calcifer_record(ctx, &holder).print(ctx);
    if !machines.is_empty() {
        println!();
        let headers = ["MACHINE", "KIND", "STATE", "RUNNING", "QUEUED", "HUB"];
        let mut table = output::Table::new(ctx, headers);
        for m in &machines {
            let hub = match ctx.config.resolve(&m.hub) {
                Ok(hub) if !m.hub.is_empty() => hub.label().to_owned(),
                _ => m.hub.clone(),
            };
            table = table.row([
                m.name.clone(),
                word(m.kind),
                m.boot.clone().unwrap_or_else(|| word(m.state)),
                m.running.to_string(),
                m.queued.to_string(),
                hub,
            ]);
        }
        table.print();
    }
    if !runs.is_empty() {
        println!();
        let mut table = output::Table::new(ctx, ["RUN", "TARGET", "MACHINE", "STATE"]);
        for run in &runs {
            let state = run.machine_state.as_deref().or(run.waiting.as_deref());
            table = table.row([
                run.number.to_string(),
                run.target.clone(),
                run.machine.clone().unwrap_or_default(),
                state.unwrap_or("queued").to_owned(),
            ]);
        }
        table.print();
    }
    Ok(ExitCode::SUCCESS)
}

/// A wire enum as it is spelled on the wire.
fn word(value: impl serde::Serialize) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

#[derive(clap::Args, Debug)]
pub struct Logs {
    /// Whose log.
    #[arg(value_enum, default_value_t = Which::Calcifer)]
    which: Which,
    /// Keep printing as it is written.
    #[arg(short, long)]
    follow: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    Calcifer,
    Machine,
}

pub async fn logs(ctx: &Context, args: Logs) -> Result<ExitCode, Error> {
    match args.which {
        Which::Calcifer => calcifer_log(ctx, args.follow).await,
        Which::Machine => crate::machine::logs(ctx, "local", None, args.follow).await,
    }
}

/// Calcifer's log, read from its file: there whether or not Calcifer is.
async fn calcifer_log(ctx: &Context, follow: bool) -> Result<ExitCode, Error> {
    let path = ctx.home.calcifer_log();
    let unreadable = |e: io::Error| {
        Error::new(
            Kind::Internal,
            "log_unreadable",
            format!("{}: {e}", path.display()),
        )
    };
    let mut file = match std::fs::File::open(&path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(Error::new(
                Kind::NotFound,
                "log_not_found",
                format!(
                    "no Calcifer has run here: {} does not exist",
                    path.display()
                ),
            )
            .next("cozy up"));
        }
        Err(e) => return Err(unreadable(e)),
    };
    let mut pending = String::new();
    loop {
        let mut chunk = vec![];
        file.read_to_end(&mut chunk).map_err(unreadable)?;
        pending.push_str(&String::from_utf8_lossy(&chunk));
        // Whole lines only: a line still being written waits for its end.
        while let Some(end) = pending.find('\n') {
            let line: String = pending.drain(..=end).collect();
            print_line(ctx, line.trim_end_matches(['\n', '\r']));
        }
        if !follow {
            if !pending.is_empty() {
                print_line(ctx, &pending);
            }
            return Ok(ExitCode::SUCCESS);
        }
        tokio::select! {
            _ = tokio::signal::ctrl_c() => return Ok(ExitCode::SUCCESS),
            () = tokio::time::sleep(FOLLOW) => {}
        }
        // A rotation replaces the file under the same name: carry on from the new one's start.
        let position = file.stream_position().map_err(unreadable)?;
        let length = std::fs::metadata(&path)
            .map(|m| m.len())
            .unwrap_or(position);
        if length < position {
            file = std::fs::File::open(&path).map_err(unreadable)?;
            file.seek(SeekFrom::Start(0)).map_err(unreadable)?;
        }
    }
}

fn print_line(ctx: &Context, line: &str) {
    if ctx.json {
        output::ndjson(&json!({ "text": line }));
    } else {
        println!("{line}");
    }
}

fn running(ctx: &Context, holder: &Holder) -> serde_json::Value {
    json!({
        "running": true,
        "address": holder.address,
        "pid": holder.pid,
        "version": holder.version,
        "started_at_ms": holder.started_at_ms,
        "log": ctx.home.calcifer_log(),
    })
}

fn calcifer_record(ctx: &Context, holder: &Holder) -> output::Record {
    output::Record::new()
        .field(
            "calcifer",
            format!("running, up {}", output::age(holder.started_at_ms)),
        )
        .field("address", holder.address)
        .field("pid", holder.pid)
        .field("version", &holder.version)
        .field("log", output::link(ctx, &ctx.home.calcifer_log()))
}
