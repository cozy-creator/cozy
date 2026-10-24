//! `cozy status`, the logs `cozy machine logs` reads (Calcifer's too), and Calcifer's restart (`cozy update`, `cozy doctor`). Any cozy command starts
//! Calcifer; restarting it never stops work: machines run what they accepted, and the next Calcifer
//! resumes what it owed.
use std::process::ExitCode;

use clap::ValueEnum;
use protocol::{
    calcifer::{Holder, MachineState},
    error::{Error, Kind},
};
use jiff::{SignedDuration, Timestamp};
use serde_json::json;

use platform::tail::{Log, Start};

use crate::{Context, calcifer_client::Client, output};

/// Stops a running Calcifer and starts the installed `cozy`'s in its place, after something
/// replaced that binary. `false`: none was running, and none is started.
pub(crate) async fn restart(ctx: &Context) -> Result<bool, Error> {
    let Some(client) = Client::running(ctx).await? else {
        return Ok(false);
    };
    client.stop().await?;
    client.ready().await?;
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
            output::next(ctx, &["cozy doctor"]);
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

/// Which of a machine's logs, and how much of it.
#[derive(clap::Args, Debug)]
pub struct LogQuery {
    /// The machine's own log, the TensorFS transport or memory log it keeps, or (on `local`)
    /// Calcifer's, this computer's controller.
    #[arg(value_enum, default_value_t = LogName::Tensord)]
    log: LogName,
    /// Keep printing as it grows.
    #[arg(short, long)]
    follow: bool,
    /// The newest N lines; 0 prints every line it keeps.
    #[arg(short = 'n', long, value_name = "N", default_value_t = 200)]
    lines: usize,
    /// Only lines from the last DURATION, such as 90s, 30m or 2h.
    #[arg(long, value_name = "DURATION", value_parser = since)]
    since: Option<SignedDuration>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LogName {
    Tensord,
    Tensorfs,
    Memory,
    Calcifer,
}

impl LogName {
    fn as_str(self) -> &'static str {
        match self {
            LogName::Tensord => "tensord",
            LogName::Tensorfs => "tensorfs",
            LogName::Memory => "memory",
            LogName::Calcifer => "calcifer",
        }
    }
}

fn since(text: &str) -> Result<SignedDuration, String> {
    text.parse()
        .map_err(|_| format!("{text:?} is not a duration such as 90s, 30m or 2h"))
}

/// A machine's log through Calcifer: its newest lines, then, with `--follow`, each line as the
/// machine writes it. A machine keeps its logs, so reading never needs it to have run anything
/// here.
pub(crate) async fn machine_log(
    ctx: &Context,
    name: &str,
    q: &LogQuery,
) -> Result<ExitCode, Error> {
    if let LogName::Calcifer = q.log {
        if name != "local" {
            return Err(Error::new(
                Kind::Usage,
                "log_not_kept",
                format!("{name} keeps no Calcifer log: Calcifer is this computer's"),
            )
            .next("cozy machine logs local calcifer"));
        }
        return calcifer_log(ctx, q).await;
    }
    let path = format!("{}/logs/{}", crate::machine::route(name)?, q.log.as_str());
    let calcifer = ctx.calcifer().await?;
    let mut shown = Shown {
        since: q
            .since
            .map(|window| Timestamp::now().saturating_sub(window).unwrap_or_default()),
        any: false,
    };
    if !q.follow {
        let text = calcifer.text(&format!("{path}?lines={}", q.lines)).await?;
        shown.print(ctx, name, q.log, &text);
        if !shown.any {
            output::say(
                ctx,
                &format!("{name} has logged nothing in its {} log", q.log.as_str()),
            );
        }
        return Ok(ExitCode::SUCCESS);
    }
    let mut lines = calcifer.follow_log(&path, q.lines as u64);
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => return Ok(ExitCode::SUCCESS),
            batch = lines.next() => shown.print(ctx, name, q.log, &batch?.text),
        }
    }
}

/// Lines printed so far, and the cutoff the first one shown must be stamped at or after. A
/// line's stamp is its first word: an RFC 3339 instant (TensorD's own log) or Unix seconds (the
/// TensorFS and memory logs). A line without one goes with the line before it.
struct Shown {
    since: Option<Timestamp>,
    any: bool,
}

impl Shown {
    fn print(&mut self, ctx: &Context, machine: &str, log: LogName, text: &str) {
        for line in text.lines().filter(|l| !l.is_empty()) {
            if let Some(cutoff) = self.since {
                if stamp(line).is_none_or(|at| at < cutoff) {
                    continue;
                }
                self.since = None;
            }
            self.any = true;
            print_log_line(ctx, machine, log, line);
        }
    }
}

fn stamp(line: &str) -> Option<Timestamp> {
    let word = line.split_whitespace().next()?;
    word.parse().ok().or_else(|| {
        let seconds: f64 = word.parse().ok()?;
        Timestamp::from_millisecond((seconds * 1000.0) as i64).ok()
    })
}

fn print_log_line(ctx: &Context, machine: &str, log: LogName, line: &str) {
    if ctx.json {
        output::ndjson(&json!({ "machine": machine, "log": log.as_str(), "text": line }));
    } else {
        println!("{line}");
    }
}

/// Calcifer's log, read from its file: there whether or not Calcifer is, its newest `lines` from
/// the last `since`. Followed, each line as it is written, woken by a watch on its directory
/// (`tail -F`, across a rotation to `calcifer.log.1`).
async fn calcifer_log(ctx: &Context, q: &LogQuery) -> Result<ExitCode, Error> {
    let path = ctx.home.calcifer_log();
    if !path.exists() {
        return Err(Error::new(
            Kind::NotFound,
            "log_not_found",
            format!(
                "no Calcifer has run here: {} does not exist",
                path.display()
            ),
        )
        .next("cozy doctor"));
    }
    let unreadable = |e: &dyn std::fmt::Display| {
        Error::new(
            Kind::Internal,
            "log_unreadable",
            format!("{}: {e}", path.display()),
        )
    };
    let log = Log::rotated(ctx.home.calcifer(), "calcifer.log");
    let start = Start::Last(q.lines as u64);
    // Lines before `since` are skipped up to the first one stamped at or after it.
    let mut cutoff = q
        .since
        .map(|window| Timestamp::now().saturating_sub(window).unwrap_or_default());
    let mut emit = |text: &str| {
        for line in text.lines().filter(|l| !l.is_empty()) {
            if let Some(at) = cutoff {
                if stamp(line).is_none_or(|stamped| stamped < at) {
                    continue;
                }
                cutoff = None;
            }
            print_line(ctx, line);
        }
    };
    if !q.follow {
        emit(&log.read(start).map_err(|e| unreadable(&e))?);
        return Ok(ExitCode::SUCCESS);
    }
    let follow = log
        .follow(start)
        .map_err(|e| match e.is_unsupported() {
            true => Error::new(Kind::Unavailable, "log_follow_unsupported", e.to_string()),
            false => unreadable(&e),
        })?;
    let (sender, mut batches) = tokio::sync::mpsc::channel(16);
    let reading = tokio::task::spawn_blocking(move || {
        follow.run(
            |batch| sender.blocking_send(batch).is_ok(),
            || !sender.is_closed(),
        )
    });
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => return Ok(ExitCode::SUCCESS),
            batch = batches.recv() => match batch {
                Some(batch) => emit(&batch.text),
                None => {
                    return match reading.await {
                        Ok(Err(e)) => Err(unreadable(&e)),
                        _ => Ok(ExitCode::SUCCESS),
                    };
                }
            },
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
