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
use jiff::{SignedDuration, Timestamp};
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
        Which::Machine => {
            let query = LogQuery {
                log: LogName::Tensord,
                follow: args.follow,
                lines: 200,
                since: None,
            };
            machine_log(ctx, "local", &query).await
        }
    }
}

/// Which of a machine's logs, and how much of it.
#[derive(clap::Args, Debug)]
pub struct LogQuery {
    /// The machine's own log, or the TensorFS transport or memory log it keeps.
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
}

impl LogName {
    fn as_str(self) -> &'static str {
        match self {
            LogName::Tensord => "tensord",
            LogName::Tensorfs => "tensorfs",
            LogName::Memory => "memory",
        }
    }
}

fn since(text: &str) -> Result<SignedDuration, String> {
    text.parse()
        .map_err(|_| format!("{text:?} is not a duration such as 90s, 30m or 2h"))
}

/// One read's worth while following: more than a busy machine writes between two reads.
const FOLLOW_TAIL: u64 = 256 << 10;

/// A machine's log through Calcifer: its newest lines, then, with `--follow`, each line as it
/// appears, read again every sampling period. A machine keeps its logs, so reading never needs
/// it to have run anything here.
pub(crate) async fn machine_log(
    ctx: &Context,
    name: &str,
    q: &LogQuery,
) -> Result<ExitCode, Error> {
    let path = format!("{}/logs/{}", crate::machine::route(name)?, q.log.as_str());
    let calcifer = ctx.calcifer().await?;
    // Lines run to a few hundred bytes: read generously, then keep the newest N.
    let tail = match q.lines {
        0 => 0,
        n => (n as u64).saturating_mul(512).max(64 << 10),
    };
    let text = calcifer.text(&format!("{path}?tail_bytes={tail}")).await?;
    let read = whole_lines(&text);
    let mut shown = match q.since {
        Some(window) => from(
            &read,
            Timestamp::now().saturating_sub(window).unwrap_or_default(),
        ),
        None => &read[..],
    };
    if q.lines > 0 && shown.len() > q.lines {
        shown = &shown[shown.len() - q.lines..];
    }
    if shown.is_empty() && !q.follow {
        output::say(
            ctx,
            &format!("{name} has logged nothing in its {} log", q.log.as_str()),
        );
    }
    for line in shown {
        print_log_line(ctx, name, q.log, line);
    }
    if !q.follow {
        return Ok(ExitCode::SUCCESS);
    }
    let mut last = read.last().map(|l| l.to_string());
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => return Ok(ExitCode::SUCCESS),
            () = tokio::time::sleep(FOLLOW) => {}
        }
        let text = calcifer
            .text(&format!("{path}?tail_bytes={FOLLOW_TAIL}"))
            .await?;
        let read = whole_lines(&text);
        // The lines after the last one printed; all of them when it rolled out of the tail.
        let fresh = match &last {
            Some(last) => match read.iter().rposition(|l| l == last) {
                Some(at) => &read[at + 1..],
                None => &read[..],
            },
            None => &read[..],
        };
        for line in fresh {
            print_log_line(ctx, name, q.log, line);
        }
        if let Some(line) = read.last() {
            last = Some(line.to_string());
        }
    }
}

/// The complete lines of a read: one still being written waits for its end.
fn whole_lines(text: &str) -> Vec<&str> {
    let complete = text.rfind('\n').map_or("", |end| &text[..end]);
    complete.lines().filter(|l| !l.is_empty()).collect()
}

/// The lines from the first one stamped at or after `cutoff`. A line's stamp is its first word:
/// an RFC 3339 instant (TensorD's own log) or Unix seconds (the TensorFS and memory logs). A
/// line without one goes with the line before it.
fn from<'a>(lines: &'a [&'a str], cutoff: Timestamp) -> &'a [&'a str] {
    let stamp = |line: &&str| -> Option<Timestamp> {
        let word = line.split_whitespace().next()?;
        word.parse().ok().or_else(|| {
            let seconds: f64 = word.parse().ok()?;
            Timestamp::from_millisecond((seconds * 1000.0) as i64).ok()
        })
    };
    match lines
        .iter()
        .position(|line| stamp(line).is_some_and(|at| at >= cutoff))
    {
        Some(at) => &lines[at..],
        None => &[],
    }
}

fn print_log_line(ctx: &Context, machine: &str, log: LogName, line: &str) {
    if ctx.json {
        output::ndjson(&json!({ "machine": machine, "log": log.as_str(), "text": line }));
    } else {
        println!("{line}");
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
