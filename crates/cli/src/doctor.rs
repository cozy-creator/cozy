//! `cozy doctor`: what is wrong with this computer's cozy, fixed where it can be. Calcifer first: its
//! lock record, whether it answers, whether it runs the installed cozy, and its service. A Calcifer
//! that does not start, answers nothing while its process makes no progress, or runs another cozy
//! than the installed one is restarted, which never stops work (machines run what they accepted).
//! Then this computer's tools, and a machine's driver, GPUs, toolchain and kernels as its Status
//! reports them. Exit 1 while a problem remains.
use std::{
    path::Path,
    process::{Command, ExitCode, Stdio},
    time::{Duration, Instant},
};

use platform::{
    process::{Birth, Exact, meter},
    tools::Side,
};
use protocol::{
    calcifer::{Holder, Machine},
    error::Error,
};
use serde::Serialize;

use crate::{
    Context,
    calcifer_client::{Client, Lock, installed},
    machine::{route, status_fields},
    output,
};

/// How often a silent Calcifer's progress is read: a sampling period, never a verdict.
const SAMPLE: Duration = Duration::from_secs(1);
/// Samples in a row with no answer and no progress (CPU or I/O) that make a Calcifer stuck.
const STILL: u32 = 5;
/// How often a wait says it is still waiting.
const REPORT: Duration = Duration::from_secs(5);

#[derive(clap::Args, Debug)]
pub struct Args {
    /// The machine to check; without it, this computer's machine, which starts for it.
    #[arg(long)]
    on: Option<String>,
}

/// One finding: what was looked at, what it is, and what doctor did about it.
#[derive(Serialize)]
struct Check {
    check: &'static str,
    ok: bool,
    state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    fixed: Option<String>,
}

impl Check {
    fn ok(check: &'static str, state: impl Into<String>) -> Self {
        Self { check, ok: true, state: state.into(), fixed: None }
    }
    fn bad(check: &'static str, state: impl Into<String>) -> Self {
        Self { check, ok: false, state: state.into(), fixed: None }
    }
    fn fixed(check: &'static str, state: impl Into<String>, fixed: impl Into<String>) -> Self {
        Self { check, ok: true, state: state.into(), fixed: Some(fixed.into()) }
    }
}

pub async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    let mut checks = calcifer(ctx).await;
    let up = checks.iter().all(|c| c.ok);
    checks.push(service(ctx));
    let tools = crate::tools::rows(ctx, Side::Computer);
    let on = args.on.unwrap_or_else(|| "local".into());
    // A machine is reached through Calcifer: with none answering there is nothing to ask.
    let machine: Option<Machine> = match up {
        true => match ctx.calcifer().await?.get(&route(&on)?).await {
            Ok(m) => Some(m),
            Err(e) => {
                checks.push(Check::bad("machine", format!("{on} did not start: {}", e.message)));
                None
            }
        },
        false => None,
    };
    if let Some(m) = &machine {
        checks.push(match (&m.status, &m.unreachable) {
            (Some(_), None) => Check::ok("machine", format!("{} reports itself", m.name)),
            (Some(_), Some(why)) => Check::ok(
                "machine",
                format!("{} did not answer just now ({why}); its last report follows", m.name),
            ),
            (None, why) => Check::bad(
                "machine",
                format!(
                    "{} cannot be checked: {}",
                    m.name,
                    why.as_deref().unwrap_or("it has not reported its status yet")
                ),
            ),
        });
    }
    let healthy = checks.iter().all(|c| c.ok);
    let code = if healthy { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    if ctx.json {
        output::json(&serde_json::json!({
            "healthy": healthy, "checks": checks, "tools": tools, "machine": machine,
        }));
        return Ok(code);
    }
    let mut table = output::Table::new(ctx, ["CHECK", "STATE"]);
    for c in &checks {
        let state = match (&c.fixed, c.ok) {
            (Some(fixed), _) => format!("{fixed}: {}", c.state),
            (None, true) => c.state.clone(),
            (None, false) => format!("PROBLEM: {}", c.state),
        };
        table = table.row([c.check.to_owned(), state]);
    }
    table.print();
    println!();
    let mut table = output::Table::new(ctx, ["THIS COMPUTER", "STATE", "PATH OR INSTALL"]);
    for row in &tools {
        table = table.row(crate::tools::cells(row));
    }
    table.print();
    if let Some(m) = &machine
        && let Some(status) = &m.status
    {
        println!();
        let mut record = output::Record::new().field("machine", &m.name);
        for (key, value) in status_fields(status) {
            record = record.field(&key, value);
        }
        if status.toolchain.is_none() {
            record = record.note(
                "this machine is older than this cozy: it reports no driver floor, toolchain or kernels",
            );
        }
        record.print(ctx);
    }
    if !healthy {
        output::next(ctx, &["cozy machine logs local calcifer", "cozy machine logs local"]);
    }
    Ok(code)
}

/// Calcifer's lock record and whether it answers, running the installed cozy: each problem found,
/// and the start or restart that fixed it.
async fn calcifer(ctx: &Context) -> Vec<Check> {
    let client = match Client::unconnected(ctx) {
        Ok(client) => client,
        Err(e) => return vec![Check::bad("calcifer", e.message)],
    };
    let mut checks = vec![];
    // What it needs, and why.
    let (stop, why) = match client.lock_state() {
        Lock::Free { stale } => {
            if stale {
                checks.push(Check::ok(
                    "lock record",
                    "named a Calcifer killed before it let go; the next one replaces it",
                ));
            }
            (false, "not running".to_owned())
        }
        Lock::Unreadable => {
            checks.push(Check::bad(
                "lock record",
                format!("{} is held but does not read as Calcifer's", ctx.home.calcifer_lock().display()),
            ));
            return checks;
        }
        Lock::Held(holder) => {
            checks.push(Check::ok(
                "lock record",
                format!("held by pid {} at {}", holder.pid, holder.address),
            ));
            if let Some(silence) = silence(ctx, &client, &holder).await {
                if let Err(e) = end(holder.pid) {
                    checks.push(Check::bad("calcifer", format!("{silence}, and ending it failed: {e}")));
                    return checks;
                }
                (false, silence)
            } else if let Some(old) = stale(ctx, &holder) {
                (true, format!("it ran cozy {old}, not the installed cozy"))
            } else {
                checks.push(Check::ok("calcifer", running(&holder)));
                return checks;
            }
        }
    };
    // One that answers stops cleanly; a silent one was ended above.
    if stop {
        let stopped = match Client::running(ctx).await {
            Ok(Some(live)) => live.stop().await.map(drop),
            Ok(None) => Ok(()),
            Err(e) => Err(e),
        };
        if let Err(e) = stopped {
            checks.push(Check::bad("calcifer", format!("{why}, and stopping it failed: {}", e.message)));
            return checks;
        }
    }
    match Client::connect(ctx).await.map(|fresh| fresh.holder()) {
        Ok(Some(holder)) => {
            let verb = if why == "not running" { "started" } else { "restarted" };
            checks.push(Check::fixed("calcifer", running(&holder), format!("{verb} ({why})")));
        }
        Ok(None) => checks.push(Check::bad("calcifer", format!("{why}, and it published no record"))),
        Err(e) => checks.push(Check::bad("calcifer", format!("{why}, and it does not start: {}", e.message))),
    }
    checks
}

/// `None` once the Calcifer `holder` names answers; else why it counts as stuck: its process made
/// no progress, CPU or I/O, for `STILL` samples in a row while it answered nothing. Where progress
/// cannot be read (Windows so far) it is waited on, said every few seconds.
async fn silence(ctx: &Context, client: &Client, holder: &Holder) -> Option<String> {
    let answered = client.answers(holder);
    tokio::pin!(answered);
    let (began, mut reported) = (Instant::now(), Instant::now());
    let (mut last, mut still) = (meter(holder.pid), 0);
    loop {
        tokio::select! {
            ok = &mut answered => {
                return match ok {
                    true => None,
                    false => Some("it holds this computer's cozy but refuses connections".into()),
                };
            }
            _ = tokio::time::sleep(SAMPLE) => {
                let now = meter(holder.pid);
                still = match (now, last) {
                    (Some(now), Some(last)) if now == last => still + 1,
                    _ => 0,
                };
                last = now;
                if still >= STILL {
                    return Some(format!(
                        "it answered nothing and made no progress for {} s",
                        began.elapsed().as_secs()
                    ));
                }
                if reported.elapsed() >= REPORT {
                    reported = Instant::now();
                    output::say(ctx, &format!(
                        "waiting for Calcifer (pid {}) to answer ({} s so far)",
                        holder.pid,
                        began.elapsed().as_secs()
                    ));
                }
            }
        }
    }
}

/// The cozy a Calcifer that answers runs, when it is not the installed one's.
fn stale(ctx: &Context, holder: &Holder) -> Option<String> {
    let installed = installed(&ctx.home).ok()?;
    let version = version(&installed)?;
    (holder.version != version).then(|| holder.version.clone())
}

/// The release a `cozy` binary says it is (`cozy 0.11.17`).
fn version(path: &Path) -> Option<String> {
    let out = Command::new(path).arg("--version").stdin(Stdio::null()).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.trim().strip_prefix("cozy ").map(str::to_owned)
}

/// Ends a stuck Calcifer's own process, and only it: the machine it started runs on its own.
fn end(pid: u32) -> Result<(), String> {
    let birth = Birth::of(pid).map_err(|e| e.to_string())?;
    match Exact::open(&birth).map_err(|e| e.to_string())? {
        Some(process) => {
            let _ = process.kill();
            process.wait().map_err(|e| e.to_string())
        }
        None => Ok(()),
    }
}

fn running(h: &Holder) -> String {
    format!("answering: cozy {}, pid {}, up {}", h.version, h.pid, output::age(h.started_at_ms))
}

/// Calcifer's service: defined for the installed cozy, so a login starts it; rewritten when not.
fn service(ctx: &Context) -> Check {
    let client = match Client::unconnected(ctx) {
        Ok(client) => client,
        Err(e) => return Check::bad("service", e.message),
    };
    let (services, spec, ours) = match client.service() {
        Ok(service) => service,
        Err(e) => return Check::bad("service", e.message),
    };
    if !services.available() {
        return Check::ok(
            "service",
            "no user service manager here: Calcifer runs detached, and the next cozy command starts it",
        );
    }
    if services.current(&spec) {
        return Check::ok("service", "defined for the installed cozy");
    }
    if !ours {
        return Check::bad(
            "service",
            format!("not defined for the installed cozy; run {} doctor", spec.program.display()),
        );
    }
    match services.ensure(&spec) {
        Ok(_) => Check::fixed("service", "defined for the installed cozy", "rewrote its definition"),
        Err(e) => Check::bad("service", format!("not defined for the installed cozy, and writing it failed: {e}")),
    }
}
