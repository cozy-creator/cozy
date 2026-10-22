//! `cozy machine end`, `keepalive` and `update`: asks Calcifer makes of one machine.
use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

use protocol::{
    calcifer::{
        self, Ended, Keepalive, Machine, MachineKind, MachineList, MachineState, MachineStatus, On,
        RunsEvent, StagedInput, SubmitRun,
    },
    error::{Error, Kind},
    ids::Digest,
    machine as v1,
};

use super::{Following, clock, machine, now_ms, route, usd};
use crate::{
    Context,
    calcifer_client::{Client, Watch},
    context::hub_answer,
    output,
};

/// Ends a rental, returning once its Hub affirms the pod gone, or removes an added machine. It
/// waits first for its runs' outputs to land here, unless `force`. Both waits follow Calcifer's
/// streams: its runs' changes, then the machine's.
pub async fn end(ctx: &Context, name: &str, force: bool) -> Result<ExitCode, Error> {
    if name == "local" {
        return Err(Error::new(
            Kind::Validation,
            "machine_local",
            "`local` is this computer's machine; `end` ends rentals and removes added machines",
        )
        .next("cozy machine disable"));
    }
    let calcifer = ctx.calcifer().await?;
    let path = route(name)?;
    let asked = match force {
        true => format!("{path}?force=true"),
        false => path.clone(),
    };
    let mut said = String::new();
    // The runs stream, and the machine's name its runs carry.
    let mut runs: Option<(Watch<RunsEvent>, String)> = None;
    let Ended {
        mut machine,
        changed,
        abandoned,
    } = loop {
        match calcifer.delete(&asked).await {
            Err(e) if e.code == "outputs_exporting" => {
                if e.message != said {
                    output::say(ctx, &format!("Waiting to end {name}: {}", e.message));
                    said = e.message;
                }
                let Some((watch, on)) = runs.as_mut() else {
                    // Subscribed after the refusal: asked again first, a landing in between counts.
                    let watch = calcifer.watch("/v1/runs/events").await?;
                    let machine: Machine = calcifer.get(&path).await?;
                    runs = Some((watch, machine.name));
                    continue;
                };
                tokio::select! {
                    moved = run_moved(watch, on) => if !moved {
                        runs = None;
                    },
                    _ = tokio::signal::ctrl_c() => {
                        return Err(Error::new(
                            Kind::Unavailable,
                            "end_waiting",
                            format!("stopped waiting; {name} was not ended and its outputs keep landing"),
                        )
                        .remedy("--force ends it now and gives up the outputs that have not landed")
                        .next(format!("cozy machine end {name}")));
                    }
                }
            }
            ended => break ended?,
        }
    };
    if !abandoned.is_empty() {
        output::warn(&format!(
            "Gave up outputs that had not landed: {}",
            abandoned.join(", ")
        ));
    }
    if machine.state == MachineState::Ending {
        output::say(ctx, &format!("Shutting down {}…", machine.name));
        // Followed by its rental id: the name is the Hub's and an alias may be what was typed.
        let id = machine
            .rental_id
            .clone()
            .unwrap_or_else(|| machine.name.clone());
        let (following, now) = Following::open(&calcifer, &id).await?;
        machine = released(&calcifer, following, now, &route(&id)?).await?;
    }
    if ctx.json {
        output::json(&Ended {
            machine,
            changed,
            abandoned,
        });
        return Ok(ExitCode::SUCCESS);
    }
    let said = match (machine.kind, changed) {
        (MachineKind::Added, true) => format!(
            "{} is no longer one of this computer's machines.",
            machine.name
        ),
        (MachineKind::Added, false) => format!("{} was already removed.", machine.name),
        (_, true) => format!(
            "{} shut down. Temporary pod files are gone. Local outputs and uploaded checkpoints remain.",
            machine.name
        ),
        (_, false) => format!("{} is already shut down.", machine.name),
    };
    println!("{said}");
    if let Ok(list) = calcifer.get::<MachineList>("/v1/machines").await {
        let live: Vec<_> = list
            .machines
            .iter()
            .filter(|m| m.state != MachineState::Ended)
            .filter_map(|m| m.rental.as_ref())
            .collect();
        let burn: i64 = live.iter().map(|r| r.usd_micros_per_hour).sum();
        println!("Rentals running: {} · {}/hour now", live.len(), usd(burn));
    }
    output::next(ctx, &["cozy machine list"]);
    Ok(ExitCode::SUCCESS)
}

/// Waits for a change to a run on `machine` (an output landed or lost, a run moved on); false
/// once the stream ended.
async fn run_moved(runs: &mut Watch<RunsEvent>, machine: &str) -> bool {
    loop {
        match runs.next().await {
            Some(RunsEvent::Updated { run }) if run.machine.as_deref() == Some(machine) => {
                return true;
            }
            Some(_) => {}
            None => return false,
        }
    }
}

/// Follows an ending rental until its Hub affirms the pod gone.
async fn released(
    calcifer: &Client,
    mut following: Following,
    mut machine: Machine,
    path: &str,
) -> Result<Machine, Error> {
    while machine.state == MachineState::Ending {
        if machine.reason.as_deref() == Some("release_unconfirmed") {
            return Err(Error::new(
                Kind::Conflict,
                "release_unconfirmed",
                format!(
                    "the Hub has not confirmed {}'s pod destroyed, and its missing record is not proof",
                    machine.name
                ),
            )
            .remedy("Calcifer keeps the record and keeps asking; the pod may still be billing")
            .next(format!("cozy machine show {}", machine.name)));
        }
        tokio::select! {
            next = following.next(calcifer) => machine = match next? {
                Some(next) => next,
                // Left the list: as Calcifer records it now.
                None => calcifer.get(path).await?,
            },
            _ = tokio::signal::ctrl_c() => {
                return Err(Error::new(
                    Kind::Unavailable,
                    "end_unconfirmed",
                    format!("stopped watching; the release of {} continues in Calcifer", machine.name),
                )
                .next(format!("cozy machine show {}", machine.name)));
            }
        }
    }
    Ok(machine)
}

/// Resets a rental's idle clock once; it answers only after the machine acknowledged.
pub async fn keepalive(ctx: &Context, name: &str) -> Result<ExitCode, Error> {
    let path = format!("{}/keepalive", route(name)?);
    let answer: Keepalive = ctx
        .calcifer()
        .await?
        .post(&path, &serde_json::json!({}), None)
        .await?;
    if ctx.json {
        output::json(&answer);
        return Ok(ExitCode::SUCCESS);
    }
    let ends = match answer.idle_deadline_ms {
        Some(at) => format!("in {} unless work arrives", clock(at - now_ms())),
        None => "after its jobs".into(),
    };
    output::Record::new()
        .field("machine", name)
        .field("ends", ends)
        .print(ctx);
    Ok(ExitCode::SUCCESS)
}

/// Updates a machine's Runtime and TensorFS: to wheels when named (a local file is written to the
/// machine with the run; an `https://…#sha256=<hex>` URL the machine fetches itself; a TensorFS not
/// named is the machine's own, or the one uv resolves for the Runtime), else to its Hub's target. A numbered run of kind update, which
/// waits for the work the machine accepted and is followed like any run; once it succeeds, the
/// machine must report what it asked for.
pub async fn update(
    ctx: &Context,
    name: &str,
    runtime_wheel: Option<PathBuf>,
    tensorfs_wheel: Option<PathBuf>,
    tensord: Option<PathBuf>,
    gpu_memory_limit: Option<u64>,
) -> Result<ExitCode, Error> {
    let mut payload = serde_json::Map::new();
    let (mut staged, mut inputs, mut named, mut want) = (vec![], vec![], vec![], vec![]);
    let dev = tensord.is_some();
    // An empty distribution: the dev `cozy` binary.
    for (member, distribution, path) in [
        ("runtime", "cozy_runtime", runtime_wheel),
        ("tensorfs", "tensorfs", tensorfs_wheel),
        ("binary", "", tensord),
    ] {
        let Some(path) = path else { continue };
        if let Some(at) = path.to_str().filter(|p| p.contains("://")) {
            let (file, fetched) = fetched(at, distribution)?;
            want.push(reports(member, &file));
            named.push(format!("{member} {file}"));
            payload.insert(member.into(), fetched);
            continue;
        }
        let (file, input) = wheel(&path, distribution)?;
        want.push(reports(member, &file));
        named.push(format!("{member} {file}"));
        inputs.push(v1::InputFile {
            field: member.into(),
            digest: input.digest.to_string(),
            length: input.length,
            media_type: match distribution {
                "" => "application/octet-stream",
                _ => "application/zip",
            }
            .into(),
            order: 0,
        });
        payload.insert(member.into(), file.into());
        staged.push(input);
    }
    if let Some(limit) = gpu_memory_limit {
        named.push(match limit {
            0 => "no GPU memory limit".into(),
            bytes => format!("a GPU memory limit of {}", output::bytes(bytes)),
        });
        payload.insert("gpu_memory_limit".into(), limit.into());
    }
    let m = machine(ctx, name).await?;
    let hub = match m.hub.as_str() {
        "" => ctx.hub()?,
        origin => ctx.config.resolve(origin)?,
    };
    // A dev binary swaps and a limit restarts the machine: either way its warm models reload.
    if (dev || gpu_memory_limit.is_some())
        && m.status
            .as_ref()
            .is_some_and(|s| s.warm.iter().any(|w| !w.level.is_empty()))
    {
        output::warn(&format!("{}'s warm models reload after it restarts", m.name));
    }
    if payload.is_empty() {
        let target = hub_answer(ctx.hub_anonymous(&hub).get_software().await)?;
        super::cohort::require_binary(&m.name, m.status.as_ref(), &target.runtime, &target.tensorfs, false)?;
        let Some(status) = &m.status else {
            return Err(Error::new(
                Kind::Unavailable,
                "machine_unanswered",
                format!("{} has not reported the software it runs", m.name),
            )
            .next(format!("cozy machine show {}", m.name)));
        };
        // An unpinned target older than the machine's software is no reason to go back.
        if (status.runtime == target.runtime && status.tensorfs == target.tensorfs)
            || (!target.pinned && newer(&status.runtime, &target.runtime))
        {
            return up_to_date(ctx, &m, &target.runtime);
        }
        named.push(format!("runtime {}", target.runtime));
        named.push(format!("tensorfs {}", target.tensorfs));
        want.push(("runtime", target.runtime.as_str().into()));
        want.push(("tensorfs", target.tensorfs.as_str().into()));
        payload.insert("runtime".into(), target.runtime.into());
        payload.insert("tensorfs".into(), target.tensorfs.into());
    }
    let requested = |member| want.iter().find(|(field, _)| *field == member)
        .and_then(|(_, value)| value.as_str()).unwrap_or("");
    super::cohort::require_binary(&m.name, m.status.as_ref(), requested("runtime"), requested("tensorfs"), dev)?;
    let spec = v1::RunSpec {
        kind: v1::RunKind::Update,
        payload: Some(
            v1::Json::parse(serde_json::Value::Object(payload).to_string()).expect("JSON"),
        ),
        inputs,
        ..Default::default()
    };
    let submit = SubmitRun {
        agreed: None,
        on: On::Machine(m.name.clone()),
        hub: hub.origin.into(),
        kind: calcifer::Kind::Update,
        target: named.join(", "),
        spec,
        inputs: staged,
        out: None,
        warnings: vec![],
    };
    let submitted = crate::run::submit(ctx, submit, None).await?;
    let run = crate::run::follow(ctx, submitted.run, false).await?;
    if run.status == calcifer::Status::Completed && !want.is_empty() {
        verify(ctx, &m.name, run.id.as_str(), &want).await?;
    }
    if dev && m.kind == MachineKind::Local && run.status == calcifer::Status::Completed {
        // `local` runs the installed cozy, the dev binary now: Calcifer runs it too.
        crate::daemon::restart(ctx).await?;
        let version = run.result.as_ref().and_then(|r| r["to"]["cozy"].as_str());
        output::warn(&format!(
            "this computer's cozy is now {}, a dev build; `cozy update` returns it to a release",
            version.unwrap_or("a dev build")
        ));
    }
    Ok(crate::run::exit_code(&run))
}

/// What a machine running `file` as an update's `member` reports: a wheel's version, or a dev cozy.
fn reports(member: &'static str, file: &str) -> (&'static str, serde_json::Value) {
    match member {
        "binary" => ("dev", true.into()),
        _ => (member, file.split('-').nth(1).unwrap_or_default().into()),
    }
}

/// A succeeded update must leave the machine reporting what it asked for (`want`): one that
/// ignored what it could not carry out fails here, with its log's line about the update.
async fn verify(ctx: &Context, name: &str, run: &str, want: &[(&str, serde_json::Value)]) -> Result<(), Error> {
    let m = machine(ctx, name).await?;
    let status = m.status.unwrap_or_default();
    if m.unreachable.is_none() && runs(&status, want) {
        return Ok(());
    }
    let log = format!("{}/logs/tensord?lines=1000", route(name)?);
    let log = ctx.calcifer().await?.text(&log).await.unwrap_or_default();
    let line = m.unreachable.or_else(|| log.lines().rfind(|l| l.contains(run)).map(str::to_owned));
    Err(unapplied(name, &status, want, line.as_deref()))
}

fn runs(status: &MachineStatus, want: &[(&str, serde_json::Value)]) -> bool {
    let status = serde_json::to_value(status).unwrap_or_default();
    want.iter().all(|(field, value)| status[*field] == *value)
}

fn unapplied(name: &str, s: &MachineStatus, want: &[(&str, serde_json::Value)], line: Option<&str>) -> Error {
    let asked: Vec<String> = (want.iter())
        .map(|(field, value)| match (*field, value.as_str()) {
            ("runtime", Some(version)) => format!("cozy-runtime {version}"),
            (field, Some(version)) => format!("{field} {version}"),
            _ => "dev build".into(),
        })
        .collect();
    let build = if s.dev { "dev build" } else { "release" };
    let mut message = format!(
        "{name}'s update succeeded but it runs cozy {} ({build} {}), cozy-runtime {}, tensorfs {}, not the requested {}",
        s.version, s.commit.get(..12).unwrap_or(&s.commit), s.runtime, s.tensorfs, asked.join(", ")
    );
    if let Some(line) = line {
        message = format!("{message}: {}", line.trim());
    }
    Error::new(Kind::Conflict, "update_unapplied", message).next(format!("cozy machine logs {name}"))
}

/// `cozy machine toolchain install`: CUDA's compiler, headers and ninja onto a machine, as an
/// update run, so cards without a prebuilt kernel compile their own. Never run on its own: a run
/// that would have used it says so and names this command.
pub async fn toolchain(ctx: &Context, name: &str) -> Result<ExitCode, Error> {
    let m = machine(ctx, name).await?;
    if let Some(toolchain) = m.status.as_ref().and_then(|s| s.toolchain.as_ref())
        && toolchain.state == "ready"
    {
        if ctx.json {
            output::json(
                &serde_json::json!({"machine": m.name, "changed": false, "toolchain": toolchain}),
            );
        } else {
            output::say(
                ctx,
                &format!("{}'s toolchain is ready: {}", m.name, toolchain.nvcc),
            );
        }
        return Ok(ExitCode::SUCCESS);
    }
    let hub = match m.hub.as_str() {
        "" => ctx.hub()?,
        origin => ctx.config.resolve(origin)?,
    };
    let payload = serde_json::json!({"toolchain": "install"});
    let spec = v1::RunSpec {
        kind: v1::RunKind::Update,
        payload: Some(v1::Json::parse(payload.to_string()).expect("JSON")),
        ..Default::default()
    };
    let submit = SubmitRun {
        agreed: None,
        on: On::Machine(m.name.clone()),
        hub: hub.origin.into(),
        kind: calcifer::Kind::Update,
        target: "toolchain (CUDA 13.0 compiler, ninja)".into(),
        spec,
        inputs: vec![],
        out: None,
        warnings: vec![],
    };
    let submitted = crate::run::submit(ctx, submit, None).await?;
    let run = crate::run::follow(ctx, submitted.run, false).await?;
    Ok(crate::run::exit_code(&run))
}

/// A wheel of `distribution` the machine fetches itself, `https://…/<wheel>#sha256=<hex>`: its
/// file name and the payload member. Nothing is sent from here.
/// `distribution`'s wheel at `https://…#sha256=<hex>`, or (an empty distribution) a `cozy` binary.
fn fetched(at: &str, distribution: &str) -> Result<(String, serde_json::Value), Error> {
    let refused = |why: String| Error::new(Kind::Validation, "wheel_refused", why);
    let (url, sha256) = at.split_once("#sha256=").unwrap_or((at, ""));
    let path = url
        .strip_prefix("https://")
        .map(|rest| rest.split('?').next().unwrap_or_default());
    let Some(path) = path else {
        return Err(refused(format!("{at}: a wheel URL is https")));
    };
    let file = path
        .split_once('/')
        .map_or("", |(_, p)| p.rsplit('/').next().unwrap_or_default());
    let named = file.starts_with(&format!("{distribution}-")) && file.ends_with(".whl");
    if !distribution.is_empty() && !named {
        return Err(refused(format!(
            "{url} names no {distribution} wheel ({distribution}-<version>-….whl)"
        )));
    }
    if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(refused(format!(
            "{at}: the URL ends #sha256=<64 hex>, the digest the machine holds its bytes to"
        )));
    }
    let member = serde_json::json!({"url": url, "sha256": sha256.to_ascii_lowercase()});
    Ok((file.to_owned(), member))
}

/// A local wheel of `distribution` (or, empty, a `cozy` binary), measured once: its file name and
/// what the run stages.
fn wheel(path: &Path, distribution: &str) -> Result<(String, StagedInput), Error> {
    let refused = |why: String| Error::new(Kind::Validation, "wheel_refused", why);
    let file = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let regular = std::fs::symlink_metadata(path).is_ok_and(|m| m.is_file());
    let named = file.starts_with(&format!("{distribution}-")) && file.ends_with(".whl");
    if !regular || (!distribution.is_empty() && !named) {
        return Err(refused(match distribution {
            "" => format!("{} is not a file", path.display()),
            _ => format!(
                "{} is not a {distribution} wheel file ({distribution}-<version>-….whl)",
                path.display()
            ),
        }));
    }
    let bytes = std::fs::read(path).map_err(|e| refused(e.to_string()))?;
    let input = StagedInput {
        path: std::path::absolute(path).map_err(|e| refused(e.to_string()))?,
        digest: Digest::of(&bytes),
        length: bytes.len() as u64,
    };
    Ok((file.to_owned(), input))
}

fn up_to_date(ctx: &Context, m: &Machine, target: &str) -> Result<ExitCode, Error> {
    let runtime = m.status.as_ref().map_or("", |s| s.runtime.as_str());
    if ctx.json {
        output::json(
            &serde_json::json!({"machine": m.name, "changed": false, "runtime": runtime, "target": target}),
        );
    } else {
        output::say(
            ctx,
            &format!(
                "{} runs cozy-runtime {runtime}; the Hub's target is {target}: nothing to update",
                m.name
            ),
        );
    }
    Ok(ExitCode::SUCCESS)
}

/// Whether release `a` is newer than `b`, by their leading dotted numbers.
fn newer(a: &str, b: &str) -> bool {
    let parts = |v: &str| -> Vec<u64> { v.split('.').map_while(|p| p.parse().ok()).collect() };
    parts(a) > parts(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TensorD 0.10.1 ignores an update's `binary` and succeeds unapplied: the update fails, typed,
    /// naming what it asked for, what runs, and the machine's own line about it.
    #[test]
    fn an_update_that_succeeded_unapplied_fails_naming_requested_and_running() {
        let old = MachineStatus {
            version: "0.10.1".into(), commit: "3f2a1b9c4d5e6f708192".into(), runtime: "0.27.4".into(),
            tensorfs: "0.10.0".into(), ..Default::default()
        };
        let want = [reports("binary", "cozy"), reports("runtime", "cozy_runtime-0.30.1-py3-none-any.whl")];
        assert!(!runs(&old, &want));
        let line = r#"WARN update r1: ignoring unknown payload field "binary""#;
        let error = unapplied("pod", &old, &want, Some(line));
        assert_eq!((error.kind, error.code.as_str()), (Kind::Conflict, "update_unapplied"));
        assert_eq!(error.message, "pod's update succeeded but it runs cozy 0.10.1 (release 3f2a1b9c4d5e), \
            cozy-runtime 0.27.4, tensorfs 0.10.0, not the requested dev build, cozy-runtime 0.30.1: \
            WARN update r1: ignoring unknown payload field \"binary\"");
        let new = MachineStatus { dev: true, runtime: "0.30.1".into(), ..old };
        assert!(runs(&new, &want));
    }
}
