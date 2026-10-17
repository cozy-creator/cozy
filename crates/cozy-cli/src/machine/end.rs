//! `cozy machine end`, `keepalive` and `update`: asks Calcifer makes of one machine.
use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

use cozy_proto::{
    calcifer::{
        self, Ended, Keepalive, Machine, MachineKind, MachineList, MachineState, On, RunsEvent,
        StagedInput, SubmitRun,
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
        output::warn(
            ctx,
            &format!(
                "Gave up outputs that had not landed: {}",
                abandoned.join(", ")
            ),
        );
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
/// machine with the run; an `https://…#sha256=<hex>` URL the machine fetches itself; a member not
/// named keeps the machine's own), else to its Hub's target. A numbered run of kind update, which
/// waits for the work the machine accepted and is followed like any run.
pub async fn update(
    ctx: &Context,
    name: &str,
    runtime_wheel: Option<PathBuf>,
    tensorfs_wheel: Option<PathBuf>,
) -> Result<ExitCode, Error> {
    let mut payload = serde_json::Map::new();
    let (mut staged, mut inputs, mut named) = (vec![], vec![], vec![]);
    for (member, distribution, path) in [
        ("runtime", "cozy_runtime", runtime_wheel),
        ("tensorfs", "tensorfs", tensorfs_wheel),
    ] {
        let Some(path) = path else { continue };
        if let Some(at) = path.to_str().filter(|p| p.contains("://")) {
            let (file, fetched) = fetched(at, distribution)?;
            named.push(format!("{member} {file}"));
            payload.insert(member.into(), fetched);
            continue;
        }
        let (file, input) = wheel(&path, distribution)?;
        named.push(format!("{member} {file}"));
        inputs.push(v1::InputFile {
            field: member.into(),
            digest: input.digest.to_string(),
            length: input.length,
            media_type: "application/zip".into(),
            order: 0,
        });
        payload.insert(member.into(), file.into());
        staged.push(input);
    }
    let m = machine(ctx, name).await?;
    let hub = match m.hub.as_str() {
        "" => ctx.hub()?,
        origin => ctx.config.resolve(origin)?,
    };
    if payload.is_empty() {
        let Some(status) = &m.status else {
            return Err(Error::new(
                Kind::Unavailable,
                "machine_unanswered",
                format!("{} has not reported the software it runs", m.name),
            )
            .next(format!("cozy machine show {}", m.name)));
        };
        let target = hub_answer(ctx.hub_anonymous(&hub).get_software().await)?;
        // An unpinned target older than the machine's software is no reason to go back.
        if (status.runtime == target.runtime && status.tensorfs == target.tensorfs)
            || (!target.pinned && newer(&status.runtime, &target.runtime))
        {
            return up_to_date(ctx, &m, &target.runtime);
        }
        named.push(format!("runtime {}", target.runtime));
        named.push(format!("tensorfs {}", target.tensorfs));
        payload.insert("runtime".into(), target.runtime.into());
        payload.insert("tensorfs".into(), target.tensorfs.into());
    }
    payload.insert("agent".into(), "bundled".into());
    let spec = v1::RunSpec {
        kind: v1::RunKind::Update,
        payload: Some(v1::Json::parse(serde_json::Value::Object(payload).to_string()).expect("JSON")),
        inputs,
        ..Default::default()
    };
    let submit = SubmitRun {
        on: On::Machine(m.name.clone()),
        hub: hub.origin,
        kind: calcifer::Kind::Update,
        target: named.join(", "),
        spec,
        inputs: staged,
        out: None,
        warnings: vec![],
    };
    let submitted = crate::run::submit(ctx, submit, None).await?;
    let run = crate::run::follow(ctx, submitted.run, false).await?;
    Ok(crate::run::exit_code(&run))
}

/// A wheel of `distribution` the machine fetches itself, `https://…/<wheel>#sha256=<hex>`: its
/// file name and the payload member. Nothing is sent from here.
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
    if !file.starts_with(&format!("{distribution}-")) || !file.ends_with(".whl") {
        return Err(refused(format!(
            "{url} names no {distribution} wheel ({distribution}-<version>-….whl)"
        )));
    }
    if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(refused(format!(
            "{at}: a wheel URL ends #sha256=<64 hex>, the digest the machine holds its bytes to"
        )));
    }
    let member = serde_json::json!({"url": url, "sha256": sha256.to_ascii_lowercase()});
    Ok((file.to_owned(), member))
}

/// A local wheel of `distribution`, measured once: its file name and what the run stages.
fn wheel(path: &Path, distribution: &str) -> Result<(String, StagedInput), Error> {
    let refused = |why: String| Error::new(Kind::Validation, "wheel_refused", why);
    let file = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let regular = std::fs::symlink_metadata(path).is_ok_and(|m| m.is_file());
    if !regular || !file.starts_with(&format!("{distribution}-")) || !file.ends_with(".whl") {
        return Err(refused(format!(
            "{} is not a {distribution} wheel file ({distribution}-<version>-….whl)",
            path.display()
        )));
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
