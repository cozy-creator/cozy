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

/// Updates a machine's Runtime, TensorFS and TensorD (its `cozy` binary) in one run and one
/// restart: each named as a release version the machine fetches itself (PyPI for the wheels, the
/// GitHub release for `cozy`), or a local path, a dev build the CLI sends (a wheel or a project to
/// build; a `cozy` binary or the cozy-creator checkout to build). With none named, each moves to
/// its Hub's target. A numbered run of kind update, which waits for the work the machine accepted
/// and is followed like any run; once it succeeds, the machine must report what it asked for.
pub async fn update(
    ctx: &Context,
    name: &str,
    runtime: Option<String>,
    tensorfs: Option<String>,
    tensord: Option<String>,
    gpu_memory_limit: Option<u64>,
) -> Result<ExitCode, Error> {
    let mut payload = serde_json::Map::new();
    let (mut staged, mut inputs, mut named, mut want) = (vec![], vec![], vec![], vec![]);
    let mut asked = vec![];
    for (member, distribution, value) in [
        ("runtime", "cozy_runtime", runtime),
        ("tensorfs", "tensorfs", tensorfs),
        ("tensord", "", tensord),
    ] {
        if let Some(value) = value {
            asked.push((member, distribution, Component::of(&value, distribution)?));
        }
    }
    let m = machine(ctx, name).await?;
    let hub = match m.hub.as_str() {
        "" => ctx.hub()?,
        origin => ctx.config.resolve(origin)?,
    };
    if asked.is_empty() && gpu_memory_limit.is_none() {
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
        let moves = |running: &str, wanted: &str| running != wanted && (target.pinned || newer(wanted, running));
        let cozy = target.cozy.as_deref().filter(|cozy| moves(&status.version, cozy));
        for (member, distribution, running, wanted) in [
            ("runtime", "cozy_runtime", status.runtime.as_str(), Some(target.runtime.as_str())),
            ("tensorfs", "tensorfs", status.tensorfs.as_str(), Some(target.tensorfs.as_str())),
            ("tensord", "", status.version.as_str(), cozy),
        ] {
            if let Some(wanted) = wanted.filter(|wanted| moves(running, wanted)) {
                asked.push((member, distribution, Component::Release(wanted.into())));
            }
        }
        if asked.is_empty() {
            return up_to_date(ctx, &m, status);
        }
    }
    // A TensorD before 0.11.16 takes its binary as `binary`, a file it is sent: a release is
    // fetched here for it (version skew).
    let status = m.status.clone().unwrap_or_default();
    let older = !names_tensord(&status.version);
    // A dev `cozy` built or picked here, which `local` then runs (Calcifer too).
    let mut dev = false;
    for (member, distribution, component) in asked {
        let key = if member == "tensord" && older { "binary" } else { member };
        let (path, version) = match component {
            Component::Release(version) if member == "tensord" && older => {
                let target = dist_target(&status.platform)?;
                let fetched = crate::update::release_binary(&version, &target).await?;
                (kept(ctx, fetched, &format!("cozy-{version}-{target}"))?, Some(version))
            }
            Component::Release(version) => {
                want.push(reports(member, &version));
                named.push(format!("{member} {version}"));
                payload.insert(key.into(), version.into());
                continue;
            }
            Component::Build(path) => (path, None),
        };
        let (file, input) = wheel(&path, distribution)?;
        dev |= member == "tensord" && version.is_none();
        want.push(match (member, &version) {
            ("tensord", Some(version)) => ("version", version.as_str().into()),
            ("tensord", None) => built_version(&path).map_or(("dev", true.into()), |v| ("version", v.into())),
            _ => reports(member, &file),
        });
        named.push(match (&version, reports(member, &file)) {
            (Some(version), _) => format!("{member} {version}"),
            (None, (_, serde_json::Value::String(stamped))) if member != "tensord" => format!("{member} {stamped} (dev)"),
            (None, _) => format!("{member} {file} (dev)"),
        });
        inputs.push(v1::InputFile {
            field: key.into(),
            digest: input.digest.to_string(),
            length: input.length,
            media_type: match distribution {
                "" => "application/octet-stream",
                _ => "application/zip",
            }
            .into(),
            order: 0,
        });
        payload.insert(key.into(), file.into());
        staged.push(input);
    }
    let restarts = payload.contains_key("tensord") || payload.contains_key("binary");
    if let Some(limit) = gpu_memory_limit {
        named.push(match limit {
            0 => "no GPU memory limit".into(),
            bytes => format!("a GPU memory limit of {}", output::bytes(bytes)),
        });
        payload.insert("gpu_memory_limit".into(), limit.into());
    }
    // A new binary or a limit restarts the machine: either way its warm models reload.
    if (restarts || gpu_memory_limit.is_some())
        && m.status
            .as_ref()
            .is_some_and(|s| s.warm.iter().any(|w| !w.level.is_empty()))
    {
        output::warn(&format!("{}'s warm models reload after it restarts", m.name));
    }
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
    let run = crate::run::follow(ctx, submitted.run).await?;
    if run.status == calcifer::Status::Completed {
        updated(ctx, &m.name, &verify(ctx, &m.name, run.id.as_str(), &want).await?);
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

/// Whether a TensorD at `version` names its binary `tensord` and fetches a release itself
/// (0.11.16 on); one that does not parse is taken as current.
fn names_tensord(version: &str) -> bool {
    semver::Version::parse(version).map_or(true, |v| (v.major, v.minor, v.patch) >= (0, 11, 16))
}

/// The dist target of a machine's `platform` (`linux/x86_64`).
fn dist_target(platform: &str) -> Result<String, Error> {
    let target = match platform {
        "linux/x86_64" => "x86_64-unknown-linux-gnu",
        "linux/aarch64" => "aarch64-unknown-linux-gnu",
        "macos/aarch64" => "aarch64-apple-darwin",
        "macos/x86_64" => "x86_64-apple-darwin",
        "windows/x86_64" => "x86_64-pc-windows-msvc",
        "windows/aarch64" => "aarch64-pc-windows-msvc",
        other => return Err(Error::new(Kind::Unavailable, "platform_unknown", format!("no cozy release is built for {other:?}"))),
    };
    Ok(target.into())
}

/// A fetched release kept under this computer's cozy home until the run sends it.
fn kept(ctx: &Context, fetched: tempfile::TempPath, name: &str) -> Result<PathBuf, Error> {
    let dir = ctx.home.root().join("tmp");
    let path = dir.join(name);
    std::fs::create_dir_all(&dir)
        .and_then(|()| fetched.persist(&path).map_err(|e| e.error))
        .map_err(|e| Error::new(Kind::Internal, "update_failed", format!("keep {}: {e}", path.display())))?;
    Ok(path)
}

/// The release a `cozy` binary built here says it is, when it runs here.
fn built_version(path: &Path) -> Option<String> {
    let out = std::process::Command::new(path).arg("--version").stdin(std::process::Stdio::null()).output().ok()?;
    String::from_utf8_lossy(&out.stdout).trim().strip_prefix("cozy ").map(str::to_owned).filter(|_| out.status.success())
}

/// What `cozy machine update` installs of one component: a release version, or a local path.
enum Component {
    Release(String),
    /// A wheel or `cozy` binary file, built here first when the path is a project to build.
    Build(PathBuf),
}

impl Component {
    fn of(value: &str, distribution: &str) -> Result<Component, Error> {
        let path = Path::new(value);
        if path.is_file() {
            return Ok(Component::Build(path.to_path_buf()));
        }
        if path.is_dir() {
            return build(path, distribution).map(Component::Build);
        }
        let version = value.trim_start_matches('v');
        if !version.is_empty() && version.bytes().all(|c| c.is_ascii_alphanumeric() || b".-+".contains(&c)) {
            return Ok(Component::Release(version.into()));
        }
        Err(Error::new(Kind::Validation, "component_refused", format!("{value:?} is neither a release version nor a path here")))
    }
}

/// Builds the project at `dir`: a Runtime or TensorFS wheel (`uv build`), else the `cozy` binary
/// of a cozy-creator checkout (`cargo build --release`). The built file.
fn build(dir: &Path, distribution: &str) -> Result<PathBuf, Error> {
    let failed = |why: String| Error::new(Kind::Validation, "build_failed", format!("{}: {why}", dir.display()));
    let out = dir.join("target").join("cozy-update");
    let status = match distribution {
        "" => std::process::Command::new("cargo")
            .args(["build", "--release", "--bin", "cozy"])
            .current_dir(dir)
            .status(),
        _ => std::process::Command::new("uv")
            .args(["build", "--wheel", "--out-dir"])
            .arg(&out)
            .arg(dir)
            .status(),
    };
    match status {
        Ok(status) if status.success() => {}
        Ok(status) => return Err(failed(format!("the build exited {status}"))),
        Err(e) => return Err(failed(e.to_string())),
    }
    if distribution.is_empty() {
        return Ok(dir.join("target/release").join(if cfg!(windows) { "cozy.exe" } else { "cozy" }));
    }
    let newest = std::fs::read_dir(&out)
        .map_err(|e| failed(e.to_string()))?
        .flatten()
        .map(|entry| entry.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(&format!("{distribution}-")) && n.ends_with(".whl")))
        .max_by_key(|p| p.metadata().and_then(|m| m.modified()).ok());
    let built = newest.ok_or_else(|| failed(format!("the build left no {distribution} wheel in {}", out.display())))?;
    stamp(&built, &local_version(dir, &built)).map_err(|e| failed(format!("stamping {}: {e}", built.display())))
}

/// A dev build's PEP 440 local version: `dev.g<commit>`, `.dirty` with uncommitted changes; a tree
/// git does not know is named by the built wheel's digest.
fn local_version(dir: &Path, wheel: &Path) -> String {
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
        out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };
    match git(&["rev-parse", "--short=12", "HEAD"]) {
        Some(commit) => {
            let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
            format!("dev.g{commit}{}", if dirty { ".dirty" } else { "" })
        }
        None => {
            let digest = std::fs::read(wheel).map(|bytes| Digest::of(&bytes).to_string()).unwrap_or_default();
            format!("dev.{}", digest.trim_start_matches("sha256:").get(..12).unwrap_or("0"))
        }
    }
}

/// `wheel` restamped with `local` as its version's local part (`0.31.0+dev.g<commit>`): its file
/// name, `.dist-info` (and `.data`) folder, METADATA and RECORD, so the machine reports which
/// build it runs. The original is removed.
fn stamp(wheel: &Path, local: &str) -> std::io::Result<PathBuf> {
    use base64::Engine as _;
    use sha2::Digest as _;
    use std::io::{Read as _, Write as _};
    let invalid = |why: String| std::io::Error::new(std::io::ErrorKind::InvalidData, why);
    let file = wheel.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let (distribution, rest) = file.strip_suffix(".whl").and_then(|s| s.split_once('-')).ok_or_else(|| invalid(format!("{file} is not a wheel name")))?;
    let (version, tags) = rest.split_once('-').ok_or_else(|| invalid(format!("{file} is not a wheel name")))?;
    let stamped = match version.contains('+') {
        true => format!("{version}.{local}"),
        false => format!("{version}+{local}"),
    };
    let (info, data) = (format!("{distribution}-{version}.dist-info/"), format!("{distribution}-{version}.data/"));
    let new_info = format!("{distribution}-{stamped}.dist-info/");
    let renamed = |name: &str| match (name.strip_prefix(&info), name.strip_prefix(&data)) {
        (Some(rest), _) => format!("{new_info}{rest}"),
        (_, Some(rest)) => format!("{distribution}-{stamped}.data/{rest}"),
        _ => name.to_owned(),
    };
    let mut archive = zip::ZipArchive::new(std::fs::File::open(wheel)?).map_err(std::io::Error::other)?;
    let mut read = |name: &str| -> std::io::Result<String> {
        let mut text = String::new();
        archive.by_name(name).map_err(|_| invalid(format!("{file} has no {name}")))?.read_to_string(&mut text)?;
        Ok(text)
    };
    let (metadata, record) = (read(&format!("{info}METADATA"))?, read(&format!("{info}RECORD"))?);
    // The header's Version line only, never one in the description below it.
    let mut replaced = false;
    let metadata: String = metadata
        .split_inclusive('\n')
        .map(|line| match !replaced && line.starts_with("Version:") {
            true => {
                replaced = true;
                format!("Version: {stamped}\n")
            }
            false => line.to_owned(),
        })
        .collect();
    let hash = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(sha2::Sha256::digest(metadata.as_bytes()));
    let record: String = record
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let path = line.rsplitn(3, ',').nth(2).unwrap_or(line);
            match path.strip_prefix(&info) {
                Some("METADATA") => format!("{new_info}METADATA,sha256={hash},{}\n", metadata.len()),
                Some("RECORD") => format!("{new_info}RECORD,,\n"),
                _ => format!("{}{}\n", renamed(path), &line[path.len()..]),
            }
        })
        .collect();
    let out = wheel.with_file_name(format!("{distribution}-{stamped}-{tags}.whl"));
    let mut writer = zip::ZipWriter::new(std::fs::File::create(&out)?);
    for i in 0..archive.len() {
        let entry = archive.by_index_raw(i).map_err(std::io::Error::other)?;
        let name = entry.name().map_err(std::io::Error::other)?.into_owned();
        if name != format!("{info}METADATA") && name != format!("{info}RECORD") {
            writer.raw_copy_file_rename(entry, renamed(&name)).map_err(std::io::Error::other)?;
        }
    }
    let options = zip::write::SimpleFileOptions::default();
    writer.start_file(format!("{new_info}METADATA"), options).map_err(std::io::Error::other)?;
    writer.write_all(metadata.as_bytes())?;
    writer.start_file(format!("{new_info}RECORD"), options).map_err(std::io::Error::other)?;
    writer.write_all(record.as_bytes())?;
    writer.finish().map_err(std::io::Error::other)?;
    std::fs::remove_file(wheel)?;
    Ok(out)
}

/// The machine's three components after an update.
fn updated(ctx: &Context, name: &str, s: &MachineStatus) {
    if ctx.json {
        output::json(&serde_json::json!({"machine": name, "tensord": s.version, "runtime": s.runtime, "tensorfs": s.tensorfs}));
    } else {
        let build = if s.dev { " (dev build)" } else { "" };
        output::say(ctx, &format!("{name} runs TensorD (cozy) {}{build}, Runtime {}, TensorFS {}", s.version, s.runtime, s.tensorfs));
    }
}

/// What a machine running `named` (a version, or a wheel file) as `member` reports.
fn reports(member: &'static str, named: &str) -> (&'static str, serde_json::Value) {
    match (member, named.strip_suffix(".whl")) {
        ("tensord", _) => ("version", named.into()),
        (_, Some(wheel)) => (member, wheel.split('-').nth(1).unwrap_or_default().into()),
        (_, None) => (member, named.into()),
    }
}

/// A succeeded update must leave the machine reporting what it asked for (`want`): one that
/// ignored what it could not carry out fails here, with its log's line about the update.
async fn verify(ctx: &Context, name: &str, run: &str, want: &[(&str, serde_json::Value)]) -> Result<MachineStatus, Error> {
    let m = machine(ctx, name).await?;
    let status = m.status.unwrap_or_default();
    if m.unreachable.is_none() && runs(&status, want) {
        return Ok(status);
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

fn up_to_date(ctx: &Context, m: &Machine, s: &MachineStatus) -> Result<ExitCode, Error> {
    if ctx.json {
        output::json(&serde_json::json!({"machine": m.name, "changed": false, "tensord": s.version, "runtime": s.runtime, "tensorfs": s.tensorfs}));
    } else {
        output::say(ctx, &format!(
            "{} runs TensorD (cozy) {}, Runtime {}, TensorFS {}: the Hub's target, nothing to update",
            m.name, s.version, s.runtime, s.tensorfs
        ));
    }
    Ok(ExitCode::SUCCESS)
}

/// Whether release `a` is newer than `b`, by their leading dotted numbers.
fn newer(a: &str, b: &str) -> bool {
    let parts = |v: &str| -> Vec<u64> { v.split('.').map_while(|p| p.parse().ok()).collect() };
    parts(a) > parts(b)
}
