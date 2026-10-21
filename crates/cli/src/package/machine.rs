//! `cozy package install|update|remove|list`: runs on a machine through Calcifer. The machine
//! resolves releases from the Hub and owns its environments; the CLI reads nothing from the Hub.
use std::{collections::BTreeMap, path::Path, process::ExitCode};

use protocol::{
    calcifer::{self, Environment, Machine, MachineState, On, Run, Status, SubmitRun},
    config::Origin,
    error::{Error, Kind},
    machine::{self as v1, Source},
};
use serde::{Deserialize, Serialize};

use super::{PackageRef, release_order};
use crate::{
    Context,
    machine::{Notices, route, until_ready},
    output::{self, Table},
    run,
};

fn spec(kind: v1::RunKind, source: Source, catalog_revision: String) -> v1::RunSpec {
    v1::RunSpec {
        kind,
        source: Some(source),
        catalog_revision,
        ..Default::default()
    }
}

fn release(package: &str, release: &str) -> Source {
    Source::Release(v1::Release {
        package: package.to_owned(),
        release: release.to_owned(),
    })
}

async fn submit(
    ctx: &Context,
    on: On,
    hub: &str,
    kind: calcifer::Kind,
    target: String,
    spec: v1::RunSpec,
    detach: bool,
) -> Result<Run, Error> {
    let ask = SubmitRun {
        agreed: None,
        on,
        hub: hub.to_owned(),
        kind,
        target,
        spec,
        inputs: vec![],
        out: None,
        warnings: vec![],
    };
    Ok(run::submit(ctx, ask, None, detach).await?.run)
}

/// Waits for `run` to settle, quietly: the verb reports the outcome itself.
async fn settle(ctx: &Context, run: Run) -> Result<Run, Error> {
    let calcifer = ctx.calcifer().await?;
    let mut events = calcifer.events(run.id.as_str(), 0);
    while let Some(event) = events.next().await {
        if event?.is_terminal() {
            break;
        }
    }
    calcifer.run(run.id.as_str()).await
}

/// Why a settled run did not succeed, as `machine: reason (code)`.
fn failure(run: &Run) -> String {
    let machine = run.machine.as_deref().unwrap_or("the machine");
    match &run.reason {
        Some(reason) => format!("{machine}: {} ({})", reason.message, reason.code),
        None => format!("{machine}: the run ended {:?}", run.status).to_lowercase(),
    }
}

#[derive(Deserialize)]
struct Installed {
    #[serde(default)]
    release: String,
    #[serde(default)]
    interface: serde_json::Value,
}

/// A warm run naming the release (none: the Hub's newest, as `cozy run` takes it) and no
/// function: the package environment only, no weights.
pub async fn install(
    ctx: &Context,
    package: &PackageRef,
    on: On,
    detach: bool,
) -> Result<ExitCode, Error> {
    let hub = ctx.hub()?;
    let name = package.package();
    let chosen = match &package.release {
        Some(chosen) => chosen.clone(),
        None => run::latest(ctx, &hub, &name).await?,
    };
    let warm = spec(v1::RunKind::Warm, release(&name, &chosen), String::new());
    let target = format!("{name}@{chosen}");
    let submitted = submit(ctx, on, &hub.origin, calcifer::Kind::Warm, target, warm, detach).await?;
    installed(ctx, submitted, &name, detach).await
}

/// A directory or a PEP 723 script, installed live: the next run reads its files as they are
/// then. This computer's machine reads them in place; another is sent what it lacks.
pub async fn install_directory(
    ctx: &Context,
    path: &Path,
    on: On,
    detach: bool,
) -> Result<ExitCode, Error> {
    let in_place = matches!(&on, On::Machine(name) if name.eq_ignore_ascii_case("local"));
    let local = super::local::source(ctx, path, in_place).await?;
    let warm = spec(
        v1::RunKind::Warm,
        Source::Local(local.source.clone()),
        String::new(),
    );
    let ask = SubmitRun {
        agreed: None,
        on,
        hub: ctx.hub()?.origin.into(),
        kind: calcifer::Kind::Warm,
        target: local.package.clone(),
        spec: warm,
        inputs: local.inputs.clone(),
        out: None,
        warnings: vec![],
    };
    let submitted = run::submit(ctx, ask, None, detach).await?.run;
    installed(ctx, submitted, &local.package, detach).await
}

/// Follows an install to its outcome and says what the machine holds now.
async fn installed(
    ctx: &Context,
    submitted: Run,
    package: &str,
    detach: bool,
) -> Result<ExitCode, Error> {
    let settled = run::follow(ctx, submitted, detach).await?;
    if detach || ctx.json || settled.status != Status::Completed {
        return Ok(run::exit_code(&settled));
    }
    let installed: Option<Installed> = settled
        .result
        .clone()
        .and_then(|r| serde_json::from_value(r).ok());
    let machine = settled.machine.clone().unwrap_or_default();
    let mut next = vec![format!("cozy package list --on {machine}")];
    if let Some(installed) = installed {
        output::say(
            ctx,
            &format!("{package}@{} is installed on {machine}", installed.release),
        );
        if !package.starts_with("local/") {
            let on = On::Machine(machine.clone());
            let held = machines(ctx, Some(&on)).await?.remove(0);
            let older: Vec<&str> = (environments(ctx, &held)?.into_iter())
                .filter(|env| env.package == package && env.release != installed.release)
                .map(|env| env.release.as_str())
                .collect();
            if !older.is_empty() {
                let older = older.join(", ");
                output::say(ctx, &format!("{machine} also holds {package}@{older}"));
                next.insert(0, format!("cozy package update {package} --on {machine}"));
            }
        }
        let first = run::interface::Interface::parse(&installed.interface)
            .ok()
            .and_then(|interface| interface.public().first().map(|name| name.to_string()));
        if let Some(callable) = first {
            next.insert(0, format!("cozy run {package}/{callable} --describe"));
        }
    }
    output::next(ctx, &next.iter().map(String::as_str).collect::<Vec<_>>());
    Ok(ExitCode::SUCCESS)
}

/// The machines this command reads: `on` alone, read from the machine now (so a run that just
/// settled shows), or every ready one as Calcifer last saw it. A booting `on` is waited for, as a
/// run sent to it waits.
async fn machines(ctx: &Context, on: Option<&On>) -> Result<Vec<Machine>, Error> {
    let calcifer = ctx.calcifer().await?;
    let Some(on) = on else {
        let all = calcifer.machines().await?.machines;
        return Ok(all
            .into_iter()
            .filter(|m| m.state == MachineState::Ready)
            .collect());
    };
    let mut machine: Machine = calcifer
        .get(&route(&named(on)?.to_ascii_lowercase())?)
        .await?;
    if machine.state == MachineState::Booting {
        let name = machine.name.clone();
        output::say(ctx, &format!("waiting for {name} to start"));
        machine = until_ready(ctx, machine, Notices::default())
            .await?
            .ok_or_else(|| {
                let message = format!("{name} has not started; nothing was sent to it");
                Error::new(Kind::Unavailable, "machine_booting", message)
            })?;
    }
    let path = route(&machine.name.to_ascii_lowercase())?;
    let machine: Machine = calcifer.get_query(&path, &[("fresh", "true")]).await?;
    Ok(vec![machine])
}

/// The machine `--on` names, whose packages can be read.
pub(crate) fn named(on: &On) -> Result<&str, Error> {
    match on {
        On::Machine(name) => Ok(name),
        _ => Err(Error::new(
            Kind::Usage,
            "machine_name_required",
            format!("--on {on} names no machine whose packages can be read"),
        )),
    }
}

/// A machine's installations within `--tensorhub`'s scope: that Hub's and local code.
fn environments<'a>(ctx: &Context, machine: &'a Machine) -> Result<Vec<&'a Environment>, Error> {
    let narrowed = ctx.narrowed()?;
    let scope = |env: &Environment| match &narrowed {
        Some(hub) => env.hub.is_empty() || origin(&env.hub).as_ref() == Some(&hub.origin),
        None => true,
    };
    Ok(machine
        .status
        .iter()
        .flat_map(|s| &s.environments)
        .filter(|env| scope(env))
        .collect())
}

/// An installation's Hub in its one spelling (a machine may write `https://tensorhub.com:443`);
/// none for local code.
fn origin(hub: &str) -> Option<Origin> {
    Origin::parse(hub).ok()
}

/// What a run on an installation's Hub names it: its one spelling, or as the machine wrote it.
fn canonical(hub: &str) -> String {
    origin(hub).map_or_else(|| hub.to_owned(), String::from)
}

/// A Hub origin as this computer names it; local code has none.
fn hub_label(ctx: &Context, hub: &str) -> String {
    if hub.is_empty() {
        return "local".into();
    }
    let Some(origin) = origin(hub) else {
        return hub.to_owned();
    };
    let known = ctx.config.hubs().into_iter().find(|h| h.origin == origin);
    known
        .map(|hub| hub.label().to_owned())
        .unwrap_or_else(|| origin.to_string())
}

/// Refuses a name the machine does not hold, before any run.
fn held(names: &[String], envs: &[&Environment], machine: &str) -> Result<(), Error> {
    match names
        .iter()
        .find(|name| !envs.iter().any(|env| &&env.package == name))
    {
        Some(name) => Err(Error::new(
            Kind::NotFound,
            "package_not_installed",
            format!("{name} is not installed on {machine}"),
        )
        .next(format!("cozy package list --on {machine}"))),
        None => Ok(()),
    }
}

pub(crate) const HEADERS: [&str; 5] = ["MACHINE", "PACKAGE", "RELEASE", "LEVEL", "HUB"];

/// Each package `machines` hold within `--tensorhub`'s scope.
pub(crate) fn rows<'a>(
    ctx: &Context,
    machines: impl IntoIterator<Item = &'a Machine>,
) -> Result<Vec<Held<'a>>, Error> {
    let mut rows = vec![];
    for machine in machines {
        for env in environments(ctx, machine)? {
            rows.push(Held {
                machine: &machine.name,
                package: &env.package,
                release: &env.release,
                level: &env.level,
                hub: hub_label(ctx, &env.hub),
                warning: &env.warning,
            });
        }
    }
    Ok(rows)
}

#[derive(Serialize)]
pub(crate) struct Held<'a> {
    machine: &'a str,
    package: &'a str,
    release: &'a str,
    level: &'a str,
    hub: String,
    /// Why the environment does not run the machine's own Runtime; empty when it does.
    warning: &'a str,
}

impl Held<'_> {
    pub(crate) fn cells(&self) -> Vec<String> {
        let (machine, package, release) = (self.machine, self.package, self.release);
        vec![
            machine.into(),
            package.into(),
            release.into(),
            self.level.into(),
            self.hub.clone(),
        ]
    }
}

pub async fn list(ctx: &Context, on: Option<On>, once: bool) -> Result<ExitCode, Error> {
    if !once && !ctx.json && ctx.term.stdout {
        let view = super::live::Packages::open(ctx, on).await?;
        crate::tui::show(ctx, crate::tui::Screen::Packages(view)).await?;
        return Ok(ExitCode::SUCCESS);
    }
    let machines = machines(ctx, on.as_ref()).await?;
    let rows = rows(ctx, &machines)?;
    if ctx.json {
        output::json(&serde_json::json!({ "packages": rows }));
        return Ok(ExitCode::SUCCESS);
    }
    if rows.is_empty() {
        println!("No packages installed.");
        output::next(
            ctx,
            &["cozy package search", "cozy package install <org/name>"],
        );
        return Ok(ExitCode::SUCCESS);
    }
    let mut table = Table::new(ctx, HEADERS);
    for row in &rows {
        table = table.row(row.cells());
    }
    for row in rows.iter().filter(|row| !row.warning.is_empty()) {
        table = table.note(format!("{}@{}: {}", row.package, row.release, row.warning));
    }
    table.print();
    Ok(ExitCode::SUCCESS)
}

#[derive(Deserialize, Default)]
struct Removal {
    #[serde(default)]
    removed: Vec<Removed>,
    #[serde(default)]
    reclaimed_bytes: u64,
    #[serde(default)]
    retained: Vec<String>,
}

#[derive(Deserialize, Serialize)]
struct Removed {
    package: String,
    release: String,
    #[serde(default)]
    hub: String,
}

/// One REMOVE run per (package, Hub) the machine holds: every release of it from that Hub.
pub async fn remove(ctx: &Context, names: &[String], on: On) -> Result<ExitCode, Error> {
    let machine = machines(ctx, Some(&on)).await?.remove(0);
    let envs = environments(ctx, &machine)?;
    held(names, &envs, &machine.name)?;
    let default_hub = ctx.hub()?.origin;
    let mut targets: Vec<(&str, String)> = envs
        .iter()
        .filter(|env| names.contains(&env.package))
        .map(|env| (env.package.as_str(), canonical(&env.hub)))
        .collect();
    targets.sort();
    targets.dedup();
    let (mut removed, mut reclaimed, mut retained, mut failed) = (vec![], 0, false, vec![]);
    for (package, hub) in targets {
        let hub = if hub.is_empty() {
            default_hub.as_str()
        } else {
            hub.as_str()
        };
        let spec = spec(v1::RunKind::Remove, release(package, ""), String::new());
        let on = On::Machine(machine.name.clone());
        let ran = settle(
            ctx,
            submit(
                ctx,
                on,
                hub,
                calcifer::Kind::Remove,
                package.to_owned(),
                spec,
                false,
            )
            .await?,
        )
        .await?;
        if ran.status != Status::Completed {
            failed.push(format!("{package}: {}", failure(&ran)));
            continue;
        }
        let result: Removal = ran
            .result
            .and_then(|r| serde_json::from_value(r).ok())
            .unwrap_or_default();
        reclaimed += result.reclaimed_bytes;
        retained |= !result.retained.is_empty();
        removed.extend(result.removed);
    }
    if ctx.json {
        output::json(&serde_json::json!({
            "machine": machine.name, "removed": removed, "reclaimed_bytes": reclaimed, "failed": failed,
        }));
    } else {
        let mut table = Table::new(ctx, ["PACKAGE", "RELEASE", "HUB"]);
        for row in &removed {
            table = table.row([
                row.package.clone(),
                row.release.clone(),
                hub_label(ctx, &row.hub),
            ]);
        }
        table = table.note(format!(
            "{} reclaimed on {}",
            output::bytes(reclaimed),
            machine.name
        ));
        if retained {
            table = table.note("Files a running executor still uses stay until it ends.");
        }
        for failure in &failed {
            table = table.note(format!("not removed: {failure}"));
        }
        table.print();
    }
    Ok(if failed.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

#[derive(Serialize)]
struct Updated {
    package: String,
    hub: String,
    from: Vec<String>,
    to: String,
    status: &'static str,
    detail: String,
}

/// Each published (package, Hub) the machine holds moves to that Hub's newest release, then its
/// older releases leave. A newer one stays (a pre-release, or one the Hub yanked since).
pub async fn update(ctx: &Context, names: &[String], on: On) -> Result<ExitCode, Error> {
    let machine = machines(ctx, Some(&on)).await?.remove(0);
    let envs = environments(ctx, &machine)?;
    held(names, &envs, &machine.name)?;
    let mut groups: BTreeMap<(&str, String), Vec<&Environment>> = BTreeMap::new();
    for env in envs
        .iter()
        .filter(|env| names.is_empty() || names.contains(&env.package))
    {
        groups
            .entry((&env.package, canonical(&env.hub)))
            .or_default()
            .push(env);
    }
    let on = || On::Machine(machine.name.clone());
    let mut rows = vec![];
    for ((package, hub), held) in groups {
        let hub = hub.as_str();
        let mut row = Updated {
            package: package.to_owned(),
            hub: hub_label(ctx, hub),
            from: held.iter().map(|env| env.release.clone()).collect(),
            to: String::new(),
            status: "skipped",
            detail: String::new(),
        };
        if hub.is_empty() || package.starts_with("local/") {
            row.detail = "local code: install its directory again to change it".into();
            rows.push(row);
            continue;
        }
        // A catalog revision no ask used before makes the machine read the Hub's newest again.
        let warm = spec(v1::RunKind::Warm, release(package, ""), fresh());
        let ran = settle(
            ctx,
            submit(
                ctx,
                on(),
                hub,
                calcifer::Kind::Warm,
                package.to_owned(),
                warm,
                false,
            )
            .await?,
        )
        .await?;
        if ran.status != Status::Completed {
            (row.status, row.detail) = ("failed", failure(&ran));
            rows.push(row);
            continue;
        }
        row.to = ran
            .result
            .and_then(|r| r["release"].as_str().map(str::to_owned))
            .unwrap_or_default();
        row.status = if row.from.contains(&row.to) {
            "current"
        } else {
            "updated"
        };
        for env in held
            .iter()
            .filter(|env| release_order(&env.release) < release_order(&row.to))
        {
            let source = Source::Installation(env.installation.clone());
            let target = format!("{package}@{}", env.release);
            let removal = spec(v1::RunKind::Remove, source, String::new());
            let ran = settle(
                ctx,
                submit(ctx, on(), hub, calcifer::Kind::Remove, target, removal, false).await?,
            )
            .await?;
            row.status = "updated";
            if ran.status != Status::Completed {
                row.detail = format!("{package}@{} stays: {}", env.release, failure(&ran));
            }
        }
        rows.push(row);
    }
    let failed = rows.iter().any(|row| row.status == "failed");
    if ctx.json {
        output::json(&serde_json::json!({ "machine": machine.name, "packages": rows }));
    } else {
        let mut table = Table::new(ctx, ["PACKAGE", "FROM", "TO", "STATUS", "HUB", "DETAIL"]);
        for row in &rows {
            let from = row.from.join(", ");
            let cells = [
                &row.package,
                &from,
                &row.to,
                row.status,
                &row.hub,
                &row.detail,
            ];
            table = table.row(cells.map(|cell| cell.to_string()));
        }
        table.note("Model downloads: skipped.").print();
    }
    Ok(if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}

/// A catalog revision no earlier ask used.
fn fresh() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("update-{:x}-{:x}", now.as_nanos(), std::process::id())
}
