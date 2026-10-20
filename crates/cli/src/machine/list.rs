//! `cozy machine list` and `show`: Calcifer's picture of every machine, which never waits on a Hub.
use std::process::ExitCode;

use protocol::{
    calcifer::{Machine, MachineKind, MachineList, MachineState, MachineStatus, Rental},
    error::Error,
};

use super::{Notices, clock, machine, now_ms, usd};
use crate::{Context, output, tui};

/// A Hub failure code: the provider stopped the pod. The Hub keeps the rental and never replaces it.
const PROVIDER_STOPPED: &str = "readiness.provider_stopped";

const IDLE_RULE: &str = "Rentals end themselves after 15 minutes idle (no queued or running work).";

pub(crate) const HEADERS: [&str; 12] = [
    "MACHINE", "KIND", "GPUS", "FAMILY", "STATE", "$/HOUR", "SPENT", "UPTIME", "RUNNING", "QUEUED",
    "ENDS", "HUB",
];

pub async fn list(ctx: &Context, once: bool) -> Result<ExitCode, Error> {
    if !once && !ctx.json && ctx.term.stdout {
        let view = super::live::Machines::open(ctx).await?;
        crate::tui::show(ctx, crate::tui::Screen::Machines(view)).await?;
        return Ok(ExitCode::SUCCESS);
    }
    let list: MachineList = ctx
        .calcifer()
        .await?
        .get("/v1/machines?refresh=true")
        .await?;
    let code = if list.unanswered.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    };
    if ctx.json {
        output::json(&list);
        return Ok(code);
    }
    let live = || {
        list.machines
            .iter()
            .filter(|m| m.state != MachineState::Ended)
    };
    let rentals: Vec<&Rental> = live().filter_map(|m| m.rental.as_ref()).collect();
    if !rentals.is_empty() {
        let burn: i64 = rentals.iter().map(|r| r.usd_micros_per_hour).sum();
        println!(
            "Rentals running: {} · {}/hour now{}",
            rentals.len(),
            usd(burn),
            accrued(&rentals)
        );
    }
    for problem in &list.unanswered {
        println!(
            "hub {} did not answer: {}; its rentals may still be billing",
            hub_label(ctx, &problem.hub),
            problem.message
        );
    }
    for hub in &list.signed_out {
        let login = match ctx.config.resolve(hub) {
            Ok(hub) => ctx.login(&hub),
            Err(_) => format!("cozy login <email> --tensorhub {hub}"),
        };
        println!("hub {}: not logged in; `{login}`", hub_label(ctx, hub));
    }
    let mut table = output::Table::new(ctx, HEADERS);
    for row in rows(ctx, &list) {
        table = table.row(row);
    }
    table.print();
    for note in live().filter_map(provider_stopped) {
        println!("{note}");
    }
    for m in live() {
        for note in Notices::default().fresh(m) {
            println!("{}: {note}", m.name);
        }
        if unreachable_since(m).is_some() {
            println!("{}: `cozy machine end {}` ends it", m.name, m.name);
        }
    }
    if !rentals.is_empty() {
        println!("{IDLE_RULE}");
    }
    Ok(code)
}

pub(crate) fn rows(ctx: &Context, list: &MachineList) -> Vec<Vec<String>> {
    let live = list
        .machines
        .iter()
        .filter(|m| m.state != MachineState::Ended);
    live.map(|m| {
        let rental = m.rental.as_ref();
        vec![
            m.name.clone(),
            match m.status.as_ref().is_some_and(|s| s.dev) {
                true => format!("{} dev", kind(m)),
                false => kind(m).into(),
            },
            gpus_cell(m),
            family_cell(m.status.as_ref()),
            state_cell(m),
            rental.map_or("—".into(), |r| usd(r.usd_micros_per_hour)),
            rental.map_or("—".into(), spent),
            uptime(m),
            m.running.to_string(),
            m.queued.to_string(),
            ends_cell(m),
            if m.hub.is_empty() {
                "—".into()
            } else {
                hub_label(ctx, &m.hub)
            },
        ]
    })
    .collect()
}

/// One machine: a live view on a terminal while it lives, else its record once (or JSON).
pub async fn show(ctx: &Context, name: &str, once: bool) -> Result<ExitCode, Error> {
    let m = machine(ctx, name).await?;
    if !once && !ctx.json && ctx.term.stdout && m.state != MachineState::Ended {
        let view = super::live::Shown::open(ctx, name).await?;
        tui::show(ctx, tui::Screen::Machine(Box::new(view))).await?;
        return Ok(ExitCode::SUCCESS);
    }
    if ctx.json {
        output::json(&m);
        return Ok(ExitCode::SUCCESS);
    }
    record(ctx, &m).print(ctx);
    Ok(ExitCode::SUCCESS)
}

/// `cozy machine show`'s record of `m`, printed once or drawn live.
pub(crate) fn record(ctx: &Context, m: &Machine) -> output::Record {
    let mut record = output::Record::new()
        .field("machine", &m.name)
        .field("kind", kind(m))
        .field("state", state_cell(m));
    if let Some(alias) = m.alias.as_deref().filter(|a| *a != m.name) {
        record = record.field("alias", alias);
    }
    if !m.hub.is_empty() {
        record = record.field("hub", hub_label(ctx, &m.hub));
    }
    if let Some(id) = &m.rental_id {
        record = record.field("rental", id);
    }
    if let Some(address) = &m.address {
        record = record.field("address", address);
    }
    if let Some(fingerprint) = &m.fingerprint {
        record = record.field("fingerprint", fingerprint);
    }
    if let Some(r) = &m.rental {
        record = record
            .field("sku", &r.sku)
            .field("gpus", gpus_of_rental(r))
            .field("$/hour", usd(r.usd_micros_per_hour))
            .field("spent", spent(r))
            .field("disk", format!("{} GB", r.disk_gb))
            .field(
                "rented",
                format!("{} ago", clock(now_ms() - r.rented_at_ms)),
            );
        if let Some(ended) = r.ended_at_ms {
            record = record.field("ended", format!("{} ago", clock(now_ms() - ended)));
        }
        if let Some(failure) = &r.failure {
            record = record.field("failure", failure);
        }
        if let Some(note) = &r.placement_note {
            record = record.field("placement", note);
        }
        if let Some(warning) = &r.provider_warning {
            record = record.field("provider", warning);
        }
        if let Some(digest) = &r.base_worker_image_digest {
            record = record.field("image", digest);
        }
    }
    if let Some(since) = unreachable_since(m) {
        record = record.note(format!(
            "the Hub cannot reach it since {} ago, though its provider reports it running; \
             `cozy machine end {}` ends it",
            clock(now_ms() - since),
            m.name
        ));
    }
    if let Some(reason) = &m.reason {
        record = record.field("reason", reason);
    }
    if m.state != MachineState::Ended {
        record = record.field("running", m.running).field("queued", m.queued);
    }
    if let Some(why) = &m.unreachable {
        record = record.note(format!(
            "it did not answer just now ({why}); its facts are from its last report"
        ));
    }
    match &m.status {
        Some(status) => {
            for (key, value) in status_fields(status) {
                record = record.field(&key, value);
            }
            // The warm set: each member at its level, or what it holds instead and why.
            for item in &status.warm {
                let held = match item.holds.as_str() {
                    holds if holds.is_empty() || holds == item.level => item.level.clone(),
                    holds => format!("{}, holds {holds}: {}", item.level, item.held_back),
                };
                record = record.field(&format!("warm {}/{}", item.source, item.entrypoint), held);
            }
            record = record.field("ends", ends_cell(m));
        }
        None if m.state == MachineState::Ready => {
            record = record.note("the machine has not reported its status yet");
        }
        None => {}
    }
    if let Some(note) = provider_stopped(m) {
        record = record.note(note);
    }
    record
}

/// What a machine reports of itself, one fact per line, GPUs by number.
pub fn status_fields(status: &MachineStatus) -> Vec<(String, String)> {
    let mut fields = vec![
        (
            "version".into(),
            match status.dev {
                true => format!("{} (dev)", status.version),
                false => status.version.clone(),
            },
        ),
        ("platform".into(), status.platform.clone()),
        ("phase".into(), status.phase.clone()),
    ];
    let commit = [("commit", &status.commit)];
    for (key, value) in commit
        .into_iter()
        .chain([("runtime", &status.runtime), ("tensorfs", &status.tensorfs)])
    {
        if !value.is_empty() {
            fields.push((key.into(), value.clone()));
        }
    }
    if status.gpus.is_empty() {
        fields.push(("gpus".into(), "none: runs use the CPU".into()));
    }
    for gpu in &status.gpus {
        let mut cell = format!("{} · {} GiB", gpu.name, gpu.memory_bytes >> 30);
        for part in [&gpu.family, &gpu.memory_class].into_iter().flatten() {
            cell += &format!(" · {part}");
        }
        if !gpu.driver.is_empty() {
            cell += &format!(" · driver {}", gpu.driver);
        }
        fields.push((format!("GPU {}", gpu.index), cell));
    }
    // Beside each card's own size, so a limited card is not mistaken for a smaller one.
    let limited: Vec<_> = status.gpus.iter().filter(|g| g.memory_limit_bytes > 0).collect();
    if let Some(first) = limited.first() {
        let cards: Vec<String> = limited.iter().map(|g| format!("GPU {}'s {} GiB", g.index, g.memory_bytes >> 30)).collect();
        let limit = crate::output::bytes(first.memory_limit_bytes);
        fields.push(("gpu memory limit".into(), format!("{limit} of {}", cards.join(", "))));
    }
    if status.disk_total_bytes > 0 {
        fields.push((
            "disk free".into(),
            format!(
                "{:.1} of {:.1} GiB",
                status.disk_free_bytes as f64 / (1u64 << 30) as f64,
                status.disk_total_bytes as f64 / (1u64 << 30) as f64
            ),
        ));
    }
    fields.push(("live runs".into(), status.live_runs.to_string()));
    if !status.driver_floor.is_empty() {
        let old: Vec<String> = status
            .gpus
            .iter()
            .filter(|gpu| older(&gpu.driver, &status.driver_floor))
            .map(|gpu| {
                format!(
                    "GPU {}'s driver {} is older: update it",
                    gpu.index, gpu.driver
                )
            })
            .collect();
        let floor = match old.is_empty() {
            true => format!("{} (met)", status.driver_floor),
            false => format!("{}; {}", status.driver_floor, old.join("; ")),
        };
        fields.push(("driver floor".into(), floor));
    }
    if let Some(toolchain) = &status.toolchain {
        let cell = match toolchain.state.as_str() {
            "ready" => format!("ready: {} · {}", toolchain.nvcc, toolchain.host_compiler),
            "installing" => "installing".into(),
            state => {
                let mut cell = format!(
                    "{state}: missing {}; cards without a prebuilt kernel use the slower SDPA/kitchen path",
                    toolchain.missing.join(", ")
                );
                if !toolchain.detail.is_empty() {
                    cell += &format!(" ({})", toolchain.detail);
                }
                cell += "; cozy machine toolchain install";
                if !toolchain.install.is_empty() {
                    cell += &format!(", and for a host compiler: {}", toolchain.install);
                }
                cell
            }
        };
        fields.push(("toolchain".into(), cell));
    }
    for kernel in &status.kernels {
        let verdict = match kernel.ok {
            true => format!("verified: {}", kernel.detail),
            false => format!("refused: {}", kernel.detail),
        };
        fields.push((
            format!("kernel {} sm{}", kernel.kernel, kernel.sm),
            format!("{verdict} (driver {})", kernel.driver),
        ));
    }
    fields
}

/// Whether driver `version` is older than `floor`, by their dotted numbers; unreadable is not.
fn older(version: &str, floor: &str) -> bool {
    let parts =
        |v: &str| -> Option<Vec<u32>> { v.trim().split('.').map(|p| p.parse().ok()).collect() };
    matches!((parts(version), parts(floor)), (Some(v), Some(f)) if v < f)
}

fn kind(m: &Machine) -> &'static str {
    match (m.kind, m.recorded) {
        (MachineKind::Local, _) => "local",
        (MachineKind::Added, _) => "added",
        (MachineKind::Rental, true) => "rental",
        (MachineKind::Rental, false) => "rental (other client)",
    }
}

pub fn state_cell(m: &Machine) -> String {
    let mut cell = match m.state {
        _ if provider_stopped(m).is_some() => "stopped by provider".to_string(),
        MachineState::Booting => match &m.boot {
            Some(boot) => format!("booting · {boot}"),
            None => "booting".into(),
        },
        MachineState::Ready => match unreachable_since(m) {
            Some(since) => format!(
                "unreachable from the Hub since {} ago",
                clock(now_ms() - since)
            ),
            None => "ready".into(),
        },
        MachineState::Ending => "ending".into(),
        MachineState::Ended => "ended".into(),
    };
    if m.unverified {
        cell += " (unverified)";
    }
    match m.unreachable.as_deref() {
        Some(_) if m.state == MachineState::Ready && unreachable_since(m).is_none() => {
            cell += " (unreachable)"
        }
        _ => {}
    }
    cell
}

/// The GPUs as the machine reports them, else as the rental was bought.
fn gpus_cell(m: &Machine) -> String {
    match (&m.status, &m.rental) {
        (Some(status), _) if !status.gpus.is_empty() => {
            format!("{}× {}", status.gpus.len(), status.gpus[0].name)
        }
        (_, Some(rental)) => gpus_of_rental(rental),
        _ => "—".into(),
    }
}

fn gpus_of_rental(r: &Rental) -> String {
    if r.gpus == 0 || r.accelerator.eq_ignore_ascii_case("cpu") {
        "—".into()
    } else {
        format!("{}× {}", r.gpus, r.accelerator)
    }
}

fn family_cell(status: Option<&MachineStatus>) -> String {
    let gpu = status.and_then(|s| s.gpus.first());
    let parts: Vec<&str> = gpu
        .into_iter()
        .flat_map(|g| [g.family.as_deref(), g.memory_class.as_deref()])
        .flatten()
        .collect();
    if parts.is_empty() {
        "—".into()
    } else {
        parts.join(" · ")
    }
}

fn spent(r: &Rental) -> String {
    match (r.spend_usd_micros, r.spend_basis.as_deref()) {
        (Some(micros), Some("charged")) => usd(micros),
        (Some(micros), Some("unmetered")) => format!("{} unmetered", usd(micros)),
        (Some(micros), _) => format!("{} est.", usd(micros)),
        (None, _) => "—".into(),
    }
}

/// The rentals' accrued total, on its weakest basis; a rental whose Hub said nothing makes it a floor.
fn accrued(rentals: &[&Rental]) -> String {
    let known: Vec<&&Rental> = rentals
        .iter()
        .filter(|r| r.spend_usd_micros.is_some())
        .collect();
    if known.is_empty() {
        return String::new();
    }
    let total: i64 = known.iter().filter_map(|r| r.spend_usd_micros).sum();
    let basis = |b: &str| known.iter().any(|r| r.spend_basis.as_deref() == Some(b));
    let suffix = if known.len() < rentals.len() || basis("estimate") {
        " est."
    } else if basis("unmetered") {
        " unmetered"
    } else {
        ""
    };
    let floor = if known.len() < rentals.len() {
        "at least "
    } else {
        ""
    };
    format!(" · accrued {floor}{}{suffix}", usd(total))
}

fn uptime(m: &Machine) -> String {
    let since = m
        .rental
        .as_ref()
        .map(|r| r.rented_at_ms)
        .or(m.status.as_ref().map(|s| s.started_at_ms));
    since.map_or("—".into(), |at| clock(now_ms() - at))
}

/// When an idle machine ends itself: a rental with work names no deadline and ends after it.
fn ends_cell(m: &Machine) -> String {
    match (m.status.as_ref().map(|s| s.idle_deadline_ms), m.kind) {
        (Some(Some(at)), _) if at > now_ms() => format!("in {}", clock(at - now_ms())),
        (Some(Some(_)), _) => "now".into(),
        (Some(None), MachineKind::Rental) => "after its jobs".into(),
        _ => "—".into(),
    }
}

/// Since when the Hub cannot reach a live rental its provider reports running.
fn unreachable_since(m: &Machine) -> Option<i64> {
    let since = m.rental.as_ref()?.unreachable_since_ms?;
    (m.state == MachineState::Ready).then_some(since)
}

fn provider_stopped(m: &Machine) -> Option<String> {
    let r = m.rental.as_ref()?;
    (r.failure.as_deref() == Some(PROVIDER_STOPPED)).then(|| {
        format!(
            "{} was stopped by its provider. The Hub keeps it but never replaces it: it returns if the \
             provider resumes it, and ends after 15 minutes idle. To keep working: `cozy machine rent {}`.",
            m.name, r.sku
        )
    })
}

fn hub_label(ctx: &Context, origin: &str) -> String {
    let hubs = ctx.config.hubs();
    hubs.iter()
        .find(|hub| hub.origin == origin)
        .map_or(origin, |hub| hub.label())
        .to_string()
}
