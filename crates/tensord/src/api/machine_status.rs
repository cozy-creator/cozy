//! Status (G/API.md, D2). Anyone who reaches the machine gets one identity frame with the sealed
//! readiness receipt; a machine-scope cap gets the whole picture, sent again whenever it changes.
//! An open stream is not activity: only `keepalive` resets the idle deadline, once.
use super::{auth::VerifiedActor, backend::MachineBackend, domain, http::blocking, v1, MachineIdentity};
use protocol::error::Error;
use std::{sync::Arc, time::Duration};
use tokio_stream::{wrappers::IntervalStream, Stream, StreamExt};

/// What this machine serves, for clients that adapt to it.
/// `run-gpus/1`: a run may request an exact GPU group, inherited and reserved by its calls.
/// `warm/1`: a warm run with no entrypoint installs its code and makes its model choices
/// present. `warm/2`: a warm run's `set` replaces the caller's warm set, and Status reports the
/// set and each environment's level. `upload/1`: a warm run of one provider source puts it in
/// its weights destination. `remove/1`: a remove run drops the caller's installations and
/// deletes their environments once nothing holds them; Status names each one's Hub.
/// `live-source/1`: a local source's manifest may name a live project file by file (or its
/// directory on this computer), installed editable in one tree and environment per package;
/// and a slot a run names no model for takes its declared default. `live-source/2`: that
/// manifest may also name the uv indexes its project's sources use without declaring them and the
/// local path dependencies its lock installs, and a locked project installs whatever indexes
/// they name.
pub const CAPABILITIES: &[&str] = &[
    "status/1",
    "run/1",
    "run-gpus/1",
    "control/1",
    "read/1",
    "warm/1",
    "warm/2",
    "upload/1",
    "local-models/1",
    "remove/1",
    "live-source/1",
    "live-source/2",
];

const LIVE: [&str; 3] = ["queued", "starting", "running"];

/// The whole picture, sent again whenever it changes, until the caller leaves.
pub(super) fn watch<B: MachineBackend>(
    identity: Arc<MachineIdentity>,
    backend: Arc<B>,
    actor: VerifiedActor,
) -> impl Stream<Item = Result<v1::StatusFrame, Error>> + Send {
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last: Option<v1::StatusFrame> = None;
    IntervalStream::new(tick)
        .then(move |_| {
            let (identity, backend) = (identity.clone(), backend.clone());
            async move { frame(&identity, &backend, actor).await }
        })
        .filter_map(move |next| match next {
            Ok(next) if last.as_ref().is_some_and(|last| !changed(last, &next)) => None,
            Ok(next) => {
                last = Some(next.clone());
                Some(Ok(next))
            }
            Err(error) => Some(Err(error)),
        })
}

/// Disk use moves all the time; it alone is news once it moved by a hundredth of the disk.
fn changed(last: &v1::StatusFrame, next: &v1::StatusFrame) -> bool {
    let moved = match (&last.disk, &next.disk) {
        (Some(a), Some(b)) => a.free_bytes.abs_diff(b.free_bytes) >= b.total_bytes / 100,
        (a, b) => a != b,
    };
    moved
        || v1::StatusFrame {
            disk: None,
            ..last.clone()
        } != v1::StatusFrame {
            disk: None,
            ..next.clone()
        }
}

pub(super) fn identity_frame(identity: &MachineIdentity) -> v1::StatusFrame {
    v1::StatusFrame {
        worker_id: identity.authority.worker_id.clone(),
        boot_id: identity.authority.boot_id.clone(),
        version: env!("CARGO_PKG_VERSION").into(),
        commit: protocol::COMMIT.into(),
        dev: identity.updates.as_ref().is_some_and(|u| u.dev()),
        capabilities: capabilities(identity),
        receipt: identity.readiness.envelope().unwrap_or_default(),
        ..Default::default()
    }
}

pub(super) fn capabilities(identity: &MachineIdentity) -> Vec<String> {
    let update = identity.updates.as_ref().map(|_| "update/1");
    CAPABILITIES
        .iter()
        .copied()
        .chain(update)
        .map(str::to_string)
        .collect()
}

pub(super) async fn frame<B: MachineBackend>(
    identity: &Arc<MachineIdentity>,
    backend: &Arc<B>,
    actor: VerifiedActor,
) -> Result<v1::StatusFrame, Error> {
    let store = identity.store.clone();
    let ((runs, environments), warm, models) = blocking(backend, move |backend| {
        let models = store.as_deref().map(crate::held_models::listing);
        let held = held(backend, actor)?;
        Ok((held, backend.warm_set(actor)?, models.unwrap_or_default()))
    })
    .await?;
    let software = identity
        .updates
        .as_ref()
        .map(|updates| updates.software())
        .unwrap_or_default();
    let lifecycle = identity.lifecycle.as_ref();
    Ok(v1::StatusFrame {
        idle_deadline_unix_ms: lifecycle.map_or(0, |l| l.deadline_ms()),
        runs,
        phase: match lifecycle {
            Some(l) if l.released() => "releasing",
            _ if !identity.readiness.proved() => "booting",
            _ => "ready",
        }
        .into(),
        platform: format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH),
        started_at_unix_ms: identity.started_at_ms as i64,
        gpus: identity
            .gpus
            .iter()
            .map(|g| v1::Gpu {
                index: g.device_index,
                name: g.device_name.clone(),
                uuid: g.device_uuid.clone(),
                pci_bus_id: g.pci_bus_id.clone(),
                memory_bytes: g.memory_bytes,
                driver: g.driver_version.clone(),
                family: match g.compute_cap.as_str() {
                    "" => String::new(),
                    cc => crate::precision::Gpus::of(&[cc]).name,
                },
                memory_class: g.memory_class.clone(),
                memory_limit_bytes: identity.gpu_memory_limit.filter(|&l| l < g.memory_bytes).unwrap_or(0),
            })
            .collect(),
        hubs: identity
            .hubs
            .iter()
            .map(|(origin, machine_id)| v1::Hub {
                origin: origin.clone(),
                machine_id: machine_id.clone(),
            })
            .collect(),
        runtime: software.runtime,
        tensorfs: software.tensorfs,
        environments,
        warm,
        disk: identity.store.as_deref().and_then(disk),
        models: models.models,
        models_bytes: models.bytes,
        webrtc: identity.webrtc_port.map(|port| {
            crate::machine::player::endpoint(port, &identity.player, &identity.cert_der)
        }),
        driver_floor: match identity.gpus.is_empty() {
            true => String::new(),
            false => platform::gpu::DRIVER_FLOOR.into(),
        },
        toolchain: identity.updates.as_ref().map(|u| u.toolchain()).unwrap_or_default(),
        kernels: identity.updates.as_ref().map(|u| u.kernels()).unwrap_or_default(),
        ..identity_frame(identity)
    })
}

fn held(
    backend: &impl MachineBackend,
    actor: VerifiedActor,
) -> Result<(Vec<v1::RunState>, Vec<v1::Environment>), Error> {
    let query = domain::MachineExecutionListQuery {
        limit: 256,
        states: LIVE.map(String::from).to_vec(),
        ..Default::default()
    };
    let runs = held_or_none(backend.list(actor, query).map(|l| l.executions))?
        .iter()
        .map(|run| super::machine_runs::state(&run.request_id, run))
        .collect();
    let packages = held_or_none(
        backend
            .list_packages(actor, domain::PackageListQuery::default())
            .map(|l| l.packages),
    )?;
    let levels = backend.levels(actor)?;
    let environments = packages
        .into_iter()
        .map(|p| v1::Environment {
            level: levels.get(&p.installation_id).copied().unwrap_or_default().into(),
            installation: p.installation_id,
            package: p.package,
            release: p.release,
            entrypoints: p.entrypoints,
            runtime: p
                .sdk
                .into_iter()
                .find(|d| d.distribution == "cozy-runtime")
                .map(|d| d.version)
                .unwrap_or_default(),
            warning: p.warning,
            hub: p.origin,
        })
        .collect();
    Ok((runs, environments))
}

/// A backend that keeps no such record holds none.
fn held_or_none<T>(listed: Result<Vec<T>, Error>) -> Result<Vec<T>, Error> {
    match listed {
        Err(error) if error.code == "capability_unavailable" => Ok(vec![]),
        other => other,
    }
}

fn disk(path: &std::path::Path) -> Option<v1::Disk> {
    let space = platform::durable::space(path).ok()?;
    Some(v1::Disk {
        total_bytes: space.total,
        free_bytes: space.free,
    })
}
