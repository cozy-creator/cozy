//! A checkpoint made whole here (`cozy model download`): what is absent of its closure,
//! fetched by the Store's [`Fetcher`] as any request's objects are, then proved and recorded
//! under its repository so the GC keeps it.

use std::sync::mpsc::{channel, RecvTimeoutError};
use std::sync::Arc;
use std::time::Duration;

use super::{Access, FetchPlan, Fetcher, Origin, Priority};
use crate::err::{refuse, Code, Result};
use crate::transport::{self, Closure, Deadline, Ledger, PullCancellation};

/// How often `progress` is told. It decides nothing.
const TELL: Duration = Duration::from_millis(100);

/// What `refspec` at `lane` names at `hub` now, asked of the hub once; the Fetcher holds
/// its closure for the origin answered.
pub fn resolve(
    fetcher: &Fetcher,
    hub: &str,
    refspec: &str,
    lane: &str,
    access: &dyn Access,
    stop: &PullCancellation,
) -> Result<(Origin, Arc<Closure>)> {
    let ledger = Ledger::with_resolution(fetcher.sample()).with_cancellation(stop.clone());
    let policy = transport::hub_policy(hub, access.policy());
    let closure = transport::closure_all(transport::shared_client(), hub, refspec, lane, access.credential(), &policy, Deadline::none(), &ledger)?;
    let origin = Origin { hub: hub.into(), refspec: refspec.into(), lane: lane.into(), manifest: closure.manifest.clone() };
    fetcher.seed(&origin, closure);
    let closure = fetcher.closure(&origin).expect("seeded");
    Ok((origin, closure))
}

/// `origin`'s checkpoint whole and recorded: what is absent of its closure at `at`. The
/// closure is the one the Fetcher holds for `origin` ([`Fetcher::seed`]), else asked of the
/// hub once. `progress` hears the bytes here and in all; the answer is the bytes that landed
/// from the hub and from the volume ([`super::Ticket::landed`]).
pub fn whole(
    fetcher: &Fetcher,
    origin: &Origin,
    access: Arc<dyn Access>,
    at: Priority,
    stop: &PullCancellation,
    progress: &(dyn Fn(u64, u64) + Sync),
) -> Result<(u64, u64)> {
    let store = fetcher.store();
    // A killed download strands its partial temps, and GC never runs while a model is held;
    // each temp carries its writer's lock, so this takes only the dead ones.
    let _ = store.reap();
    let closure = match fetcher.closure(origin) {
        Some(closure) => closure,
        None => {
            let (asked, closure) = resolve(fetcher, &origin.hub, &origin.refspec, &origin.lane, &*access, stop)?;
            if asked.manifest != origin.manifest {
                let (refspec, lane) = (&origin.refspec, &origin.lane);
                return refuse(Code::NOT_CONTAINED, format!("{refspec} lane {lane:?} now names {}, not {}", asked.manifest.id(), origin.manifest.id()));
            }
            closure
        }
    };
    let session = format!("whole-{}", &closure.manifest.sha256[..12]);
    let (plan, _) = FetchPlan::of(store, &session, &closure.manifest, &closure.objects)?;
    let total = plan.declared_bytes();
    if !plan.wanted.is_empty() {
        fetcher.admit_whole(origin, &plan.held, plan.wanted_bytes(), plan.wanted.len() as u64)?;
    }
    // What is here too: held from GC until the end, and its bytes in the progress.
    let every: Vec<_> = plan.held.iter().chain(&plan.wanted).cloned().collect();
    let ticket = fetcher.want(origin, access, &every, at);
    let (answer, answered) = channel();
    std::thread::scope(|scope| {
        let waiting = std::thread::Builder::new().spawn_scoped(scope, || answer.send(ticket.wait(&[], stop)));
        // No thread for the wait: it runs here, untold.
        let Ok(waiting) = waiting else {
            return ticket.wait(&[], stop);
        };
        let waited = loop {
            progress(ticket.progress().0.min(total), total);
            match answered.recv_timeout(TELL) {
                Ok(waited) => break waited,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break refuse(Code::IO_FAILED, "the wait ended without its answer"),
            }
        };
        let _ = waiting.join();
        waited
    })?;
    match closure.scope.as_str() {
        "runtime" => plan.complete_cozytensors(store),
        _ => plan.complete(store),
    }?;
    transport::record(store, &closure)?;
    crate::origin::retire(store, &closure.manifest)?;
    progress(total, total);
    Ok(ticket.landed())
}
