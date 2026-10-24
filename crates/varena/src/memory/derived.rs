//! The derived-weights disk cache in Memory (varena-leases §6, M1/M4; [`crate::derived`]; S1's
//! frames, S5's store). A declared set's page names its derived entry in its weights ask; a
//! grant that brings one (or one this process holds) refills the page with no transform; a
//! transform that landed is offered to the write-behind writer when deriving it again costs more
//! than writing it and reading it back.

use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock, Weak};

use parking_lot::Mutex;
use sha2::{Digest, Sha256};

use super::{Inner, Memory, State};
use crate::backend::{Event, Va};
use crate::derived::{pays, Hold, Job, Key as Derived, Refill, Writer};
use crate::table::{Fill, Key};
use crate::Result;

/// Memory's side of the derived cache.
#[derive(Default)]
pub(super) struct Cache {
    writer: OnceLock<Arc<Writer>>,
    /// Refills granted, for the fill threads: by page and generation.
    refills: Mutex<HashMap<(Key, u64), Arc<Refill>>>,
    /// Entries this process wrote: it holds them (the holder set is the open fds).
    entries: Mutex<HashMap<Derived, File>>,
    /// The page each offered entry is written from.
    pages: Mutex<HashMap<Derived, Key>>,
    /// Transforms that landed, measured once both events complete: (page, the bind's event,
    /// the transform's).
    landed: Mutex<Vec<(Key, Event, Event)>>,
    /// The store's measured read bandwidth, bytes per second (M4's read side).
    read_bps: AtomicU64,
}

impl Memory {
    /// The writer, once leased: its hold is this Memory's pages.
    pub(super) fn start_derived(&self) -> Result<()> {
        let writer = Writer::new(self.inner.backend.clone(), Arc::new(Holder(Arc::downgrade(&*self.inner))))?;
        let _ = self.inner.derived.writer.set(Arc::new(writer));
        Ok(())
    }
}

impl Inner {
    /// A declared set's page: the key of its derived bytes.
    pub(super) fn derived_key(&self, st: &State, key: Key) -> Option<Derived> {
        let m = st.mech(key.0).ok()?;
        let Fill::Declared(identity) = &m.spec.fingerprint.fill else { return None };
        let digest = Sha256::digest(m.key.as_bytes());
        let fingerprint = digest.get(..8).and_then(|b| <[u8; 8]>::try_from(b).ok()).map(u64::from_le_bytes)?;
        Some(Derived::new(fingerprint, key.1, identity))
    }

    /// The fill key a page's weights ask names (S5's store key), when its set is declared.
    pub(super) fn fill_key(&self, key: Key) -> Option<String> {
        let st = self.state.lock();
        self.derived_key(&st, key).map(|k| k.store())
    }

    /// A page's grant (not adopted): its entry, the manager's or one this process holds, refills
    /// it on the fill threads.
    pub(super) fn granted(&self, key: Key, gen: u64, span: u64, entry: Option<File>) {
        let derived = {
            let st = self.state.lock();
            self.derived_key(&st, key)
        };
        let Some(dk) = derived else { return };
        let entry = entry.or_else(|| self.derived.entries.lock().get(&dk).and_then(|f| f.try_clone().ok()));
        let mut refills = self.derived.refills.lock();
        refills.retain(|(k, g), _| *k != key || *g > gen);
        if let Some(entry) = entry {
            refills.insert((key, gen), Arc::new(Refill::new(&dk, entry, span)));
        }
    }

    /// The fill threads' refill for a page, when its grant brought one.
    pub(super) fn derived_refill(&self, key: Key, gen: u64) -> Option<Arc<Refill>> {
        self.derived.refills.lock().remove(&(key, gen))
    }

    /// A batch read from disk: M4's read bandwidth.
    pub(super) fn read_measured(&self, bytes: u64, ns: u64) {
        if ns == 0 {
            return;
        }
        let now = (u128::from(bytes) * 1_000_000_000 / u128::from(ns)) as u64;
        let was = self.derived.read_bps.load(Relaxed);
        self.derived.read_bps.store(if was == 0 { now } else { (was * 3 + now) / 4 }, Relaxed);
    }

    /// Whether a transform on this page is measured for the writer: leased, with a writer.
    pub(super) fn measures(&self) -> bool {
        self.derived.writer.get().is_some()
    }

    /// A transform landed (`Pin::filled`): `from` (the bind's event, now the cache's to retire)
    /// to `to` (the page's ready event) is its time on the device.
    pub(super) fn transformed(&self, key: Key, from: Event, to: Event) {
        self.derived.landed.lock().push((key, from, to));
    }

    /// Transforms whose events completed: each is offered to the writer when deriving it again
    /// costs more than writing it and reading it back (M4; before the writer measured its own
    /// bandwidth, the store's read stands in). Under the state lock; nothing waits.
    pub(super) fn offer_derived(&self, st: &State, agent: &crate::lease::Agent) {
        let Some(writer) = self.derived.writer.get() else { return };
        let mut landed = self.derived.landed.lock();
        landed.retain(|&(key, from, to)| {
            let ns = match self.backend.elapsed(from, to) {
                Ok(None) => return true,
                Ok(Some(ns)) => Some(ns),
                Err(_) => None,
            };
            self.backend.retire(from);
            let (Some(ns), Some(dk), Ok((va, off, span))) = (ns, self.derived_key(st, key), st.place(key)) else { return false };
            let read = self.derived.read_bps.load(Relaxed) as f64;
            if !pays(ns, span, writer.write_bps().unwrap_or(read), read) || self.derived.entries.lock().contains_key(&dk) {
                return false;
            }
            self.derived.pages.lock().insert(dk, key);
            let (writer, va) = (writer.clone(), Va { base: va.base + off, len: span });
            let asked = agent.derive(dk.store(), span, Box::new(move |entry: Option<crate::lease::DerivedEntry>| {
                if let Some(e) = entry {
                    writer.offer(Job { key: dk, entry: e.file, entry_id: e.id, va, ready: Event::DONE });
                }
            }));
            if let Err(error) = asked {
                tracing::debug!("varena: a derived entry for {key:?}: {error}");
            }
            false
        });
    }

    /// No request runs: the writer may write.
    pub(super) fn derived_idle(&self) {
        if let Some(w) = self.derived.writer.get() {
            w.wake();
        }
    }
}

/// The writer's hold on this Memory's pages.
struct Holder(Weak<Inner>);

impl Hold for Holder {
    /// The page, filled with these derived bytes and quiet, pinned for its copy: a revoke
    /// declines it meanwhile, and never waits on the disk write after (M1).
    fn pin(&self, dk: &Derived) -> Option<Box<dyn Send>> {
        let inner = self.0.upgrade()?;
        let key = *inner.derived.pages.lock().get(dk)?;
        let mut st = inner.state.lock();
        let fit = st.homes.get(&key).is_some_and(|h| h.filled && h.phys.is_some() && !h.filling && !h.transforming) && inner.derived_key(&st, key) == Some(*dk);
        if !fit {
            return None;
        }
        let id = st.tick();
        st.pins.insert(id, (key, None, std::thread::current().id()));
        st.sync(key);
        Some(Box::new(Held { inner: Arc::downgrade(&inner), id, key }))
    }

    fn idle(&self) -> bool {
        self.0.upgrade().is_some_and(|i| i.state.lock().table.requests.is_empty())
    }

    /// Written whole: TensorD hears which entry (its id), and this process holds the entry for
    /// its own refills.
    fn written(&self, dk: Derived, entry_id: u64, entry: File) {
        let Some(inner) = self.0.upgrade() else { return };
        inner.derived.pages.lock().remove(&dk);
        if let Some(agent) = inner.lease.get() {
            if let Err(error) = agent.derived(dk.store(), entry_id) {
                tracing::debug!("varena: a derived entry's written ack: {error}");
            }
        }
        inner.derived.entries.lock().insert(dk, entry);
    }
}

/// The writer's pin on a page for its copy.
struct Held {
    inner: Weak<Inner>,
    id: u64,
    key: Key,
}

impl Drop for Held {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.upgrade() {
            let mut st = inner.state.lock();
            st.pins.remove(&self.id);
            st.sync(self.key);
            drop(st);
            inner.landed.notify_all();
        }
    }
}
