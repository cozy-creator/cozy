//! Kept pages. An executor exports each filled, read-only page into custody while it maps it,
//! and withdraws it when it unmaps it; custody holds the handles by (fingerprint, page). When an
//! exporter's connection ends (it died, or TensorD reaped it), what it held becomes orphans
//! (class O), which the next process to load that fingerprint imports instead of copying: one
//! load path. Custody runs in whichever participant binds the GPU's service name (TensorD when
//! it exists), so a standalone engine runs the same code.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use platform::handles::Owned;
use serde::{Deserialize, Serialize};

use crate::ledger::Row;
use crate::os::{self, Stream};
use crate::{Error, Result};

#[derive(Debug, Serialize, Deserialize)]
enum Call {
    Export { fp: String, page: u32, len: u64 },
    Withdraw { fp: String, page: u32 },
    Import { fp: String, page: u32 },
    Release { bytes: u64 },
    Reap { pid: u32 },
    Fault,
}

#[derive(Debug, Serialize, Deserialize)]
enum Reply {
    Done,
    Imported { found: bool },
    Released { freed: u64 },
}

struct Kept {
    handle: Owned,
    len: u64,
    /// The connection whose process maps it; `None`: an orphan.
    owner: Option<u64>,
    /// When it last changed hands (a logical clock): orphans go least recently used first.
    used: u64,
}

#[derive(Default)]
struct State {
    pages: HashMap<(String, u32), Kept>,
    /// Connection → its process.
    conns: HashMap<u64, u32>,
    clock: u64,
}

impl State {
    fn orphans(&self) -> u64 {
        self.pages.values().filter(|k| k.owner.is_none()).map(|k| k.len).sum()
    }

    fn orphan(&mut self, conn: u64) {
        self.pages.values_mut().filter(|k| k.owner == Some(conn)).for_each(|k| k.owner = None);
    }
}

fn name(gpu: &str) -> String {
    format!("custody-{gpu}")
}

/// Binds the GPU's custody name and serves it from this process while it lives; false when
/// another participant does. Its orphan bytes are published in `row`.
fn start(gpu: &str, row: Arc<Row>) -> Result<bool> {
    let Some(listener) = os::listen(&name(gpu))? else { return Ok(false) };
    let state = Arc::new(Mutex::new(State::default()));
    std::thread::Builder::new()
        .name("varena-custody".into())
        .spawn(move || {
            for stream in os::incoming(listener) {
                let (s, r) = (state.clone(), row.clone());
                let _ = std::thread::Builder::new().name("varena-custody-conn".into()).spawn(move || serve(stream, &s, &r));
            }
        })
        .map_err(|e| Error::io("spawn custody", e))?;
    Ok(true)
}

fn serve(stream: Stream, state: &Mutex<State>, row: &Row) {
    let pid = os::peer(&stream).unwrap_or(0);
    let conn = {
        let mut s = state.lock();
        s.clock += 1;
        let conn = s.clock;
        s.conns.insert(conn, pid);
        conn
    };
    while let Ok(Some((bytes, handle))) = os::recv(&stream) {
        let Ok(call) = os::decode::<Call>(&bytes) else { break };
        let mut s = state.lock();
        s.clock += 1;
        let used = s.clock;
        let (reply, out) = match call {
            Call::Export { fp, page, len } => {
                if let Some(handle) = handle {
                    s.pages.insert((fp, page), Kept { handle, len, owner: Some(conn), used });
                }
                (Reply::Done, None)
            }
            Call::Withdraw { fp, page } => {
                if s.pages.get(&(fp.clone(), page)).is_some_and(|k| k.owner == Some(conn)) {
                    s.pages.remove(&(fp, page));
                }
                (Reply::Done, None)
            }
            Call::Import { fp, page } => match s.pages.get_mut(&(fp, page)).filter(|k| k.owner.is_none()) {
                Some(k) => match os::dup(&k.handle) {
                    Ok(h) => {
                        k.owner = Some(conn);
                        k.used = used;
                        (Reply::Imported { found: true }, Some(h))
                    }
                    Err(_) => (Reply::Imported { found: false }, None),
                },
                None => (Reply::Imported { found: false }, None),
            },
            Call::Release { bytes } => {
                let mut orphans: Vec<((String, u32), u64, u64)> = s.pages.iter().filter(|(_, k)| k.owner.is_none()).map(|(key, k)| (key.clone(), k.used, k.len)).collect();
                orphans.sort_by_key(|(key, used, _)| (*used, key.clone()));
                let mut freed = 0;
                for (key, _, len) in orphans {
                    if freed >= bytes {
                        break;
                    }
                    s.pages.remove(&key);
                    freed += len;
                }
                (Reply::Released { freed }, None)
            }
            Call::Reap { pid } => {
                let conns: Vec<u64> = s.conns.iter().filter(|(_, p)| **p == pid).map(|(c, _)| *c).collect();
                conns.into_iter().for_each(|c| s.orphan(c));
                (Reply::Done, None)
            }
            Call::Fault => {
                s.pages.clear();
                (Reply::Done, None)
            }
        };
        row.set_orphans(s.orphans());
        drop(s);
        let sent = os::encode(&reply).and_then(|b| os::send(&stream, &(b, out)));
        if sent.is_err() {
            break;
        }
    }
    let mut s = state.lock();
    s.orphan(conn);
    s.conns.remove(&conn);
    row.set_orphans(s.orphans());
}

/// A participant's connection to custody.
pub(crate) struct Client {
    stream: Mutex<Stream>,
}

impl Client {
    /// Connects to the GPU's custody, serving it from this process when nobody does.
    pub fn open(gpu: &str, row: &Arc<Row>) -> Result<Client> {
        loop {
            if let Some(stream) = os::connect(&name(gpu))? {
                return Ok(Client { stream: Mutex::new(stream) });
            }
            // Nobody serves it: bind it; a racer that bound first is connected to next.
            start(gpu, row.clone())?;
        }
    }

    fn call(&self, call: &Call, handle: Option<Owned>) -> Result<(Reply, Option<Owned>)> {
        os::call(&self.stream.lock(), call, handle)
    }

    pub fn export(&self, fp: &str, page: u32, len: u64, handle: Owned) -> Result<()> {
        self.call(&Call::Export { fp: fp.into(), page, len }, Some(handle)).map(|_| ())
    }

    pub fn withdraw(&self, fp: &str, page: u32) -> Result<()> {
        self.call(&Call::Withdraw { fp: fp.into(), page }, None).map(|_| ())
    }

    /// The orphan kept under (fingerprint, page), now this process's.
    pub fn import(&self, fp: &str, page: u32) -> Result<Option<Owned>> {
        match self.call(&Call::Import { fp: fp.into(), page }, None)? {
            (Reply::Imported { found: true }, Some(h)) => Ok(Some(h)),
            _ => Ok(None),
        }
    }

    /// `pid` is gone: what it held becomes orphans (TensorD, after `waitpid`).
    pub fn reap(&self, pid: u32) -> Result<()> {
        self.call(&Call::Reap { pid }, None).map(|_| ())
    }

    /// A GPU fault or sticky error: every kept page goes.
    pub fn fault(&self) -> Result<()> {
        self.call(&Call::Fault, None).map(|_| ())
    }

    /// Drops orphans, least recently used first, until `bytes`; returns what it freed.
    pub fn release(&self, bytes: u64) -> Result<u64> {
        match self.call(&Call::Release { bytes }, None)? {
            (Reply::Released { freed }, _) => Ok(freed),
            _ => Ok(0),
        }
    }
}
