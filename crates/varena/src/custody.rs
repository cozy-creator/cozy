//! Kept pages. An executor exports each filled, read-only page into custody while it maps it,
//! and withdraws it when it unmaps it; custody holds the handles by (fingerprint key, page).
//! When an exporter's connection ends (it died, or TensorD reaped it), what it held becomes
//! orphans (class O), which the next process to load that fingerprint imports instead of
//! copying: one load path. Custody runs in whichever participant binds the GPU's service name
//! (TensorD when it exists), under a row of its own that publishes the orphans, so a
//! standalone engine runs the same code and no executor's row outlives it.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use platform::handles::Owned;
use serde::{Deserialize, Serialize};

use crate::ledger::{Row, Shared};
use crate::os::{self, Stream};
use crate::{Error, Result};

#[derive(Debug, Serialize, Deserialize)]
enum Call {
    Export { fp: String, page: u32, len: u64 },
    Withdraw { fp: String, page: u32 },
    Import { fp: String, page: u32 },
    /// An import this process did not use goes back to orphans.
    Disown { fp: String, page: u32 },
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

/// Binds the GPU's custody name and serves it from this process while it lives, under a row of
/// its own; false when another participant serves it.
fn start(shared: &Arc<Shared>) -> Result<bool> {
    let Some(listener) = os::listen(&name(&shared.name))? else { return Ok(false) };
    let row = Arc::new(shared.claim(std::process::id())?);
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
    // Only this user's processes: an abstract name has no permissions of its own.
    let Ok(pid) = os::peer(&stream) else { return };
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
            Call::Disown { fp, page } => {
                if let Some(k) = s.pages.get_mut(&(fp, page)).filter(|k| k.owner == Some(conn)) {
                    k.owner = None;
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

/// A participant's connection to custody. A broken connection reconnects (serving custody when
/// nobody does); a call that still fails means "no kept page", never a failed bind.
pub(crate) struct Client {
    shared: Arc<Shared>,
    stream: Mutex<Option<Stream>>,
}

impl Client {
    /// Connects to the GPU's custody, serving it from this process when nobody does.
    pub fn open(shared: &Arc<Shared>) -> Result<Client> {
        let client = Client { shared: shared.clone(), stream: Mutex::new(None) };
        *client.stream.lock() = Some(client.connect()?);
        Ok(client)
    }

    fn connect(&self) -> Result<Stream> {
        loop {
            if let Some(stream) = os::connect(&name(&self.shared.name))? {
                return Ok(stream);
            }
            // Nobody serves it: bind it; a racer that bound first is connected to next.
            start(&self.shared)?;
        }
    }

    /// One call; a broken connection is replaced once (the service's holder exited).
    fn call(&self, call: &Call, handle: Option<&Owned>) -> Result<(Reply, Option<Owned>)> {
        let mut stream = self.stream.lock();
        let mut last = Error::Closed("custody".into());
        for _ in 0..2 {
            let s = match stream.take() {
                Some(s) => s,
                None => self.connect()?,
            };
            let handle = handle.map(os::dup).transpose()?;
            match os::call(&s, call, handle) {
                Ok(reply) => {
                    *stream = Some(s);
                    return Ok(reply);
                }
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    pub fn export(&self, fp: &str, page: u32, len: u64, handle: Owned) -> Result<()> {
        self.call(&Call::Export { fp: fp.into(), page, len }, Some(&handle)).map(|_| ())
    }

    pub fn withdraw(&self, fp: &str, page: u32) -> Result<()> {
        self.call(&Call::Withdraw { fp: fp.into(), page }, None).map(|_| ())
    }

    pub fn disown(&self, fp: &str, page: u32) -> Result<()> {
        self.call(&Call::Disown { fp: fp.into(), page }, None).map(|_| ())
    }

    /// The orphan kept under (fingerprint key, page), now this process's; `None` when custody
    /// has none (or cannot be reached).
    pub fn import(&self, fp: &str, page: u32) -> Option<Owned> {
        match self.call(&Call::Import { fp: fp.into(), page }, None) {
            Ok((Reply::Imported { found: true }, Some(h))) => Some(h),
            _ => None,
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
    pub fn release(&self, bytes: u64) -> u64 {
        match self.call(&Call::Release { bytes }, None) {
            Ok((Reply::Released { freed }, _)) => freed,
            _ => 0,
        }
    }
}
