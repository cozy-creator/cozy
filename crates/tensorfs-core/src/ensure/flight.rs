//! One flight per manifest per Store, across processes. The first `ensure` leads and holds
//! `tmp/ensure/<manifest>.lock`; later ones wait on that lock, reading the leader's record
//! for progress, and then find the closure resident. The record is also the flight's disk
//! reservation: admission subtracts what live flights have yet to write.

use std::fs::{self, File, OpenOptions};
use std::io::ErrorKind;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::Duration;

use fs2::FileExt;

use crate::canon::{self, Fields, Value};
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::ObjectRef;
use crate::storage::{self, HeldKey};
use crate::store::Store;
use crate::transport::PullCancellation;

use super::admission::Space;

fn io(what: &str, error: std::io::Error) -> Refusal {
    crate::store::classify_io(format!("ensure {what}"), error)
}

fn dir(store: &Store) -> Result<PathBuf> {
    let dir = store.root().join("tmp/ensure");
    if !dir.is_dir() {
        fs::create_dir_all(&dir).map_err(|e| io("mkdir", e))?;
        // Every uid that writes this Store takes part in its flights (as tmp/leases).
        let _ = fs::set_permissions(&dir, fs::Permissions::from_mode(0o1777));
    }
    Ok(dir)
}

/// A lock file another uid created may be read-only to us; flock needs only a descriptor.
fn open_lock(path: &Path) -> Result<File> {
    match OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o666)
        .open(path)
    {
        Err(e) if e.kind() == ErrorKind::PermissionDenied => File::open(path),
        other => other,
    }
    .map_err(|e| io("open lock", e))
}

/// Whether `file`'s flock is held by someone else; takes it when it is free.
fn contended(file: &File) -> Result<bool> {
    match file.try_lock_exclusive() {
        Ok(()) => Ok(false),
        Err(e) if e.kind() == ErrorKind::WouldBlock => Ok(true),
        Err(e) => Err(io("lock", e)),
    }
}

/// What a leader says about its flight, rewritten at every sample.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Record {
    pub model: String,
    pub step: String,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub wanted_bytes: u64,
    pub wanted_objects: u64,
    pub landed_bytes: u64,
    pub landed_objects: u64,
}

impl Record {
    fn to_bytes(&self) -> Vec<u8> {
        canon::write(&Value::obj(vec![
            ("bytes_done", Value::uint(self.bytes_done)),
            ("bytes_total", Value::uint(self.bytes_total)),
            ("landed_bytes", Value::uint(self.landed_bytes)),
            ("landed_objects", Value::uint(self.landed_objects)),
            ("model", Value::str(self.model.as_str())),
            ("step", Value::str(self.step.as_str())),
            ("wanted_bytes", Value::uint(self.wanted_bytes)),
            ("wanted_objects", Value::uint(self.wanted_objects)),
        ]))
    }

    fn parse(bytes: &[u8]) -> Result<Record> {
        let value = canon::parse(bytes, 64 * 1024)?;
        let mut f = Fields::new("ensure record", &value)?;
        let record = Record {
            bytes_done: f.req_uint("bytes_done")?,
            bytes_total: f.req_uint("bytes_total")?,
            landed_bytes: f.req_uint("landed_bytes")?,
            landed_objects: f.req_uint("landed_objects")?,
            model: f.req_str("model")?.to_string(),
            step: f.req_str("step")?.to_string(),
            wanted_bytes: f.req_uint("wanted_bytes")?,
            wanted_objects: f.req_uint("wanted_objects")?,
        };
        f.rest();
        Ok(record)
    }

    fn read(path: &Path) -> Option<Record> {
        Record::parse(&fs::read(path).ok()?).ok()
    }

    /// What this flight has yet to write: its unlanded bytes and their inodes.
    pub fn remaining(&self) -> Space {
        let objects = self.wanted_objects.saturating_sub(self.landed_objects);
        Space {
            bytes: self.wanted_bytes.saturating_sub(self.landed_bytes),
            inodes: if objects == 0 {
                0
            } else {
                super::admission::METADATA_INODES + super::admission::OBJECT_INODES * objects
            },
        }
    }
}

/// The lead of one manifest's flight. Dropping it ends the flight.
pub(super) struct Flight {
    _lock: File,
    record: PathBuf,
    /// The closure this flight holds from GC (see [`held_objects`]).
    objects: PathBuf,
    pub manifest: String,
    /// Another process's flight for this manifest was running when this one asked.
    pub joined: bool,
}

impl Flight {
    /// Lead the flight for `manifest` (bare hex), waiting while another process leads it.
    /// `waiting` sees the leader's record once per `tick`; the tick is resolution only.
    pub fn lead(
        store: &Store,
        manifest: &str,
        cancellation: Option<&PullCancellation>,
        tick: Duration,
        waiting: &dyn Fn(Option<Record>),
    ) -> Result<Flight> {
        let dir = dir(store)?;
        let lock = open_lock(&dir.join(format!("{manifest}.lock")))?;
        let record = dir.join(format!("{manifest}.json"));
        let mut joined = false;
        while contended(&lock)? {
            joined = true;
            waiting(Record::read(&record));
            if cancellation.is_some_and(PullCancellation::is_cancelled) {
                return refuse(
                    Code::TRANSFER_FAILED,
                    "the pull was cancelled by its caller",
                );
            }
            std::thread::sleep(tick);
        }
        Ok(Flight {
            _lock: lock,
            objects: record.with_extension("objects"),
            record,
            manifest: manifest.to_string(),
            joined,
        })
    }

    /// Hold `objects` (the closure, manifest included) from GC for the life of the flight.
    pub fn hold(&self, objects: &[ObjectRef]) -> Result<()> {
        let lines: String = objects
            .iter()
            .map(|o| format!("{} {}\n", o.sha256, o.length))
            .collect();
        fs::write(&self.objects, lines).map_err(|e| io("write held objects", e))
    }

    pub fn record(&self, record: &Record) -> Result<()> {
        let temp = self
            .record
            .with_extension(format!("{}.tmp", crate::meta::now_nanos_unique()));
        fs::write(&temp, record.to_bytes())
            .and_then(|()| fs::rename(&temp, &self.record))
            .map_err(|e| {
                let _ = fs::remove_file(&temp);
                io("write record", e)
            })
    }
}

impl Drop for Flight {
    fn drop(&mut self) {
        // The record goes first: once the lock is free, nothing may read a stale one.
        let _ = fs::remove_file(&self.record);
        let _ = fs::remove_file(&self.objects);
    }
}

/// Serializes admission across processes: measure, subtract, record.
pub(super) fn gate(store: &Store) -> Result<File> {
    let gate = open_lock(&dir(store)?.join("admission.lock"))?;
    gate.lock_exclusive().map_err(|e| io("admission lock", e))?;
    Ok(gate)
}

/// Every live flight but `except`: its manifest and its record, when it has written one.
pub(super) fn live(store: &Store, except: &str) -> Result<Vec<(String, Option<Record>)>> {
    let dir = dir(store)?;
    let mut flights = Vec::new();
    for entry in fs::read_dir(&dir).map_err(|e| io("list", e))? {
        let path = entry.map_err(|e| io("list", e))?.path();
        let Some(manifest) = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".lock"))
        else {
            continue;
        };
        if manifest == except || crate::ids::hex64("flight", manifest).is_err() {
            continue;
        }
        // Taking a free lock proves its leader is gone; the kernel released it with them.
        if contended(&open_lock(&path)?)? {
            flights.push((
                manifest.to_string(),
                Record::read(&path.with_extension("json")),
            ));
        }
    }
    Ok(flights)
}

/// Every resident object a live flight's closure names, as GC holds. An ensure counts the
/// objects already resident as held when it plans, and lands the rest unreferenced until its
/// pull is recorded; a collection in between (its own admission's, or another process's)
/// must not take either, or the pull re-buys them past the space it was admitted for.
pub(crate) fn held_objects(store: &Store) -> Result<Vec<HeldKey>> {
    if !store.root().join("tmp/ensure").is_dir() {
        return Ok(Vec::new());
    }
    let dir = dir(store)?;
    let mut holds = Vec::new();
    for (manifest, _) in live(store, "")? {
        let Ok(lines) = fs::read_to_string(dir.join(format!("{manifest}.objects"))) else {
            continue;
        };
        for line in lines.lines() {
            let Some((hex, length)) = line.split_once(' ') else {
                continue;
            };
            let Ok(length) = length.parse() else {
                continue;
            };
            let (kind, key) = if store.manifest_path(hex).is_file() {
                ("manifest", storage::manifest_key(hex)?)
            } else if store.contains(hex) {
                ("blob", storage::blob_key(hex)?)
            } else {
                continue;
            };
            holds.push(HeldKey {
                key,
                kind: kind.into(),
                length,
                sha256: hex.to_string(),
            });
        }
    }
    Ok(holds)
}
