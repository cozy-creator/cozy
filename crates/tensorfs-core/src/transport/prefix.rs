//! One durable contiguous source prefix. No parallel range graph and no saved hash internals.
//! This is called only while the selected-source writer holds its native owner fence.
use super::{
    pull::{self, FetchObserver, Fetched, Opened},
    CredentialProvider, Deadline, Ledger, SourcePolicy, FETCH_ATTEMPTS,
};
use crate::{
    canon::{Fields, Value},
    err::{refuse, Code, Refusal, Result},
    fetch::{Admitted, DeliveryGrant, FetchPlan},
    ids::{prefixed, ObjectRef, Plain},
    sha256::{hex, Sha256},
    store::{Fault, Store, Verified},
};
use fs2::FileExt;
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
};

#[derive(Clone, Debug)]
pub struct SourceDownload {
    pub checkpoint_bytes: u64,
    pub fault: Fault,
}
impl Default for SourceDownload {
    fn default() -> Self {
        Self {
            checkpoint_bytes: super::PART_BYTES,
            fault: Fault::default(),
        }
    }
}
#[derive(Clone)]
struct State {
    owner: String,
    object: ObjectRef,
    extent: u64,
    prefix_sha256: String,
}
impl Plain for State {
    const MAX_BYTES: usize = 4096;
    fn from_value(value: &Value) -> Result<Self> {
        let mut f = Fields::new("SourcePrefix", value)?;
        let owner = prefixed("source prefix owner", f.req_str("owner")?)?;
        let object = ObjectRef::from_value("source prefix object", f.req("object")?)?;
        let extent = f.req_uint("extent")?;
        let prefix_sha256 = crate::ids::hex64("source prefix digest", f.req_str("prefix_sha256")?)?;
        if f.req_str("representation")? != "identity" {
            return refuse(Code::WRONG_TYPE, "source prefix has another representation");
        }
        f.done()?;
        if extent > object.length {
            return refuse(Code::LENGTH_MISMATCH, "prefix exceeds object");
        }
        Ok(Self {
            owner,
            object,
            extent,
            prefix_sha256,
        })
    }
    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("owner", Value::str(&self.owner)),
            ("object", self.object.to_value()),
            ("extent", Value::uint(self.extent)),
            ("prefix_sha256", Value::str(&self.prefix_sha256)),
            ("representation", Value::str("identity")),
        ])
    }
}
fn io(e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("source prefix: {e}"),
    }
}
fn area(store: &Store) -> PathBuf {
    store.root().join("roots/source-downloads")
}
fn owner_dir(store: &Store, owner: &str) -> Result<PathBuf> {
    prefixed("prefix owner", owner)?;
    Ok(area(store).join(&owner[7..]))
}
struct Prefix {
    dir: PathBuf,
    file: File,
    state: State,
    hash: Sha256,
    _lock: File,
}
impl Prefix {
    fn open(store: &Store, owner: &str, object: &ObjectRef) -> Result<Self> {
        Self::open_mode(store, owner, object, true)
    }
    fn open_mode(store: &Store, owner: &str, object: &ObjectRef, writable: bool) -> Result<Self> {
        crate::ids::hex64("prefix object", &object.sha256)?;
        let dir = owner_dir(store, owner)?.join(&object.sha256);
        fs::create_dir_all(&dir).map_err(io)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join("lock"))
            .map_err(io)?;
        lock.try_lock_exclusive().map_err(|_| Refusal {
            code: Code::STORE_BUSY,
            detail: "source prefix has a live writer".into(),
        })?;
        let mut state = State {
            owner: owner.into(),
            object: object.clone(),
            extent: 0,
            prefix_sha256: hex(&Sha256::new().finish()),
        };
        match File::open(dir.join("state.json")) {
            Ok(file) => {
                let mut bytes = vec![];
                file.take(State::MAX_BYTES as u64 + 1)
                    .read_to_end(&mut bytes)
                    .map_err(io)?;
                state = State::parse(&bytes)?;
                if state.owner != owner || state.object != *object {
                    return refuse(
                        Code::TRANSACTION_CONFLICT,
                        "source prefix has another immutable identity",
                    );
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(io(e)),
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(writable)
            .create(writable)
            .truncate(false)
            .open(dir.join("bytes.part"))
            .map_err(io)?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0u8; 1 << 20];
        let mut left = state.extent;
        while left > 0 {
            let n = file
                .read(&mut buffer[..left.min(1 << 20) as usize])
                .map_err(io)?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
            left -= n as u64;
        }
        let corrupt = left != 0 || hex(&hash.clone().finish()) != state.prefix_sha256;
        if corrupt && !writable {
            return refuse(
                Code::OBJECT_CORRUPT,
                "source donor prefix failed integrity validation",
            );
        }
        if corrupt {
            // Untrusted local bytes are never an input to a resumed completed object.
            state.extent = 0;
            hash = Sha256::new();
            state.prefix_sha256 = hex(&hash.clone().finish());
        }
        if writable {
            file.set_len(state.extent).map_err(io)?;
        }
        file.seek(SeekFrom::Start(state.extent)).map_err(io)?;
        let mut prefix = Self {
            dir,
            file,
            state,
            hash,
            _lock: lock,
        };
        if writable {
            prefix.checkpoint(&Fault::default())?;
        }
        if corrupt {
            return refuse(Code::OBJECT_CORRUPT, "retained source prefix failed integrity validation and was discarded; retry starts this object at zero");
        }
        Ok(prefix)
    }
    fn checkpoint(&mut self, fault: &Fault) -> Result<()> {
        fault.hit("prefix-written");
        self.file.sync_all().map_err(io)?;
        fault.hit("prefix-fsync");
        self.state.prefix_sha256 = hex(&self.hash.clone().finish());
        let temp = self.dir.join("state.tmp");
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp)
            .map_err(io)?;
        file.write_all(&self.state.canonical_bytes())
            .and_then(|()| file.sync_all())
            .map_err(io)?;
        fs::rename(temp, self.dir.join("state.json")).map_err(io)?;
        // Sync all newly-created ancestors before acknowledging retained progress.
        for dir in [
            Some(self.dir.as_path()),
            self.dir.parent(),
            self.dir.parent().and_then(|p| p.parent()),
            self.dir
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent()),
        ]
        .into_iter()
        .flatten()
        {
            File::open(dir).and_then(|f| f.sync_all()).map_err(io)?;
        }
        fault.hit("prefix-checkpoint");
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.state.extent = 0;
        self.hash = Sha256::new();
        self.file.set_len(0).map_err(io)?;
        self.file.seek(SeekFrom::Start(0)).map_err(io)?;
        self.checkpoint(&Fault::default())
    }
}
fn range(value: Option<&str>) -> Option<(u64, u64, u64)> {
    let (span, total) = value?.strip_prefix("bytes ")?.split_once('/')?;
    let (first, last) = span.split_once('-')?;
    Some((first.parse().ok()?, last.parse().ok()?, total.parse().ok()?))
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn fetch_source(
    store: &Store,
    owner: &str,
    grant: &DeliveryGrant,
    url: &str,
    policy: &SourcePolicy,
    credentials: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    options: &SourceDownload,
    mut observer: Option<&mut dyn FetchObserver>,
) -> Result<Fetched> {
    let mut prefix = Prefix::open(store, owner, &grant.object)?;
    store.admits(grant.object.length - prefix.state.extent)?;
    let mut transferred = 0;
    let checkpoint = options.checkpoint_bytes.clamp(1 << 20, 1 << 30);
    let mut failures = 0;
    while prefix.state.extent < grant.object.length {
        let start = prefix.state.extent;
        let end = (start + checkpoint).min(grant.object.length) - 1;
        let opened = pull::open_object(
            pull::shared_client(),
            url,
            policy,
            credentials,
            deadline,
            ledger,
            &[("range".into(), format!("bytes={start}-{end}"))],
            &mut |checked, status| {
                if let Some(w) = observer.as_mut() {
                    w.hop(failures + 1, &checked.host, status);
                }
            },
        );
        let response = match opened {
            Ok(Opened::Body(response)) => response,
            Ok(Opened::Retryable(_)) | Ok(Opened::Expired(_)) => {
                failures += 1;
                if failures < FETCH_ATTEMPTS {
                    continue;
                }
                return refuse(
                    Code::TRANSFER_FAILED,
                    "source refused every range attempt; refresh its delivery authority and retry",
                );
            }
            Ok(Opened::Refused(e)) => return Err(e),
            Err(e) => {
                failures += 1;
                if e.code == Code::TRANSFER_FAILED && failures < FETCH_ATTEMPTS {
                    continue;
                }
                return Err(e);
            }
        };
        // The common HTTP opener enforces identity representation before exposing a body.
        let limit = if response.status == 206 {
            if range(response.header("content-range")) != Some((start, end, grant.object.length)) {
                return refuse(
                    Code::LENGTH_MISMATCH,
                    "source range differs from requested immutable extent",
                );
            }
            end + 1
        } else if response.status == 200 {
            prefix.reset()?;
            grant.object.length
        } else {
            return refuse(Code::TRANSFER_FAILED, "source did not return object bytes");
        };
        let mut body = response.into_body();
        let mut buffer = vec![0u8; 1 << 20];
        let mut last_checkpoint = prefix.state.extent;
        loop {
            let n = match body.read(&mut buffer) {
                Ok(n) => n,
                Err(error) => {
                    prefix.checkpoint(&Fault::default())?;
                    failures += 1;
                    if failures >= FETCH_ATTEMPTS {
                        return Err(body.failure().unwrap_or_else(|| io(error)));
                    }
                    break;
                }
            };
            if n == 0 {
                if prefix.state.extent < limit {
                    failures += 1;
                    if failures >= FETCH_ATTEMPTS {
                        return refuse(
                            Code::TRANSFER_FAILED,
                            "source ended before the requested extent",
                        );
                    }
                }
                break;
            }
            transferred += n as u64;
            if prefix.state.extent + n as u64 > limit {
                return refuse(Code::LENGTH_MISMATCH, "source body exceeds declared range");
            }
            prefix.file.write_all(&buffer[..n]).map_err(io)?;
            prefix.hash.update(&buffer[..n]);
            prefix.state.extent += n as u64;
            store.moved(n as u64);
            if let Some(w) = observer.as_mut() {
                w.moved(transferred);
            }
            if prefix.state.extent > checkpoint {
                options.fault.hit("prefix-next-written");
            }
            if prefix.state.extent - last_checkpoint >= checkpoint {
                prefix.checkpoint(&options.fault)?;
                last_checkpoint = prefix.state.extent;
            }
        }
        prefix.checkpoint(&options.fault)?;
        if prefix.state.extent > start {
            failures = 0;
        }
    }
    if hex(&prefix.hash.clone().finish()) != grant.object.sha256 {
        prefix.reset()?;
        return refuse(
            Code::OBJECT_ID_MISMATCH,
            "complete source bytes do not match the pinned SHA-256",
        );
    }
    options.fault.hit("prefix-verified");
    let object = grant.object.clone();
    let Prefix { dir, file, .. } = prefix;
    let result = store.admit_verified(
        Verified {
            tmp: dir.join("bytes.part"),
            file,
            sha256: object.sha256.clone(),
            length: object.length,
        },
        &options.fault,
        None,
    )?;
    options.fault.hit("prefix-admitted");
    // Keep the lock inode stable. An old process must never lock an unlinked predecessor.
    let _ = fs::remove_file(dir.join("state.json"));
    Ok(Fetched {
        admitted: Admitted {
            object,
            first_writer: result.admitted,
        },
        transferred,
    })
}

pub(crate) fn discard_owner(store: &Store, owner: &str) -> Result<()> {
    let dir = owner_dir(store, owner)?;
    match fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(io(e)),
    }
}
pub(crate) fn bytes(store: &Store) -> Result<u64> {
    let owners = match fs::read_dir(area(store)) {
        Ok(rows) => rows,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(io(e)),
    };
    let mut total = 0u64;
    for owner in owners {
        for object in fs::read_dir(owner.map_err(io)?.path()).map_err(io)? {
            let path = object.map_err(io)?.path().join("bytes.part");
            match fs::symlink_metadata(path) {
                Ok(md) if md.is_file() => {
                    total = total.checked_add(md.len()).ok_or_else(|| Refusal {
                        code: Code::ARITH_OVERFLOW,
                        detail: "prefix bytes overflow".into(),
                    })?
                }
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(io(e)),
            }
        }
    }
    Ok(total)
}
/// Copy an independently owned verified prefix while both source and recipient are fenced.
pub(crate) fn adopt_owner(
    store: &Store,
    source: &str,
    owner: &str,
    mut landed: impl FnMut(&ObjectRef) -> Result<()>,
) -> Result<()> {
    let entries = match fs::read_dir(owner_dir(store, source)?) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io(e)),
    };
    for entry in entries {
        let dir = entry.map_err(io)?.path();
        let file = match File::open(dir.join("state.json")) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(io(e)),
        };
        let mut bytes = vec![];
        file.take(State::MAX_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(io)?;
        let state = State::parse(&bytes)?;
        if state.owner != source
            || dir.file_name().and_then(|name| name.to_str()) != Some(&state.object.sha256)
        {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "source prefix has another immutable identity",
            );
        }
        // Admission may have linked a read-only CAS inode or removed bytes.part
        // before the source writer recorded this member. Reuse the native verified
        // object and record recipient custody before considering any prefix copy.
        let (plan, _) = FetchPlan::of_objects(
            store,
            "source-prefix-adoption",
            std::slice::from_ref(&state.object),
        )?;
        if !plan.held.is_empty() {
            landed(&state.object)?;
            continue;
        }
        let mut from = Prefix::open_mode(store, source, &state.object, false)?;
        from.file.seek(SeekFrom::Start(0)).map_err(io)?;
        if from.state.extent == from.state.object.length {
            // A stopped chmod/link admission may leave only a read-only prefix.
            // Copy through normal digest-checked admission; never chmod, truncate,
            // consume or unlink the donor inode before recipient custody is durable.
            store.put_stream(
                &mut (&mut from.file).take(from.state.extent),
                Some(&from.state.object),
                &Fault::default(),
            )?;
            landed(&from.state.object)?;
            continue;
        }
        let mut to = Prefix::open(store, owner, &state.object)?;
        if to.state.extent >= from.state.extent {
            continue;
        }
        store.admits(from.state.extent - to.state.extent)?;
        to.reset()?;
        from.file.seek(SeekFrom::Start(0)).map_err(io)?;
        let mut buffer = vec![0u8; 1 << 20];
        let mut left = from.state.extent;
        while left > 0 {
            let n = from
                .file
                .read(&mut buffer[..left.min(1 << 20) as usize])
                .map_err(io)?;
            if n == 0 {
                return refuse(Code::LENGTH_MISMATCH, "adopted prefix was truncated");
            }
            to.file.write_all(&buffer[..n]).map_err(io)?;
            to.hash.update(&buffer[..n]);
            to.state.extent += n as u64;
            left -= n as u64;
        }
        to.checkpoint(&Fault::default())?;
    }
    Ok(())
}
