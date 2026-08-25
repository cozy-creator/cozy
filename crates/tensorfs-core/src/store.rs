//! The object store (tfs-002): shape-agnostic mixed CAS at `objects/sha256/xx/yy/<hex>`.
//!
//! Tensor parts, blob segments and documents admit through ONE put path; the store knows
//! no shapes (tensorfs.md §1). Identity is `sha256(stored_bytes)`, so an identity-bearing
//! document admits only as its exact canonical bytes — a pretty twin lands at a different
//! id and refuses `put_expected`.
//!
//! Admission is `writing → verified-temp → installed`: temp in `tmp/`, hash while writing,
//! fsync, 0444, then a no-clobber `link()` into place plus a parent-directory fsync. A
//! crash at any stage leaves a temp, never a partial object under `objects/`.
//!
//! `contains` is a PRESENCE HINT and nothing more. Trusting bytes requires a durable
//! verification record bound to object id, length, file identity (dev, ino), size and
//! change token (mtime), and the store generation. Any mismatch invalidates the record and
//! forces a rehash; a rehash that disagrees quarantines.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::canon::{Fields, Value};
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{check_format, hex64, Doc, ObjectRef};
use crate::sha256::{self, Sha256};

pub const ERA: &str = "v2";
const BUF: usize = 1 << 20;

#[cfg(target_os = "linux")]
const O_NOFOLLOW: i32 = 0o400_000;
#[cfg(target_os = "macos")]
const O_NOFOLLOW: i32 = 0x0100;

fn io(what: impl AsRef<str>, e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {e}", what.as_ref()),
    }
}

// ---------------------------------------------------------------- era marker

/// `<root>/store.json` — the era marker (tensorfs.md §10b). Opening an unmarked or
/// v1-era root as v2 refuses; the generation binds every verification record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Marker {
    pub era: String,
    pub generation: u64,
    pub object_hash: String,
}

impl Doc for Marker {
    const FORMAT: &'static str = "tensorfs.store/1";
    const MAX_BYTES: usize = 4096;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("Marker", v)?;
        let era = f.req_str("era")?.to_string();
        check_format("Marker", f.req_str("format")?, Self::FORMAT)?;
        let generation = f.req_uint("generation")?;
        let object_hash = f.req_str("object_hash")?.to_string();
        f.done()?;
        if object_hash != "sha256" {
            return refuse(
                Code::UNKNOWN_FORMAT,
                format!("store object_hash {object_hash:?} is not sha256"),
            );
        }
        Ok(Marker {
            era,
            generation,
            object_hash,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("era", Value::str(self.era.clone())),
            ("format", Value::str(Self::FORMAT)),
            ("generation", Value::uint(self.generation)),
            ("object_hash", Value::str(self.object_hash.clone())),
        ])
    }
}

// ---------------------------------------------------- durable verification record

/// `<root>/verified/xx/yy/<hex>` — the durable fact that these exact bytes were hashed
/// once, on admission. Bound to everything that could change underneath it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VRecord {
    pub sha256: String,
    pub length: u64,
    pub dev: u64,
    pub ino: u64,
    pub mtime_s: u64,
    pub mtime_ns: u64,
    pub generation: u64,
}

impl Doc for VRecord {
    const FORMAT: &'static str = "tensorfs.verification/1";
    const MAX_BYTES: usize = 4096;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("VRecord", v)?;
        let dev = f.req_uint("dev")?;
        check_format("VRecord", f.req_str("format")?, Self::FORMAT)?;
        let generation = f.req_uint("generation")?;
        let ino = f.req_uint("ino")?;
        let length = f.req_uint("length")?;
        let mtime_ns = f.req_uint("mtime_ns")?;
        let mtime_s = f.req_uint("mtime_s")?;
        let sha256 = hex64("VRecord", f.req_str("sha256")?)?;
        f.done()?;
        Ok(VRecord {
            sha256,
            length,
            dev,
            ino,
            mtime_s,
            mtime_ns,
            generation,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("dev", Value::uint(self.dev)),
            ("format", Value::str(Self::FORMAT)),
            ("generation", Value::uint(self.generation)),
            ("ino", Value::uint(self.ino)),
            ("length", Value::uint(self.length)),
            ("mtime_ns", Value::uint(self.mtime_ns)),
            ("mtime_s", Value::uint(self.mtime_s)),
            ("sha256", Value::str(self.sha256.clone())),
        ])
    }
}

impl VRecord {
    fn of(hex: &str, md: &fs::Metadata, generation: u64) -> VRecord {
        VRecord {
            sha256: hex.to_string(),
            length: md.len(),
            dev: md.dev(),
            ino: md.ino(),
            mtime_s: md.mtime().max(0) as u64,
            mtime_ns: md.mtime_nsec().max(0) as u64,
            generation,
        }
    }
    /// The first disagreeing bound fact, or None when the record still stands.
    fn disagreement(&self, hex: &str, md: &fs::Metadata, generation: u64) -> Option<String> {
        let now = VRecord::of(hex, md, generation);
        if self.sha256 != now.sha256 {
            return Some(format!("object id {} != {}", self.sha256, now.sha256));
        }
        if self.generation != now.generation {
            return Some(format!(
                "store generation {} != {}",
                self.generation, now.generation
            ));
        }
        if self.dev != now.dev || self.ino != now.ino {
            return Some(format!(
                "file identity {}:{} != {}:{}",
                self.dev, self.ino, now.dev, now.ino
            ));
        }
        if self.length != now.length {
            return Some(format!("size {} != {}", self.length, now.length));
        }
        if self.mtime_s != now.mtime_s || self.mtime_ns != now.mtime_ns {
            return Some(format!(
                "change token {}.{:09} != {}.{:09}",
                self.mtime_s, self.mtime_ns, now.mtime_s, now.mtime_ns
            ));
        }
        None
    }
}

// ---------------------------------------------------------------- fault injection

/// Dev-only kill points. `tfs put --fault <stage>` reaches the stage, announces it, and
/// waits to be killed from outside — a real SIGKILL to a real process, never a simulation.
#[derive(Debug, Clone, Default)]
pub struct Fault {
    pub stage: Option<String>,
    pub ready: Option<PathBuf>,
}

pub const FAULT_STAGES: &[&str] = &[
    "open",    // temp created, no bytes
    "partial", // first chunk written, temp is a partial object
    "written", // all bytes written, not yet durable
    "fsync",   // durable temp, still 0600, not linked
    "chmod",   // 0444 verified temp, one link() from installed
    "linked",  // installed, parent dir not yet fsynced, temp not yet unlinked
    "record",  // verification record written; nothing left to lose
];

impl Fault {
    fn hit(&self, stage: &str) {
        if self.stage.as_deref() != Some(stage) {
            return;
        }
        if let Some(p) = &self.ready {
            let _ = fs::write(p, format!("{stage} {}\n", std::process::id()));
        }
        eprintln!("FAULT-READY {stage} pid={}", std::process::id());
        loop {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
}

// ---------------------------------------------------------------- the store

#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
    marker: Marker,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Put {
    pub obj: ObjectRef,
    /// This process won the no-clobber admission (vs. found the id already installed).
    pub admitted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Verified { rehashed: bool },
    Invalidated { why: String },
    Quarantined { to: PathBuf, why: String },
}

#[derive(Debug, Default, Clone)]
pub struct Scrub {
    pub scanned: usize,
    pub sampled: usize,
    pub rehashed: usize,
    pub quarantined: usize,
    pub bytes_hashed: u64,
}

impl Store {
    pub fn init(root: &Path) -> Result<Store> {
        for d in ["objects/sha256", "verified", "tmp", "quarantine"] {
            fs::create_dir_all(root.join(d)).map_err(|e| io(format!("mkdir {d}"), e))?;
        }
        let p = root.join("store.json");
        if p.exists() {
            return Store::open(root);
        }
        let marker = Marker {
            era: ERA.to_string(),
            generation: now_secs(),
            object_hash: "sha256".to_string(),
        };
        write_doc(&p, &marker.canonical_bytes(), true)?;
        Ok(Store {
            root: root.to_path_buf(),
            marker,
        })
    }

    /// Typed refusal for an unmarked or non-v2 root (tensorfs.md §10b).
    pub fn open(root: &Path) -> Result<Store> {
        let p = root.join("store.json");
        let bytes = match fs::read(&p) {
            Ok(b) => b,
            Err(_) => {
                let v1: Vec<&str> = ["metadata.sqlite3", "refs", "objects", "blobs", "store.toml"]
                    .into_iter()
                    .filter(|n| root.join(n).exists())
                    .collect();
                return refuse(
                    Code::STORE_ERA,
                    if v1.is_empty() {
                        format!("{}: no era marker — not a tensorfs store", root.display())
                    } else {
                        format!(
                            "{}: no v2 era marker beside {v1:?} — refusing to open unknown \
                             or v1-era state as v2",
                            root.display()
                        )
                    },
                );
            }
        };
        let marker = Marker::parse(&bytes)?;
        if marker.era != ERA {
            return refuse(
                Code::STORE_ERA,
                format!(
                    "store era {:?} != {ERA:?} — v1-era state never opens as v2 (§10b)",
                    marker.era
                ),
            );
        }
        Ok(Store {
            root: root.to_path_buf(),
            marker,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn generation(&self) -> u64 {
        self.marker.generation
    }

    /// A new generation invalidates every verification record at once — the recovery lever
    /// after a restore, a clone, or any event that breaks file identity wholesale.
    pub fn bump_generation(&mut self) -> Result<u64> {
        self.marker.generation = now_secs().max(self.marker.generation + 1);
        write_doc(
            &self.root.join("store.json"),
            &self.marker.canonical_bytes(),
            true,
        )?;
        Ok(self.marker.generation)
    }

    pub fn object_path(&self, hex: &str) -> PathBuf {
        self.root
            .join("objects/sha256")
            .join(&hex[0..2])
            .join(&hex[2..4])
            .join(hex)
    }
    fn record_path(&self, hex: &str) -> PathBuf {
        self.root
            .join("verified")
            .join(&hex[0..2])
            .join(&hex[2..4])
            .join(hex)
    }

    /// PRESENCE HINT ONLY. True says an object file exists at that id; it says nothing
    /// about its bytes. Resume must consult `record` or rehash.
    pub fn contains(&self, hex: &str) -> bool {
        self.object_path(hex).is_file()
    }

    // ------------------------------------------------------------ admission

    pub fn put_stream<R: Read>(
        &self,
        r: &mut R,
        expect: Option<&ObjectRef>,
        fault: &Fault,
    ) -> Result<Put> {
        let tmp = self.root.join("tmp").join(format!(
            "put-{}-{}",
            std::process::id(),
            now_nanos_unique()
        ));
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)
            .map_err(|e| io(format!("create temp {}", tmp.display()), e))?;
        fault.hit("open");

        let mut h = Sha256::new();
        let mut buf = vec![0u8; BUF];
        let mut len: u64 = 0;
        let mut first = true;
        loop {
            let n = match r.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => return cleanup(&tmp, io("read source", e)),
            };
            h.update(&buf[..n]);
            if let Err(e) = f.write_all(&buf[..n]) {
                return cleanup(&tmp, io("write temp", e));
            }
            len += n as u64;
            if first {
                first = false;
                fault.hit("partial");
            }
        }
        fault.hit("written");
        if let Err(e) = f.sync_all() {
            return cleanup(&tmp, io("fsync temp", e));
        }
        drop(f);
        fault.hit("fsync");

        let hex = sha256::hex(&h.finish());
        if let Some(want) = expect {
            if want.length != len {
                return cleanup(
                    &tmp,
                    Refusal {
                        code: Code::LENGTH_MISMATCH,
                        detail: format!("expected {} bytes, streamed {len}", want.length),
                    },
                );
            }
            if want.sha256 != hex {
                return cleanup(
                    &tmp,
                    Refusal {
                        code: Code::OBJECT_ID_MISMATCH,
                        detail: format!("expected sha256:{}, streamed sha256:{hex}", want.sha256),
                    },
                );
            }
        }

        if let Err(e) = fs::set_permissions(&tmp, fs::Permissions::from_mode(0o444)) {
            return cleanup(&tmp, io("chmod 0444", e));
        }
        fault.hit("chmod");

        let dst = self.object_path(&hex);
        let dir = dst.parent().unwrap().to_path_buf();
        if let Err(e) = fs::create_dir_all(&dir) {
            return cleanup(&tmp, io("mkdir fan-out", e));
        }
        // No-clobber admission: link() refuses to replace an installed object.
        let admitted = match fs::hard_link(&tmp, &dst) {
            Ok(()) => true,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => false,
            Err(e) => return cleanup(&tmp, io(format!("link {}", dst.display()), e)),
        };
        fault.hit("linked");
        if admitted {
            fsync_dir(&dir)?;
        }
        let _ = fs::remove_file(&tmp);

        // A record is written only for bytes THIS process hashed. Losing the race means we
        // never saw the installed file's bytes, so we make no claim about them.
        if admitted {
            let md = stat(&dst)?;
            self.write_record(&VRecord::of(&hex, &md, self.marker.generation))?;
        }
        fault.hit("record");
        Ok(Put {
            obj: ObjectRef {
                sha256: hex,
                length: len,
            },
            admitted,
        })
    }

    pub fn put_file(&self, src: &Path, expect: Option<&ObjectRef>, fault: &Fault) -> Result<Put> {
        let mut f = File::open(src).map_err(|e| io(format!("open {}", src.display()), e))?;
        self.put_stream(&mut f, expect, fault)
    }

    // ------------------------------------------------------------ records

    fn write_record(&self, rec: &VRecord) -> Result<()> {
        let p = self.record_path(&rec.sha256);
        fs::create_dir_all(p.parent().unwrap()).map_err(|e| io("mkdir verified", e))?;
        // Not fsynced, deliberately: a record is a CACHED FACT. Losing one costs a rehash,
        // and a torn one cannot be observed (atomic rename) nor believed (canonical parse
        // + inode/mtime/generation binding). Paying fsync here doubled small-object cost.
        write_doc(&p, &rec.canonical_bytes(), false)
    }

    fn record(&self, hex: &str) -> Option<VRecord> {
        VRecord::parse(&fs::read(self.record_path(hex)).ok()?).ok()
    }

    fn drop_record(&self, hex: &str) {
        let _ = fs::remove_file(self.record_path(hex));
    }

    /// The one question the read and resume paths ask: may these bytes be trusted without
    /// hashing them again?
    pub fn record_valid(&self, hex: &str) -> std::result::Result<VRecord, String> {
        let md = match stat(&self.object_path(hex)) {
            Ok(md) => md,
            Err(e) => return Err(e.detail),
        };
        let rec = self
            .record(hex)
            .ok_or_else(|| "no verification record".to_string())?;
        match rec.disagreement(hex, &md, self.marker.generation) {
            Some(why) => Err(why),
            None => Ok(rec),
        }
    }

    // ------------------------------------------------------------ reads

    /// Open for reading without following a symlink at the final component, and refuse
    /// anything that is not a regular file.
    pub fn open_nofollow(&self, hex: &str) -> Result<File> {
        let p = self.object_path(hex);
        let f = OpenOptions::new()
            .read(true)
            .custom_flags(O_NOFOLLOW)
            .open(&p)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Refusal {
                        code: Code::OBJECT_ABSENT,
                        detail: format!("sha256:{hex} not in {}", self.root.display()),
                    }
                } else {
                    io(format!("open {}", p.display()), e)
                }
            })?;
        let md = f.metadata().map_err(|e| io("stat object", e))?;
        if !md.is_file() {
            return refuse(
                Code::NOT_REGULAR_FILE,
                format!("{} is not a regular file", p.display()),
            );
        }
        Ok(f)
    }

    /// A verified handle: trusted from a valid record, otherwise rehashed. Bytes that
    /// disagree with their id are quarantined, never returned.
    pub fn open_verified(&self, hex: &str) -> Result<File> {
        match self.verify(hex, false)? {
            Verdict::Quarantined { to, why } => refuse(
                Code::OBJECT_CORRUPT,
                format!("sha256:{hex} quarantined to {}: {why}", to.display()),
            ),
            _ => self.open_nofollow(hex),
        }
    }

    /// Bounded range read over verified bytes. Out-of-bounds refuses instead of clamping.
    pub fn read_range(&self, hex: &str, off: u64, len: u64) -> Result<Vec<u8>> {
        let mut f = self.open_verified(hex)?;
        let size = f.metadata().map_err(|e| io("stat object", e))?.len();
        let end = off.checked_add(len).ok_or(Refusal {
            code: Code::RANGE_BOUNDS,
            detail: format!("offset {off} + length {len} overflows"),
        })?;
        if end > size {
            return refuse(
                Code::RANGE_BOUNDS,
                format!("range [{off}, {end}) leaves object of {size} bytes"),
            );
        }
        f.seek(SeekFrom::Start(off)).map_err(|e| io("seek", e))?;
        let mut out = vec![0u8; len as usize];
        f.read_exact(&mut out).map_err(|e| io("read range", e))?;
        Ok(out)
    }

    /// Stream a verified object out. The read path does not hash; `verify` decided already.
    pub fn read_into<W: Write>(&self, hex: &str, w: &mut W) -> Result<u64> {
        let mut f = self.open_verified(hex)?;
        let mut buf = vec![0u8; BUF];
        let mut n = 0u64;
        loop {
            let k = f.read(&mut buf).map_err(|e| io("read object", e))?;
            if k == 0 {
                return Ok(n);
            }
            w.write_all(&buf[..k]).map_err(|e| io("write out", e))?;
            n += k as u64;
        }
    }

    // ------------------------------------------------------------ verify / quarantine

    /// `paranoid` always rehashes. Otherwise a valid record is the answer and a rehash
    /// happens exactly when the record is missing or any bound fact moved.
    pub fn verify(&self, hex: &str, paranoid: bool) -> Result<Verdict> {
        let md = stat(&self.object_path(hex))?;
        let disagreement = match self.record(hex) {
            None => Some("no verification record".to_string()),
            Some(rec) => rec.disagreement(hex, &md, self.marker.generation),
        };
        if disagreement.is_none() && !paranoid {
            return Ok(Verdict::Verified { rehashed: false });
        }
        let got = self.hash_object(hex)?;
        if got != hex {
            let why = disagreement.unwrap_or_else(|| "paranoid rehash".to_string());
            let to = self.quarantine(hex)?;
            return Ok(Verdict::Quarantined {
                to,
                why: format!("{why}; rehash gave sha256:{got}"),
            });
        }
        let md = stat(&self.object_path(hex))?;
        self.write_record(&VRecord::of(hex, &md, self.marker.generation))?;
        Ok(match disagreement {
            None => Verdict::Verified { rehashed: true },
            Some(why) => Verdict::Invalidated { why },
        })
    }

    pub fn hash_object(&self, hex: &str) -> Result<String> {
        let mut f = self.open_nofollow(hex)?;
        let mut h = Sha256::new();
        let mut buf = vec![0u8; BUF];
        loop {
            let n = f.read(&mut buf).map_err(|e| io("read object", e))?;
            if n == 0 {
                return Ok(sha256::hex(&h.finish()));
            }
            h.update(&buf[..n]);
        }
    }

    /// Take the NAME first: rename out of `objects/` so exactly one remover wins and the
    /// bad bytes can never satisfy another read.
    pub fn quarantine(&self, hex: &str) -> Result<PathBuf> {
        let dir = self.root.join("quarantine");
        fs::create_dir_all(&dir).map_err(|e| io("mkdir quarantine", e))?;
        let mut to = dir.join(hex);
        let mut n = 1;
        while to.exists() {
            to = dir.join(format!("{hex}.{n}"));
            n += 1;
        }
        fs::rename(self.object_path(hex), &to).map_err(|e| io("quarantine rename", e))?;
        self.drop_record(hex);
        Ok(to)
    }

    // ------------------------------------------------------------ scrub / reap

    pub fn objects(&self) -> Result<Vec<String>> {
        let mut out = Vec::new();
        let base = self.root.join("objects/sha256");
        for a in read_dir(&base)? {
            for b in read_dir(&a)? {
                for o in read_dir(&b)? {
                    if let Some(n) = o.file_name().and_then(|s| s.to_str()) {
                        if hex64("object", n).is_ok() {
                            out.push(n.to_string());
                        }
                    }
                }
            }
        }
        out.sort();
        Ok(out)
    }

    /// ONE knob. `rate` is the fraction of objects rehashed this pass, chosen
    /// deterministically from (pass, id). Everything else gets its record checked, and a
    /// record that disagrees is rehashed anyway — so there is no second paranoia dial.
    pub fn scrub(&self, rate: f64, pass: u64) -> Result<Scrub> {
        let mut s = Scrub::default();
        for hex in self.objects()? {
            s.scanned += 1;
            let sampled = sample(pass, &hex) < rate;
            if sampled {
                s.sampled += 1;
            }
            let size = stat(&self.object_path(&hex)).map(|m| m.len()).unwrap_or(0);
            match self.verify(&hex, sampled)? {
                Verdict::Verified { rehashed } => {
                    if rehashed {
                        s.rehashed += 1;
                        s.bytes_hashed += size;
                    }
                }
                Verdict::Invalidated { .. } => {
                    s.rehashed += 1;
                    s.bytes_hashed += size;
                }
                Verdict::Quarantined { .. } => {
                    s.rehashed += 1;
                    s.bytes_hashed += size;
                    s.quarantined += 1;
                }
            }
        }
        Ok(s)
    }

    /// Age-bounded reap of abandoned admission temps. A temp is scratch by construction:
    /// nothing references it, so removing an old one can never lose an object.
    pub fn reap(&self, max_age_secs: u64) -> Result<(usize, usize)> {
        let (mut gone, mut kept) = (0, 0);
        for p in read_dir(&self.root.join("tmp"))? {
            let age = fs::metadata(&p)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| SystemTime::now().duration_since(t).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            if age >= max_age_secs && fs::remove_file(&p).is_ok() {
                gone += 1;
            } else {
                kept += 1;
            }
        }
        Ok((gone, kept))
    }
}

// ---------------------------------------------------------------- helpers

fn cleanup<T>(tmp: &Path, e: Refusal) -> Result<T> {
    let _ = fs::set_permissions(tmp, fs::Permissions::from_mode(0o600));
    let _ = fs::remove_file(tmp);
    Err(e)
}

fn stat(p: &Path) -> Result<fs::Metadata> {
    let md = fs::symlink_metadata(p).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            Refusal {
                code: Code::OBJECT_ABSENT,
                detail: format!("{} absent", p.display()),
            }
        } else {
            io(format!("stat {}", p.display()), e)
        }
    })?;
    if !md.is_file() {
        return refuse(
            Code::NOT_REGULAR_FILE,
            format!("{} is not a regular file", p.display()),
        );
    }
    Ok(md)
}

fn read_dir(p: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let rd = match fs::read_dir(p) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(io(format!("read_dir {}", p.display()), e)),
    };
    for e in rd {
        out.push(e.map_err(|e| io("dir entry", e))?.path());
    }
    out.sort();
    Ok(out)
}

/// Atomic small-document write: temp beside the target, fsync, rename, fsync the dir.
fn write_doc(p: &Path, bytes: &[u8], durable: bool) -> Result<()> {
    let dir = p.parent().unwrap();
    let tmp = dir.join(format!(
        ".{}.{}.{}",
        p.file_name().unwrap().to_string_lossy(),
        std::process::id(),
        now_nanos_unique()
    ));
    let mut f = File::create(&tmp).map_err(|e| io(format!("create {}", tmp.display()), e))?;
    f.write_all(bytes).map_err(|e| io("write doc", e))?;
    if durable {
        f.sync_all().map_err(|e| io("fsync doc", e))?;
    }
    drop(f);
    fs::rename(&tmp, p).map_err(|e| io(format!("rename {}", p.display()), e))?;
    if durable {
        fsync_dir(dir)?;
    }
    Ok(())
}

fn fsync_dir(dir: &Path) -> Result<()> {
    File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(|e| io(format!("fsync dir {}", dir.display()), e))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn now_nanos_unique() -> u128 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
        .wrapping_add(N.fetch_add(1, Ordering::Relaxed) as u128)
}

/// Deterministic [0,1) sample position for (pass, object) — the one scrub knob's input.
fn sample(pass: u64, hex: &str) -> f64 {
    let d = sha256::digest(format!("{pass}:{hex}").as_bytes());
    let v = u32::from_be_bytes([d[0], d[1], d[2], d[3]]);
    v as f64 / u32::MAX as f64
}
