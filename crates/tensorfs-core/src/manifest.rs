//! SnapshotManifest — directories and regular file→object/blob entries only.
//! `checkpoint_id = snapshot_id`, unconditionally.
//!
//! Path semantics are REDUCED (tensorfs.md §2): no symlinks, hardlink groups, sparse holes
//! or executable bits — an entry kind outside {dir, file, blob} refuses decode, so the
//! richer facts of a foreign source tree are unrepresentable rather than merely unused.
//!
//! NFC, without a normalization table: decisions row "tfs-013 canonical layer divergences"
//! makes every tensorfs document field printable-ASCII, which is strictly STRONGER than NFC
//! (an NFC twin and an NFD twin of one name are both red, at the canon layer, before any
//! path law runs). There is no table here for a normalizer to disagree with.

use crate::canon::{as_arr, as_str, Fields, Value};
use crate::err::{refuse, Code, Result};
use crate::ids::{check_format, Doc, ObjectRef};
use crate::limits;

pub const SEGMENT_PROFILE: &str = "fixed-64m/1";

/// Content-addressed contained documents (EncodingSpec objects and their conformance-vector
/// sets) live under ONE directory whose leaf names ARE their digests. The path terminal is
/// therefore checkable against the object it names, with no second naming authority.
pub const CLOSURE_DIR: &str = "closure";
pub const ASSET_DIR: &str = "assets";

/// The canonical path of one contained document.
pub fn closure_path(sha256: &str) -> String {
    format!("{CLOSURE_DIR}/{sha256}.json")
}

pub fn asset_path(sha256: &str) -> String {
    format!("{ASSET_DIR}/{sha256}.bin")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobDescriptor {
    pub logical_size: u64,
    pub whole_sha256: String,
    pub segments: Vec<ObjectRef>,
}

impl BlobDescriptor {
    pub fn of(bytes: &[u8]) -> BlobDescriptor {
        BlobDescriptor {
            logical_size: bytes.len() as u64,
            whole_sha256: crate::sha256::hex_digest(bytes),
            segments: bytes
                .chunks(limits::GRID_BYTES as usize)
                .map(ObjectRef::of)
                .collect(),
        }
    }
    fn from_value(v: &Value) -> Result<BlobDescriptor> {
        let mut f = Fields::new("BlobDescriptor", v)?;
        let logical_size = f.req_uint("logical_size")?;
        let profile = as_str(
            "BlobDescriptor",
            "segment_profile",
            f.req("segment_profile")?,
        )?;
        if profile != SEGMENT_PROFILE {
            return refuse(
                Code::UNKNOWN_FORMAT,
                format!("segment_profile {profile:?} is not {SEGMENT_PROFILE:?}"),
            );
        }
        let mut segments = Vec::new();
        for s in as_arr("BlobDescriptor", "segments", f.req("segments")?)? {
            segments.push(ObjectRef::from_value("BlobDescriptor.segment", s)?);
        }
        let whole_sha256 =
            crate::ids::hex64("BlobDescriptor.whole_sha256", f.req_str("whole_sha256")?)?;
        f.done()?;
        if segments.is_empty() || segments.len() > limits::MAX_SEGMENTS {
            return refuse(Code::COUNT_CAP, "blob segment count out of bounds");
        }
        let mut total = 0u64;
        for (i, s) in segments.iter().enumerate() {
            let last = i + 1 == segments.len();
            if s.length == 0
                || (!last && s.length != limits::GRID_BYTES)
                || s.length > limits::GRID_BYTES
            {
                return refuse(
                    Code::BYTE_LENGTH_MISMATCH,
                    format!(
                        "segment {i} length {} violates the {SEGMENT_PROFILE} profile",
                        s.length
                    ),
                );
            }
            total += s.length;
        }
        if total != logical_size {
            return refuse(
                Code::BYTE_LENGTH_MISMATCH,
                format!("segments total {total} != logical_size {logical_size}"),
            );
        }
        Ok(BlobDescriptor {
            logical_size,
            whole_sha256,
            segments,
        })
    }
    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("logical_size", Value::uint(self.logical_size)),
            ("segment_profile", Value::str(SEGMENT_PROFILE)),
            (
                "segments",
                Value::arr(self.segments.iter().map(|s| s.to_value()).collect()),
            ),
            ("whole_sha256", Value::str(self.whole_sha256.clone())),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    Dir,
    File(ObjectRef),
    Blob(BlobDescriptor),
}

/// A manifest UNDER CONSTRUCTION. Mutable, renderable, and deliberately identity-free: a
/// draft can be built wrong on purpose (that is what a planted red arm is), and the one
/// thing it can never do is advertise an id for a tree that does not satisfy the path law.
///
/// Audit fix (external review, folded in at tfs-003): `SnapshotManifest` used to carry
/// public mutable fields while `snapshot_id`/`checkpoint_id` hashed WITHOUT validating, so
/// a programmatically-built invalid manifest could obtain an id and be stored under it.
/// Identity now exists only on the sealed type.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Draft {
    pub header: Option<ObjectRef>,
    pub entries: Vec<(String, Entry)>,
}

impl Draft {
    pub fn of(m: &SnapshotManifest) -> Draft {
        Draft {
            header: m.header.clone(),
            entries: m.entries.clone(),
        }
    }
    /// The canonical rendering of a draft. Bytes for a decoder to refuse — never an id.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        crate::canon::write(&manifest_value(&self.header, &self.entries))
    }
    /// The ONLY programmatic path to a manifest: the path law runs, or nothing is produced.
    pub fn seal(self) -> Result<SnapshotManifest> {
        let m = SnapshotManifest {
            header: self.header,
            entries: self.entries,
        };
        m.validate()?;
        Ok(m)
    }
}

/// A VALIDATED snapshot tree. Constructible only through `Draft::seal` or a strict decode,
/// so every instance that exists has already satisfied the path law — which is what makes
/// `snapshot_id()` infallible AND honest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotManifest {
    /// Structural, never by pathname: a checkpoint snapshot carries exactly one.
    header: Option<ObjectRef>,
    entries: Vec<(String, Entry)>,
}

const RESERVED: [&str; 22] = [
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

pub(crate) fn check_path(p: &str) -> Result<()> {
    if p.is_empty() || p.len() > limits::MAX_KEY_BYTES {
        return refuse(
            Code::PATH_ILLEGAL,
            format!("path length {} out of bounds", p.len()),
        );
    }
    if p.starts_with('/') || p.ends_with('/') || p.contains("//") {
        return refuse(Code::PATH_ILLEGAL, format!("{p:?}: empty path component"));
    }
    for c in p.split('/') {
        if c == "." || c == ".." {
            return refuse(Code::PATH_ILLEGAL, format!("{p:?}: dot component"));
        }
        if c.ends_with('.') || c.ends_with(' ') || c.starts_with(' ') {
            return refuse(Code::PATH_ILLEGAL, format!("{p:?}: trailing dot or space"));
        }
        if c.bytes()
            .any(|b| matches!(b, b'\\' | b':' | b'*' | b'?' | b'"' | b'<' | b'>' | b'|'))
        {
            return refuse(Code::PATH_ILLEGAL, format!("{p:?}: reserved character"));
        }
        let stem = c.split('.').next().unwrap_or(c).to_ascii_lowercase();
        if RESERVED.contains(&stem.as_str()) {
            return refuse(
                Code::PATH_RESERVED,
                format!("{p:?}: Windows reserved stem {stem:?}"),
            );
        }
    }
    Ok(())
}

/// One contained-document path whose terminal must equal the digest it names.
fn check_content_addressed(path: &str, entry: &Entry) -> Result<()> {
    let (leaf, suffix) = if let Some(rest) = path.strip_prefix(CLOSURE_DIR) {
        match rest.strip_prefix('/') {
            Some(leaf) => (leaf, ".json"),
            None => return Ok(()),
        }
    } else if let Some(rest) = path.strip_prefix(ASSET_DIR) {
        match rest.strip_prefix('/') {
            Some(leaf) => (leaf, ".bin"),
            None => return Ok(()),
        }
    } else {
        return Ok(());
    };
    let obj = match entry {
        Entry::File(o) => o,
        _ => {
            return refuse(
                Code::PATH_DIGEST_MISMATCH,
                format!("{path:?}: contained documents are bounded canonical FILE objects"),
            )
        }
    };
    match leaf.strip_suffix(suffix) {
        Some(stem) if stem == obj.sha256 => Ok(()),
        _ => refuse(
            Code::PATH_DIGEST_MISMATCH,
            format!(
                "{path:?}: content-addressed terminal does not name {}",
                obj.id()
            ),
        ),
    }
}

impl SnapshotManifest {
    pub fn header(&self) -> Option<&ObjectRef> {
        self.header.as_ref()
    }
    pub fn entries(&self) -> &[(String, Entry)] {
        &self.entries
    }
    /// The pinned atom. Every checkpoint has a snapshot — the id is the snapshot's.
    /// Infallible because the type cannot exist unvalidated.
    pub fn snapshot_id(&self) -> String {
        self.object_id()
    }
    /// UNCONDITIONAL: one derivation, zero consumer branching. A tensors-only checkpoint
    /// lists exactly one `model.cozytensors` file and its id derives identically.
    pub fn checkpoint_id(&self) -> Result<String> {
        match self.header {
            Some(_) => Ok(self.snapshot_id()),
            None => refuse(
                Code::ATTACHMENT_CARDINALITY,
                "not a checkpoint snapshot: no cozytensors header attachment",
            ),
        }
    }

    /// Is this exact object listed as a file entry, and at how many paths?
    pub fn listings(&self, r: &ObjectRef) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|(_, e)| matches!(e, Entry::File(o) if o == r))
            .map(|(p, _)| p.as_str())
            .collect()
    }

    /// The path law, run by `Draft::seal` and by every decode — the two doors into the
    /// type, which is why it is private: there is no instance that skipped it.
    /// Strictly ascending raw bytes gives
    /// duplicate-path refusal and parent-before-child for free; `dirs` is a subsequence of
    /// an ascending sequence, so the parent lookup is a binary search rather than a scan.
    fn validate(&self) -> Result<()> {
        if self.entries.is_empty() {
            return refuse(Code::MISSING_FIELD, "a snapshot lists at least one entry");
        }
        let mut dirs: Vec<&str> = Vec::new();
        let mut prev: Option<&str> = None;
        let mut folded: Vec<String> = Vec::with_capacity(self.entries.len());
        let mut header_listings: Vec<&str> = Vec::new();

        for (path, entry) in &self.entries {
            check_path(path)?;
            if let Some(p) = prev {
                if p.as_bytes() >= path.as_bytes() {
                    return refuse(
                        Code::PATH_ORDER,
                        format!("{path:?} does not strictly follow {p:?}"),
                    );
                }
            }
            prev = Some(path);
            folded.push(path.to_ascii_lowercase());
            if let Some(idx) = path.rfind('/') {
                let parent = &path[..idx];
                if dirs.binary_search(&parent).is_err() {
                    return refuse(
                        Code::PATH_PARENT_MISSING,
                        format!("{path:?}: no directory entry for {parent:?}"),
                    );
                }
            }
            check_content_addressed(path, entry)?;
            match entry {
                Entry::Dir => dirs.push(path),
                Entry::File(o) => {
                    if self.header.as_ref() == Some(o) {
                        header_listings.push(path);
                    }
                }
                Entry::Blob(_) => {}
            }
        }

        folded.sort();
        if let Some(w) = folded.windows(2).find(|w| w[0] == w[1]) {
            return refuse(
                Code::PATH_CASE_COLLISION,
                format!("two entries case-fold onto {:?}", w[0]),
            );
        }

        // The attachment is TYPED, never found by pathname — but the object it names must
        // be in the tree exactly once, or the projected checkout has zero or two headers.
        if self.header.is_some() && header_listings.len() != 1 {
            return refuse(
                Code::ATTACHMENT_CARDINALITY,
                format!(
                    "the attached CozyTensorsHeader is listed at {} paths {header_listings:?}, not 1",
                    header_listings.len()
                ),
            );
        }
        Ok(())
    }
}

impl Doc for SnapshotManifest {
    const FORMAT: &'static str = "tensorfs.snapshot/1";
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("SnapshotManifest", v)?;
        check_format(
            "SnapshotManifest",
            as_str("SnapshotManifest", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;
        let header = match f.opt("attachments") {
            Some(a) => {
                let mut af = Fields::new("SnapshotManifest.attachments", a)?;
                let h = ObjectRef::from_value(
                    "attachments.cozytensors_header",
                    af.req("cozytensors_header")?,
                )?;
                af.done()?;
                Some(h)
            }
            None => None,
        };
        let raw = as_arr("SnapshotManifest", "entries", f.req("entries")?)?;
        // Refuse BEFORE allocation: the entry vector is sized only once the count is legal.
        if raw.len() > limits::MAX_ENTRIES {
            return refuse(
                Code::COUNT_CAP,
                format!(
                    "entry count {} over the {} cap",
                    raw.len(),
                    limits::MAX_ENTRIES
                ),
            );
        }
        let mut entries = Vec::with_capacity(raw.len());
        for e in raw {
            let mut ef = Fields::new("SnapshotManifest.entry", e)?;
            let kind = as_str("entry", "kind", ef.req("kind")?)?.to_string();
            let path = ef.req_str("path")?.to_string();
            let entry = match kind.as_str() {
                "dir" => Entry::Dir,
                "file" => Entry::File(ObjectRef::from_value("entry.object", ef.req("object")?)?),
                "blob" => Entry::Blob(BlobDescriptor::from_value(ef.req("blob")?)?),
                other => {
                    return refuse(
                        Code::UNKNOWN_FIELD,
                        format!("entry kind {other:?} is outside {{dir,file,blob}}"),
                    )
                }
            };
            ef.done()?;
            entries.push((path, entry));
        }
        f.done()?;
        // Strict decode IS the path law: a manifest that parses but violates it never exists
        // as a value, so no caller can forget to validate one.
        Draft { header, entries }.seal()
    }

    fn to_value(&self) -> Value {
        manifest_value(&self.header, &self.entries)
    }
}

/// ONE rendering, shared by the draft and the sealed manifest. Two spellings of the same
/// document is the drift this repo is built against.
fn manifest_value(header: &Option<ObjectRef>, entries: &[(String, Entry)]) -> Value {
    let mut fields = vec![
        ("format", Value::str(SnapshotManifest::FORMAT)),
        (
            "entries",
            Value::arr(
                entries
                    .iter()
                    .map(|(p, e)| {
                        let mut f = vec![("path", Value::str(p.clone()))];
                        match e {
                            Entry::Dir => f.push(("kind", Value::str("dir"))),
                            Entry::File(o) => {
                                f.push(("kind", Value::str("file")));
                                f.push(("object", o.to_value()));
                            }
                            Entry::Blob(b) => {
                                f.push(("blob", b.to_value()));
                                f.push(("kind", Value::str("blob")));
                            }
                        }
                        Value::obj(f)
                    })
                    .collect(),
            ),
        ),
    ];
    if let Some(h) = header {
        fields.push((
            "attachments",
            Value::obj(vec![("cozytensors_header", h.to_value())]),
        ));
    }
    Value::obj(fields)
}
