//! SnapshotManifest — directories and regular file→object/blob entries only.
//! `checkpoint_id = snapshot_id`, unconditionally.

use crate::canon::{as_arr, as_str, Fields, Value};
use crate::err::{refuse, Code, Result};
use crate::ids::{check_format, Doc, ObjectRef};
use crate::limits;

pub const SEGMENT_PROFILE: &str = "fixed-64m/1";

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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SnapshotManifest {
    /// Structural, never by pathname: a checkpoint snapshot carries exactly one.
    pub header: Option<ObjectRef>,
    pub entries: Vec<(String, Entry)>,
}

const RESERVED: [&str; 22] = [
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

fn check_path(p: &str) -> Result<()> {
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

impl SnapshotManifest {
    /// The pinned atom. Every checkpoint has a snapshot — the id is the snapshot's.
    pub fn snapshot_id(&self) -> String {
        self.object_id()
    }
    pub fn checkpoint_id(&self) -> Result<String> {
        match self.header {
            Some(_) => Ok(self.snapshot_id()),
            None => refuse(
                Code::ATTACHMENT_CARDINALITY,
                "not a checkpoint snapshot: no cozytensors header attachment",
            ),
        }
    }

    pub fn validate(&self) -> Result<()> {
        let mut folded: Vec<String> = Vec::new();
        let mut dirs: Vec<&str> = Vec::new();
        let mut prev: Option<&str> = None;
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
            let fold = path.to_ascii_lowercase();
            if folded.contains(&fold) {
                return refuse(
                    Code::PATH_CASE_COLLISION,
                    format!("{path:?} case-folds onto an earlier entry"),
                );
            }
            folded.push(fold);
            if let Some(idx) = path.rfind('/') {
                let parent = &path[..idx];
                if !dirs.contains(&parent) {
                    return refuse(
                        Code::PATH_PARENT_MISSING,
                        format!("{path:?}: no directory entry for {parent:?}"),
                    );
                }
            }
            if matches!(entry, Entry::Dir) {
                dirs.push(path);
            }
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
        let mut entries = Vec::new();
        for e in as_arr("SnapshotManifest", "entries", f.req("entries")?)? {
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
        if entries.len() > limits::MAX_TENSORS {
            return refuse(Code::COUNT_CAP, "entry count over cap");
        }
        Ok(SnapshotManifest { header, entries })
    }

    fn to_value(&self) -> Value {
        let mut fields = vec![
            ("format", Value::str(Self::FORMAT)),
            (
                "entries",
                Value::arr(
                    self.entries
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
        if let Some(h) = &self.header {
            fields.push((
                "attachments",
                Value::obj(vec![("cozytensors_header", h.to_value())]),
            ));
        }
        Value::obj(fields)
    }
}
