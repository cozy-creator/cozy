//! One location-typed virtual file tree. A manifest has only ordinary files and one optional
//! typed CozyTensors entry; directories are inferred from paths and empty directories are not data.

use crate::canon::{as_arr, as_str, Fields, Value};
use crate::err::{refuse, Code, Result};
use crate::ids::{Doc, ObjectRef};
use crate::limits;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    File(ObjectRef),
    CozyTensors(ObjectRef),
}

impl Entry {
    pub fn blob(&self) -> &ObjectRef {
        match self {
            Entry::File(blob) | Entry::CozyTensors(blob) => blob,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Draft {
    pub entries: Vec<(String, Entry)>,
}

impl Draft {
    pub fn of(manifest: &Manifest) -> Self {
        Self {
            entries: manifest.entries.clone(),
        }
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        crate::canon::write(&manifest_value(&self.entries))
    }

    pub fn seal(self) -> Result<Manifest> {
        let manifest = Manifest {
            entries: self.entries,
        };
        manifest.validate()?;
        if manifest.canonical_bytes().len() > Manifest::MAX_BYTES {
            return refuse(Code::SIZE_CAP, "manifest canonical bytes exceed cap");
        }
        Ok(manifest)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    entries: Vec<(String, Entry)>,
}

/// Traversal and ambiguity only. A manifest path is a key in a VIRTUAL tree: nothing here
/// touches a real filesystem, so nothing here is entitled to a filesystem's opinions.
/// Windows device stems and case folding are a CHECKOUT law and live in `project.rs`.
pub(crate) fn check_path(path: &str) -> Result<()> {
    if path.is_empty() || path.len() > limits::MAX_KEY_BYTES {
        return refuse(
            Code::PATH_ILLEGAL,
            format!("path length {} out of bounds", path.len()),
        );
    }
    if path.starts_with('/') || path.ends_with('/') || path.contains("//") {
        return refuse(
            Code::PATH_ILLEGAL,
            format!("{path:?}: empty path component"),
        );
    }
    for component in path.split('/') {
        if component == "." || component == ".." {
            return refuse(Code::PATH_ILLEGAL, format!("{path:?}: dot component"));
        }
        if component.ends_with('.') || component.ends_with(' ') || component.starts_with(' ') {
            return refuse(
                Code::PATH_ILLEGAL,
                format!("{path:?}: trailing dot or space"),
            );
        }
        if component
            .bytes()
            .any(|byte| matches!(byte, b'\\' | b':' | b'*' | b'?' | b'"' | b'<' | b'>' | b'|'))
        {
            return refuse(Code::PATH_ILLEGAL, format!("{path:?}: reserved character"));
        }
    }
    Ok(())
}

impl Manifest {
    pub const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    /// Build one ordinary-file tree from already measured blob references. This is the shared
    /// producer used by local callers and cloud adapters; it creates no second source-tree shape.
    pub fn from_files(entries: Vec<(String, ObjectRef)>) -> Result<Self> {
        Draft {
            entries: entries
                .into_iter()
                .map(|(path, blob)| (path, Entry::File(blob)))
                .collect(),
        }
        .seal()
    }

    pub fn entries(&self) -> &[(String, Entry)] {
        &self.entries
    }

    pub fn header(&self) -> Option<&ObjectRef> {
        self.entries.iter().find_map(|(_, entry)| match entry {
            Entry::CozyTensors(blob) => Some(blob),
            Entry::File(_) => None,
        })
    }

    pub fn manifest_id(&self) -> String {
        self.object_id()
    }

    pub fn listings(&self, blob: &ObjectRef) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|(_, entry)| entry.blob() == blob)
            .map(|(path, _)| path.as_str())
            .collect()
    }

    fn validate(&self) -> Result<()> {
        if self.entries.is_empty() {
            return refuse(Code::MISSING_FIELD, "a manifest lists at least one entry");
        }
        if self.entries.len() > limits::MAX_ENTRIES {
            return refuse(Code::COUNT_CAP, "manifest entry count exceeds cap");
        }
        let mut previous: Option<&str> = None;
        let mut cozytensors = 0usize;
        for (path, entry) in &self.entries {
            check_path(path)?;
            if previous.is_some_and(|value| value.as_bytes() >= path.as_bytes()) {
                return refuse(
                    Code::PATH_ORDER,
                    format!("{path:?} does not strictly follow {previous:?}"),
                );
            }
            previous = Some(path);
            crate::ids::hex64("Manifest.entry.blob", &entry.blob().sha256)?;
            if entry.blob().length == 0 && matches!(entry, Entry::CozyTensors(_)) {
                return refuse(Code::LENGTH_MISMATCH, format!("{path:?}: zero-length blob"));
            }
            if matches!(entry, Entry::CozyTensors(_)) {
                cozytensors += 1;
            }
        }
        if cozytensors > 1 {
            return refuse(
                Code::ATTACHMENT_CARDINALITY,
                "a manifest may contain at most one cozytensors entry",
            );
        }
        Ok(())
    }
}

impl Doc for Manifest {
    const FORMAT: &'static str = "";
    const MAX_BYTES: usize = Manifest::MAX_BYTES;

    fn from_value(value: &Value) -> Result<Self> {
        let mut fields = Fields::new("Manifest", value)?;
        let raw = as_arr("Manifest", "entries", fields.req("entries")?)?;
        if raw.len() > limits::MAX_ENTRIES {
            return refuse(Code::COUNT_CAP, "manifest entry count exceeds cap");
        }
        let mut entries = Vec::with_capacity(raw.len());
        for value in raw {
            let mut item = Fields::new("Manifest.entry", value)?;
            let blob = ObjectRef::from_value("Manifest.entry.blob", item.req("blob")?)?;
            let kind = as_str("Manifest.entry", "kind", item.req("kind")?)?;
            let path = item.req_str("path")?.to_string();
            let entry = match kind {
                "file" => Entry::File(blob),
                "cozytensors" => Entry::CozyTensors(blob),
                other => {
                    return refuse(
                        Code::UNKNOWN_FIELD,
                        format!("manifest entry kind {other:?} is outside {{file,cozytensors}}"),
                    )
                }
            };
            item.done()?;
            entries.push((path, entry));
        }
        fields.done()?;
        Draft { entries }.seal()
    }

    fn to_value(&self) -> Value {
        manifest_value(&self.entries)
    }
}

fn manifest_value(entries: &[(String, Entry)]) -> Value {
    Value::obj(vec![(
        "entries",
        Value::arr(
            entries
                .iter()
                .map(|(path, entry)| {
                    Value::obj(vec![
                        ("blob", entry.blob().to_value()),
                        (
                            "kind",
                            Value::str(match entry {
                                Entry::File(_) => "file",
                                Entry::CozyTensors(_) => "cozytensors",
                            }),
                        ),
                        ("path", Value::str(path.clone())),
                    ])
                })
                .collect(),
        ),
    )])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_has_two_entry_kinds_and_no_format_tag() {
        let manifest = Draft {
            entries: vec![
                ("config.json".into(), Entry::File(ObjectRef::of(b"{}"))),
                (
                    "model.cozytensors".into(),
                    Entry::CozyTensors(ObjectRef::of(b"header")),
                ),
            ],
        }
        .seal()
        .unwrap();
        let bytes = manifest.canonical_bytes();
        assert!(!String::from_utf8_lossy(&bytes).contains("format"));
        assert_eq!(Manifest::parse(&bytes).unwrap(), manifest);
    }

    #[test]
    fn ordinary_files_include_empty_content() {
        let manifest = Manifest::from_files(vec![("empty.py".into(), ObjectRef::of(b""))]).unwrap();
        assert_eq!(
            Manifest::parse(&manifest.canonical_bytes()).unwrap(),
            manifest
        );
    }
}
