//! A source on local disk, read from its safetensors headers only: tensor groups (a folder's
//! sharded checkpoint, or one loose file) and plain files.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::Deserialize;

use crate::err::{refuse, Code, Refusal, Result};
use crate::format::Dtype;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Index into `SourceSet::files`, and the tensor's first byte there.
    pub file: usize,
    pub start: u64,
    /// Bytes; an unknown dtype's are its declared range, read as they are.
    pub len: u64,
    /// A quantized carrier's tensors folded into one (`carriers`): `dtype` and `shape` are then
    /// the logical ones, and the location is the data part's.
    pub encoded: Option<Box<super::carriers::Folded>>,
}

#[derive(Debug)]
pub struct Group {
    /// The folder it sits in, relative to the source root (`""` at the root).
    pub folder: String,
    /// The loose file's name without `.safetensors`, or the index's stem.
    pub stem: String,
    pub tensors: BTreeMap<String, Tensor>,
    /// The files' safetensors `__metadata__`.
    pub metadata: BTreeMap<String, String>,
}

impl Group {
    pub fn label(&self) -> String {
        match self.folder.as_str() {
            "" => self.stem.clone(),
            folder => format!("{folder}/{}", self.stem),
        }
    }
}

#[derive(Debug)]
pub struct Plain {
    pub path: String,
    pub file: usize,
    pub len: u64,
}

#[derive(Debug, Default)]
pub struct SourceSet {
    pub files: Vec<PathBuf>,
    pub groups: Vec<Group>,
    pub plain: Vec<Plain>,
    digests: Mutex<BTreeMap<usize, String>>,
}

fn io(what: &Path, e: std::io::Error) -> Refusal {
    crate::err::classify_io(format!("read {}", what.display()), e)
}

#[derive(Deserialize)]
struct Entry {
    dtype: Dtype,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
}

/// A safetensors file's tensors, validated against its length (whole-byte sizes that fit the data
/// section and do not overlap), and its `__metadata__`.
pub fn read_header(path: &Path, file: usize) -> Result<Header> {
    let mut f = File::open(path).map_err(|e| io(path, e))?;
    let len_of_file = f.metadata().map_err(|e| io(path, e))?.len();
    let mut prefix = [0u8; 8];
    f.read_exact(&mut prefix).map_err(|e| io(path, e))?;
    let n = u64::from_le_bytes(prefix);
    if n > 100 << 20 || 8 + n > len_of_file {
        return refuse(
            Code::CARRIER_GEOMETRY,
            format!("{}: header length {n} outside the file", path.display()),
        );
    }
    let mut head = vec![0u8; n as usize];
    f.read_exact(&mut head).map_err(|e| io(path, e))?;
    let mut doc: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&head).map_err(|e| Refusal {
            code: Code::MALFORMED_JSON,
            detail: format!("{}: {e}", path.display()),
        })?;
    // Safetensors metadata is string to string; anything else is not read.
    let metadata = match doc.remove("__metadata__") {
        Some(serde_json::Value::Object(m)) => m
            .into_iter()
            .filter_map(|(k, v)| Some((k, v.as_str()?.to_owned())))
            .collect(),
        _ => BTreeMap::new(),
    };
    let data = 8 + n;
    let mut out = BTreeMap::new();
    let mut spans = Vec::new();
    for (key, value) in doc {
        let e: Entry = serde_json::from_value(value).map_err(|err| Refusal {
            code: Code::CARRIER_GEOMETRY,
            detail: format!("{}: {key}: {err}", path.display()),
        })?;
        let (begin, end) = e.data_offsets;
        let len = end.wrapping_sub(begin);
        let fits = match e.dtype.byte_len(&e.shape) {
            Some(want) => want == len,
            None => matches!(e.dtype, Dtype::Other(_)),
        };
        if !fits || end < begin || data + end > len_of_file {
            return refuse(
                Code::CARRIER_GEOMETRY,
                format!(
                    "{}: {key} {} {:?} at {begin}..{end}",
                    path.display(),
                    e.dtype,
                    e.shape
                ),
            );
        }
        spans.push((begin, end, key.clone()));
        out.insert(
            key,
            Tensor {
                dtype: e.dtype,
                shape: e.shape,
                file,
                start: data + begin,
                len,
                encoded: None,
            },
        );
    }
    spans.sort();
    if let Some(w) = spans.windows(2).find(|w| w[0].1 > w[1].0) {
        return refuse(
            Code::CARRIER_OVERLAP,
            format!("{}: {} overlaps {}", path.display(), w[0].2, w[1].2),
        );
    }
    Ok((out, metadata))
}

pub type Header = (BTreeMap<String, Tensor>, BTreeMap<String, String>);

fn split(path: &str) -> (&str, &str) {
    path.rsplit_once('/').unwrap_or(("", path))
}

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

impl SourceSet {
    /// The source from `(relative path, local file)` pairs: each `*.safetensors.index.json`
    /// with its shards is one group, every other `*.safetensors` file one group, and the rest
    /// plain files.
    pub fn scan(members: &[(String, PathBuf)]) -> Result<SourceSet> {
        let mut set = SourceSet::default();
        let by_path: BTreeMap<&str, &PathBuf> =
            members.iter().map(|(p, f)| (p.as_str(), f)).collect();
        let mut sharded = std::collections::BTreeSet::new();
        for (path, local) in members {
            let Some(stem) = path
                .strip_suffix(".json")
                .and_then(|p| p.split_once(".safetensors.index"))
                .map(|(s, _)| s)
            else {
                continue;
            };
            let bytes = std::fs::read(local).map_err(|e| io(local, e))?;
            let index: Index = serde_json::from_slice(&bytes).map_err(|e| Refusal {
                code: Code::MALFORMED_JSON,
                detail: format!("{path}: {e}"),
            })?;
            let (folder, stem) = split(stem);
            let mut group = Group {
                folder: folder.into(),
                stem: stem.into(),
                tensors: BTreeMap::new(),
                metadata: BTreeMap::new(),
            };
            let mut shards: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
            for (key, shard) in &index.weight_map {
                shards.entry(shard).or_default().push(key);
            }
            for (shard, keys) in shards {
                let shard_path = if folder.is_empty() {
                    shard.to_string()
                } else {
                    format!("{folder}/{shard}")
                };
                let Some(local) = by_path.get(shard_path.as_str()) else {
                    return refuse(
                        Code::MISSING_TENSOR,
                        format!("{path}: shard {shard_path} is not in the source"),
                    );
                };
                sharded.insert(shard_path.clone());
                let file = set.files.len();
                set.files.push(local.to_path_buf());
                let (tensors, metadata) = read_header(local, file)?;
                let mut named: Vec<&str> = keys.clone();
                named.sort();
                if tensors.keys().map(String::as_str).ne(named.iter().copied()) {
                    return refuse(
                        Code::MISSING_TENSOR,
                        format!("{shard_path}: its keys differ from {path}'s weight map"),
                    );
                }
                group.tensors.extend(tensors);
                group.metadata.extend(metadata);
            }
            set.groups.push(group);
        }
        for (path, local) in members {
            if path.ends_with(".safetensors") && !sharded.contains(path) {
                let file = set.files.len();
                set.files.push(local.clone());
                let (folder, name) = split(path);
                let stem = name.strip_suffix(".safetensors").unwrap_or(name);
                let (tensors, metadata) = read_header(local, file)?;
                set.groups.push(Group {
                    folder: folder.into(),
                    stem: stem.into(),
                    tensors,
                    metadata,
                });
            } else if !path.contains(".safetensors") {
                let len = std::fs::metadata(local).map_err(|e| io(local, e))?.len();
                set.plain.push(Plain {
                    path: path.clone(),
                    file: set.files.len(),
                    len,
                });
                set.files.push(local.clone());
            }
        }
        Ok(set)
    }

    /// The plain file whose bytes are `sha256` (hashed only when its length matches).
    pub fn find_raw(&self, sha256: &str, length: u64) -> Option<(usize, u64)> {
        for p in self.plain.iter().filter(|p| p.len == length) {
            let mut digests = self.digests.lock().unwrap_or_else(|e| e.into_inner());
            if let std::collections::btree_map::Entry::Vacant(e) = digests.entry(p.file) {
                e.insert(file_sha256(&self.files[p.file]).ok()?);
            }
            if digests[&p.file] == sha256 {
                return Some((p.file, p.len));
            }
        }
        None
    }

    pub fn read_plain(&self, p: &Plain) -> Result<Vec<u8>> {
        let path = &self.files[p.file];
        std::fs::read(path).map_err(|e| io(path, e))
    }

    /// The model class a group's `config.json` names: Diffusers' `_class_name`, or the first of
    /// Transformers' `architectures`.
    pub fn class_of(&self, group: &Group) -> Option<String> {
        let path = match group.folder.as_str() {
            "" => "config.json".to_owned(),
            folder => format!("{folder}/config.json"),
        };
        let plain = self.plain.iter().find(|p| p.path == path)?;
        let config: serde_json::Value =
            serde_json::from_slice(&self.read_plain(plain).ok()?).ok()?;
        let class = config
            .get("_class_name")
            .or_else(|| config.get("architectures")?.get(0));
        class?.as_str().map(str::to_owned)
    }
}

pub fn file_sha256(path: &Path) -> Result<String> {
    let mut f = File::open(path).map_err(|e| io(path, e))?;
    let mut hash = crate::sha256::Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        match f.read(&mut buf).map_err(|e| io(path, e))? {
            0 => return Ok(crate::sha256::hex(&hash.finish())),
            n => hash.update(&buf[..n]),
        }
    }
}
