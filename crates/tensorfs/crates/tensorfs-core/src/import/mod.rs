//! Import: sources become one canonical model (normalize at ingest).
//!
//! 1. List and pin each source, and select what the model is built from (`remote`).
//! 2. Download remote members into `<tmp>/import/<id>/`, never the store; resumable, since
//!    `<id>` is the pinned selection's digest, and locked (`<id>.lock`) while one import uses it.
//!    Local members are read where they are.
//! 3. Recognize from safetensors headers alone: a built-in table claims a group when its key
//!    set is the table's (`engine`). Unclaimed groups are stored as they are.
//! 4. Convert row by row into the [`Sink`], with the configs and tokenizers (the table's
//!    pinned reference supplies them when the source has none), then delete the tmp area.

pub mod engine;
pub mod remote;
pub mod source;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::err::{refuse, Code, Result};
use crate::format::Dtype;

pub use engine::{tables, Table};
pub use remote::Access;

/// Where an import's output goes: the writer, or anything that takes tensors and files.
pub trait Sink {
    /// One tensor, `data` yielding exactly its bytes.
    fn put(
        &mut self,
        component: &str,
        key: &str,
        dtype: &Dtype,
        shape: &[u64],
        data: &mut dyn Read,
    ) -> Result<()>;
    fn file(&mut self, path: &str, bytes: &[u8]) -> Result<()>;
    fn commit(&mut self, imported: &Imported) -> Result<()>;
}

/// What the model records about its import.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Imported {
    /// Every key is canonical: converted by a table, or the source's own Diffusers or
    /// Transformers layout. `false` means stored as-is from a layout nothing recognized.
    pub normalized: bool,
    pub converters: Vec<String>,
    /// The pinned sources.
    pub sources: Vec<String>,
}

pub struct Request<'a> {
    pub sources: &'a [String],
    /// A table to convert with; required for tables recognition never picks alone.
    pub converter: Option<&'a str>,
    /// A Diffusers weights variant (`fp16`), when a folder has several.
    pub variant: Option<&'a str>,
    pub access: &'a Access,
    /// The store's tmp directory.
    pub tmp: &'a Path,
    pub progress: &'a (dyn Fn(&str, u64, u64) + Sync),
}

fn digest(parts: &[&str]) -> String {
    crate::sha256::hex_digest(parts.join("\n").as_bytes())
}

/// Lands `members` under `area`: `(path, local file)` pairs.
fn land(
    members: &[remote::Member],
    area: &Path,
    access: &Access,
    progress: &(dyn Fn(&str, u64, u64) + Sync),
) -> Result<Vec<(String, PathBuf)>> {
    let total = members.iter().filter_map(|m| m.length).sum();
    let mut done = 0;
    let mut out = Vec::new();
    for m in members {
        progress("downloading", done, total);
        out.push((
            m.path.clone(),
            remote::land(m, &area.join(&m.path), access)?,
        ));
        done += m.length.unwrap_or(0);
    }
    progress("downloading", done, total);
    Ok(out)
}

fn folder(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(f, _)| f)
}

/// The model's documents: the source's own (not those of converted folders, whose configs
/// are the foreign layout's), each table's reference (its index, the produced components'
/// configs and the asset folders), and the tables' inline files.
fn documents(
    set: &source::SourceSet,
    recognized: &engine::Recognized,
    planned: &[engine::Planned],
    req: &Request,
    area: &Path,
) -> Result<BTreeMap<String, Vec<u8>>> {
    let tables = tables();
    let converted: Vec<&str> = recognized
        .claims
        .iter()
        .map(|c| set.groups[c.group].folder.as_str())
        .collect();
    let raw: Vec<usize> = recognized
        .tables
        .iter()
        .flat_map(|t| &tables[*t].components)
        .flat_map(|c| &c.rows)
        .filter_map(|r| match &r.source {
            engine::Src::Raw(raw) => set.find_raw(&raw.sha256, raw.length).map(|(f, _)| f),
            engine::Src::Key(_) => None,
        })
        .collect();
    let mut files = BTreeMap::new();
    for p in &set.plain {
        if !raw.contains(&p.file)
            && remote::document(&p.path)
            && !converted.contains(&folder(&p.path))
        {
            files.insert(p.path.clone(), set.read_plain(p)?);
        }
    }
    let produced: Vec<&str> = planned.iter().map(|p| p.component.as_str()).collect();
    for t in &recognized.tables {
        let table = &tables[*t];
        if let Some(reference) = &table.reference {
            let listing = remote::list(&remote::parse(reference)?, req.access)?;
            let selected = remote::select(listing, None, &|m| remote::read_small(m, req.access))?;
            let weighted: Vec<&str> = selected
                .iter()
                .filter(|m| !remote::document(&m.path))
                .map(|m| folder(&m.path))
                .collect();
            let docs: Vec<remote::Member> = selected
                .iter()
                .filter(|m| {
                    let f = folder(&m.path);
                    remote::document(&m.path)
                        && (f.is_empty() || produced.contains(&f) || !weighted.contains(&f))
                })
                .cloned()
                .collect();
            for (path, local) in land(&docs, &area.join("reference"), req.access, req.progress)? {
                let bytes = std::fs::read(&local)
                    .map_err(|e| crate::store::classify_io(format!("read {path}"), e))?;
                files.entry(path).or_insert(bytes);
            }
        }
        for (path, doc) in &table.files {
            files.insert(path.clone(), serde_json::to_vec(doc).expect("a JSON value"));
        }
    }
    Ok(files)
}

pub fn import(req: &Request, sink: &mut dyn Sink) -> Result<Imported> {
    let progress = req.progress;
    progress("resolving", 0, 0);
    let mut pinned = Vec::new();
    let mut members: Vec<remote::Member> = Vec::new();
    for raw in req.sources {
        let listing = remote::list(&remote::parse(raw)?, req.access)?;
        pinned.push(listing.pinned.clone());
        for m in remote::select(listing, req.variant, &|m| remote::read_small(m, req.access))? {
            if members.iter().any(|o| o.path == m.path) {
                return refuse(
                    Code::DUPLICATE_KEY,
                    format!("two sources both hold {}", m.path),
                );
            }
            members.push(m);
        }
    }
    let id = digest(&[
        &pinned.join("\n"),
        req.variant.unwrap_or(""),
        req.converter.unwrap_or(""),
    ]);
    let area = req.tmp.join("import").join(&id[..32]);
    std::fs::create_dir_all(req.tmp.join("import"))
        .map_err(|e| crate::store::classify_io("create the import area", e))?;
    // One import of a selection at a time: the kernel's lock, released when this ends.
    let _area = cozy_platform::lock::LockFile::new(area.with_extension("lock"))
        .lock(cozy_platform::lock::Kind::Exclusive)
        .map_err(|e| {
            crate::store::classify_io("lock the import area", std::io::Error::other(e.to_string()))
        })?;
    let landed = land(&members, &area.join("source"), req.access, progress)?;
    let set = source::SourceSet::scan(&landed)?;

    let tables = tables();
    let recognized = engine::recognize(&set, tables, req.converter)?;
    let converters: Vec<String> = recognized
        .tables
        .iter()
        .map(|t| tables[*t].name.clone())
        .collect();
    // The source's own Diffusers or Transformers layout is canonical by definition.
    let canonical = set.plain.iter().any(|p| {
        let (folder, name) = p.path.rsplit_once('/').unwrap_or(("", &p.path));
        name.ends_with("model_index.json")
            || (name == "config.json" && set.groups.iter().any(|g| g.folder == folder))
    });
    let normalized = recognized.unclaimed.is_empty() || canonical;

    let identity = |g: &source::Group| {
        let shared = set.groups.iter().filter(|o| o.folder == g.folder).count() > 1;
        match (g.folder.as_str(), shared) {
            ("", false) => "model".to_string(),
            ("", true) => g.stem.clone(),
            (f, false) => f.to_string(),
            (f, true) => format!("{f}/{}", g.stem),
        }
    };
    let planned = engine::plan(&set, tables, &recognized, &identity)?;
    let files = documents(&set, &recognized, &planned, req, &area)?;
    let opened: Vec<File> = set
        .files
        .iter()
        .map(|f| {
            File::open(f).map_err(|e| crate::store::classify_io(format!("open {}", f.display()), e))
        })
        .collect::<Result<_>>()?;
    let total = planned.len() as u64;
    for (n, p) in planned.iter().enumerate() {
        progress("converting", n as u64, total);
        let mut reader = engine::TensorReader::new(&opened[p.read.file], p)?;
        sink.put(&p.component, &p.key, &p.dtype, &p.shape, &mut reader)?;
    }
    for (path, bytes) in &files {
        sink.file(path, bytes)?;
    }
    let imported = Imported {
        normalized,
        converters,
        sources: pinned,
    };
    sink.commit(&imported)?;
    drop(opened);
    match std::fs::remove_dir_all(&area) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            return Err(crate::store::classify_io(
                format!("remove {}", area.display()),
                e,
            ));
        }
        _ => {}
    }
    progress("imported", total, total);
    Ok(imported)
}
