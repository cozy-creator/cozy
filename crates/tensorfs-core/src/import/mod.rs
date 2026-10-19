//! Import: sources become one canonical model (normalize at ingest).
//!
//! 1. List and pin each source, and select what the model is built from (`remote`).
//! 2. Download remote members into `<tmp>/import/<id>/`, never the store; resumable, since
//!    `<id>` is the pinned selection's digest, and locked (`<id>.lock`) while one import uses it.
//!    Local members are read where they are.
//! 3. Fold quantized carriers into encoded tensors, by their producer's declaration
//!    (`carriers`). Recognize from safetensors headers alone: a built-in table claims a group
//!    when its key set is the table's (`engine`). Unclaimed groups are stored as they are.
//! 4. Convert row by row into the [`Sink`], with the configs and tokenizers (the table's
//!    pinned reference supplies them when the source has none), then delete the tmp area.
//!
//! A GGUF file is one component of its own (`gguf`).

mod bnb;
pub mod carriers;
pub mod engine;
pub mod gguf;
pub mod order;
pub mod remote;
pub mod source;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::err::{refuse, Code, Result};
use crate::format::{Dtype, Encoding};

pub use engine::{tables, Table};
pub use order::{constructor_order, Ordered};
pub use remote::Access;

/// One stored part of an output tensor, `data` yielding exactly its bytes.
pub struct Part<'a> {
    /// `""` for a plain tensor's one part.
    pub role: &'a str,
    pub dtype: &'a Dtype,
    pub shape: &'a [u64],
    pub data: &'a mut dyn Read,
}

/// Where an import's output goes: the writer, or anything that takes tensors and files.
pub trait Sink {
    /// One tensor: plain (no encoding, one part) or encoded (its parts in order).
    fn put(
        &mut self,
        component: &str,
        key: &str,
        dtype: &Dtype,
        shape: &[u64],
        encoding: Option<&Encoding>,
        parts: Vec<Part>,
    ) -> Result<()>;
    fn file(&mut self, path: &str, bytes: &[u8]) -> Result<()>;
    fn commit(&mut self, imported: &Imported) -> Result<()>;
}

/// The header metadata key of each component's source safetensors `__metadata__`
/// (`{component: {key: value}}`), which export writes back.
pub const FILE_METADATA: &str = "safetensors_metadata";

/// `__metadata__` keys that declare a quantized carrier, which the encodings now state.
const DECLARATIONS: [&str; 2] = ["_quantization_metadata", "quantization_config"];

/// What the model records about its import.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Imported {
    /// Every key is canonical: converted by a table, or the source's own Diffusers or
    /// Transformers layout. `false` means stored as-is from a layout nothing recognized.
    pub normalized: bool,
    pub converters: Vec<String>,
    /// The pinned sources.
    pub sources: Vec<String>,
    /// Each component's source safetensors `__metadata__` (LoRA training fields, `format`, …).
    pub file_metadata: BTreeMap<String, BTreeMap<String, String>>,
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
    /// Asked between members and tensors: true ends the import (it resumes from there).
    pub stop: Option<&'a (dyn Fn() -> bool + Sync)>,
}

fn stopped(req: &Request) -> Result<()> {
    match req.stop.is_some_and(|stop| stop()) {
        true => refuse(Code::TRANSFER_FAILED, "the import was stopped"),
        false => Ok(()),
    }
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

fn name(path: &str) -> &str {
    path.rsplit_once('/').map_or(path, |(_, n)| n)
}

fn is_gguf(path: &str) -> bool {
    path.to_ascii_lowercase().ends_with(".gguf")
}

/// Adds the configs, tokenizers and license at the root of the repository a GGUF names as its
/// source to `files`, in the GGUF's folder `into`, where the source has none; the pinned reference.
fn gguf_documents(
    reference: &str,
    into: &str,
    req: &Request,
    area: &Path,
    files: &mut BTreeMap<String, Vec<u8>>,
) -> Result<String> {
    let listing = remote::list(&remote::parse(reference)?, req.access)?;
    let pin = listing.pinned.clone();
    let read = |m: &remote::Member| remote::read_small(m, req.access);
    let docs: Vec<remote::Member> = remote::select(listing, None, remote::Files::Configs, &read)?
        .into_iter()
        .filter(|m| !remote::weights(&m.path) && folder(&m.path).is_empty())
        .collect();
    let area = area.join("gguf-reference");
    for (path, local) in land(&docs, &area, req.access, req.progress)? {
        let bytes = std::fs::read(&local)
            .map_err(|e| crate::err::classify_io(format!("read {path}"), e))?;
        let path = match into {
            "" => path,
            f => format!("{f}/{path}"),
        };
        files.entry(path).or_insert(bytes);
    }
    Ok(pin)
}

/// The model's files: every non-weight file of the source, byte-exact at its path, except a
/// converted folder's `config.json` (the foreign layout's); each table's reference (its index,
/// the produced components' configs and the asset folders); the tables' inline files.
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
        let foreign = converted.contains(&folder(&p.path)) && name(&p.path) == "config.json";
        if !raw.contains(&p.file) && !foreign && !is_gguf(&p.path) {
            files.insert(p.path.clone(), set.read_plain(p)?);
        }
    }
    let produced: Vec<&str> = planned.iter().map(|p| p.component.as_str()).collect();
    for t in &recognized.tables {
        let table = &tables[*t];
        if let Some(reference) = &table.reference {
            let listing = remote::list(&remote::parse(reference)?, req.access)?;
            let read = |m: &remote::Member| remote::read_small(m, req.access);
            let selected = remote::select(listing, None, remote::Files::Configs, &read)?;
            let weighted: Vec<&str> = selected
                .iter()
                .filter(|m| remote::weights(&m.path))
                .map(|m| folder(&m.path))
                .collect();
            let docs: Vec<remote::Member> = selected
                .iter()
                .filter(|m| {
                    let f = folder(&m.path);
                    !remote::weights(&m.path)
                        && (f.is_empty() || produced.contains(&f) || !weighted.contains(&f))
                })
                .cloned()
                .collect();
            for (path, local) in land(&docs, &area.join("reference"), req.access, req.progress)? {
                let bytes = std::fs::read(&local)
                    .map_err(|e| crate::err::classify_io(format!("read {path}"), e))?;
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
        let read = |m: &remote::Member| remote::read_small(m, req.access);
        for m in remote::select(listing, req.variant, remote::Files::All, &read)? {
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
        .map_err(|e| crate::err::classify_io("create the import area", e))?;
    // One import of a selection at a time: the kernel's lock, released when this ends.
    let _area = platform::lock::LockFile::new(area.with_extension("lock"))
        .lock(platform::lock::Kind::Exclusive)
        .map_err(|e| {
            crate::err::classify_io("lock the import area", std::io::Error::other(e.to_string()))
        })?;
    stopped(req)?;
    let landed = land(&members, &area.join("source"), req.access, progress)?;
    stopped(req)?;
    let mut set = source::SourceSet::scan(&landed)?;
    carriers::fold(&mut set)?;

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

    let ggufs: Vec<&source::Plain> = set.plain.iter().filter(|p| is_gguf(&p.path)).collect();
    let component = |f: &str, stem: &str| {
        let groups = set.groups.iter().filter(|o| o.folder == f).count();
        let shared = groups + ggufs.iter().filter(|p| folder(&p.path) == f).count() > 1;
        match (f, shared) {
            ("", false) => "model".to_string(),
            ("", true) => stem.to_owned(),
            (f, false) => f.to_string(),
            (f, true) => format!("{f}/{stem}"),
        }
    };
    let identity = |g: &source::Group| component(&g.folder, &g.stem);
    let planned = engine::plan(&set, tables, &recognized, &identity)?;
    let mut files = documents(&set, &recognized, &planned, req, &area)?;
    let (mut converters, mut normalized) = (converters, normalized);
    let mut ggufs_planned = Vec::new();
    for p in &ggufs {
        let file = name(&p.path);
        let component = component(folder(&p.path), &file[..file.len() - ".gguf".len()]);
        let local = &set.files[p.file];
        let opened = File::open(local)
            .map_err(|e| crate::err::classify_io(format!("open {}", local.display()), e))?;
        let plan = gguf::plan(&gguf::read(&opened)?, &component)?;
        normalized &= plan.normalized;
        converters.extend(plan.arch.as_ref().map(|a| format!("gguf.{a}")));
        let record = serde_json::to_vec(&plan.meta).expect("the record serializes");
        files.insert(format!("{}.json", p.path), record);
        if let Some(reference) = &plan.reference {
            let into = folder(&p.path);
            pinned.push(gguf_documents(reference, into, req, &area, &mut files)?);
        }
        ggufs_planned.push((p.file, component, plan));
    }
    let opened: Vec<File> = set
        .files
        .iter()
        .map(|f| {
            File::open(f).map_err(|e| crate::err::classify_io(format!("open {}", f.display()), e))
        })
        .collect::<Result<_>>()?;
    let gguf_tensors: usize = ggufs_planned.iter().map(|(_, _, p)| p.tensors.len()).sum();
    let total = (planned.len() + gguf_tensors) as u64;
    for (n, p) in planned.iter().enumerate() {
        stopped(req)?;
        progress("converting", n as u64, total);
        let what = format!("{}/{}", p.component, p.key);
        let mut readers = (p.parts.iter())
            .map(|part| {
                engine::TensorReader::new(&opened[part.read.file], &part.read, what.clone())
            })
            .collect::<Result<Vec<_>>>()?;
        let parts = (p.parts.iter().zip(&mut readers))
            .map(|(part, reader)| Part {
                role: &part.role,
                dtype: &part.dtype,
                shape: &part.shape,
                data: reader,
            })
            .collect();
        sink.put(
            &p.component,
            &p.key,
            &p.dtype,
            &p.shape,
            p.encoding.as_ref(),
            parts,
        )?;
    }
    for (file, component, plan) in &ggufs_planned {
        for o in &plan.tensors {
            stopped(req)?;
            let mut readers: Vec<gguf::Ranges> = (o.parts.iter())
                .map(|part| gguf::Ranges::new(&opened[*file], &part.ranges))
                .collect();
            let parts = (o.parts.iter().zip(&mut readers))
                .map(|(part, reader)| Part {
                    role: part.role,
                    dtype: &part.dtype,
                    shape: &part.shape,
                    data: reader,
                })
                .collect();
            let encoding = o.encoding.as_ref();
            sink.put(component, &o.key, &o.dtype, &o.shape, encoding, parts)?;
        }
    }
    for (path, bytes) in &files {
        sink.file(path, bytes)?;
    }
    let mut file_metadata: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    let named = (recognized.claims.iter())
        .map(|c| (c.name.clone(), c.group))
        .chain(recognized.unclaimed.iter().map(|g| (identity(&set.groups[*g]), *g)));
    for (component, g) in named {
        let kept = (set.groups[g].metadata.iter())
            .filter(|(k, _)| !DECLARATIONS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()));
        file_metadata.entry(component).or_default().extend(kept);
    }
    file_metadata.retain(|_, m| !m.is_empty());
    let imported = Imported {
        normalized,
        converters,
        sources: pinned,
        file_metadata,
    };
    sink.commit(&imported)?;
    drop(opened);
    match std::fs::remove_dir_all(&area) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            return Err(crate::err::classify_io(
                format!("remove {}", area.display()),
                e,
            ));
        }
        _ => {}
    }
    progress("imported", total, total);
    Ok(imported)
}
