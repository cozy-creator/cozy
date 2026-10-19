//! `tfs export --safetensors`: a checkpoint as the Diffusers/Transformers folder it came from.
//!
//! - Every manifest file but the header is written as it is: `model_index.json`, configs,
//!   tokenizers.
//! - A component in a folder (it has files there, or `model_index.json` lists it) is written to
//!   `<component>/<weights>.safetensors`, where `<weights>` is `diffusion_pytorch_model` for a
//!   Diffusers component and `model` otherwise. Any other component is `<component>.safetensors`
//!   at the root, which for `model` is a Transformers folder.
//! - A component imported from safetensors files is written back as those files: their paths
//!   (shards and variants included), their headers byte for byte and the tensors at their
//!   offsets, and their index (header metadata `safetensors`, [`crate::import::SOURCE_FILES`]).
//!   One whose tensors are no longer the source's (an encoded lane) takes the standard layout:
//! - Files are sharded at [`SHARD`] bytes as `<weights>-00001-of-0000N.safetensors` beside the
//!   standard `<weights>.safetensors.index.json`.
//! - Each file is laid out as the safetensors library writes one: tensors by dtype (in its order,
//!   descending), then by name; the header lists them in that order with fields `dtype`, `shape`,
//!   `data_offsets`. A folder safetensors wrote is reproduced byte for byte, so its tensors sit at
//!   the same offsets: CPU kernels round differently on differently aligned weights, and a render
//!   from the export must equal upstream's exactly. Construction order stays in the checkpoint.
//! - An encoded tensor `x.weight` keeps its parts: `x.weight` (the `weight` part),
//!   `x.weight_scale`, `x.weight_scale_2`, `x.input_scale`, and `__metadata__`'s
//!   `cozytensors.encodings` names each one's scheme and params. With `dequantize`, it is decoded
//!   into that dtype instead.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde_json::{json, Map, Value as Json};

use crate::client::io;
use crate::failure::{fail, Kind, Result};
use crate::format::{decode, Body, Dtype, Location, PartBytes, Value, HEADER_PATH};
use crate::import::SOURCE_FILES;
use crate::store2::Checkpoint;

/// The largest shard, as Diffusers and Transformers write them (20 GB).
pub const SHARD: u64 = 20_000_000_000;

/// Where one safetensors tensor's bytes come from.
enum Bytes<'a> {
    Stored(Location),
    Decoded { key: &'a str, dtype: Dtype },
}

struct Entry<'a> {
    name: String,
    dtype: Dtype,
    shape: Vec<u64>,
    length: u64,
    bytes: Bytes<'a>,
}

/// The safetensors name of part `role` of tensor `key`.
fn part_name(key: &str, role: &str) -> String {
    match role {
        "weight" => key.to_owned(),
        "input_scale" => match key.strip_suffix(".weight") {
            Some(base) => format!("{base}.input_scale"),
            None => format!("{key}_input_scale"),
        },
        role => format!("{key}_{role}"),
    }
}

/// Writes `ck` into `dir` as a Diffusers/Transformers folder; the paths written, relative.
pub fn safetensors(ck: &Checkpoint, dir: &Path, dequantize: Option<&Dtype>) -> Result<Vec<String>> {
    if let Some(dtype) = dequantize {
        if !matches!(dtype, Dtype::F32 | Dtype::BF16 | Dtype::F16) {
            return fail(
                Kind::Invalid,
                format!("dequantize to {dtype}: one of F32, BF16, F16"),
            );
        }
    }
    let files = &ck.manifest().files;
    let index: BTreeMap<String, Json> = match files.get("model_index.json") {
        Some(_) => serde_json::from_slice(&ck.file("model_index.json")?).unwrap_or_default(),
        None => BTreeMap::new(),
    };
    let layouts = recorded(ck);
    let mut written = Vec::new();
    for path in files.keys().filter(|p| *p != HEADER_PATH) {
        let dest = dir.join(path);
        let parent = dest.parent().expect("a file has a directory");
        fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
        fs::write(&dest, ck.file(path)?).map_err(|e| io(&dest, e))?;
        written.push(path.clone());
    }
    for c in ck.header().components() {
        let component = &c.name;
        let in_folder = index.contains_key(component)
            || files
                .keys()
                .any(|p| p.starts_with(&format!("{component}/")));
        let (folder, weights) = if in_folder {
            (format!("{component}/"), weights_name(ck, &index, component))
        } else {
            (String::new(), component.clone())
        };
        let mut entries = Vec::new();
        let mut encodings = Map::new();
        for t in &c.tensors {
            let key = &t.key;
            match (&t.body, dequantize) {
                (Body::Data(at), _) => entries.push(Entry {
                    name: key.clone(),
                    dtype: t.dtype.clone(),
                    shape: t.shape.clone(),
                    length: at.length,
                    bytes: Bytes::Stored(*at),
                }),
                (Body::Encoded { .. }, Some(dtype)) => {
                    let Some(length) = dtype.byte_len(&t.shape) else {
                        return fail(
                            Kind::Invalid,
                            format!("{component}/{key}: {:?} overflows", t.shape),
                        );
                    };
                    entries.push(Entry {
                        name: key.clone(),
                        dtype: dtype.clone(),
                        shape: t.shape.clone(),
                        length,
                        bytes: Bytes::Decoded {
                            key,
                            dtype: dtype.clone(),
                        },
                    })
                }
                (Body::Encoded { encoding, parts }, None) => {
                    let names: Map<String, Json> = parts
                        .iter()
                        .map(|p| (p.role.clone(), Json::from(part_name(key, &p.role))))
                        .collect();
                    encodings.insert(
                        key.clone(),
                        json!({"scheme": encoding.scheme, "params": encoding.params, "parts": names}),
                    );
                    for p in parts {
                        entries.push(Entry {
                            name: part_name(key, &p.role),
                            dtype: p.dtype.clone(),
                            shape: p.shape.clone(),
                            length: p.length,
                            bytes: Bytes::Stored(p.at()),
                        });
                    }
                }
            }
        }
        if let Some(e) = entries.iter().find(|e| matches!(e.dtype, Dtype::Other(_))) {
            return fail(
                Kind::Unsupported,
                format!(
                    "{component}/{}: safetensors has no dtype {}",
                    e.name, e.dtype
                ),
            );
        }
        if let Some(files) = layouts.get(component).and_then(|rec| placed(rec, &entries)) {
            for (path, text, spans) in files {
                write_file(ck, component, &text, &spans, &dir.join(&path))?;
                written.push(path);
            }
            if let Some((path, text)) = &layouts[component].index {
                let dest = dir.join(path);
                fs::write(&dest, text).map_err(|e| io(&dest, e))?;
                written.push(path.clone());
            }
            continue;
        }
        // The safetensors writer's own order (its `prepare`): dtype descending, then name.
        entries.sort_by(|a, b| b.dtype.cmp(&a.dtype).then_with(|| a.name.cmp(&b.name)));
        let mut seen = std::collections::HashSet::new();
        if let Some(e) = entries.iter().find(|e| !seen.insert(e.name.as_str())) {
            return fail(
                Kind::Invalid,
                format!("{component}: two tensors export as {}", e.name),
            );
        }
        let mut metadata = Map::from_iter([("format".to_owned(), Json::from("pt"))]);
        if !encodings.is_empty() {
            let text = serde_json::to_string(&encodings).expect("JSON values serialize");
            metadata.insert("cozytensors.encodings".to_owned(), Json::from(text));
        }
        let mut shards: Vec<Vec<Entry>> = vec![vec![]];
        let mut size = 0;
        for e in entries {
            if size > 0 && size + e.length > SHARD {
                shards.push(vec![]);
                size = 0;
            }
            size += e.length;
            shards.last_mut().expect("a shard is open").push(e);
        }
        let n = shards.len();
        let mut weight_map = Map::new();
        let mut total = 0;
        for (i, shard) in shards.iter().enumerate() {
            let file = match n {
                1 => format!("{weights}.safetensors"),
                _ => format!("{weights}-{:05}-of-{n:05}.safetensors", i + 1),
            };
            for e in shard {
                weight_map.insert(e.name.clone(), Json::from(file.clone()));
                total += e.length;
            }
            let refs: Vec<&Entry> = shard.iter().collect();
            let mut offset = 0;
            let spans: Vec<(u64, &Entry)> = (refs.iter())
                .map(|e| {
                    offset += e.length;
                    (offset - e.length, *e)
                })
                .collect();
            let text = header_text(&refs, &metadata);
            write_file(ck, component, &text, &spans, &dir.join(&folder).join(&file))?;
            written.push(format!("{folder}{file}"));
        }
        if n > 1 {
            let file = format!("{folder}{weights}.safetensors.index.json");
            let doc = json!({"metadata": {"total_size": total}, "weight_map": weight_map});
            let dest = dir.join(&file);
            fs::write(&dest, serde_json::to_vec_pretty(&doc).expect("JSON"))
                .map_err(|e| io(&dest, e))?;
            written.push(file);
        }
    }
    Ok(written)
}

/// `diffusion_pytorch_model` for a Diffusers component, `model` for any other.
fn weights_name(ck: &Checkpoint, index: &BTreeMap<String, Json>, component: &str) -> String {
    let diffusers = match index
        .get(component)
        .and_then(|v| v.get(0))
        .and_then(Json::as_str)
    {
        Some(library) => library == "diffusers",
        None => ck
            .file(&format!("{component}/config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Map<String, Json>>(&b).ok())
            .is_some_and(|c| c.contains_key("_diffusers_version") || c.contains_key("_class_name")),
    };
    if diffusers {
        "diffusion_pytorch_model"
    } else {
        "model"
    }
    .to_owned()
}

/// As safetensors serializes a header: `__metadata__`, then each tensor in data order as
/// `{"dtype","shape","data_offsets"}`, padded with spaces to a multiple of 8.
fn header_text(entries: &[&Entry], metadata: &Map<String, Json>) -> Vec<u8> {
    let string = |s: &str| serde_json::to_string(s).expect("a string serializes");
    let mut text = format!(
        "{{\"__metadata__\":{}",
        serde_json::to_string(metadata).expect("JSON")
    );
    let mut offset = 0;
    for e in entries {
        let shape = serde_json::to_string(&e.shape).expect("JSON");
        let end = offset + e.length;
        text += &format!(
            ",{}:{{\"dtype\":{},\"shape\":{shape},\"data_offsets\":[{offset},{end}]}}",
            string(&e.name),
            string(e.dtype.name())
        );
        offset = end;
    }
    text.push('}');
    let mut text = text.into_bytes();
    text.resize(text.len().next_multiple_of(8), b' ');
    text
}

/// A safetensors file: `text`, its header, then each entry at its offset in the data section
/// (gaps zero-filled).
fn write_file(
    ck: &Checkpoint,
    component: &str,
    text: &[u8],
    entries: &[(u64, &Entry)],
    dest: &Path,
) -> Result<()> {
    let parent = dest.parent().expect("a file has a directory");
    fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
    let temp = dest.with_extension(format!("tmp-{}", std::process::id()));
    let mut out = BufWriter::with_capacity(8 << 20, File::create(&temp).map_err(|e| io(&temp, e))?);
    let wrote = (|| -> Result<()> {
        out.write_all(&(text.len() as u64).to_le_bytes())?;
        out.write_all(text)?;
        let mut end = 0;
        for (offset, e) in entries {
            io::copy(&mut io::repeat(0).take(offset - end), &mut out)?;
            match &e.bytes {
                Bytes::Stored(at) => copy(ck, at, &mut out)?,
                Bytes::Decoded { key, dtype } => {
                    out.write_all(&decode_as(ck, component, key, dtype)?)?
                }
            }
            end = offset + e.length;
        }
        out.flush()?;
        Ok(())
    })();
    drop(out);
    if let Err(e) = wrote.and_then(|()| fs::rename(&temp, dest).map_err(|e| io(dest, e))) {
        let _ = fs::remove_file(&temp);
        return Err(e);
    }
    Ok(())
}

/// The source's own weights files of a component, as import recorded them (header metadata
/// `safetensors`): each file's path and header text, and the index's path and text.
struct Recorded {
    files: Vec<(String, String)>,
    index: Option<(String, String)>,
}

fn recorded(ck: &Checkpoint) -> BTreeMap<String, Recorded> {
    let text =
        |m: &BTreeMap<String, Value>, k: &str| m.get(k).and_then(Value::as_str).map(str::to_owned);
    let mut out = BTreeMap::new();
    let Some(Value::List(layouts)) = ck.header().metadata.get(SOURCE_FILES) else {
        return out;
    };
    for layout in layouts {
        let Value::Map(m) = layout else { continue };
        let (Some(component), Some(Value::List(files))) = (text(m, "component"), m.get("files"))
        else {
            continue;
        };
        let files = files
            .iter()
            .filter_map(|f| match f {
                Value::Map(f) => Some((text(f, "path")?, text(f, "header")?)),
                _ => None,
            })
            .collect();
        let index = match m.get("index") {
            Some(Value::Map(i)) => text(i, "path").zip(text(i, "text")),
            _ => None,
        };
        out.insert(component, Recorded { files, index });
    }
    out
}

/// Where each recorded file puts `entries`, when they are exactly its tensors (names, dtypes,
/// shapes, lengths): else the component no longer is the source's (an encoded lane), and is
/// written in the standard layout.
fn placed<'e>(
    rec: &Recorded,
    entries: &'e [Entry],
) -> Option<Vec<(String, Vec<u8>, Vec<(u64, &'e Entry)>)>> {
    let by_name: BTreeMap<&str, &Entry> = entries.iter().map(|e| (e.name.as_str(), e)).collect();
    let mut seen = std::collections::BTreeSet::new();
    let mut files = Vec::new();
    for (path, text) in &rec.files {
        let header: Map<String, Json> = serde_json::from_str(text).ok()?;
        let mut spans = Vec::new();
        for (name, t) in header.iter().filter(|(k, _)| *k != "__metadata__") {
            let e = by_name.get(name.as_str())?;
            let at: (u64, u64) = serde_json::from_value(t["data_offsets"].clone()).ok()?;
            let shape: Vec<u64> = serde_json::from_value(t["shape"].clone()).ok()?;
            if t["dtype"].as_str() != Some(e.dtype.name())
                || shape != e.shape
                || at.1.checked_sub(at.0) != Some(e.length)
            {
                return None;
            }
            if !seen.insert(name.as_str()) {
                return None;
            }
            spans.push((at.0, *e));
        }
        spans.sort_by_key(|(at, _)| *at);
        if spans.windows(2).any(|w| w[0].0 + w[0].1.length > w[1].0) {
            return None;
        }
        files.push((path.clone(), text.clone().into_bytes(), spans));
    }
    (seen.len() == entries.len()).then_some(files)
}

/// Streams the stored bytes at `at` from their chunk files.
fn copy(ck: &Checkpoint, at: &Location, out: &mut impl Write) -> Result<()> {
    for span in ck.header().spans(at) {
        let path = &ck.chunk_path(span.chunk);
        let mut file = ck.store().file(&ck.header().chunks[span.chunk])?;
        file.seek(SeekFrom::Start(span.start))
            .map_err(|e| io(path, e))?;
        let n = io::copy(&mut file.take(span.length), out).map_err(|e| io(path, e))?;
        if n != span.length {
            return fail(
                Kind::Integrity,
                format!("{}: {n} of {} bytes", path.display(), span.length),
            );
        }
    }
    Ok(())
}

/// An encoded tensor decoded into `dtype` (rounded once, from the scheme's f32 values).
fn decode_as(ck: &Checkpoint, component: &str, key: &str, dtype: &Dtype) -> Result<Vec<u8>> {
    let t = ck.tensor(component, key)?;
    let Body::Encoded { encoding, parts } = &t.body else {
        unreachable!("only encoded tensors are decoded")
    };
    let bytes = parts
        .iter()
        .map(|p| {
            let mut buf = vec![0; p.length as usize];
            ck.bytes_into(&p.at(), &mut buf).map(|()| buf)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let views = parts
        .iter()
        .zip(&bytes)
        .map(|(p, b)| {
            let view = PartBytes {
                dtype: &p.dtype,
                shape: &p.shape,
                bytes: b,
            };
            (p.role.as_str(), view)
        })
        .collect();
    Ok(decode(dtype, &t.shape, encoding, &views)?)
}
