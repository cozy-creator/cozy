//! `tfs export --safetensors`: a checkpoint as the Diffusers/Transformers folder it came from.
//!
//! - Repo files (README, LICENSE, images) are written as they are, assets byte-exact into their
//!   component's folder, and `model_index.json` and each component's config (`config.json`, or
//!   its `config_file`) as the HF writers write them ([`config_json`]): equal to the source's as
//!   parsed JSON.
//! - With a pipeline, a component is the folder of its name. Without one, the component
//!   [`ROOT_COMPONENT`] is the root (a Transformers folder), another with a config or assets is
//!   its folder, and any other is `<component>.safetensors` at the root.
//! - A component's weights are `<weights>.safetensors` in its folder, where `<weights>` is
//!   `diffusion_pytorch_model` for a Diffusers component and `model` otherwise.
//! - Files are sharded at [`SHARD`] bytes as `<weights>-00001-of-0000N.safetensors` beside the
//!   standard `<weights>.safetensors.index.json`. Each file's `__metadata__` is its component's
//!   source `__metadata__` ([`crate::import::FILE_METADATA`]), with `format: pt` when it has none.
//! - Each file is laid out as the safetensors library writes one: tensors by dtype (in its order,
//!   descending), then by name; the header lists them in that order with fields `dtype`, `shape`,
//!   `data_offsets`. A folder safetensors wrote is reproduced byte for byte, so its tensors sit at
//!   the same offsets: CPU kernels round differently on differently aligned weights, and a render
//!   from the export must equal upstream's exactly. Construction order stays in the checkpoint.
//! - An encoded tensor `x.weight` keeps its parts: `x.weight` (the `weight` part),
//!   `x.weight_scale`, `x.weight_scale_2`, `x.input_scale`, and `__metadata__`'s
//!   `cozytensors.encodings` names each one's scheme and params. With `dequantize`, it is decoded
//!   into that dtype instead.

mod gguf;

pub use gguf::gguf;

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value as Json};

use crate::client::io;
use crate::failure::{fail, Kind, Result};
use crate::format::{
    decode, variant_of, Body, Component, Config, ConfigValue, Dtype, Location, PartBytes, Value,
};
use crate::import::{FILE_METADATA, PIPELINE_FILE, ROOT_COMPONENT};
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

/// Writes `ck` into `dir` as a Diffusers/Transformers folder, with the repo files `repo` (path →
/// local file; a tree's files other than its variants); the paths written, relative.
pub fn safetensors(
    ck: &Checkpoint,
    repo: &BTreeMap<String, PathBuf>,
    dir: &Path,
    dequantize: Option<&Dtype>,
) -> Result<Vec<String>> {
    if let Some(dtype) = dequantize {
        if !matches!(dtype, Dtype::F32 | Dtype::BF16 | Dtype::F16) {
            return fail(
                Kind::Invalid,
                format!("dequantize to {dtype}: one of F32, BF16, F16"),
            );
        }
    }
    let header = ck.header();
    let file_metadata = match header.metadata.get(FILE_METADATA) {
        Some(Value::Map(m)) => m.clone(),
        _ => BTreeMap::new(),
    };
    let mut written = Vec::new();
    for (path, local) in repo.iter().filter(|(p, _)| variant_of(p).is_none()) {
        let dest = dir.join(path);
        copy(&mut File::open(local).map_err(|e| io(local, e))?, &dest)?;
        written.push(path.clone());
    }
    if let Some(pipeline) = header.pipeline() {
        write(&dir.join(PIPELINE_FILE), &config_json(&pipeline.config))?;
        written.push(PIPELINE_FILE.to_owned());
    }
    for c in header.components() {
        let component = &c.name;
        let bare = c.config.is_empty() && c.config_file.is_none() && c.assets.is_empty();
        let (folder, weights) = match (header.pipeline(), component.as_str()) {
            (None, ROOT_COMPONENT) => (String::new(), weights_name(c)),
            (None, name) if bare => (String::new(), name.to_owned()),
            _ => (format!("{component}/"), weights_name(c)),
        };
        if !c.config.is_empty() || c.config_file.is_some() {
            let path = format!("{folder}{}", c.config_name());
            write(&dir.join(&path), &config_json(&c.config))?;
            written.push(path);
        }
        for (name, object) in &c.assets {
            let path = format!("{folder}{name}");
            copy(&mut ck.object(object)?, &dir.join(&path))?;
            written.push(path);
        }
        if c.tensors.is_empty() {
            continue;
        }
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
        // The safetensors writer's own order (its `prepare`): dtype descending, then name.
        entries.sort_by(|a, b| b.dtype.cmp(&a.dtype).then_with(|| a.name.cmp(&b.name)));
        let mut seen = std::collections::HashSet::new();
        if let Some(e) = entries.iter().find(|e| !seen.insert(e.name.as_str())) {
            return fail(
                Kind::Invalid,
                format!("{component}: two tensors export as {}", e.name),
            );
        }
        let mut metadata: Map<String, Json> = match file_metadata.get(component) {
            Some(Value::Map(m)) => (m.iter())
                .filter_map(|(k, v)| Some((k.clone(), Json::from(v.as_str()?))))
                .collect(),
            _ => Map::new(),
        };
        metadata.entry("format").or_insert(Json::from("pt"));
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
        let mut weight_map = Config::new();
        let mut total = 0;
        for (i, shard) in shards.iter().enumerate() {
            let file = match n {
                1 => format!("{weights}.safetensors"),
                _ => format!("{weights}-{:05}-of-{n:05}.safetensors", i + 1),
            };
            for e in shard {
                weight_map.insert(e.name.clone(), ConfigValue::Str(file.clone()));
                total += e.length;
            }
            write_file(
                ck,
                component,
                shard,
                &metadata,
                &dir.join(&folder).join(&file),
            )?;
            written.push(format!("{folder}{file}"));
        }
        if n > 1 {
            let file = format!("{folder}{weights}.safetensors.index.json");
            let total = Config::from([("total_size".to_owned(), ConfigValue::Int(total.into()))]);
            let doc = Config::from([
                ("metadata".to_owned(), ConfigValue::Map(total)),
                ("weight_map".to_owned(), ConfigValue::Map(weight_map)),
            ]);
            write(&dir.join(&file), &config_json(&doc))?;
            written.push(file);
        }
    }
    Ok(written)
}

/// `diffusion_pytorch_model` for a Diffusers component, `model` for any other.
fn weights_name(c: &Component) -> String {
    match c.class.starts_with("diffusers.") || c.config.contains_key("_diffusers_version") {
        true => "diffusion_pytorch_model",
        false => "model",
    }
    .to_owned()
}

fn write(dest: &Path, bytes: &[u8]) -> Result<()> {
    let parent = dest.parent().expect("a file has a directory");
    fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
    fs::write(dest, bytes).map_err(|e| io(dest, e))
}

/// An asset or repo file, byte-exact.
fn copy(from: &mut File, dest: &Path) -> Result<()> {
    let parent = dest.parent().expect("a file has a directory");
    fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
    let mut out = File::create(dest).map_err(|e| io(dest, e))?;
    io::copy(from, &mut out).map_err(|e| io(dest, e))?;
    Ok(())
}

/// A config as the Diffusers and Transformers writers write one, `json.dumps(config, indent=2,
/// sort_keys=True) + "\n"`, with Python's float spelling (`1e-05`, `1.0`).
pub fn config_json(config: &Config) -> Vec<u8> {
    let mut out = String::new();
    map(&mut out, config, 0);
    out.push('\n');
    out.into_bytes()
}

fn map(out: &mut String, m: &Config, depth: usize) {
    nested(out, '{', '}', depth, m.iter(), |out, (k, v)| {
        out.push_str(&serde_json::to_string(k).expect("a string serializes"));
        out.push_str(": ");
        value(out, v, depth + 1);
    });
}

fn nested<T>(
    out: &mut String,
    open: char,
    close: char,
    depth: usize,
    items: impl ExactSizeIterator<Item = T>,
    mut each: impl FnMut(&mut String, T),
) {
    out.push(open);
    let empty = items.len() == 0;
    for (i, item) in items.enumerate() {
        out.push_str(if i == 0 { "\n" } else { ",\n" });
        out.push_str(&"  ".repeat(depth + 1));
        each(out, item);
    }
    if !empty {
        out.push('\n');
        out.push_str(&"  ".repeat(depth));
    }
    out.push(close);
}

fn value(out: &mut String, v: &ConfigValue, depth: usize) {
    match v {
        ConfigValue::Null => out.push_str("null"),
        ConfigValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        ConfigValue::Int(i) => out.push_str(&i.to_string()),
        ConfigValue::Float(f) => out.push_str(&python_float(*f)),
        ConfigValue::Str(s) => out.push_str(&serde_json::to_string(s).expect("a string serializes")),
        ConfigValue::List(l) => nested(out, '[', ']', depth, l.iter(), |out, v| value(out, v, depth + 1)),
        ConfigValue::Map(m) => map(out, m, depth),
    }
}

/// Python's `repr` of a float: the shortest round-trip digits (ryu's, the closest of them),
/// positional for decimal exponents -4 through 15, else `d.ddde±XX`.
fn python_float(f: f64) -> String {
    if !f.is_finite() {
        return match f {
            f if f.is_nan() => "NaN",
            f if f > 0.0 => "Infinity",
            _ => "-Infinity",
        }
        .to_owned();
    }
    let ryu = serde_json::to_string(&f).expect("a finite float");
    let (sign, ryu) = ryu.strip_prefix('-').map_or(("", ryu.as_str()), |r| ("-", r));
    let (mantissa, exp) = ryu.split_once('e').unwrap_or((ryu, "0"));
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let all = format!("{int}{frac}");
    let digits = all.trim_start_matches('0').trim_end_matches('0');
    if digits.is_empty() {
        return format!("{sign}0.0");
    }
    let lead = (all.len() - all.trim_start_matches('0').len()) as i32;
    let exp = exp.parse::<i32>().expect("an integer exponent") + int.len() as i32 - 1 - lead;
    if (-4..16).contains(&exp) {
        let point = exp + 1;
        if point <= 0 {
            return format!("{sign}0.{}{digits}", "0".repeat(-point as usize));
        }
        let point = point as usize;
        return match digits.len() <= point {
            true => format!("{sign}{digits}{}.0", "0".repeat(point - digits.len())),
            false => format!("{sign}{}.{}", &digits[..point], &digits[point..]),
        };
    }
    let mantissa = match digits.len() {
        1 => digits.to_owned(),
        _ => format!("{}.{}", &digits[..1], &digits[1..]),
    };
    format!("{sign}{mantissa}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
}

fn write_file(
    ck: &Checkpoint,
    component: &str,
    entries: &[Entry],
    metadata: &Map<String, Json>,
    dest: &Path,
) -> Result<()> {
    // As safetensors serializes its header: `__metadata__`, then each tensor in data order as
    // `{"dtype","shape","data_offsets"}`, padded with spaces to a multiple of 8.
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
    let parent = dest.parent().expect("a file has a directory");
    fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
    let temp = dest.with_extension(format!("tmp-{}", std::process::id()));
    let mut out = BufWriter::with_capacity(8 << 20, File::create(&temp).map_err(|e| io(&temp, e))?);
    let wrote = (|| -> Result<()> {
        out.write_all(&(text.len() as u64).to_le_bytes())?;
        out.write_all(&text)?;
        for e in entries {
            match &e.bytes {
                Bytes::Stored(at) => stored(ck, at, &mut out)?,
                Bytes::Decoded { key, dtype } => {
                    out.write_all(&decode_as(ck, component, key, dtype)?)?
                }
            }
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

/// Streams the stored bytes at `at` from their chunk files.
fn stored(ck: &Checkpoint, at: &Location, out: &mut impl Write) -> Result<()> {
    for span in ck.header().spans(at) {
        let path = &ck.chunk_path(span.chunk);
        let mut file = ck.chunk(span.chunk)?;
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
