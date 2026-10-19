//! `tfs export --gguf`: every GGUF a checkpoint was imported from, rebuilt from its record
//! (`<file>.json`, `import::gguf`): the KV and tensor infos in the source's order, then each
//! tensor's bytes from its stored parts (concatenated, llama's Q/K rows re-permuted), padded to
//! the alignment as llama.cpp and gguf-py write them.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::client::io;
use crate::failure::{fail, Kind, Result};
use crate::format::decode::gguf::Type;
use crate::format::Body;
use crate::import::gguf::{self, Meta};
use crate::store2::Checkpoint;

/// Writes each GGUF of `ck` into `dir`; the paths written, relative.
pub fn gguf(ck: &Checkpoint, dir: &Path) -> Result<Vec<String>> {
    let records: Vec<String> = (ck.manifest().files.keys())
        .filter(|p| p.to_ascii_lowercase().ends_with(".gguf.json"))
        .cloned()
        .collect();
    if records.is_empty() {
        return fail(Kind::Invalid, "the checkpoint was not imported from a GGUF");
    }
    let mut written = Vec::new();
    for record in records {
        let meta: Meta = match serde_json::from_slice(&ck.file(&record)?) {
            Ok(meta) => meta,
            Err(e) => return fail(Kind::Invalid, format!("{record}: {e}")),
        };
        let path = record
            .strip_suffix(".json")
            .expect("filtered on it")
            .to_owned();
        write(ck, &meta, &dir.join(&path))?;
        written.push(path);
    }
    Ok(written)
}

fn write(ck: &Checkpoint, meta: &Meta, dest: &Path) -> Result<()> {
    let invalid = |what: String| fail(Kind::Invalid, format!("{}: {what}", dest.display()));
    let mut infos = Vec::new();
    for row in &meta.tensors {
        match Type::by_name(&row.ty) {
            Some(ty) if ty.bytes(row.ne.iter().product()).is_some() => {
                infos.push((row.name.clone(), row.ne.clone(), ty))
            }
            _ => return invalid(format!("{} is not whole {} blocks", row.name, row.ty)),
        }
    }
    let alignment = gguf::alignment(&meta.kv)? as usize;
    let parent = dest.parent().expect("a file has a directory");
    fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
    let temp = dest.with_extension(format!("tmp-{}", std::process::id()));
    let mut out = BufWriter::with_capacity(8 << 20, File::create(&temp).map_err(|e| io(&temp, e))?);
    let wrote = (|| -> Result<()> {
        out.write_all(&gguf::header(meta.version, &meta.kv, &infos)?)
            .map_err(|e| io(&temp, e))?;
        for (row, (_, ne, ty)) in meta.tensors.iter().zip(&infos) {
            let mut bytes = Vec::new();
            for (component, key, role) in &row.from {
                let t = ck.tensor(component, key)?;
                let at = match (&t.body, role.as_str()) {
                    (Body::Data(at), "") => *at,
                    (Body::Encoded { parts, .. }, role) => match parts
                        .iter()
                        .find(|p| p.role == role)
                    {
                        Some(p) => p.at(),
                        None => return invalid(format!("{component}/{key} has no part {role}")),
                    },
                    _ => return invalid(format!("{component}/{key} is plain")),
                };
                let from = bytes.len();
                bytes.resize(from + at.length as usize, 0);
                ck.bytes_into(&at, &mut bytes[from..])?;
            }
            let want = ty.bytes(ne.iter().product()).expect("checked above");
            if bytes.len() as u64 != want {
                return invalid(format!(
                    "{}: {} bytes, {want} expected",
                    row.name,
                    bytes.len()
                ));
            }
            if let Some(heads) = row.permute {
                bytes = permute(&bytes, ne.get(1).copied().unwrap_or(ne[0]), heads);
            }
            out.write_all(&bytes).map_err(|e| io(&temp, e))?;
            let padding = bytes.len().next_multiple_of(alignment) - bytes.len();
            out.write_all(&vec![0; padding]).map_err(|e| io(&temp, e))?;
        }
        out.flush().map_err(|e| io(&temp, e))
    })();
    drop(out);
    if let Err(e) = wrote.and_then(|()| fs::rename(&temp, dest).map_err(|e| io(dest, e))) {
        let _ = fs::remove_file(&temp);
        return Err(e);
    }
    Ok(())
}

/// Upstream rows back into llama's converter order: GGUF row `r` is upstream row `perm⁻¹(r)`.
fn permute(upstream: &[u8], rows: u64, heads: u64) -> Vec<u8> {
    let row = upstream.len() / rows as usize;
    let mut out = vec![0; upstream.len()];
    for u in 0..rows {
        let g = gguf::permuted(rows, heads, u) as usize;
        out[g * row..(g + 1) * row]
            .copy_from_slice(&upstream[u as usize * row..(u as usize + 1) * row]);
    }
    out
}
