//! Current D10 reader validation. Reads headers and small assets, never tensor chunks.
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
};
use tensorfs::format::{Header, ObjectRef};
fn visit(path: &Path, rows: &mut Vec<serde_json::Value>) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            visit(&entry.path(), rows)?;
        } else if entry.file_name() == "model.cozytensors" {
            let raw = fs::read(entry.path())?;
            let header = Header::from_cbor(&raw)?;
            let folder = entry
                .path()
                .parent()
                .ok_or("header has no folder")?
                .to_owned();
            let mut assets = 0;
            for (_, _, at) in header.assets() {
                let bytes = fs::read(folder.join("chunks").join(&at.sha256))?;
                if ObjectRef::of(&bytes) != *at {
                    return Err("D10 asset identity mismatch".into());
                }
                assets += 1;
            }
            rows.push(json!({"header": ObjectRef::of(&raw), "folder": folder,
                "components": header.components().iter().map(|c| &c.name).collect::<Vec<_>>(),
                "tensors": header.tensors().count(), "assets": assets,
                "retained_chunks": header.chunks.len(), "source": header.metadata.get("source")}));
        }
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("usage: d10_validate THIN_ROOT")?,
    );
    let mut rows = Vec::new();
    visit(&root, &mut rows)?;
    if rows.is_empty() {
        return Err("no thin D10 headers found".into());
    }
    println!(
        "{}",
        json!({"d10_source": option_env!("D10_NATIVE_SOURCE"), "headers": rows, "tensor_chunk_reads": 0})
    );
    Ok(())
}
