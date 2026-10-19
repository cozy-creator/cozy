//! TensorFS v2 checkpoints through the plane's `register`: the header, in its order (the
//! construction order, #624), becomes a read plan over the chunk files, and those files are the
//! weight set's source. A part's bytes keep their chunk's 4 KiB alignment in the layout, so a
//! region's reads are runs of whole chunks.

use std::collections::HashSet;
use std::fs::File;

use tensorfs_core::format::Header;
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::read::{ObjectRange, Order, PlanItem, ReadPlan, Source};
use tensorfs_core::store2::Checkpoint;

use crate::{Error, Result};

/// Every part of `components` (header order), named `component/key#role` (`value` for a plain
/// tensor's data): a part longer than the rest of its chunk is one item per chunk it spans.
pub fn read_plan(header: &Header, components: &[&str]) -> ReadPlan {
    let (mut items, mut dest) = (Vec::new(), 0);
    for c in header.components().iter().filter(|c| components.contains(&c.name.as_str())) {
        for t in &c.tensors {
            for (role, at) in t.locations() {
                let what = format!("{}/{}#{}", c.name, t.key, if role.is_empty() { "value" } else { role });
                let spans = header.spans(&at);
                if spans.is_empty() {
                    items.push(PlanItem { what, dest_off: dest, len: 0, source: Source::Inline(Vec::new()) });
                    continue;
                }
                for s in spans {
                    let obj = &header.chunks[s.chunk];
                    let obj = ObjectRef { sha256: obj.sha256.clone(), length: obj.length };
                    items.push(PlanItem { what: what.clone(), dest_off: dest, len: s.length, source: Source::Object(ObjectRange { obj, off: s.start, len: s.length }) });
                    dest += s.length;
                }
            }
        }
    }
    ReadPlan { io_bytes: dest, bytes: dest, items, inline_parts: 0, order: Order::Destination }
}

/// The chunk files `plan` reads, opened.
pub fn chunk_files(ck: &Checkpoint, plan: &ReadPlan) -> Result<Vec<(ObjectRef, File)>> {
    let mut used: HashSet<&str> = plan.items.iter().filter_map(|it| match &it.source {
        Source::Object(o) => Some(o.obj.sha256.as_str()),
        Source::Inline(_) => None,
    }).collect();
    let mut out = Vec::new();
    for (i, chunk) in ck.header().chunks.iter().enumerate() {
        if !used.remove(chunk.sha256.as_str()) {
            continue;
        }
        let path = &ck.chunk_paths()[i];
        let file = File::open(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        out.push((ObjectRef { sha256: chunk.sha256.clone(), length: chunk.length }, file));
    }
    Ok(out)
}
