//! TensorFS checkpoints through `register` and `seal`: the parts a weight set's regions name, in
//! header order (the construction order, #624), become a read plan over their chunk files, and
//! those files are the weight set's source. Nothing else of the checkpoint is walked or opened,
//! so a set registers in time proportional to its regions. A part's bytes keep their chunk's
//! 4 KiB alignment in the layout; a chunk two regions share is read once per region.

use std::fs::File;

use tensorfs::format::Header;
use tensorfs::ids::ObjectRef;
use tensorfs::read::{ObjectRange, Order, PlanItem, ReadPlan, Source};
use tensorfs::store2::Checkpoint;

use crate::{Error, Result};

/// A checkpoint as varena reads it: its header, and the stored checkpoint when a store holds
/// it. A header parsed alone (an executor's, granted by TensorD) has none: the grant's files
/// or a sealed host layout are its only source.
#[derive(Clone, Copy)]
pub struct Ck<'a> {
    pub header: &'a Header,
    pub stored: Option<&'a Checkpoint>,
}

impl<'a> From<&'a Checkpoint> for Ck<'a> {
    fn from(ck: &'a Checkpoint) -> Self {
        Ck { header: ck.header(), stored: Some(ck) }
    }
}

/// The parts `regions` name (`component/key` for all of a tensor's parts, `component/key#role`
/// for one, `value` being a plain tensor's data), in header order, each named
/// `component/key#role`; a part longer than the rest of its chunk is one item per chunk it
/// spans. Also the chunks they read (ascending `Header::chunks` indices).
pub fn read_plan(header: &Header, regions: &[Vec<String>]) -> Result<(ReadPlan, Vec<usize>)> {
    let mut wanted = Vec::new();
    for name in regions.iter().flatten() {
        wanted.push(locate(header, name).ok_or_else(|| Error::Invalid(format!("{name:?} is not in the checkpoint")))?);
    }
    wanted.sort_unstable();
    wanted.dedup();
    let (mut items, mut chunks, mut dest) = (Vec::new(), Vec::new(), 0);
    for (ci, ti, only) in wanted {
        let c = &header.components()[ci];
        let t = &c.tensors[ti];
        for (role, at) in t.locations() {
            let role = if role.is_empty() { "value" } else { role };
            if only.is_some_and(|r| r != role) {
                continue;
            }
            let what = format!("{}/{}#{role}", c.name, t.key);
            let spans = header.spans(&at);
            if spans.is_empty() {
                items.push(PlanItem { what, dest_off: dest, len: 0, source: Source::Inline(Vec::new()) });
                continue;
            }
            for s in spans {
                let obj = &header.chunks[s.chunk];
                let obj = ObjectRef { sha256: obj.sha256.clone(), length: obj.length };
                items.push(PlanItem { what: what.clone(), dest_off: dest, len: s.length, source: Source::Object(ObjectRange { obj, off: s.start, len: s.length }) });
                chunks.push(s.chunk);
                dest += s.length;
            }
        }
    }
    chunks.sort_unstable();
    chunks.dedup();
    Ok((ReadPlan { io_bytes: dest, bytes: dest, items, inline_parts: 0, order: Order::Destination }, chunks))
}

/// A region's name: (component index, tensor index, the one part it names, if so).
fn locate<'n>(header: &Header, name: &'n str) -> Option<(usize, usize, Option<&'n str>)> {
    let (component, key) = name.split_once('/')?;
    if let Some((ci, ti)) = header.position(component, key) {
        return Some((ci, ti, None));
    }
    let (key, role) = key.rsplit_once('#')?;
    let (ci, ti) = header.position(component, key)?;
    let t = &header.components()[ci].tensors[ti];
    let has = t.locations().iter().any(|(r, _)| *r == role || (r.is_empty() && role == "value"));
    has.then_some((ci, ti, Some(role)))
}

/// Chunks `chunks` (indices into `Header::chunks`) of `ck`, opened.
pub fn chunk_files(ck: &Checkpoint, chunks: &[usize]) -> Result<Vec<(ObjectRef, File)>> {
    chunks
        .iter()
        .map(|&i| {
            let (chunk, path) = (&ck.header().chunks[i], ck.chunk_path(i));
            let file = File::open(&path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
            Ok((ObjectRef { sha256: chunk.sha256.clone(), length: chunk.length }, file))
        })
        .collect()
}
