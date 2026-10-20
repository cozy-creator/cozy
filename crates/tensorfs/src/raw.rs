//! Advanced interop shared by language bindings and placement integrations.
//! Ordinary Rust clients drive async work on their own runtime.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::format::{Component, Header, Tensor};
use crate::{Failure, Result};

pub use crate::io::decode::{Output as DecodeOutput, Part as DecodePart};
pub use crate::session::PartData;

/// Check encoded-part descriptors through the shared format registry without reading data.
pub fn check_decode(
    encoding: &crate::format::Encoding,
    dtype: &crate::format::Dtype,
    shape: &[u64],
    parts: &[crate::format::decode::PartMetadata<'_>],
) -> Result<()> {
    crate::io::decode::check(encoding, dtype, shape, parts)
}

/// Decode existing part buffers into an existing, exactly typed destination.
///
/// # Safety
/// Every span must satisfy [`crate::io::decode::into`]'s validity, placement and stream
/// requirements. Retain all source and destination owners through checked completion.
/// After an error, retain them until [`crate::io::drain`] and every producer stream
/// succeed; an error does not prove quiescence. Output is unspecified after a runtime
/// or value error, while descriptor and overlap refusals precede writes.
#[allow(unsafe_code)]
pub unsafe fn decode_into(
    encoding: &crate::format::Encoding,
    dtype: &crate::format::Dtype,
    shape: &[u64],
    parts: &[DecodePart<'_>],
    out: DecodeOutput<'_>,
) -> Result<()> {
    // SAFETY: the caller supplies and retains all buffers under the contract above.
    unsafe { crate::io::decode::into(encoding, dtype, shape, parts, out) }
}

/// Download exactly these objects after checking that they belong to the checkpoint.
pub async fn fetch_objects(
    checkpoint: &crate::Checkpoint,
    objects: &[crate::format::ObjectRef],
    want: &crate::Download<'_>,
) -> Result<crate::Fetched> {
    checkpoint.fetch_objects_exact(objects, want).await
}

/// Fill caller-owned host or device destinations through the existing byte engine.
///
/// # Safety
/// Every source and destination must satisfy [`crate::io::read`]'s ownership and
/// pointer requirements. An error does not prove GPU completion: retain device
/// destinations and direct host/device source owners until [`crate::io::drain`]
/// succeeds for every participating device and any producer stream also drains.
#[allow(unsafe_code)]
pub unsafe fn read_into(
    checkpoint: &crate::Checkpoint,
    targets: &[crate::io::Target<'_>],
    options: &crate::io::ReadOpts,
) -> Result<crate::io::ReadStats> {
    // SAFETY: the caller supplies and retains every target under the contract above.
    unsafe { checkpoint.read_into_raw(targets, options) }
}

// For the Python binding only: runs `future` on the process-owned driver. Importing the
// binding neither calls it nor starts native workers.
#[doc(hidden)]
pub fn block_on<T>(future: impl std::future::Future<Output = Result<T>>) -> Result<T> {
    runtime()?.block_on(future)
}

pub(crate) fn runtime() -> Result<Arc<tokio::runtime::Runtime>> {
    crate::fetch::runtime().map_err(|error| {
        let detail = error.detail.clone();
        Failure {
            detail,
            ..Failure::from(error)
        }
    })
}

/// `header`'s tensors grouped into regions for a weight plane, as `<component>/<key>` names.
/// Tensors sharing a chunk stay in one group, its members in header order; groups follow the
/// header order of their first tensor, and neighbouring groups merge while their chunks total
/// at most `max_bytes` (a larger group is a region alone) and they hold the same `boundaries`.
/// A boundary is a `<component>/<key prefix>` module path (`prefix` or `prefix.*`), such as a
/// block: a region never holds tensors of two of them, or of one and of none, unless a shared
/// chunk does. A region never spans components.
///
/// `only`, when not empty, names what a load wants in the same form (blocks, or single
/// tensors): the work and the result cover just those tensors and the ones sharing a chunk with
/// them, never the rest of the checkpoint. A selection of whole boundaries gets the regions the
/// full call gives its tensors.
pub fn regions(header: &Header, max_bytes: u64, boundaries: &[String], only: &[String]) -> Vec<Vec<String>> {
    let split = |paths: &[String]| -> Vec<(String, String)> {
        let pairs = paths.iter().filter_map(|b| b.split_once('/'));
        pairs.map(|(c, p)| (c.to_owned(), p.to_owned())).collect()
    };
    let (boundaries, only) = (split(boundaries), split(only));
    let mut out = Vec::new();
    for c in header.components() {
        let of = |paths: &[(String, String)]| -> Vec<String> {
            paths.iter().filter(|(pc, _)| *pc == c.name).map(|(_, p)| p.clone()).collect()
        };
        let members = if only.is_empty() {
            (0..c.tensors.len()).collect()
        } else {
            let want = of(&only);
            if want.is_empty() {
                continue;
            }
            let picked = (0..c.tensors.len()).filter(|&i| want.iter().any(|p| under(&c.tensors[i].key, p)));
            sharing(header, &c.tensors, picked.collect())
        };
        out.extend(component_regions(header, c, max_bytes, &of(&boundaries), &members));
    }
    out
}

/// `key` is the module path `prefix` or inside it.
fn under(key: &str, prefix: &str) -> bool {
    key == prefix || key.strip_prefix(prefix).is_some_and(|rest| rest.starts_with('.'))
}

/// The chunks (first, last) each location of `t` covers.
fn chunk_ranges<'h>(header: &'h Header, t: &'h Tensor) -> impl Iterator<Item = (usize, usize)> + 'h {
    t.locations().into_iter().filter(|(_, at)| at.length > 0).map(|(_, at)| {
        let (mut chunk, mut end) = (at.chunk as usize, at.offset + at.length);
        while end > header.chunks[chunk].length {
            end -= header.chunks[chunk].length;
            chunk += 1;
        }
        (at.chunk as usize, chunk)
    })
}

/// `picked` (ascending tensor indices) and every tensor sharing a chunk with them, transitively.
fn sharing(header: &Header, tensors: &[Tensor], mut picked: Vec<usize>) -> Vec<usize> {
    let mut chunks: BTreeSet<usize> = BTreeSet::new();
    let mut taken = vec![false; tensors.len()];
    for &i in &picked {
        taken[i] = true;
        chunks.extend(chunk_ranges(header, &tensors[i]).flat_map(|(a, b)| a..=b));
    }
    loop {
        let before = picked.len();
        for (i, t) in tensors.iter().enumerate() {
            if taken[i] {
                continue;
            }
            let ranges: Vec<(usize, usize)> = chunk_ranges(header, t).collect();
            if ranges.iter().any(|&(a, b)| chunks.range(a..=b).next().is_some()) {
                taken[i] = true;
                picked.push(i);
                chunks.extend(ranges.into_iter().flat_map(|(a, b)| a..=b));
            }
        }
        if picked.len() == before {
            picked.sort_unstable();
            return picked;
        }
    }
}

/// [`regions`] of component `c` over `members` (ascending tensor indices, closed under chunk
/// sharing), with `boundaries` the component's key prefixes.
fn component_regions(header: &Header, c: &Component, max_bytes: u64, boundaries: &[String], members: &[usize]) -> Vec<Vec<String>> {
    let name = |i: usize| format!("{}/{}", c.name, c.tensors[i].key);
    // Groups of tensors (by index) and the chunks they use; the group each chunk is in.
    let mut groups: Vec<(Vec<usize>, BTreeSet<usize>)> = Vec::new();
    let mut empty: Vec<String> = Vec::new();
    let mut owner: HashMap<usize, usize> = HashMap::new();
    for &i in members {
        let chunks: BTreeSet<usize> = chunk_ranges(header, &c.tensors[i]).flat_map(|(a, b)| a..=b).collect();
        if chunks.is_empty() {
            empty.push(name(i));
            continue;
        }
        let mut joined: Vec<usize> = chunks.iter().filter_map(|k| owner.get(k).copied()).collect();
        joined.sort_unstable();
        joined.dedup();
        let g = match joined.split_first() {
            None => {
                groups.push((Vec::new(), BTreeSet::new()));
                groups.len() - 1
            }
            Some((&first, rest)) => {
                // Fold later groups into the earliest; their members keep header order.
                for &other in rest.iter().rev() {
                    let (m, used) = std::mem::take(&mut groups[other]);
                    groups[first].0.extend(m);
                    groups[first].1.extend(used);
                }
                groups[first].0.sort_unstable();
                first
            }
        };
        groups[g].0.push(i);
        groups[g].1.extend(chunks);
        for &k in &groups[g].1 {
            owner.insert(k, g);
        }
    }
    // The deepest boundary holding a key, if any.
    let label = |key: &str| boundaries.iter().filter(|p| under(key, p)).max_by_key(|p| p.len()).cloned();
    // Neighbours in the header: no tensor outside `members` lies between them.
    let rank = |i: usize| members.partition_point(|&m| m < i);
    // (names, chunk bytes, the boundaries its tensors are in, its last tensor)
    type Region = (Vec<String>, u64, BTreeSet<Option<String>>, usize);
    let mut regions: Vec<Region> = Vec::new();
    for (m, used) in groups.into_iter().filter(|g| !g.0.is_empty()) {
        let n: u64 = used.iter().map(|k| header.chunks[*k].length).sum();
        let labels: BTreeSet<Option<String>> = m.iter().map(|&i| label(&c.tensors[i].key)).collect();
        let (first, last) = (m[0], m[m.len() - 1]);
        let keys = m.into_iter().map(name);
        match regions.last_mut() {
            Some((r, bytes, l, end))
                if *bytes + n <= max_bytes && *l == labels && (first <= *end || first - *end == rank(first) - rank(*end)) =>
            {
                r.extend(keys);
                *bytes += n;
                *end = (*end).max(last);
            }
            _ => regions.push((keys.collect(), n, labels, last)),
        }
    }
    match regions.first_mut() {
        Some((r, ..)) => r.extend(empty),
        None if !empty.is_empty() => regions.push((empty, 0, BTreeSet::new(), 0)),
        None => {}
    }
    regions.into_iter().map(|(r, ..)| r).collect()
}
