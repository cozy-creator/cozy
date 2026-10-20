//! The cozytensors level: a tensor's stored bytes become byte ranges of its chunk files. A
//! location longer than the rest of its chunk continues at offset 0 of the following chunks.

#[cfg(feature = "store")]
use std::collections::{btree_map::Entry, BTreeMap};

use super::Dst;
#[cfg(feature = "store")]
use super::{Copy, ReadOpts, ReadStats, Src};
#[cfg(feature = "store")]
use crate::format::{self, Body, Location, Tensor};
#[cfg(feature = "store")]
use crate::store2::{Checkpoint, Error, Result};

/// `dst.len()` stored bytes of one tensor from `offset`: a plain tensor's data (`role` `""`),
/// or one part of an encoded tensor.
#[derive(Clone, Copy, Debug)]
pub struct Target<'a> {
    pub component: &'a str,
    pub key: &'a str,
    pub role: &'a str,
    pub offset: u64,
    pub dst: Dst,
}

#[cfg(feature = "store")]
impl Checkpoint {
    /// Every target's bytes through the engine ([`super::read`]), each chunk opened once and
    /// read in chunk order.
    ///
    /// # Safety
    /// As [`super::read`].
    pub unsafe fn read_into(&self, targets: &[Target], opts: &ReadOpts) -> Result<ReadStats> {
        let mut spans: Vec<(usize, u64, Dst)> = Vec::new();
        for t in targets {
            let at = stored(self.tensor(t.component, t.key)?, t)?;
            let mut skip = t.offset;
            let mut done = 0;
            for s in self.header().spans(&at) {
                if done == t.dst.len() {
                    break;
                }
                if skip >= s.length {
                    skip -= s.length;
                    continue;
                }
                let n = ((s.length - skip) as usize).min(t.dst.len() - done);
                spans.push((s.chunk, s.start + skip, t.dst.slice(done, n)));
                (skip, done) = (0, done + n);
            }
        }
        spans.sort_by_key(|&(chunk, start, _)| (chunk, start));
        let mut files = BTreeMap::new();
        for &(chunk, ..) in &spans {
            if let Entry::Vacant(e) = files.entry(chunk) {
                e.insert(self.chunk(chunk)?);
            }
        }
        let copies: Vec<Copy> = (spans.iter())
            .map(|&(chunk, offset, dst)| Copy { src: Src::File(&files[&chunk], offset), dst })
            .collect();
        // SAFETY: per the contract.
        unsafe { super::read(&copies, opts) }.map_err(|e| Error::Io {
            op: "read",
            path: self.chunk_path(spans[e.copy].0),
            source: e.source,
        })
    }
}

/// The location `t` reads, checked to hold its range.
#[cfg(feature = "store")]
fn stored(tensor: &Tensor, t: &Target) -> Result<Location> {
    let what = || format!("{}/{}", t.component, t.key);
    let at = match (&tensor.body, t.role) {
        (Body::Data(at), "") => *at,
        (Body::Encoded { .. }, "") => {
            return Err(format::Error::Invalid(format!("{} is encoded: name a part", what())).into())
        }
        _ => match tensor.part(t.role) {
            Some(p) => p.at(),
            None => return Err(format::Error::Invalid(format!("{} has no part {:?}", what(), t.role)).into()),
        },
    };
    if t.offset.checked_add(t.dst.len() as u64).is_none_or(|end| end > at.length) {
        let why = format!("{}: {} bytes from {} of {}", what(), t.dst.len(), t.offset, at.length);
        return Err(format::Error::Invalid(why).into());
    }
    Ok(at)
}
