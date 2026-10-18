//! One byte layout per weight set, identical in a sealed host layout and in the device VA.
//! A region is a union of whole chunks: it starts on the VMM granularity so it maps on its own,
//! and its chunks sit at 4 KiB offsets in the given order, so a chunk reads with O_DIRECT in
//! place and moves to the device without repacking. Parts are 4 KiB aligned in their chunks
//! (cozytensors/2), so every part is a legal view of any dtype.

use std::collections::HashMap;

use tensorfs_core::format::{Body, Header};

use crate::{Error, Result};

pub const PART_ALIGN: u64 = 4096;
/// The VMM mapping granularity of every NVIDIA GPU the plane supports; devices reporting a
/// granularity that does not divide it are refused at open.
pub const REGION_ALIGN: u64 = 2 << 20;

pub fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

/// A chunk's place in the layout.
#[derive(Debug, Clone)]
pub struct ChunkAt {
    /// Its index in `Header::chunks`.
    pub index: u32,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone)]
pub struct Region {
    pub offset: u64,
    /// Bytes from `offset` to the end of the last chunk.
    pub nbytes: u64,
    /// `nbytes` rounded up to `REGION_ALIGN`: the mapped span.
    pub span: u64,
    pub chunks: Vec<ChunkAt>,
}

/// A plain tensor's data (`role` empty) or one part of an encoded tensor.
#[derive(Debug, Clone)]
pub struct Part {
    pub component: String,
    pub key: String,
    pub role: String,
    pub region: u32,
    pub offset: u64,
    pub nbytes: u64,
}

impl Part {
    /// `component/key`, or `component/key#role` for an encoded tensor's part.
    pub fn what(&self) -> String {
        match self.role.as_str() {
            "" => format!("{}/{}", self.component, self.key),
            role => format!("{}/{}#{role}", self.component, self.key),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Layout {
    pub regions: Vec<Region>,
    /// Every non-empty part whose chunks are registered, by region, then offset.
    pub parts: Vec<Part>,
    /// Total span (a multiple of `REGION_ALIGN`).
    pub nbytes: u64,
    /// Two layouts with the same digest place the same chunks at the same offsets: what a
    /// sealed host layout is checked against.
    pub digest: String,
}

impl Layout {
    /// `regions[r]` lists region `r`'s chunks (indexes into `header.chunks`) in layout order. No
    /// chunk may appear twice. A tensor is in the weight set when its chunks are; a tensor with
    /// only some of its chunks registered, or spread over regions, is refused.
    pub fn build(header: &Header, regions: &[Vec<u32>]) -> Result<Layout> {
        let mut owner: HashMap<u32, u32> = HashMap::new();
        let mut out = Vec::with_capacity(regions.len());
        let mut ident = String::new();
        let mut at = 0u64;
        for (r, chunks) in regions.iter().enumerate() {
            if chunks.is_empty() {
                return Err(Error::Invalid(format!("region {r} has no chunk")));
            }
            let mut placed = Vec::with_capacity(chunks.len());
            let mut end = at;
            for &c in chunks {
                let chunk = header.chunks.get(c as usize).ok_or_else(|| {
                    Error::Invalid(format!("region {r} names chunk {c} of {}", header.chunks.len()))
                })?;
                if owner.insert(c, r as u32).is_some() {
                    return Err(Error::Invalid(format!("chunk {c} is in two regions")));
                }
                let offset = align_up(end, PART_ALIGN);
                ident.push_str(&format!("{c}:{}:{}@{offset};", chunk.sha256, chunk.length));
                placed.push(ChunkAt { index: c, offset, length: chunk.length });
                end = offset + chunk.length;
            }
            let span = align_up((end - at).max(1), REGION_ALIGN);
            ident.push_str(&format!("region{r}={at}+{span};"));
            out.push(Region { offset: at, nbytes: end - at, span, chunks: placed });
            at += span;
        }
        let place: HashMap<u32, &ChunkAt> = out.iter().flat_map(|r| r.chunks.iter().map(|c| (c.index, c))).collect();
        let mut parts = Vec::new();
        for (component, c) in &header.components {
            for (key, t) in &c.tensors {
                let located: Vec<(&str, _)> = match &t.body {
                    Body::Data(at) => vec![("", *at)],
                    Body::Encoded { parts, .. } => parts.iter().map(|(role, p)| (role.as_str(), p.at())).collect(),
                };
                for (role, at) in located {
                    let spans = header.spans(&at);
                    let mine: Vec<_> = spans.iter().filter_map(|s| place.get(&(s.chunk as u32))).collect();
                    if mine.is_empty() {
                        continue;
                    }
                    let what = format!("{component}/{key}{}{role}", if role.is_empty() { "" } else { "#" });
                    // The bytes must be one contiguous run of the layout: every chunk registered,
                    // in one region, each continuing exactly where the previous one ends.
                    let contiguous = mine.len() == spans.len()
                        && mine.windows(2).all(|w| w[1].offset == w[0].offset + w[0].length)
                        && mine.iter().all(|c| owner[&c.index] == owner[&mine[0].index]);
                    if !contiguous {
                        return Err(Error::Invalid(format!("{what} is not one run of one region's chunks")));
                    }
                    parts.push(Part {
                        component: component.clone(),
                        key: key.clone(),
                        role: role.to_owned(),
                        region: owner[&mine[0].index],
                        offset: mine[0].offset + at.offset,
                        nbytes: at.length,
                    });
                }
            }
        }
        parts.sort_by_key(|p| (p.region, p.offset));
        Ok(Layout {
            regions: out,
            parts,
            nbytes: at,
            digest: tensorfs_core::sha256::hex_digest(ident.as_bytes()),
        })
    }

    pub fn region(&self, r: u32) -> Result<&Region> {
        self.regions
            .get(r as usize)
            .ok_or_else(|| Error::Invalid(format!("region {r} of {}", self.regions.len())))
    }
}
