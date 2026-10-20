//! One byte layout per weight set, identical in every tier: host memfd offsets, device VA
//! offsets and offsets inside a streamed copy are the same numbers. Regions start on the VMM granularity so
//! each maps independently; parts start on a 4 KiB boundary so O_DIRECT lands in place and a
//! region moves in one copy.

#[cfg(feature = "store")]
use std::collections::{HashMap, HashSet};

#[cfg(feature = "store")]
use tensorfs::io::Span;

use crate::{Error, Result};

pub const PART_ALIGN: u64 = 4096;
/// The VMM mapping granularity of every NVIDIA GPU the plane supports; devices reporting a
/// granularity that does not divide it are refused at open.
pub const REGION_ALIGN: u64 = 2 << 20;

pub fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

pub use tensorfs::io::source::{Item, ItemSource};

#[derive(Debug, Clone)]
pub struct Part {
    pub what: String,
    pub region: u32,
    pub offset: u64,
    pub nbytes: u64,
}

#[derive(Debug, Clone)]
pub struct Region {
    pub offset: u64,
    /// Bytes that carry data (last part's end, relative to `offset`).
    pub nbytes: u64,
    /// `nbytes` rounded up to `REGION_ALIGN`: the mapped span.
    pub span: u64,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub struct Layout {
    pub parts: Vec<Part>,
    pub regions: Vec<Region>,
    /// Total span (a multiple of `REGION_ALIGN`).
    pub nbytes: u64,
    /// Identity of the layout: two layouts with the same digest place the same bytes at the
    /// same offsets. Guards adoption of a host memfd written by another process.
    pub digest: String,
    /// The `regions` it was built from; a sealed host layout carries them (`host::sealed_regions`).
    pub names: Vec<Vec<String>>,
}

#[cfg(feature = "store")]
fn tensor_of(what: &str) -> &str {
    what.rsplit_once('#').map_or(what, |(t, _)| t)
}

impl Layout {
    /// `regions[r]` names the tensors ("component/key", every role) or single parts
    /// ("component/key#role") of region `r`. Every part of `spans` belongs to exactly one
    /// region; a region's parts keep their (construction) order.
    #[cfg(feature = "store")]
    pub fn build(spans: &[Span], regions: &[Vec<String>]) -> Result<Layout> {
        let mut owner: HashMap<&str, u32> = HashMap::new();
        for (r, tensors) in regions.iter().enumerate() {
            if tensors.is_empty() {
                return Err(Error::Invalid(format!("region {r} names no tensor")));
            }
            for t in tensors {
                if owner.insert(t.as_str(), r as u32).is_some() {
                    return Err(Error::Invalid(format!("tensor {t:?} is in two regions")));
                }
            }
        }
        // Group spans by part, preserving their order.
        let mut parts: Vec<(&str, Vec<&Span>)> = Vec::new();
        for it in spans {
            match parts.last_mut() {
                Some((w, v)) if *w == it.what => v.push(it),
                _ => parts.push((&it.what, vec![it])),
            }
        }
        let mut seen: HashSet<&str> = HashSet::new();
        let mut per_region: Vec<Vec<usize>> = vec![Vec::new(); regions.len()];
        for (i, (what, _)) in parts.iter().enumerate() {
            let (key, r) = match owner.get_key_value(*what).or_else(|| owner.get_key_value(tensor_of(what))) {
                Some((k, r)) => (*k, *r),
                None => return Err(Error::Invalid(format!("part {what:?} is in no region"))),
            };
            if owner.contains_key(*what) && owner.contains_key(tensor_of(what)) {
                return Err(Error::Invalid(format!("{what:?} is named both as a part and by its tensor")));
            }
            seen.insert(key);
            per_region[r as usize].push(i);
        }
        if let Some(t) = owner.keys().find(|t| !seen.contains(*t)) {
            return Err(Error::Invalid(format!("region names {t:?}, which the checkpoint does not carry")));
        }

        let mut out_parts = Vec::with_capacity(parts.len());
        let mut out_regions = Vec::with_capacity(regions.len());
        let mut at = 0u64;
        let mut ident = String::new();
        for (r, idx) in per_region.iter().enumerate() {
            let start = at;
            let mut items = Vec::new();
            let mut end = start;
            for &i in idx {
                let (what, its) = &parts[i];
                let off = align_up(end, PART_ALIGN);
                let mut n = 0u64;
                for (j, it) in its.iter().enumerate() {
                    items.push(Item {
                        offset: off + n,
                        len: it.len,
                        source: it.source.clone(),
                        part_end: j + 1 == its.len(),
                    });
                    match &it.source {
                        ItemSource::Object(o) => ident.push_str(&format!("{}:{}:{}@{};", o.obj.sha256, o.off, o.len, off + n)),
                        ItemSource::Inline(bytes) => ident.push_str(&format!("inline:{}:{}@{};", tensorfs::sha256::hex_digest(bytes), it.len, off + n)),
                    }
                    n += it.len;
                }
                out_parts.push(Part {
                    what: what.to_string(),
                    region: r as u32,
                    offset: off,
                    nbytes: n,
                });
                ident.push_str(&format!("{what}={off}+{n};"));
                end = off + n;
            }
            let nbytes = end - start;
            let span = align_up(nbytes.max(1), REGION_ALIGN);
            ident.push_str(&format!("region{r}={start}+{span};"));
            out_regions.push(Region {
                offset: start,
                nbytes,
                span,
                items,
            });
            at = start + span;
        }
        Ok(Layout {
            parts: out_parts,
            regions: out_regions,
            nbytes: at,
            digest: tensorfs::sha256::hex_digest(ident.as_bytes()),
            names: regions.to_vec(),
        })
    }

    pub fn region(&self, r: u32) -> Result<&Region> {
        self.regions
            .get(r as usize)
            .ok_or_else(|| Error::Invalid(format!("region {r} of {}", self.regions.len())))
    }
}
