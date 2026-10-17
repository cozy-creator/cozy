//! The writer's chunk rule (spec "Chunks", informative): a pure planner from ordered parts to
//! chunk lengths and locations. Readers never need it; they follow offsets.

use std::cmp::Ordering;

use super::Location;

/// A part this long or longer is its own object(s).
pub const STANDALONE: u64 = 16 << 20;
/// Standalone parts split every `GRID` bytes from their first byte.
pub const GRID: u64 = 64 << 20;
/// A pack never grows past this.
pub const PACK: u64 = 64 << 20;
/// A pack holding this much closes at the next block boundary.
pub const BLOCK_CLOSE: u64 = 16 << 20;
/// Every packed part starts at a multiple of this.
pub const ALIGN: u64 = 4096;

/// One stored part: a plain tensor's data (`role` empty) or one part of an encoded tensor.
#[derive(Debug, Clone, Copy)]
pub struct Item<'a> {
    pub component: &'a str,
    pub key: &'a str,
    pub role: &'a str,
    pub length: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Components by name, then keys in natural order (block indices ascend), then roles.
    Natural,
    /// The items' own order: the writer's explicit execution order.
    Given,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// Chunk lengths, in `Header::chunks` order.
    pub chunks: Vec<u64>,
    /// Where each input item lives, in input order. Empty items get `{0, 0, 0}`.
    pub places: Vec<Location>,
}

pub fn plan(items: &[Item], order: Order) -> Plan {
    let mut idx: Vec<usize> = (0..items.len()).collect();
    if order == Order::Natural {
        idx.sort_by(|&a, &b| {
            let (x, y) = (&items[a], &items[b]);
            (x.component.cmp(y.component))
                .then_with(|| natural_cmp(x.key, y.key))
                .then_with(|| x.role.cmp(y.role))
        });
    }
    let mut chunks: Vec<u64> = Vec::new();
    let mut places = vec![Location::default(); items.len()];
    // The open pack: its chunk, component, and the block of its last part.
    let mut pack: Option<(usize, &str, &str)> = None;
    for i in idx {
        let it = &items[i];
        if it.length == 0 {
            continue;
        }
        if it.length >= STANDALONE {
            places[i] = Location {
                chunk: chunks.len() as u64,
                offset: 0,
                length: it.length,
            };
            let mut left = it.length;
            while left > 0 {
                chunks.push(left.min(GRID));
                left -= left.min(GRID);
            }
            continue;
        }
        let block = block_of(it.key);
        let fits = pack.and_then(|(c, component, last)| {
            let start = chunks[c].next_multiple_of(ALIGN);
            let close = component != it.component
                || (last != block && chunks[c] >= BLOCK_CLOSE)
                || start + it.length > PACK;
            (!close).then_some((c, start))
        });
        let (c, offset) = fits.unwrap_or_else(|| {
            chunks.push(0);
            (chunks.len() - 1, 0)
        });
        chunks[c] = offset + it.length;
        places[i] = Location {
            chunk: c as u64,
            offset,
            length: it.length,
        };
        pack = Some((c, it.component, block));
    }
    Plan { chunks, places }
}

impl Plan {
    /// The chunks' bytes, given each item's bytes in input order (zero padding between parts).
    pub fn assemble(&self, data: &[&[u8]]) -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = self.chunks.iter().map(|n| vec![0; *n as usize]).collect();
        for (at, bytes) in self.places.iter().zip(data) {
            let (mut chunk, mut offset, mut rest) = (at.chunk as usize, at.offset as usize, *bytes);
            while !rest.is_empty() {
                let n = rest.len().min(out[chunk].len() - offset);
                out[chunk][offset..offset + n].copy_from_slice(&rest[..n]);
                (chunk, offset, rest) = (chunk + 1, 0, &rest[n..]);
            }
        }
        out
    }
}

/// The deepest numbered prefix of a dotted key: `a.1.b.2.c` → `a.1.b.2`; `""` without one.
pub fn block_of(key: &str) -> &str {
    let mut end = 0;
    let mut pos = 0;
    for seg in key.split('.') {
        pos += seg.len();
        if !seg.is_empty() && seg.bytes().all(|c| c.is_ascii_digit()) {
            end = pos;
        }
        pos += 1;
    }
    &key[..end]
}

/// Text runs compare by bytes, ASCII digit runs by value; ties fall back to plain order.
pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    fn run(s: &[u8], digits: bool) -> (&[u8], &[u8]) {
        let n = s
            .iter()
            .position(|c| c.is_ascii_digit() != digits)
            .unwrap_or(s.len());
        s.split_at(n)
    }
    let (mut x, mut y) = (a.as_bytes(), b.as_bytes());
    loop {
        let ((tx, rx), (ty, ry)) = (run(x, false), run(y, false));
        let ((dx, rx), (dy, ry)) = (run(rx, true), run(ry, true));
        let (nx, ny) = (trim_zeros(dx), trim_zeros(dy));
        let o = tx
            .cmp(ty)
            .then_with(|| dx.is_empty().cmp(&dy.is_empty()).reverse())
            .then_with(|| nx.len().cmp(&ny.len()))
            .then_with(|| nx.cmp(ny));
        if o != Ordering::Equal {
            return o;
        }
        if dx.is_empty() && dy.is_empty() {
            return a.cmp(b);
        }
        (x, y) = (rx, ry);
    }
}

fn trim_zeros(d: &[u8]) -> &[u8] {
    &d[d.iter().position(|c| *c != b'0').unwrap_or(d.len())..]
}
