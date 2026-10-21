//! One process's facts: its sets and their pages, the plans of its requests, their cursors and
//! the measured rooms; plus the other rows' published bytes. Classes are derived here, never
//! stored: [`Table::class`] answers from the facts each time.

use std::sync::Arc;

use sha2::{Digest, Sha256};
use tensorfs::io::source::Source;

use crate::layout::Layout;

/// How far [`crate::Memory::load`] brings a set: into the machine's RAM tier, or onto the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Ram,
    Gpu,
}

/// What turns a page's stored bytes into its weights.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Fill {
    /// The stored bytes are the weights: a page is filled when its copy lands.
    Stored,
    /// The caller transforms each fresh page (a LoRA bake) and calls [`crate::Pin::filled`].
    /// The string is the transform's identity: the LoRA content digests, scales, bake recipe
    /// and torch/GPU arch. Declared: its pages may be kept and fetched speculatively.
    Declared(String),
    /// A transform whose identity is not known yet: never kept, never fetched speculatively.
    Undeclared,
}

/// One stored part's identity, as the fingerprint covers it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartId {
    pub name: String,
    pub encoding: String,
    pub dtype: String,
}

/// What a filled page's bytes are: sha256 over the layout digest, each part's stored name,
/// encoding and dtype, the fill identity, the package release (the bake code's identity), the
/// rank shard and the GPU UUID. Kept pages are found
/// by it, so two pages with one fingerprint hold the same bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fingerprint {
    pub digest: String,
    pub fill: Fill,
}

impl Fingerprint {
    /// `release` names the package release, `shard` the rank's slice ("0/1" whole), `gpu` the
    /// device's UUID.
    pub fn new(layout: &Layout, parts: &[PartId], fill: Fill, release: &str, shard: &str, gpu: &str) -> Fingerprint {
        let mut h = Sha256::new();
        let mut field = |b: &[u8]| {
            h.update((b.len() as u64).to_le_bytes());
            h.update(b);
        };
        field(layout.digest.as_bytes());
        for p in parts {
            field(p.name.as_bytes());
            field(p.encoding.as_bytes());
            field(p.dtype.as_bytes());
        }
        match &fill {
            Fill::Stored => field(b"stored"),
            Fill::Declared(id) => {
                field(b"declared");
                field(id.as_bytes());
            }
            Fill::Undeclared => field(b"undeclared"),
        }
        field(release.as_bytes());
        field(shard.as_bytes());
        field(gpu.as_bytes());
        let digest = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        Fingerprint { digest, fill }
    }

    /// Its pages may be kept, and fetched before a declared use.
    pub fn declared(&self) -> bool {
        self.fill != Fill::Undeclared
    }
}

/// A weight set: its layout (one page per region), where its bytes come from, and what they are.
#[derive(Clone)]
pub struct SetSpec {
    /// Unique per package release (`<package>@<release>/<component>`): measured peaks and cut
    /// levels are kept under it.
    pub name: String,
    pub layout: Arc<Layout>,
    pub source: Arc<Source>,
    pub fingerprint: Fingerprint,
}

/// A request's declared use: its steps in order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    pub steps: Vec<Step>,
}

/// One step: these sets run together (no order among them), each `passes` times.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub sets: Vec<String>,
    pub passes: u32,
}
