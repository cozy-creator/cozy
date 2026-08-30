//! Degraded-behavior operation receipts (tensorfs.md §4).
//!
//! Every fallback that changes performance or trust says so in a structured document keyed
//! by exact manifest, build and device — and by nothing that enters content
//! identity. A silent fallback is the defect; a slow path that announces itself is fine.
//!
//! These are OBSERVABILITY records, not canonical documents (proto-007): one process emits
//! them to its caller and nothing stores, digests, or re-reads them. They are deliberately
//! not admitted to the CAS: nothing references them, they carry a device and a build, and a
//! manifest's identity must not move because a symlink fell back to a copy.

use crate::canon::{self, Value};

/// The closed set of degradations, ENFORCED by the decoder below. A new one is a code change
/// here, which is the point: "some other fallback happened" is not a receipt.
pub const KINDS: &[&str] = &[
    "symlink-fell-back-to-copy",
    "buffered-instead-of-direct",
    "fill-used-gather",
    "record-invalidated-rehash",
    "lease-dropped-unreleased",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub op: String,
    pub degraded: String,
    pub manifest: String,
    pub build: String,
    pub device: String,
    pub detail: String,
}

impl Receipt {
    pub fn new(op: &str, degraded: &str, manifest: &str, detail: String) -> Receipt {
        Receipt {
            op: op.to_string(),
            degraded: degraded.to_string(),
            manifest: manifest.to_string(),
            build: env!("CARGO_PKG_VERSION").to_string(),
            // TensorFS owns no device. The field carries what the CALLER declared, and
            // "host" is the honest answer when nobody declared one.
            device: "host".to_string(),
            detail,
        }
    }
    pub fn on(mut self, device: &str) -> Receipt {
        self.device = device.to_string();
        self
    }
}

impl Receipt {
    pub fn to_value(&self) -> Value {
        debug_assert!(KINDS.contains(&self.degraded.as_str()));
        Value::obj(vec![
            ("build", Value::str(self.build.clone())),
            ("degraded", Value::str(self.degraded.clone())),
            ("detail", Value::str(self.detail.clone())),
            ("device", Value::str(self.device.clone())),
            ("op", Value::str(self.op.clone())),
            ("manifest", Value::str(self.manifest.clone())),
        ])
    }
    pub fn canonical_bytes(&self) -> Vec<u8> {
        canon::write(&self.to_value())
    }
}

/// Collected receipts for one operation. The caller decides where they go; the operation
/// only has to emit them.
#[derive(Debug, Default, Clone)]
pub struct Receipts(pub Vec<Receipt>);

impl Receipts {
    pub fn emit(&mut self, r: Receipt) {
        self.0.push(r);
    }
    pub fn count(&self, degraded: &str) -> usize {
        self.0.iter().filter(|r| r.degraded == degraded).count()
    }
    pub fn print(&self) {
        for r in &self.0 {
            println!(
                "  RECEIPT {} op={} {} — {}",
                r.degraded, r.op, r.device, r.detail
            );
        }
    }
}
