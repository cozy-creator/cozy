//! Degraded-behavior operation receipts (tensorfs.md §4).
//!
//! Every fallback that changes performance or trust says so in a structured document keyed
//! by exact snapshot, build, device and generation — and by nothing that enters content
//! identity. A silent fallback is the defect; a slow path that announces itself is fine.
//!
//! These are OBSERVABILITY documents. They are canonical so a consumer can diff them, and
//! they are deliberately not admitted to the CAS: nothing references them, they carry a
//! device and a build, and a snapshot's identity must not move because a symlink fell back
//! to a copy.

use crate::canon::{Fields, Value};
use crate::err::Result;
use crate::ids::{check_format, Doc};
use crate::limits;

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
    pub snapshot: String,
    pub build: String,
    pub device: String,
    pub generation: u64,
    pub detail: String,
}

impl Receipt {
    pub fn new(
        op: &str,
        degraded: &str,
        snapshot: &str,
        generation: u64,
        detail: String,
    ) -> Receipt {
        Receipt {
            op: op.to_string(),
            degraded: degraded.to_string(),
            snapshot: snapshot.to_string(),
            build: env!("CARGO_PKG_VERSION").to_string(),
            // TensorFS owns no device. The field carries what the CALLER declared, and
            // "host" is the honest answer when nobody declared one.
            device: "host".to_string(),
            generation,
            detail,
        }
    }
    pub fn on(mut self, device: &str) -> Receipt {
        self.device = device.to_string();
        self
    }
}

impl Doc for Receipt {
    const FORMAT: &'static str = "tensorfs.operation_receipt/1";
    const MAX_BYTES: usize = limits::SPEC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("Receipt", v)?;
        let build = f.req_str("build")?.to_string();
        let degraded = f.req_str("degraded")?.to_string();
        let detail = f.req_str("detail")?.to_string();
        let device = f.req_str("device")?.to_string();
        check_format("Receipt", f.req_str("format")?, Self::FORMAT)?;
        let generation = f.req_uint("generation")?;
        let op = f.req_str("op")?.to_string();
        let snapshot = f.req_str("snapshot")?.to_string();
        f.done()?;
        if !KINDS.contains(&degraded.as_str()) {
            return crate::err::refuse(
                crate::err::Code::UNKNOWN_FIELD,
                format!("degraded kind {degraded:?} is outside the closed set {KINDS:?}"),
            );
        }
        Ok(Receipt {
            op,
            degraded,
            snapshot,
            build,
            device,
            generation,
            detail,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("build", Value::str(self.build.clone())),
            ("degraded", Value::str(self.degraded.clone())),
            ("detail", Value::str(self.detail.clone())),
            ("device", Value::str(self.device.clone())),
            ("format", Value::str(Self::FORMAT)),
            ("generation", Value::uint(self.generation)),
            ("op", Value::str(self.op.clone())),
            ("snapshot", Value::str(self.snapshot.clone())),
        ])
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
