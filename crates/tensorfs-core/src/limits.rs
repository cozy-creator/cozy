//! PROVISIONAL constants (cozytensors.md §7). Ratification waits on proto-001's REAL
//! selected-H3 tensor-schema corpus + SDXL; nothing here is frozen until that lands.

pub const PROVISIONAL: &str = "provisional: awaiting proto-001 H3+SDXL corpus (tfs-013/tfs-015)";

// Measured evidence banked toward ratification — recorded, never treated as ratification:
//
// tfs-013: the 64 MiB DOC_MAX_BYTES cap binds at ~85,000 tensors, so MAX_TENSORS (100,000)
//   is unreachable — the two caps are not mutually consistent (decisions row 167).
// tfs-015: the ONE uniform tensor form costs 467 B/tensor over a mixed plain/fp8/nvfp4 tree
//   with REAL-length H3 keys (the shipped DiT header's 943 logical keys average 37.7 B) —
//   1.65 MiB at the real selected-H3 cardinality (3,699 tensors), 1.12 MiB at SDXL (2,515).
//   Both are ~2.5% of DOC_MAX_BYTES: the uniform form is not what pressures it.
// tfs-015: a real 382 MiB, 64-tensor, 100-part checkpoint produced 77 segments and 24 inline
//   bodies under GRID_BYTES=64 MiB / INLINE_MAX_BYTES=256 B. The 256 B threshold caught
//   every norm-scale and rank-0 scalar role and no weight role, which is the intent; a
//   real H3 tree at proto-001 is what decides whether it should move.
// tfs-015: MAX_TOTAL_REFS=65,536 is the whole-document reference total. A 500 GB artifact on
//   the 64 MiB grid needs ~8,000 — the cap leaves 8x headroom while bounding the fetch plan
//   a single header can demand.

pub const GRID_BYTES: u64 = 64 * 1024 * 1024;
pub const INLINE_MAX_BYTES: u64 = 256;

pub const DOC_MAX_BYTES: usize = 64 * 1024 * 1024;
pub const SPEC_MAX_BYTES: usize = 64 * 1024;
pub const DEPTH_MAX: usize = 8;

pub const MAX_COMPONENTS: usize = 256;
pub const MAX_TENSORS: usize = 100_000;
pub const MAX_PARTS: usize = 16;
pub const MAX_RANK: usize = 32;
pub const MAX_KEY_BYTES: usize = 4096;
pub const MAX_NAME_BYTES: usize = 128;
pub const MAX_SEGMENTS: usize = 65_536;
/// Manifest file-map entries. A real checkpoint tree is ~10; the cap bounds a hostile map
/// BEFORE the entry vector is allocated.
pub const MAX_ENTRIES: usize = 65_536;
/// Whole-DOCUMENT totals, not only per-part counts: a header may not multiply small legal
/// parts into an unbounded fetch plan or an unbounded byte claim.
pub const MAX_TOTAL_REFS: usize = 65_536;
pub const MAX_TOTAL_BYTES: u64 = 1 << 50;
pub const MAX_RELATION_NODES: usize = 64;
pub const MAX_DOC_TEXT_BYTES: usize = 4096;
pub const MAX_ELEMENTS: u64 = 1 << 48;

// ---------------------------------------------------------------- ingest border (tfs-003)

/// Foreign carrier header cap, enforced BEFORE the allocation: an 8-byte length field is
/// read, checked, and only then is a buffer made. Real evidence: the shipped H3 DiT header
/// is 486 KiB over 1,667 keys and SDXL's largest is 121 KiB over 2,516 — 16 MiB leaves 30x.
pub const CARRIER_HEADER_MAX_BYTES: usize = 16 * 1024 * 1024;
/// Maximum source role admitted for a permutation-class op. Source runs stream through
/// the fixed writer buffer; this is a declared work bound, not a resident allocation.
/// Identity/inherit ops are not subject to this permutation-specific limit.
pub const CONVERT_TENSOR_MAX_BYTES: u64 = 512 * 1024 * 1024;

/// Interoperable integer range (I-JSON): numbers outside it refuse.
pub const INT_MAX: i64 = (1i64 << 53) - 1;
pub const INT_MIN: i64 = -((1i64 << 53) - 1);
