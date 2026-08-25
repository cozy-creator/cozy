//! PROVISIONAL constants (cozytensors.md §7). Ratification waits on proto-001's REAL
//! selected-H3 topology corpus + SDXL; nothing here is frozen until that lands.

pub const PROVISIONAL: &str = "provisional: awaiting proto-001 H3+SDXL corpus (tfs-013/tfs-015)";

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
pub const MAX_RELATION_NODES: usize = 64;
pub const MAX_DOC_TEXT_BYTES: usize = 4096;
pub const MAX_ELEMENTS: u64 = 1 << 48;

/// Interoperable integer range (I-JSON): numbers outside it refuse.
pub const INT_MAX: i64 = (1i64 << 53) - 1;
pub const INT_MIN: i64 = -((1i64 << 53) - 1);
