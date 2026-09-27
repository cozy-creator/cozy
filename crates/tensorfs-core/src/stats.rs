//! Process-wide counters for the verify-once contract (proto-061). Monotonic; callers diff
//! two snapshots around the work they measure.

use std::sync::atomic::{AtomicU64, Ordering};

static CATALOG_OPENS: AtomicU64 = AtomicU64::new(0);
static PRESENCE_PASSES: AtomicU64 = AtomicU64::new(0);
static REPO_CACHE_VERIFIED_READS: AtomicU64 = AtomicU64::new(0);
static ADMISSION_BATCHES: AtomicU64 = AtomicU64::new(0);
static ADMITTED_OBJECTS: AtomicU64 = AtomicU64::new(0);
static ADMISSION_NANOS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// SQLite connections opened to any Store catalog.
    pub catalog_opens: u64,
    /// Walks that prove every object of one declared set (a manifest closure) resident.
    pub presence_passes: u64,
    /// Repository-cache entries read and hashed.
    pub repo_cache_verified_reads: u64,
    /// Blob admission batches committed (one transaction and commit each).
    pub admission_batches: u64,
    /// Blobs those batches published and recorded.
    pub admitted_objects: u64,
    /// Wall time those batches spent linking, fsyncing and committing.
    pub admission_nanos: u64,
}

pub fn snapshot() -> Stats {
    Stats {
        catalog_opens: CATALOG_OPENS.load(Ordering::Relaxed),
        presence_passes: PRESENCE_PASSES.load(Ordering::Relaxed),
        repo_cache_verified_reads: REPO_CACHE_VERIFIED_READS.load(Ordering::Relaxed),
        admission_batches: ADMISSION_BATCHES.load(Ordering::Relaxed),
        admitted_objects: ADMITTED_OBJECTS.load(Ordering::Relaxed),
        admission_nanos: ADMISSION_NANOS.load(Ordering::Relaxed),
    }
}

pub fn catalog_opens() -> u64 {
    CATALOG_OPENS.load(Ordering::Relaxed)
}

pub fn presence_passes() -> u64 {
    PRESENCE_PASSES.load(Ordering::Relaxed)
}

pub fn repo_cache_verified_reads() -> u64 {
    REPO_CACHE_VERIFIED_READS.load(Ordering::Relaxed)
}

pub(crate) fn catalog_opened() {
    CATALOG_OPENS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn presence_pass() {
    PRESENCE_PASSES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn repo_cache_verified_read() {
    REPO_CACHE_VERIFIED_READS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn admission_batch(objects: usize, spent: std::time::Duration) {
    ADMISSION_BATCHES.fetch_add(1, Ordering::Relaxed);
    ADMITTED_OBJECTS.fetch_add(objects as u64, Ordering::Relaxed);
    ADMISSION_NANOS.fetch_add(
        u64::try_from(spent.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}
