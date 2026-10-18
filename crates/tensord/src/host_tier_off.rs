//! The sealed host tier where the OS has no sealed shared memory (macOS, Windows): the same types, and
//! a tier that holds nothing. Every executor request is refused as `Unsupported`, so the executor
//! reads its weights from its other sources; facts report an empty tier.
use crate::access::Fetch;
use cozy_platform::contain::host_memory::HostMemory;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs::File, io, sync::Arc, time::Duration};
use tensorfs_core::{header::Header, store::Store};

#[derive(Clone, Debug)]
pub struct HostTierConfig {
    pub fill_threads: usize,
    pub ttl: Duration,
}
impl Default for HostTierConfig {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism().map_or(4, |n| n.get());
        Self {
            fill_threads: cpus.clamp(1, 8),
            ttl: Duration::from_secs(30 * 60),
        }
    }
}

pub trait TierLimit: Send + Sync {
    fn limit(&self, host: &HostMemory, charged: u64) -> u64;
    fn staging(&self, host: &HostMemory, charged: u64) -> u64 {
        self.limit(host, charged).saturating_sub(charged)
    }
}
pub struct HalfOfHeadroom;
impl TierLimit for HalfOfHeadroom {
    fn limit(&self, _host: &HostMemory, _charged: u64) -> u64 {
        0
    }
}

#[derive(Clone)]
pub struct HostGrant {
    pub manifest: String,
    pub header: Header,
    pub components: BTreeSet<String>,
}

pub struct SealedRequest<'a> {
    pub sha256: &'a str,
    pub length: u64,
    pub stage: Option<&'a [u32]>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Fill {
    pub name: String,
    pub bytes: u64,
    pub ms: f64,
    pub cached_bytes: u64,
    pub direct_bytes: u64,
    pub buffered_bytes: u64,
    pub disk_read_bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Ledger {
    pub fills: Vec<Fill>,
    pub hits: u64,
    pub windows_opened: u64,
    pub streamed_bytes: u64,
    pub failed: u64,
    pub released: u64,
    pub released_bytes: u64,
    pub freed_bytes: u64,
    pub stranded_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HostTierFacts {
    pub host: HostMemory,
    pub limit: u64,
    pub charged_bytes: u64,
    pub held_bytes: u64,
    pub filling_bytes: u64,
    pub windows: usize,
    pub window_bytes: u64,
    pub window_read_bytes: u64,
    pub entries: usize,
    pub peers: usize,
    #[serde(flatten)]
    pub ledger: Ledger,
}

pub struct HostTier;

fn off() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, "the sealed host tier is a Unix machine's")
}

impl HostTier {
    pub fn new(_store: Arc<Store>, _config: HostTierConfig, _limit: Box<dyn TierLimit>) -> io::Result<Arc<Self>> {
        Ok(Arc::new(Self))
    }

    pub fn prepare(self: &Arc<Self>, _grants: Vec<HostGrant>) {}

    pub fn register_peer(&self, _pidfd: File) -> u64 {
        0
    }

    pub fn seal(
        &self,
        _peer: u64,
        _grants: &[HostGrant],
        _request: SealedRequest<'_>,
        _plan: File,
        _fetch: Option<&Arc<Fetch>>,
    ) -> io::Result<Option<File>> {
        Err(off())
    }

    pub fn stage(
        &self,
        _peer: u64,
        _grants: &[HostGrant],
        _request: SealedRequest<'_>,
        _plan: File,
        _fetch: Option<&Arc<Fetch>>,
    ) -> io::Result<Vec<u32>> {
        Err(off())
    }

    pub fn object_files(
        &self,
        _peer: u64,
        _grants: &[HostGrant],
        _request: SealedRequest<'_>,
        _plan: File,
        _fetch: Option<&Arc<Fetch>>,
    ) -> io::Result<Vec<(tensorfs_core::ids::ObjectRef, File)>> {
        Err(off())
    }

    pub fn release(&self, _want: u64) -> u64 {
        0
    }

    pub fn facts(&self) -> HostTierFacts {
        HostTierFacts {
            host: cozy_platform::contain::host_memory::read(),
            limit: 0,
            charged_bytes: 0,
            held_bytes: 0,
            filling_bytes: 0,
            windows: 0,
            window_bytes: 0,
            window_read_bytes: 0,
            entries: 0,
            peers: 0,
            ledger: Ledger::default(),
        }
    }
}

/// The SHA-256 (hex) of `files`' objects in order, one `<sha256>:<length>\n` line each.
pub fn objects_digest(files: &[(tensorfs_core::ids::ObjectRef, File)]) -> String {
    let lines: String = files
        .iter()
        .map(|(obj, _)| format!("{}:{}\n", obj.sha256, obj.length))
        .collect();
    format!("{:x}", Sha256::digest(lines.as_bytes()))
}
