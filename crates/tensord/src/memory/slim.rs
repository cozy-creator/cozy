//! TensorD's placement interface. The ledger owns custody and RAM service; TensorD never
//! opens a CUDA context or reconstructs the executor's eviction order.

use std::os::fd::BorrowedFd;
use std::sync::Arc;
pub use varena::ledger::Grant;
pub use varena::ledger::HostCache;
use varena::ledger::{Ledger, Status};

pub struct Memory(Arc<Ledger>);
impl Memory {
    pub fn create(gpu_uuid: &str, card_bytes: u64, limit: Option<u64>) -> varena::Result<Self> {
        Self::create_with_host(gpu_uuid, card_bytes, limit, Arc::new(HostCache::default()))
    }
    pub fn create_with_host(
        gpu_uuid: &str,
        card_bytes: u64,
        limit: Option<u64>,
        host: Arc<HostCache>,
    ) -> varena::Result<Self> {
        Ledger::create_with_host(
            gpu_uuid,
            limit.map_or(card_bytes, |cap| cap.min(card_bytes)),
            host,
        )
        .map(|l| Self(Arc::new(l)))
    }
    pub fn grant_unbound(&self) -> varena::Result<Grant> {
        self.0.grant_unbound()
    }
    pub fn grant(&self, pid: u32) -> varena::Result<Grant> {
        self.0.grant(pid)
    }
    pub fn reap(&self, pid: u32) -> varena::Result<()> {
        self.0.reap(pid)
    }
    pub fn fits(&self, package: &str, shape: &str) -> varena::Result<bool> {
        self.0.fits(package, shape)
    }
    pub fn status(&self) -> varena::Result<Status> {
        self.0.status()
    }
    pub fn events(&self) -> BorrowedFd<'_> {
        self.0.events()
    }
    pub fn release_unused_ram(&self) -> bool {
        self.0.release_unused_ram()
    }
    pub fn prepare_host(&self, reservation: u64, need: Option<u64>) {
        self.0.prepare_host(reservation, need);
    }
    pub fn finish_host(&self, reservation: u64) {
        self.0.finish_host(reservation);
    }
    pub fn capacity(&self) -> u64 {
        self.0.capacity()
    }
    pub fn ledger(&self) -> &Arc<Ledger> {
        &self.0
    }

    /// Register once in the existing reactor. Each await consumes the eventfd counter; all
    /// pending changes coalesce into one wake, and the caller then retries queue admission.
    pub fn event_source(&self) -> std::io::Result<tokio::io::unix::AsyncFd<std::fs::File>> {
        let file = std::fs::File::from(self.events().try_clone_to_owned()?);
        // SAFETY: AsyncFd owns this File exclusively; its descriptor stays open and stable
        // until AsyncFd is dropped. The descriptor is an eventfd, already nonblocking.
        unsafe { tokio::io::unix::AsyncFd::register(file) }.map_err(Into::into)
    }
    pub async fn changed(source: &tokio::io::unix::AsyncFd<std::fs::File>) -> std::io::Result<()> {
        loop {
            let mut ready = source.readable().await?;
            match ready.try_io(|fd| {
                let mut count = [0u8; 8];
                std::io::Read::read_exact(&mut fd.get_ref(), &mut count)
            }) {
                Ok(result) => return result,
                Err(_) => continue,
            }
        }
    }
}
