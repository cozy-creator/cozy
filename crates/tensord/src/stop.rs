//! One caller's cooperative stop signal, and the work it stops: how often that work moved.
//! Clones name the same intent; a resumed operation takes a new one and keeps what the store
//! already holds.
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

#[derive(Clone, Debug, Default)]
pub struct Stop(Arc<(AtomicBool, AtomicU64)>);

impl Stop {
    pub fn cancel(&self) {
        self.0 .0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0 .0.load(Ordering::Acquire)
    }
    /// Its work moved: bytes landed, a stage began, its installer's meter advanced.
    pub fn moved(&self) {
        self.0 .1.fetch_add(1, Ordering::Relaxed);
    }
    pub fn progress(&self) -> u64 {
        self.0 .1.load(Ordering::Relaxed)
    }
}
