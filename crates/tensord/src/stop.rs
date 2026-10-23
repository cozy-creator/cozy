//! One caller's cooperative stop signal, and the work it stops: how often that work moved.
//! Clones name the same intent; a resumed operation takes a new one and keeps what the store
//! already holds.
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

#[derive(Clone, Debug, Default)]
pub struct Stop(Arc<(AtomicBool, AtomicU64, AtomicU64)>);

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
    /// Its work told where it is (a stage, bytes of a total): it moved when that is not where
    /// it last told. Told again unchanged, a stuck transfer is not moving.
    pub fn at(&self, stage: &str, done: u64, total: u64) {
        use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher};
        let place = BuildHasherDefault::<DefaultHasher>::default().hash_one((stage, done, total));
        if self.0 .2.swap(place, Ordering::Relaxed) != place {
            self.moved();
        }
    }
    pub fn progress(&self) -> u64 {
        self.0 .1.load(Ordering::Relaxed)
    }
}
