//! Panic points for the recovery tests: arm one, and the next pass through it panics.

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Point {
    Decide = 1,
    Fill = 2,
}

static ARMED: AtomicU32 = AtomicU32::new(0);

/// The next pass through `point` panics, once.
pub fn arm(point: Point) {
    ARMED.store(point as u32, Relaxed);
}

#[allow(clippy::panic)]
pub(crate) fn hit(point: Point) {
    if ARMED.compare_exchange(point as u32, 0, Relaxed, Relaxed).is_ok() {
        panic!("injected fault at {point:?}");
    }
}
