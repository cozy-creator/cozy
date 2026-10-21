//! Fault points for the recovery and race tests: arm one, and the next pass through it panics;
//! hold one, and the next thread through it waits there until released.

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use parking_lot::{Condvar, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Point {
    Decide = 1,
    Fill = 2,
    /// A reaper between reading the rows and freeing the dead ones.
    Reap = 3,
}

static ARMED: AtomicU32 = AtomicU32::new(0);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Hold {
    Free,
    Set(Point),
    Reached(Point),
}

static HOLD: Mutex<Hold> = Mutex::new(Hold::Free);
static MOVED: Condvar = Condvar::new();

/// The next pass through `point` panics, once.
pub fn arm(point: Point) {
    ARMED.store(point as u32, Relaxed);
}

/// The next thread through `point` waits there until [`release`].
pub fn hold(point: Point) {
    *HOLD.lock() = Hold::Set(point);
}

/// Blocks until a thread waits at the held point.
pub fn reached() {
    let mut h = HOLD.lock();
    while !matches!(*h, Hold::Reached(_)) {
        MOVED.wait(&mut h);
    }
}

pub fn release() {
    *HOLD.lock() = Hold::Free;
    MOVED.notify_all();
}

#[allow(clippy::panic)]
pub(crate) fn hit(point: Point) {
    if ARMED.compare_exchange(point as u32, 0, Relaxed, Relaxed).is_ok() {
        panic!("injected fault at {point:?}");
    }
    let mut h = HOLD.lock();
    if *h == Hold::Set(point) {
        *h = Hold::Reached(point);
        MOVED.notify_all();
        while *h == Hold::Reached(point) {
            MOVED.wait(&mut h);
        }
    }
}
