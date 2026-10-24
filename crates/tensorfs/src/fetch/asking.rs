//! The threads blocked on a TensorFS exchange now. Each exchange is judged by the transport's own
//! rule on what it moves (`ask`, the Fetcher), so an observer judging the thread's work by its
//! progress leaves that wait to the exchange's rule instead of calling it wedged first (Runtime
//! `liveness.py`: an outside observer gives the inner ledger its own chance).
use std::sync::Mutex;
use std::thread::ThreadId;

static ASKING: Mutex<Vec<(ThreadId, u32)>> = Mutex::new(Vec::new());

/// This thread's wait on an exchange, until dropped.
pub struct Asking(ThreadId);

/// Marks this thread as waiting on an exchange.
pub fn begin() -> Asking {
    let thread = std::thread::current().id();
    let mut asking = ASKING.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    match asking.iter_mut().find(|(t, _)| *t == thread) {
        Some((_, n)) => *n += 1,
        None => asking.push((thread, 1)),
    }
    Asking(thread)
}

impl Drop for Asking {
    fn drop(&mut self) {
        let mut asking = ASKING.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(i) = asking.iter().position(|(t, _)| *t == self.0) {
            asking[i].1 -= 1;
            if asking[i].1 == 0 {
                asking.swap_remove(i);
            }
        }
    }
}

/// Whether `thread` waits on an exchange now.
pub fn waits(thread: ThreadId) -> bool {
    ASKING.lock().unwrap_or_else(std::sync::PoisonError::into_inner).iter().any(|(t, _)| *t == thread)
}
