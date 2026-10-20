//! Host-tier and GPU completion fault points, armed only by tests (cargo feature `faults`).
//! Without the feature every point is a no-op.

#[cfg(feature = "faults")]
mod imp {
    use std::sync::Mutex;

    use crate::cuda;

    static ARMED: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
    static STALLS: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());

    /// Sleep `ms` at every pass through `point` from now on (0 disarms).
    pub fn arm_stall(point: &'static str, ms: u64) {
        let mut s = STALLS.lock().unwrap();
        s.retain(|(p, _)| *p != point);
        if ms > 0 {
            s.push((point, ms));
        }
    }

    pub(crate) fn stall(point: &'static str) {
        let ms = STALLS.lock().unwrap().iter().find(|(p, _)| *p == point).map(|(_, ms)| *ms);
        if let Some(ms) = ms {
            std::thread::sleep(std::time::Duration::from_millis(ms));
        }
    }

    /// Fail the `n`th (1-based) next pass through `point` as a driver out-of-memory. Replaces
    /// any earlier arming of `point`.
    pub fn arm(point: &'static str, n: u64) {
        let mut a = ARMED.lock().unwrap();
        a.retain(|(p, _)| *p != point);
        a.push((point, n));
    }

    pub(crate) fn hit(point: &'static str) -> crate::Result<()> {
        let mut armed = ARMED.lock().unwrap();
        let Some(i) = armed.iter().position(|(p, _)| *p == point) else { return Ok(()) };
        armed[i].1 -= 1;
        if armed[i].1 > 0 {
            return Ok(());
        }
        armed.remove(i);
        Err(crate::Error::Cuda { op: point, code: cuda::ERROR_OUT_OF_MEMORY, name: "injected".into() })
    }
}

#[cfg(feature = "faults")]
pub use imp::{arm, arm_stall};
#[cfg(feature = "faults")]
pub(crate) use imp::{hit, stall};

#[cfg(not(feature = "faults"))]
#[inline(always)]
pub(crate) fn stall(_point: &'static str) {}

#[cfg(not(feature = "faults"))]
#[inline(always)]
pub(crate) fn hit(_point: &'static str) -> crate::Result<()> {
    Ok(())
}

/// One-shot CPU phase acknowledgements, absent from library builds and Python bindings.
#[cfg(all(test, feature = "faults"))]
pub(crate) mod phases {
    use std::sync::{mpsc, Mutex};
    static WATCHERS: Mutex<Vec<(&'static str, mpsc::Sender<()>)>> = Mutex::new(Vec::new());

    pub(crate) fn watch(point: &'static str) -> mpsc::Receiver<()> {
        let (tx, rx) = mpsc::channel();
        let mut watchers = WATCHERS.lock().unwrap();
        assert!(!watchers.iter().any(|(p, _)| *p == point), "duplicate proof phase");
        watchers.push((point, tx));
        rx
    }

    pub(crate) fn reached(point: &'static str) {
        let mut watchers = WATCHERS.lock().unwrap();
        if let Some(i) = watchers.iter().position(|(p, _)| *p == point) {
            let (_, tx) = watchers.swap_remove(i);
            let _ = tx.send(());
        }
    }
}
