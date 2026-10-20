//! Fault points on the host tier, armed only by tests (cargo feature `faults`).
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
