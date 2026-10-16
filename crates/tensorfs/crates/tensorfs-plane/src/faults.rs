//! Fault points on the copy and staging paths, armed only by tests (cargo feature `faults`).
//! Without the feature every point is a no-op.

#[cfg(feature = "faults")]
mod imp {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use crate::cuda::{self, CUstream};

    static ARMED: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
    static GATES: Mutex<Vec<(&'static str, Arc<AtomicBool>)>> = Mutex::new(Vec::new());
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

    /// Holds the next copy job's stream at `point` on the device until opened (or dropped).
    pub struct Gate(Arc<AtomicBool>);
    impl Gate {
        pub fn open(&self) {
            self.0.store(true, Ordering::Release);
        }
        pub fn is_open(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
        pub fn opener(&self) -> impl FnOnce() + Send + 'static {
            let f = self.0.clone();
            move || f.store(true, Ordering::Release)
        }
    }
    impl Drop for Gate {
        fn drop(&mut self) {
            self.open();
        }
    }

    pub fn arm_gate(point: &'static str) -> Gate {
        let f = Arc::new(AtomicBool::new(false));
        GATES.lock().unwrap().push((point, f.clone()));
        Gate(f)
    }

    pub(crate) fn gate(point: &'static str, stream: CUstream) -> crate::Result<()> {
        let f = {
            let mut g = GATES.lock().unwrap();
            let Some(i) = g.iter().position(|(p, _)| *p == point) else { return Ok(()) };
            g.remove(i).1
        };
        extern "C" fn hold(p: *mut std::ffi::c_void) {
            // SAFETY: the Arc leaked below, reclaimed exactly once here.
            let f = unsafe { Arc::from_raw(p as *const AtomicBool) };
            while !f.load(Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        let d = cuda::driver()?;
        let p = Arc::into_raw(f) as *mut std::ffi::c_void;
        // SAFETY: our stream; the host function only waits on the flag.
        cuda::check("cuLaunchHostFunc", unsafe { (d.launch_host_func)(stream, hold, p) })
    }
}

#[cfg(feature = "faults")]
pub use imp::{arm, arm_gate, arm_stall, Gate};
#[cfg(feature = "faults")]
pub(crate) use imp::{gate, hit, stall};

#[cfg(not(feature = "faults"))]
#[inline(always)]
pub(crate) fn stall(_point: &'static str) {}

#[cfg(not(feature = "faults"))]
#[inline(always)]
pub(crate) fn hit(_point: &'static str) -> crate::Result<()> {
    Ok(())
}

#[cfg(not(feature = "faults"))]
#[inline(always)]
pub(crate) fn gate(_point: &'static str, _stream: crate::cuda::CUstream) -> crate::Result<()> {
    Ok(())
}
