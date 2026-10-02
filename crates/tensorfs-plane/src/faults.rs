//! Fault points on the copy path, armed only by tests (cargo feature `faults`). Without the
//! feature every point is a no-op.

#[cfg(feature = "faults")]
mod imp {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use crate::cuda::{self, CUstream};

    static ARMED: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
    static GATES: Mutex<Vec<(&'static str, Arc<AtomicBool>)>> = Mutex::new(Vec::new());

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
pub use imp::{arm, arm_gate, Gate};
#[cfg(feature = "faults")]
pub(crate) use imp::{gate, hit};

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
