//! Region registrations retained with a placement mapping. The mapping owner drops every
//! registration before unmapping or punching its bytes; TensorFS owns CUDA registration.
use super::cuda::{self, HostPin};
use std::collections::BTreeMap;
use std::sync::Mutex;

#[derive(Clone, Copy)]
enum State {
    Fresh,
    Once,
    Pinned,
    Refused,
}
struct Inner {
    regions: Vec<State>,
    guards: BTreeMap<u64, HostPin>,
}
pub struct PinnedRanges {
    ptr: usize,
    len: usize,
    read_only: bool,
    inner: Mutex<Inner>,
}

impl PinnedRanges {
    /// # Safety
    /// The mapping stays valid until all guards are cleared, and a registered range is never
    /// unmapped or punched before `unpin_region`/`clear` returns.
    pub unsafe fn new(ptr: *mut u8, len: usize, regions: usize, read_only: bool) -> Self {
        Self {
            ptr: ptr as usize,
            len,
            read_only,
            inner: Mutex::new(Inner {
                regions: vec![State::Fresh; regions],
                guards: BTreeMap::new(),
            }),
        }
    }
    fn register_into(&self, inner: &mut Inner, off: u64, len: u64) -> cuda::Result<()> {
        if inner.guards.contains_key(&off) {
            return Ok(());
        }
        if off.checked_add(len).is_none_or(|end| end > self.len as u64) {
            return Err(cuda::Error::CudaUnavailable(
                "registration exceeds its host mapping".into(),
            ));
        }
        // SAFETY: caller of new guarantees this live mapping; bounds checked above.
        let guard = unsafe {
            cuda::pin_region(
                (self.ptr as *mut u8).add(off as usize),
                len as usize,
                self.read_only,
            )
        }?;
        inner.guards.insert(off, guard);
        Ok(())
    }
    pub fn register(&self, off: u64, len: u64) -> cuda::Result<()> {
        self.register_into(&mut self.inner.lock().unwrap(), off, len)
    }
    /// Stage the first copy. Reuse pays registration once; refused ranges stay staged.
    pub fn pin_region(&self, region: u32, off: u64, len: u64) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let Some(state) = inner.regions.get(region as usize).copied() else {
            return false;
        };
        match state {
            State::Pinned => true,
            State::Refused => false,
            State::Fresh => {
                inner.regions[region as usize] = State::Once;
                false
            }
            State::Once => match self.register_into(&mut inner, off, len) {
                Ok(()) => {
                    inner.regions[region as usize] = State::Pinned;
                    true
                }
                Err(cuda::Error::Cuda {
                    code: cuda::ERROR_NOT_SUPPORTED,
                    ..
                }) if self.read_only => {
                    inner.regions.fill(State::Refused);
                    false
                }
                Err(_) => {
                    inner.regions[region as usize] = State::Refused;
                    false
                }
            },
        }
    }
    pub fn unpin_region(&self, region: u32, off: u64) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(state) = inner.regions.get_mut(region as usize) {
            *state = State::Fresh;
        }
        inner.guards.remove(&off);
    }
    pub fn unregister(&self, off: u64) {
        self.inner.lock().unwrap().guards.remove(&off);
    }
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.guards.clear();
        inner.regions.fill(State::Fresh);
    }
}
