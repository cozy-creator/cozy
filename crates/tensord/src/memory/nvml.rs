//! Each GPU's memory sampler thread over `platform::gpu::Device` (NVML, opened at run time;
//! no CUDA). Read only on that thread, never on the API or journal loop: a driver stall stalls
//! that thread alone.
use super::policy::Sample;
use platform::gpu::Device;
use std::{
    io,
    sync::{Arc, Condvar, Mutex},
    time::Duration,
};

struct State {
    latest: Option<Sample>,
    asked: u64,
    answered: u64,
    stopped: bool,
}

/// One GPU's sampler thread: a reading every second, and one on demand before a decision.
pub struct Sampler {
    shared: Arc<(Mutex<State>, Condvar)>,
}

impl Sampler {
    /// `observe` runs on the sampler thread after every reading (the floor watchdog).
    pub fn start(device: Device, observe: impl Fn(&Sample) + Send + 'static) -> io::Result<Self> {
        let shared = Arc::new((
            Mutex::new(State {
                latest: None,
                asked: 0,
                answered: 0,
                stopped: false,
            }),
            Condvar::new(),
        ));
        let thread = shared.clone();
        std::thread::Builder::new()
            .name("gpu-memory-sampler".into())
            .spawn(move || loop {
                let reading = device.read().ok().map(|reading| Sample {
                    total: reading.total,
                    free: reading.free,
                    display: reading.display,
                    processes: reading.processes,
                });
                let (lock, changed) = &*thread;
                let mut state = lock.lock().unwrap();
                state.latest = reading.clone();
                state.answered = state.asked;
                changed.notify_all();
                if state.stopped {
                    return;
                }
                drop(state);
                if let Some(sample) = &reading {
                    observe(sample);
                }
                let state = lock.lock().unwrap();
                let _ = changed
                    .wait_timeout_while(state, Duration::from_secs(1), |s| {
                        s.asked == s.answered && !s.stopped
                    })
                    .unwrap();
            })?;
        Ok(Self { shared })
    }

    /// A reading taken after this call: waits for the sampler thread, not for a clock.
    pub fn now(&self) -> Option<Sample> {
        let (lock, changed) = &*self.shared;
        let mut state = lock.lock().unwrap();
        state.asked += 1;
        let ticket = state.asked;
        changed.notify_all();
        let state = changed
            .wait_while(state, |s| s.answered < ticket && !s.stopped)
            .unwrap();
        state.latest.clone()
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        let (lock, changed) = &*self.shared;
        lock.lock().unwrap().stopped = true;
        changed.notify_all();
    }
}
