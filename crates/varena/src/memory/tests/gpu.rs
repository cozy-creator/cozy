//! Private real-CUDA controls. Every selected case runs in its own bounded process.
//! Ordinary CPU tests never load CUDA; explicit GPU selection refuses missing opt-in/capability.

use super::*;
use crate::cuda::proof;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(10);
const WATCHDOG: Duration = Duration::from_secs(20);
const BYTES: u64 = REGION_ALIGN;
const CASES: &[&str] = &[
    "source_handoff", "consumer_dependency", "home_record", "streamed_record",
    "staging_prior", "last_pin", "healthy_drain", "unregister",
];

#[derive(Default)]
struct GateFlags { entered: bool, released: bool, exited: bool, expired: bool }
#[derive(Default)]
struct GateState { flags: Mutex<GateFlags>, cv: Condvar }

impl GateState {
    fn release(&self) {
        self.flags.lock().unwrap_or_else(|e| e.into_inner()).released = true;
        self.cv.notify_all();
    }

    fn wait(&self, what: &str, ready: impl Fn(&GateFlags) -> bool) {
        let state = self.flags.lock().unwrap_or_else(|e| e.into_inner());
        let (state, _) = self.cv.wait_timeout_while(state, WAIT, |s| !ready(s))
            .unwrap_or_else(|e| e.into_inner());
        assert!(ready(&state) && !state.expired, "REFUSED: gate {what} was not acknowledged");
    }
}

// No CUDA, user callback, panic, logging, or dependency on later GPU work in this function.
unsafe extern "C" fn host_gate(data: *mut c_void) {
    // SAFETY: arm passes one retained Arc raw reference; only this callback consumes it.
    let gate = unsafe { Arc::<GateState>::from_raw(data.cast()) };
    let mut state = gate.flags.lock().unwrap_or_else(|e| e.into_inner());
    state.entered = true;
    gate.cv.notify_all();
    while !state.released {
        state = gate.cv.wait(state).unwrap_or_else(|e| e.into_inner());
    }
    state.exited = true;
    gate.cv.notify_all();
}

struct Gate {
    state: Arc<GateState>,
    done: Option<mpsc::Sender<()>>,
    watchdog: Option<JoinHandle<()>>,
}

impl Gate {
    fn new() -> Self {
        let state = Arc::new(GateState::default());
        let (done, rx) = mpsc::channel();
        let guard = state.clone();
        let watchdog = std::thread::spawn(move || {
            if matches!(rx.recv_timeout(WATCHDOG), Err(mpsc::RecvTimeoutError::Timeout)) {
                guard.flags.lock().unwrap_or_else(|e| e.into_inner()).expired = true;
                guard.release();
            }
        });
        Self { state, done: Some(done), watchdog: Some(watchdog) }
    }

    fn arm(stream: CUstream) -> Self {
        let api = proof::symbols().expect("REFUSED: optional CUDA proof symbols unavailable");
        let gate = Self::new();
        let data = Arc::into_raw(gate.state.clone()) as *mut c_void;
        // SAFETY: a live precreated stream; raw userdata remains retained even on ambiguous error.
        let result = unsafe { (api.host)(stream, host_gate, data) };
        cuda::check("proof cuLaunchHostFunc", result).expect("REFUSED: CUDA host gate not accepted");
        gate.state.wait("entered", |s| s.entered);
        gate
    }

    fn release(&self) {
        self.state.release();
        self.state.wait("exited", |s| s.exited);
    }

    fn check(&self) {
        assert!(!self.state.flags.lock().unwrap_or_else(|e| e.into_inner()).expired,
            "REFUSED: watchdog released the gate; not a proof");
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        self.state.release();
        if let Some(done) = self.done.take() { let _ = done.send(()); }
        if let Some(watchdog) = self.watchdog.take() { let _ = watchdog.join(); }
    }
}

#[test]
fn cpu_gate_release_acknowledges_callback_without_cuda() {
    let gate = Gate::new();
    let raw = Arc::into_raw(gate.state.clone()) as usize;
    let callback = std::thread::spawn(move || {
        // SAFETY: this CPU control transfers exactly one Arc raw reference to the callback.
        unsafe { host_gate(raw as *mut c_void) };
    });
    gate.state.wait("entered", |s| s.entered);
    gate.release();
    callback.join().unwrap();
    gate.check();
}

#[test]
fn cpu_gate_drop_releases_callback_without_cuda() {
    let gate = Gate::new();
    let state = gate.state.clone();
    let raw = Arc::into_raw(state.clone()) as usize;
    let callback = std::thread::spawn(move || {
        // SAFETY: one retained raw Arc, consumed once by this callback.
        unsafe { host_gate(raw as *mut c_void) };
    });
    state.wait("entered", |s| s.entered);
    drop(gate);
    state.wait("exited", |s| s.exited);
    callback.join().unwrap();
}

#[test]
#[ignore = "requires explicit opt-in, exact source and one real CUDA device; one process per case"]
fn gated_cuda_custody() {
    assert_eq!(std::env::var("VARENA_GPU_PROOF").as_deref(), Ok("1"), "REFUSED: explicit opt-in missing");
    let source = option_env!("VARENA_GPU_PROOF_SOURCE").unwrap_or("");
    assert!(source.len() == 40 && source.bytes().all(|b| b.is_ascii_hexdigit()), "REFUSED: embedded source absent");
    assert_eq!(std::env::var("VARENA_GPU_PROOF_SOURCE").as_deref(), Ok(source), "REFUSED: source differs");
    let case = std::env::var("VARENA_GPU_CASE").expect("REFUSED: case missing");
    assert!(CASES.contains(&case.as_str()), "REFUSED: unknown case");
    panic!("REFUSED: source implementation in progress; no GPU proof available");
}
