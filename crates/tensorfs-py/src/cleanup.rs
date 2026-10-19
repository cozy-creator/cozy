//! Final native releases must not wait under the GIL, or on a mover that a release joins.
//! One process-owned worker keeps the actual payload alive until its destructor completes.

use std::sync::{mpsc, OnceLock};

type Task = Box<dyn FnOnce() + Send + 'static>;
static QUEUE: OnceLock<Result<mpsc::Sender<Task>, String>> = OnceLock::new();

fn sender() -> Result<&'static mpsc::Sender<Task>, &'static String> {
    QUEUE
        .get_or_init(|| {
            let (sender, receiver) = mpsc::channel::<Task>();
            std::thread::Builder::new()
                .name("tensorfs-cleanup".into())
                .spawn(move || {
                    for task in receiver {
                        // A panicking destructor must not abandon every later queued release.
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
                    }
                })
                .map_err(|error| error.to_string())?;
            Ok(sender)
        })
        .as_ref()
}

/// Called during module initialization so a worker-creation failure is an import error,
/// before Python can own a native value that needs an asynchronous final release.
pub(crate) fn init() -> std::io::Result<()> {
    sender()
        .map(|_| ())
        .map_err(|error| std::io::Error::other(error.clone()))
}

/// Move the real final owner, not an extra clone. Returns without waiting for its release;
/// the worker never attaches to Python. A `Py<T>` in the payload follows PyO3's normal
/// deferred decref rules. Callers waiting on weakrefs/resources must allow this queued drop.
pub(crate) fn defer<T: Send + 'static>(value: T) {
    // init() has already succeeded before any class is installed. The receiver catches
    // destructor panics and cannot close while this static sender keeps it alive.
    let sender = sender().expect("TensorFS cleanup initialized before native values");
    sender
        .send(Box::new(move || drop(value)))
        .unwrap_or_else(|_| unreachable!("TensorFS cleanup receiver remains live"));
}
