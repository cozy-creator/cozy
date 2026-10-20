//! Async ownership and Result callbacks around the existing import engine.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use crate::failure::{fail, Failure, Kind, Result};
use crate::format::{Body, Tree};
use crate::import::{self, remote::Access};

use super::{provider, Checkpoint, Client};

/// A failing callback stops conversion before finalization starts.
pub type ConvertProgress<'a> = dyn Fn(&str, u64, u64) -> Result<()> + Sync + 'a;

#[derive(Default)]
pub struct Convert<'a> {
    pub converter: Option<&'a str>,
    pub variant: Option<&'a str>,
    pub progress: Option<&'a ConvertProgress<'a>>,
}

#[derive(Default)]
struct Control {
    cancelled: Arc<AtomicBool>,
    notify: Arc<tokio::sync::Notify>,
}

impl Control {
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    fn stopped(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn check(&self) -> Result<()> {
        if self.stopped() {
            fail(
                Kind::Network,
                "the conversion was stopped before publication",
            )
        } else {
            Ok(())
        }
    }
}

struct CancelOnDrop(Arc<Control>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        // No wait, filesystem cleanup, runtime creation or callback from Drop.
        self.0.cancel();
    }
}

struct Event {
    stage: String,
    held: u64,
    total: u64,
    ack: mpsc::SyncSender<()>,
}

/// Borrowed callbacks stay on the awaiting caller; the worker sees only an owned event/ack
/// channel. Returning a callback error waits for the worker to stop, preserving its exact
/// family/detail. Dropping the future releases pending acks and wakes stalled native I/O.
async fn blocking<T: Send + 'static>(
    control: Arc<Control>,
    progress: Option<&ConvertProgress<'_>>,
    work: impl FnOnce(tokio::sync::mpsc::UnboundedSender<Event>) -> Result<T> + Send + 'static,
) -> Result<T> {
    let _cancel = CancelOnDrop(control.clone());
    let (send, mut receive) = tokio::sync::mpsc::unbounded_channel();
    let worker = tokio::task::spawn_blocking(move || work(send));
    let mut first_error = None;
    while let Some(event) = receive.recv().await {
        if first_error.is_none() {
            if let Some(report) = progress {
                if let Err(error) = report(&event.stage, event.held, event.total) {
                    first_error = Some(error);
                    control.cancel();
                }
            }
        }
        // Capacity one: this never blocks the async caller waiting for a scheduled worker.
        let _ = event.ack.send(());
    }
    let result = match worker.await {
        Ok(result) => result,
        Err(error) => fail(Kind::Io, format!("conversion worker failed: {error}")),
    };
    match first_error {
        Some(error) => Err(error),
        None => result,
    }
}

impl Client {
    fn import_access(&self, sources: &[String], control: &Control) -> Result<Access> {
        let runtime = tokio::runtime::Handle::try_current().map_err(|_| Failure {
            kind: Kind::Unsupported,
            detail: "conversion needs the caller's active Tokio runtime".into(),
        })?;
        Ok(Access {
            // A supplied loopback source is the existing explicit local-source carve-out.
            allow_local: sources
                .iter()
                .any(|s| crate::fetch::policy::is_loopback_origin(s)),
            token: self.config.token.clone().map(|p| {
                Arc::new(provider::NativeProvider(p)) as Arc<dyn crate::fetch::TokenProvider>
            }),
            runtime: Some(runtime),
            cancelled: Some(control.cancelled.clone()),
            cancel_notify: Some(control.notify.clone()),
            ca_der: self.config.ca_der.clone(),
            ..crate::import::Access::from_env()
        })
    }

    /// Convert into the managed store with the same native Writer, layouts and byte engine.
    /// Callbacks may refuse until finalization starts; an atomic commit already in progress
    /// cannot be undone by dropping its awaiting future.
    pub async fn convert_to_cozytensors(&self, sources: &[String], options: &Convert<'_>) -> Result<Checkpoint> {
        if sources.is_empty() {
            return fail(Kind::Invalid, "convert needs a source");
        }
        let control = Arc::new(Control::default());
        let access = self.import_access(sources, &control)?;
        let (sources, store) = (sources.to_owned(), self.store.clone());
        let converter = options.converter.map(str::to_owned);
        let variant = options.variant.map(str::to_owned);
        let worker_control = control.clone();
        blocking(control, options.progress, move |events| {
            let report = |stage: &str, held, total| {
                let (ack, wait) = mpsc::sync_channel(1);
                if events
                    .send(Event {
                        stage: stage.into(),
                        held,
                        total,
                        ack,
                    })
                    .is_err()
                    || wait.recv().is_err()
                {
                    worker_control.cancel();
                }
            };
            let stopped = || worker_control.stopped();
            let tmp = store.root().join("tmp");
            let request = import::Request {
                sources: &sources,
                converter: converter.as_deref(),
                variant: variant.as_deref(),
                access: &access,
                tmp: &tmp,
                progress: &report,
                stop: Some(&stopped),
            };
            let mut writer = store.writer(None, None)?;
            import::convert(&request, &mut writer).map_err(provider::from_native)?;
            worker_control.check()?;
            let header = writer.committed().ok_or_else(|| Failure {
                kind: Kind::Integrity,
                detail: "conversion sealed no checkpoint".into(),
            })?;
            // Acquire the checkpoint before releasing the Writer's strong native owner.
            let raw = store.checkpoint(header)?;
            let variant = import::main_variant(raw.header().tensors().map(|(_, tensor)| {
                let (encoding, length) = match &tensor.body {
                    Body::Data(at) => (None, at.length),
                    Body::Encoded { encoding, parts } => {
                        (Some(encoding), parts.iter().map(|p| p.length).sum())
                    }
                };
                (&tensor.dtype, encoding, length)
            }));
            let mut parsed = Tree {
                files: writer.repo_files().clone(),
            };
            parsed.set_variant(&variant, header.clone());
            worker_control.check()?;
            let tree = store.put_tree(&parsed)?;
            let extra = std::iter::once(tree.clone())
                .chain(writer.repo_files().values().cloned())
                .collect();
            let checkpoint = Checkpoint::managed(
                raw,
                None,
                Some(tree.clone()),
                None,
                extra,
                store.clone(),
                Some(variant),
            )?;
            store.cache_touch(&tree)?;
            Ok(checkpoint)
        })
        .await
    }
}
