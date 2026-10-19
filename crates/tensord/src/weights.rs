//! An attempt's weights: the executor's `weights_writer` requests answered over the derive and
//! source channels (`derive`). A source is granted when it is one of the call's `Weights[M]`
//! arguments, an output when its function declares it. An adopted output is kept as a local
//! model, becomes a product of the run's log (its digest the output's manifest) and, with a
//! weights destination, is published there before the adopt is answered.
use crate::{
    access::Fetch,
    derive::{self, Derivation, Hooks},
    device_executor::{Answer, Frame},
    hub, origin,
    storage::{self, Store},
};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tensorfs_core::{format::ObjectRef, sha256};

/// The media type of a weights output's product: its bytes are the output's manifest.
pub const MANIFEST_MEDIA: &str = "application/vnd.cozy.model-manifest";
/// Executor-written exchange files stay this small (TensorFS bounds its own documents).
const EXCHANGE_MAX: u64 = 64 << 20;

type BindOutput = Arc<dyn Fn(&str, &str) -> io::Result<()> + Send + Sync>;

/// Execution-owner hooks: current attempt authority, durable output association, the
/// sources' fetch, and the identity a committed derivation replays under (its function's code,
/// declaration and request; none: this run's own).
pub struct GrantAuthority {
    pub current: Arc<dyn Fn() -> bool + Send + Sync>,
    pub bind_output: BindOutput,
    pub fetch: Option<Arc<Fetch>>,
    pub replay: Option<String>,
}

/// Where a run's weights outputs are published: the repository and its Hub, written under the run's
/// capability there.
#[derive(Clone)]
pub struct Destination {
    pub repository: String,
    pub hub: hub::Source,
}

/// One attempt's weights grant.
pub struct Grant {
    pub actor: String,
    pub run: String,
    pub spool: PathBuf,
    /// Its model inputs' manifests ("sha256:<hex>") and lengths.
    pub sources: BTreeMap<String, u64>,
    /// Its declared outputs and their byte bounds.
    pub outputs: BTreeMap<String, u64>,
    pub destination: Option<Destination>,
    /// Refuses once the attempt is no longer current (canceled, paused, ended).
    pub current: Arc<dyn Fn() -> bool + Send + Sync>,
    /// Persist the run/output association before any native transaction can begin.
    bind_output: BindOutput,
    fetch: Option<Arc<Fetch>>,
    replay: Option<String>,
    opened: Mutex<BTreeMap<String, String>>,
    /// Publications attempted and not yet in the run's log.
    uploads: Mutex<Vec<Upload>>,
}

/// One output's publication into its destination: when it ran, and its refusal if any.
pub struct Upload {
    pub output: String,
    pub transaction: String,
    pub destination: String,
    pub called_ms: i64,
    pub finished_ms: i64,
    pub error: Option<String>,
}

impl Grant {
    pub fn new(
        actor: &str,
        run: &str,
        spool: &Path,
        sources: BTreeMap<String, u64>,
        outputs: BTreeMap<String, u64>,
        destination: Option<Destination>,
        authority: GrantAuthority,
    ) -> Self {
        let GrantAuthority {
            current,
            bind_output,
            fetch,
            replay,
        } = authority;
        Self {
            actor: actor.into(),
            run: run.into(),
            spool: spool.into(),
            sources,
            outputs,
            destination,
            current,
            bind_output,
            fetch,
            replay,
            opened: Mutex::new(BTreeMap::new()),
            uploads: Mutex::new(Vec::new()),
        }
    }

    /// The publications attempted since the last call, for the run's log.
    pub fn take_uploads(&self) -> Vec<Upload> {
        std::mem::take(&mut *self.uploads.lock().unwrap())
    }
}

/// An adopted output, as the run's log records it.
pub struct Adopted {
    pub output: String,
    pub manifest: ObjectRef,
}

pub struct Weights {
    store: Arc<Store>,
    /// Local custody transitions only. Channel IO and publication never hold this.
    custody: Mutex<()>,
}

type Refusal = (&'static str, String);

impl Weights {
    pub fn new(store: Arc<Store>) -> io::Result<Self> {
        Ok(Self { store, custody: Mutex::new(()) })
    }

    /// One `weights_writer` request: its answer, the channel end that follows it, and the
    /// output it adopted.
    pub fn answer(&self, grant: &Grant, frame: &Frame) -> (Answer, Option<File>, Option<Adopted>) {
        let answered = match frame.operation.as_str() {
            "source" => self.source(grant, frame).map(|(a, f)| (a, Some(f), None)),
            "output" => self.output(grant, frame).map(|(a, f)| (a, Some(f), None)),
            "adopt" => self
                .adopt(grant, frame)
                .map(|(a, adopted)| (a, None, Some(adopted))),
            other => Err((
                "weights_writer_refused",
                format!("unknown weights operation {other:?}"),
            )),
        };
        answered
            .unwrap_or_else(|(code, detail)| (Answer::refused(frame.seq, code, detail), None, None))
    }

    fn source(&self, grant: &Grant, frame: &Frame) -> Result<(Answer, File), Refusal> {
        let length = grant.sources.get(&frame.manifest).copied().ok_or((
            "weights_source_ungranted",
            "source is outside the invocation".to_string(),
        ))?;
        let source = reference(&frame.manifest, length)?;
        let (client, server) = platform::ipc::pair().map_err(io::Error::from).map_err(io_refusal)?;
        let (store, current, fetch) = (self.store.clone(), grant.current.clone(), grant.fetch.clone());
        std::thread::Builder::new()
            .name("weights-source".into())
            .spawn(move || {
                let granted = [source.clone()];
                let ensure = |objects: &[ObjectRef]| ensure(&store, fetch.as_deref(), &granted, objects);
                let hooks = Hooks { current: &*current, ensure: &ensure };
                if let Err(error) = derive::serve_source(&store, &source, server, &hooks) {
                    tracing::warn!("weights source {}: {error}", source.id());
                }
            })
            .map_err(io_refusal)?;
        let mut answer = Answer::ok(frame.seq);
        (answer.descriptor, answer.length) = (true, length);
        Ok((answer, File::from(platform::handles::Owned::from(client))))
    }

    fn output(&self, grant: &Grant, frame: &Frame) -> Result<(Answer, File), Refusal> {
        let slot = &frame.output_slot;
        let bound = grant.outputs.get(slot).copied().ok_or((
            "weights_output_ungranted",
            format!("{slot:?} is not a declared weights output"),
        ))?;
        if !(grant.current)() {
            return Err(("weights_writer_closed", "output attempt is not running".into()));
        }
        let raw = exchange(&grant.spool, slot, "derivation", frame.length)?;
        let derivation = Derivation::parse(&raw).map_err(io_refusal)?;
        if derivation.max_new_bytes != bound {
            return Err(("weights_writer_ungranted", "output exceeds invocation grant".into()));
        }
        if derivation.granted().any(|s| grant.sources.get(&s.id()) != Some(&s.length)) {
            return Err(("weights_source_ungranted", "source is outside invocation grant".into()));
        }
        // The output's work is its code and request, its slot and what it declares (its sources
        // among it): any run of the same resumes or replays the same transaction.
        let work = sha256::hex_digest(&raw);
        let code = grant.replay.as_deref().unwrap_or(&grant.run);
        let identity = format!("{}\0{code}\0{slot}\0{work}", grant.actor);
        let transaction = format!("sha256:{}", sha256::hex_digest(identity.as_bytes()));
        (grant.bind_output)(&transaction, slot).map_err(io_refusal)?;
        grant.opened.lock().unwrap().insert(transaction.clone(), slot.clone());
        let (client, server) = platform::ipc::pair().map_err(io::Error::from).map_err(io_refusal)?;
        let (store, current, fetch) = (self.store.clone(), grant.current.clone(), grant.fetch.clone());
        let (tx, output) = (transaction.clone(), slot.clone());
        std::thread::Builder::new()
            .name("weights-output".into())
            .spawn(move || {
                let granted: Vec<ObjectRef> = derivation.granted().cloned().collect();
                let ensure = |objects: &[ObjectRef]| ensure(&store, fetch.as_deref(), &granted, objects);
                let hooks = Hooks { current: &*current, ensure: &ensure };
                if let Err(error) = derive::serve_output(&store, &tx, &derivation, server, &hooks) {
                    tracing::warn!("weights output {output}: {error}");
                }
            })
            .map_err(io_refusal)?;
        let mut answer = Answer::ok(frame.seq);
        (answer.descriptor, answer.transaction) = (true, transaction);
        Ok((answer, File::from(platform::handles::Owned::from(client))))
    }

    fn adopt(&self, grant: &Grant, frame: &Frame) -> Result<(Answer, Adopted), Refusal> {
        current_or_refuse(&*grant.current)?;
        let slot = &frame.output_slot;
        let opened = grant.opened.lock().unwrap().get(&frame.transaction).cloned();
        if opened.as_ref() != Some(slot) {
            return Err(("weights_receipt_mismatch", "receipt has no opened output".into()));
        }
        // The executor relays what `commit` answered; the store's own record decides.
        #[derive(serde::Deserialize)]
        struct Relayed {
            transaction_id: String,
            manifest: ObjectRef,
        }
        let raw = exchange(&grant.spool, slot, "native-receipt", frame.length)?;
        let relayed: Relayed = serde_json::from_slice(&raw).map_err(|e| ("weights_receipt_mismatch", e.to_string()))?;
        let custody = self.custody.lock().unwrap();
        let committed = derive::committed(&self.store, &frame.transaction).map_err(io_refusal)?;
        if committed.as_ref() != Some(&relayed.manifest) || relayed.transaction_id != frame.transaction {
            return Err(("weights_receipt_mismatch", "receipt differs from the store's commit".into()));
        }
        current_or_refuse(&*grant.current)?;
        // The output stays reachable as its own local model, whatever happens to the run.
        let name = format!(
            "output-{}",
            &sha256::hex_digest(format!("{}\0{}\0{slot}", grant.actor, grant.run).as_bytes())[..40]
        );
        let manifest = derive::adopt(&self.store, &frame.transaction, &name).map_err(io_refusal)?;
        drop(custody);
        if let Some(destination) = &grant.destination {
            current_or_refuse(&*grant.current)?;
            let called_ms = crate::machine::lifecycle::now_ms();
            let published = self.publish(grant, destination, slot, &manifest);
            grant.uploads.lock().unwrap().push(Upload {
                output: slot.clone(),
                transaction: frame.transaction.clone(),
                destination: destination.repository.trim_start_matches("model://").into(),
                called_ms,
                finished_ms: crate::machine::lifecycle::now_ms(),
                error: published.as_ref().err().map(|(_, message)| message.clone()),
            });
            published?;
        }
        let mut answer = Answer::ok(frame.seq);
        current_or_refuse(&*grant.current)?;
        answer.request_id = grant.run.clone();
        answer.manifest = manifest.id();
        answer.manifest_length = manifest.length;
        answer.receipt_digest = format!("sha256:{}", sha256::hex_digest(&raw));
        Ok((answer, Adopted { output: slot.clone(), manifest }))
    }

    /// Explicit run cancellation relinquishes only its unfinished output: what its writer
    /// journaled and its pin. An adopted output stays a local model.
    pub fn cancel_output(&self, transaction: &str) -> io::Result<()> {
        let _custody = self.custody.lock().unwrap();
        derive::cancel(&self.store, transaction)
    }

    /// One adopted output into its destination, under the run's capability.
    fn publish(&self, grant: &Grant, destination: &Destination, slot: &str, manifest: &ObjectRef) -> Result<(), Refusal> {
        let repository = destination.repository.trim_start_matches("model://");
        let publishing = hub::Catalog::for_op(&destination.hub, hub::Op::Publish { model: repository })?;
        let operation = format!(
            "output-{}",
            &sha256::hex_digest(format!("{}\0{}\0{slot}", grant.actor, grant.run).as_bytes())[..40]
        );
        publishing
            .publish(&origin::Publication {
                store: &self.store,
                destination: repository,
                manifest,
                operation: &operation,
                progress: &|_, _| (),
                streams: 8,
                stop: None,
                commit: &|| true,
            })
            .map(drop)
            .map_err(|e| {
                let message = if e.code == tensorfs_core::err::Code::HUB_REFUSED {
                    format!("publication was refused before its commit: {}", e.detail)
                } else {
                    format!("publication failed before its commit: {e}")
                };
                publishing.reason("weights_publication_failed", message)
            })
    }
}

/// `objects` of the `granted` checkpoints present: those a checkpoint held in part lacks come
/// through the call's fetch (none: every source is here whole, and an absent object refuses).
fn ensure(store: &Store, fetch: Option<&Fetch>, granted: &[ObjectRef], objects: &[ObjectRef]) -> io::Result<()> {
    let Some(fetch) = fetch else {
        return Ok(());
    };
    for manifest in granted {
        let checkpoint = storage::checkpoint(store, &manifest.id())?;
        let chunks = storage::chunks(&checkpoint, &[]);
        let mine: Vec<ObjectRef> = objects.iter().filter(|o| chunks.contains(o)).cloned().collect();
        if !mine.is_empty() {
            fetch.wait(&manifest.id(), &mine)?;
        }
    }
    Ok(())
}

/// An executor-written exchange file in the attempt's spool, exactly `length` bytes.
fn exchange(spool: &Path, slot: &str, kind: &str, length: u64) -> Result<Vec<u8>, Refusal> {
    let valid = !slot.is_empty()
        && slot.len() <= 64
        && slot
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if !valid || length > EXCHANGE_MAX {
        return Err(("weights_host_identity", "weights exchange identity is invalid".into()));
    }
    let path = spool.join(format!("weights-{slot}-{kind}.canonical"));
    let mut file = File::open(&path).map_err(io_refusal)?;
    let mut bytes = Vec::with_capacity(length as usize);
    file.by_ref()
        .take(length + 1)
        .read_to_end(&mut bytes)
        .map_err(io_refusal)?;
    if bytes.len() as u64 != length {
        return Err(("weights_writer_length", "writer bytes differ from their declared length".into()));
    }
    Ok(bytes)
}

fn reference(id: &str, length: u64) -> Result<ObjectRef, Refusal> {
    storage::check_digest("manifest", id, true).map_err(io_refusal)?;
    Ok(ObjectRef { sha256: id.trim_start_matches("sha256:").into(), length })
}

fn current_or_refuse(current: &(dyn Fn() -> bool + Send + Sync)) -> Result<(), Refusal> {
    match current() {
        true => Ok(()),
        false => Err(("weights_writer_closed", "the writer's attempt is no longer current".into())),
    }
}

fn io_refusal(error: io::Error) -> Refusal {
    ("weights_writer_refused", error.to_string())
}

#[cfg(test)]
mod tests;
