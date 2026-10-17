//! The trusted owner's half of an executor's two native weight channels: one authorized derived
//! writer ([`serve_derived`]) and one exact source ([`serve_source`]). The executor's halves are
//! `python/tensorfs/_derived_channel.py` and `derived.SourceClient`. A frame is a kind byte and a
//! big-endian u32 length, then JSON or at most 1 MiB of bytes. The peer names no Store,
//! transaction or path: the owner bound all of them before handing it the connected stream.
use super::{
    add_config, add_indexed_part, check_writer, checkpoint_indexed, commit, completed,
    config_write_bound, fence, finish_writer, inspect_source, part_write_bound, release_guards,
    transaction_id, writer_store, Additions, CheckpointCursor, ComponentDeclaration,
    ConfigDeclaration, Declaration, PartDeclaration, PartSource, ReceiptFacts, Source, SourceView,
    TensorDeclaration, MAX_COMPANION_FILES,
};
use crate::canon::Value;
use crate::dtype::Dtype;
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::Part;
use crate::ids::ObjectRef;
use crate::jcs::{self, Json};
use crate::meta::{Hold, Meta};
use crate::read::{self, ReadLease};
use crate::fetch::{Access, Fetcher, Priority, Ticket};
use crate::store::Store;
use crate::transport::PullCancellation;
use std::io::{self, Read, Write};
use cozy_platform::ipc::LocalStream;
use std::sync::{Arc, Mutex};

const JSON: u8 = 0;
const BYTES: u8 = 1;
const CHUNK: usize = 1 << 20;
const METADATA: usize = 64 << 20;
const READ_MAX: u64 = 64 << 20;

/// The executor's derivation arguments (`Derivation.native_arguments`, as JSON): sources by
/// alias, targets, configs, construction order, the output's byte bound, companion files.
pub fn declaration_from_arguments(raw: &[u8]) -> Result<Declaration> {
    let Json::Arr(arguments) = jcs::parse(raw, METADATA)? else {
        return refuse(Code::WRONG_TYPE, "derivation arguments are an array");
    };
    if !(5..=6).contains(&arguments.len()) {
        return refuse(Code::WRONG_TYPE, "derivation arguments have five or six members");
    }
    let mut sources = vec![];
    for (alias, value) in object("sources", &arguments[0])? {
        let pair = array("source", value)?;
        let [manifest, length] = pair else {
            return refuse(Code::WRONG_TYPE, "a source is [manifest, length]");
        };
        sources.push(Source {
            alias: alias.clone(),
            manifest: reference(text("source", manifest)?, uint("source", length)?)?,
        });
    }
    let mut components = vec![];
    for (target, value) in object("targets", &arguments[1])? {
        let fields = Fields::new("target", value, &["source", "source_component", "drop", "add"])?;
        let mut add = vec![];
        for (key, tensor) in object("addition", fields.req("add")?)? {
            let tensor = Fields::new("addition", tensor, &["logical_dtype", "shape", "encoding", "parts"])?;
            let mut parts = vec![];
            for (role, part) in object("parts", tensor.req("parts")?)? {
                let part = Fields::new("part", part, &["dtype", "shape", "source"])?;
                let source = match part.opt("source") {
                    None => None,
                    Some(selected) => {
                        let selected = Fields::new("part source", selected, &["source", "component", "tensor", "role"])?;
                        Some(PartSource {
                            source: selected.text("source")?,
                            component: selected.text("component")?,
                            tensor: selected.text("tensor")?,
                            role: selected.text("role")?,
                        })
                    }
                };
                parts.push(PartDeclaration {
                    role: role.clone(),
                    dtype: Dtype::parse(&part.text("dtype")?)?,
                    shape: shape(part.req("shape")?)?,
                    source,
                });
            }
            add.push(TensorDeclaration {
                key: key.clone(),
                dtype: Dtype::parse(&tensor.text("logical_dtype")?)?,
                shape: shape(tensor.req("shape")?)?,
                encoding: tensor.text("encoding")?,
                parts,
            });
        }
        components.push(ComponentDeclaration {
            target: target.clone(),
            source: fields.opt("source").map(|v| text("target", v).map(String::from)).transpose()?,
            source_component: fields
                .opt("source_component")
                .map(|v| text("target", v).map(String::from))
                .transpose()?,
            drop: array("drop", fields.req("drop")?)?
                .iter()
                .map(|v| text("drop", v).map(String::from))
                .collect::<Result<_>>()?,
            add,
        });
    }
    let mut configs = vec![];
    for (target, value) in object("configs", &arguments[2])? {
        let fields = Fields::new("config", value, &["kind", "source", "source_config"])?;
        let target = target.clone();
        let source = fields.opt("source").map(|v| text("config", v).map(String::from)).transpose()?;
        let source_config =
            fields.opt("source_config").map(|v| text("config", v).map(String::from)).transpose()?;
        configs.push(match (fields.text("kind")?.as_str(), source, source_config) {
            ("copy", Some(source), Some(source_config)) => ConfigDeclaration::Copy { target, source, source_config },
            ("derive", Some(source), Some(source_config)) => ConfigDeclaration::Derive { target, source, source_config },
            ("add", None, None) => ConfigDeclaration::Add { target },
            _ => {
                return refuse(
                    Code::MISSING_FIELD,
                    format!("config {target:?} has an invalid kind/source combination"),
                )
            }
        });
    }
    let order = array("order", &arguments[3])?
        .iter()
        .map(|pair| match array("order", pair)? {
            [component, key] => Ok((text("order", component)?.to_string(), text("order", key)?.to_string())),
            _ => refuse(Code::WRONG_TYPE, "an order row is [component, key]"),
        })
        .collect::<Result<_>>()?;
    let mut files = vec![];
    if let Some(companions) = arguments.get(5) {
        for (path, encoded) in object("files", companions)? {
            files.push((path.clone(), super::decode_file(text("file", encoded)?)?));
        }
        if files.len() > MAX_COMPANION_FILES {
            return refuse(Code::COUNT_CAP, "at most 16 companion files may be declared");
        }
    }
    let mut declaration = Declaration {
        work_fingerprint: None,
        sources,
        components,
        configs,
        files,
        objects: vec![],
        order,
        max_new_bytes: uint("max_new_bytes", &arguments[4])?,
    };
    declaration.normalize_and_validate()?;
    Ok(declaration)
}

/// One native derived writer with its holds: its work hold and its sources' read leases.
pub struct Writer {
    store: Store,
    meta: std::sync::Arc<Meta>,
    transaction: String,
    session: u64,
    sources: Vec<SourceView>,
    additions: Additions,
    cursor: Mutex<Option<CheckpointCursor>>,
    holds: Mutex<Option<Holds>>,
    /// Interest in what its sources held in part lack, for the writer's life.
    tickets: Mutex<Vec<Ticket>>,
}

/// A writer's work hold and its sources' read leases, by alias.
type Holds = (Hold, Vec<(String, ReadLease)>);

impl Writer {

    /// Open an owner-authorized replacement attempt with a durable native session ID.
    pub fn begin_next(
        store: &Store,
        meta: std::sync::Arc<Meta>,
        transaction: &str,
        declaration: Declaration,
        checkpoint: Option<&ObjectRef>,
    ) -> Result<Writer> {
        let next = super::begin_next(store, &meta, transaction, declaration, checkpoint)?;
        Self::from_begin(store, meta, transaction, next.writer_session_id, next.begun)
    }

    fn from_begin(
        store: &Store,
        meta: std::sync::Arc<Meta>,
        transaction: &str,
        session: u64,
        begun: super::Begin,
    ) -> Result<Writer> {
        let writer_store = match writer_store(store) {
            Ok(store) => store,
            Err(error) => {
                release_guards(&meta, Some(begun.writer_hold), begun.source_leases);
                return Err(error);
            }
        };
        Ok(Writer {
            store: writer_store,
            meta,
            transaction: transaction_id(transaction)?,
            session,
            sources: begun.source_views,
            additions: begun.additions,
            cursor: Mutex::new(None),
            holds: Mutex::new(Some((begun.writer_hold, begun.source_leases))),
            tickets: Mutex::default(),
        })
    }
    /// A source part's bytes, under the lease that pins it: the source's own from begin, or for
    /// a part of a source here in part, one taken when it is first read, after `fetch`.
    #[allow(clippy::too_many_arguments)]
    fn source_read_into(&self, alias: &str, component: &str, key: &str, role: &str, offset: u64, into: &mut [u8], fetch: Option<&Fetching>) -> Result<()> {
        let mut holds = self.holds.lock().unwrap();
        let (_, leases) = holds.as_mut().ok_or_else(released)?;
        let view = self.sources.iter().find(|view| view.fact.alias == alias).ok_or_else(|| Refusal {
            code: Code::ROOT_ABSENT,
            detail: format!("no source alias {alias:?} on this derived writer"),
        })?;
        if !view.fact.components.iter().any(|allowed| allowed == component) {
            return refuse(
                Code::MISSING_COMPONENT,
                format!("source alias {alias:?} component {component:?} is outside the locked target declaration"),
            );
        }
        let part = header_part(view, component, key, role)?;
        let covers = |lease: &ReadLease| part.segments().iter().all(|s| lease.objects().contains(s));
        let at = match leases.iter().position(|(held, lease)| held == alias && covers(lease)) {
            Some(at) => at,
            None => {
                let absent: Vec<ObjectRef> =
                    part.segments().iter().filter(|s| !self.store.contains(&s.sha256)).cloned().collect();
                if let (false, Some(fetch)) = (absent.is_empty(), fetch) {
                    fetch.wait(&view.fact.manifest, &absent, &mut self.tickets.lock().unwrap())?;
                }
                let lease = read::acquire(&self.store, &self.meta, &view.fact.manifest.sha256, part.segments().to_vec())?.0;
                leases.push((alias.to_string(), lease));
                leases.len() - 1
            }
        };
        check_writer(&self.meta, &self.transaction, self.session)?;
        let what = format!("{component}/{key}#{role}");
        read::read_part_into(&leases[at].1, &what, part, offset, into.len() as u64, into)?;
        check_writer(&self.meta, &self.transaction, self.session)
    }

    fn release(&self) {
        let Some((hold, leases)) = self.holds.lock().unwrap().take() else {
            return;
        };
        release_guards(&self.meta, Some(hold), leases);
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        self.release();
    }
}

fn released() -> Refusal {
    Refusal {
        code: Code::LEASE_REVOKED,
        detail: "derived writer holds were released".into(),
    }
}

fn header_part<'v>(view: &'v SourceView, component: &str, key: &str, role: &str) -> Result<&'v Part> {
    view.header
        .components
        .iter()
        .find(|(name, _)| name == component)
        .and_then(|(_, tensors)| tensors.iter().find(|(name, _)| name == key))
        .and_then(|(_, tensor)| tensor.parts.iter().find(|(name, _)| name == role))
        .map(|(_, part)| part)
        .ok_or_else(|| Refusal {
            code: Code::MISSING_TENSOR,
            detail: format!("source {} has no exact part {component}/{key}#{role}", view.fact.alias),
        })
}

/// How a reader of a source held here in part makes its absent objects present: the Store's
/// fetcher, the access the attempt fetches under, and the stop that ends its waits. They come
/// from the origin the source's custody record names ([`crate::origin`]).
pub struct Fetching<'a> {
    pub fetcher: &'a Fetcher,
    pub access: Arc<dyn Access>,
    pub stop: &'a PullCancellation,
}

impl Fetching<'_> {
    /// Wait for `absent` objects of `manifest`, under a ticket `tickets` keeps for its origin.
    fn wait(&self, manifest: &ObjectRef, absent: &[ObjectRef], tickets: &mut Vec<Ticket>) -> Result<()> {
        let origin = crate::origin::of(self.fetcher.store().root(), manifest)?.ok_or_else(|| Refusal {
            code: Code::OBJECT_ABSENT,
            detail: format!("{} lacks {} and no origin holds it", manifest.id(), absent[0].id()),
        })?;
        let at = match tickets.iter().position(|ticket| *ticket.origin() == origin) {
            Some(at) => at,
            None => {
                tickets.push(self.fetcher.want(&origin, Arc::clone(&self.access), &[], Priority::Now));
                tickets.len() - 1
            }
        };
        tickets[at].wait(absent, self.stop)
    }
}

/// What the owner keeps of a writer's progress (each checkpoint as it lands and the receipt),
/// and how it fetches what a source here in part lacks.
pub struct Hooks<'a> {
    /// Refuses once the execution that owns the writer is no longer current.
    pub check_current: &'a (dyn Fn() -> Result<()> + Sync),
    pub record_checkpoint: &'a (dyn Fn(&Json) + Sync),
    pub record_receipt: &'a (dyn Fn(&ReceiptFacts) + Sync),
    /// None: every source is here whole, and an absent object refuses.
    pub fetch: Option<&'a Fetching<'a>>,
}

/// Serve one prebound writer until its commit, fence or peer loss, then fence what is live.
pub fn serve_derived(
    writer: &Writer,
    stream: LocalStream,
    operation: &str,
    slot: &str,
    mut checkpoint: Option<ObjectRef>,
    hooks: &Hooks,
) -> Result<()> {
    let mut channel = Channel(stream);
    let served = (|| -> Result<()> {
        loop {
            let Some(frame) = channel.receive()? else {
                return Ok(());
            };
            let Frame::Json(request) = frame else {
                return refuse(Code::WRONG_TYPE, "expected a derived operation");
            };
            (hooks.check_current)()?;
            let op = request.op()?;
            let result = match op.as_str() {
                "read" => {
                    read_source(writer, &mut channel, &request, hooks)?;
                    continue;
                }
                "part" => {
                    let fields = request.only(&["op", "component", "key", "role"])?;
                    let (component, key, role) = (fields.text("component")?, fields.text("key")?, fields.text("role")?);
                    part_write_bound(&writer.meta, &writer.transaction, writer.session, &writer.additions, (&component, &key, &role))?;
                    let mut reader = Pull { channel: &mut channel, writer, hooks, ended: false };
                    let written = add_indexed_part(
                        &writer.store,
                        &writer.meta,
                        &writer.transaction,
                        writer.session,
                        &writer.additions,
                        writer.cursor.lock().unwrap().as_mut(),
                        (&component, &key, &role),
                        &mut reader,
                    )?;
                    let objects = written
                        .part
                        .segments()
                        .iter()
                        .map(|o| obj(vec![("sha256", Json::Str(o.sha256.clone())), ("length", num(o.length))]))
                        .collect::<Result<_>>()?;
                    obj(vec![
                        ("bytes", num(written.bytes)),
                        ("deduped_bytes", num(written.deduped)),
                        ("whole_sha256", Json::Str(written.whole)),
                        ("objects", Json::Arr(objects)),
                    ])?
                }
                "config" => {
                    let fields = request.only(&["op", "name"])?;
                    let name = fields.text("name")?;
                    config_write_bound(&writer.meta, &writer.transaction, writer.session, &name)?;
                    let mut reader = Pull { channel: &mut channel, writer, hooks, ended: false };
                    // A config is not a part: the next checkpoint takes the full path.
                    *writer.cursor.lock().unwrap() = None;
                    add_config(&writer.meta, &writer.transaction, writer.session, &name, &mut reader)?;
                    Json::Null
                }
                "receipt" | "completed_parts" | "completed_configs" | "checkpoint" | "commit" | "fence" => {
                    request.only(&["op"])?;
                    match op.as_str() {
                        "receipt" => Json::Null,
                        "completed_parts" => {
                            let (parts, _) = completed(&writer.meta, &writer.transaction, writer.session)?;
                            Json::Arr(
                                parts
                                    .into_iter()
                                    .map(|(c, k, r)| Json::Arr(vec![Json::Str(c), Json::Str(k), Json::Str(r)]))
                                    .collect(),
                            )
                        }
                        "completed_configs" => {
                            let (_, configs) = completed(&writer.meta, &writer.transaction, writer.session)?;
                            Json::Arr(configs.into_iter().map(Json::Str).collect())
                        }
                        "checkpoint" => {
                            let (plan, head) = checkpoint_indexed(
                                &writer.store,
                                &writer.meta,
                                &writer.transaction,
                                writer.session,
                                operation,
                                slot,
                                checkpoint.as_ref(),
                                &mut writer.cursor.lock().unwrap(),
                            )?;
                            let result = obj(vec![
                                ("plan_digest", Json::Str(plan)),
                                ("head", head.head.as_ref().map_or(Json::Null, |r| Json::Str(r.id()))),
                                ("head_length", num(head.head.as_ref().map_or(0, |r| r.length))),
                                ("index", num(head.links.saturating_sub(1))),
                                ("bytes", num(head.bytes)),
                            ])?;
                            (hooks.check_current)()?;
                            (hooks.record_checkpoint)(&result);
                            if head.head.is_some() {
                                checkpoint = head.head;
                            }
                            result
                        }
                        "commit" => {
                            {
                                let holds = writer.holds.lock().unwrap();
                                let (hold, _) = holds.as_ref().ok_or_else(released)?;
                                hold.still_registered(&writer.meta)?;
                            }
                            let facts = commit(&writer.store, &writer.meta, &writer.transaction, writer.session)?;
                            finish_writer(&writer.store)?;
                            writer.release();
                            // Recorded even if cancellation raced: the native winner is reconciled.
                            (hooks.record_receipt)(&facts);
                            json(&facts.to_value())
                        }
                        _ => Json::Bool(fence(&writer.meta, &writer.transaction, writer.session)?),
                    }
                }
                _ => return refuse(Code::SOURCE_NOT_ALLOWED, "operation is outside this derived handle"),
            };
            channel.send(&obj(vec![("result", result)])?)?;
            if op == "commit" || op == "fence" {
                return Ok(());
            }
        }
    })();
    if let Err(refusal) = &served {
        let _ = channel.refuse(refusal);
    }
    let _ = fence(&writer.meta, &writer.transaction, writer.session);
    writer.release();
    served
}

/// A completed output's receipt for an executor that reopens it (a resumed attempt).
pub fn serve_derived_replay(receipt: &ReceiptFacts, stream: LocalStream, hooks: &Hooks) -> Result<()> {
    let mut channel = Channel(stream);
    let served = (|| -> Result<()> {
        let Some(Frame::Json(request)) = channel.receive()? else {
            return refuse(Code::WRONG_TYPE, "expected a receipt request");
        };
        request.only(&["op"])?;
        if request.op()? != "receipt" {
            return refuse(Code::SOURCE_NOT_ALLOWED, "a completed output answers its receipt only");
        }
        (hooks.check_current)()?;
        (hooks.record_receipt)(receipt);
        channel.send(&obj(vec![("result", json(&receipt.to_value()))])?)
    })();
    if let Err(refusal) = &served {
        let _ = channel.refuse(refusal);
    }
    served
}

/// Serve one exact source's inspection, checks and part reads until the peer closes. A source
/// here whole is pinned under one read lease at its first check or read; one here in part (its
/// origin holds the rest, [`crate::origin`]) pins each part as it is first read, after `fetch`
/// makes its absent objects resident. Leases are held until the channel ends.
pub fn serve_source(
    store: &Store,
    meta: &Meta,
    source: &ObjectRef,
    stream: LocalStream,
    check_current: &(dyn Fn() -> Result<()> + Sync),
    fetch: Option<&Fetching>,
) -> Result<()> {
    let mut channel = Channel(stream);
    let mut leases: Vec<ReadLease> = vec![];
    let mut tickets: Vec<Ticket> = vec![];
    let mut view: Option<SourceView> = None;
    let served = (|| -> Result<()> {
        loop {
            let Some(frame) = channel.receive()? else {
                return Ok(());
            };
            let Frame::Json(request) = frame else {
                return refuse(Code::WRONG_TYPE, "expected a scoped source operation");
            };
            let op = request.op()?;
            let fields = match op.as_str() {
                "inspect" => request.only(&["op", "components", "configs"])?,
                "read" => request.only(&["op", "manifest", "manifest_length", "component", "key", "role", "offset", "length"])?,
                "check" => request.only(&["op", "manifest", "manifest_length"])?,
                _ => return refuse(Code::SOURCE_NOT_ALLOWED, "operation is outside this source handle"),
            };
            if op != "inspect" && (fields.text("manifest")? != source.id() || fields.uint("manifest_length")? != source.length) {
                return refuse(Code::TRANSACTION_CONFLICT, "source operation changed its bound identity");
            }
            check_current()?;
            if view.is_none() {
                // Exact-source validation runs even when the first operation is a read.
                // An empty selection means everything the source admits.
                let snapshot = store.read_manifest(source)?;
                let header_ref = snapshot.header().ok_or_else(|| Refusal {
                    code: Code::MISSING_FIELD,
                    detail: "scoped source has no tensor header".into(),
                })?;
                let header = crate::checkpoint::load_header(store, header_ref)?;
                let components = header.components.iter().map(|(name, _)| name.clone()).collect();
                let configs = header.configs.iter().map(|(name, _)| name.clone()).collect();
                view = Some(inspect_source(store, meta, source.clone(), components, configs)?.view);
            }
            if op != "inspect" && leases.is_empty() {
                let walk = crate::checkpoint::walk_cozytensors(store, &store.read_manifest(source)?)?;
                let mut objects: Vec<ObjectRef> = walk.distinct().into_iter().cloned().collect();
                if !objects.iter().all(|object| store.contains(&object.sha256)) {
                    objects = vec![view.as_ref().unwrap().fact.header.clone()];
                }
                leases.push(read::acquire(store, meta, &source.sha256, objects)?.0);
            }
            if op == "check" {
                for lease in &leases {
                    lease.recheck(meta)?;
                }
            }
            match op.as_str() {
                "inspect" => {
                    let names = |key: &str| -> Result<Vec<String>> {
                        array(key, fields.req(key)?)?.iter().map(|v| text(key, v).map(String::from)).collect()
                    };
                    let source_view = view.as_ref().unwrap();
                    let mut components = names("components")?;
                    if components.is_empty() {
                        components = source_view.fact.components.clone();
                    }
                    let mut selected_configs = names("configs")?;
                    if selected_configs.is_empty() {
                        selected_configs = source_view.configs.clone();
                    }
                    let inspection = inspect_source(store, meta, source.clone(), components, selected_configs)?;
                    let configs = inspection
                        .configs
                        .iter()
                        .map(|c| obj(vec![("name", Json::Str(c.name.clone())), ("length", num(c.bytes.len() as u64))]))
                        .collect::<Result<_>>()?;
                    let mut result = inspection_value(&inspection.view)?;
                    if let Json::Obj(pairs) = &mut result {
                        pairs.retain(|(k, _)| k != "configs");
                        pairs.push(("configs".into(), Json::Arr(configs)));
                        result = Json::object(std::mem::take(pairs))?;
                    }
                    channel.send(&obj(vec![("result", result)])?)?;
                    for config in &inspection.configs {
                        for chunk in config.bytes.chunks(CHUNK) {
                            check_current()?;
                            channel.data(chunk)?;
                        }
                    }
                }
                "read" => {
                    let length = fields.uint("length")?;
                    if length > READ_MAX {
                        return refuse(Code::SIZE_CAP, "source read exceeds its buffer bound");
                    }
                    let (component, key, role) = (fields.text("component")?, fields.text("key")?, fields.text("role")?);
                    let part = header_part(view.as_ref().unwrap(), &component, &key, &role)?;
                    let at = match leases.iter().position(|lease| part.segments().iter().all(|s| lease.objects().contains(s))) {
                        Some(at) => at,
                        None => {
                            let absent: Vec<ObjectRef> =
                                part.segments().iter().filter(|s| !store.contains(&s.sha256)).cloned().collect();
                            if let (false, Some(fetch)) = (absent.is_empty(), fetch) {
                                fetch.wait(source, &absent, &mut tickets)?;
                            }
                            leases.push(read::acquire(store, meta, &source.sha256, part.segments().to_vec())?.0);
                            leases.len() - 1
                        }
                    };
                    leases[at].recheck(meta)?;
                    let mut data = vec![0; length as usize];
                    let what = format!("{component}/{key}#{role}");
                    read::read_part_into(&leases[at], &what, part, fields.uint("offset")?, length, &mut data)?;
                    leases[at].recheck(meta)?;
                    check_current()?;
                    channel.send(&obj(vec![("result", num(length))])?)?;
                    for chunk in data.chunks(CHUNK) {
                        check_current()?;
                        channel.data(chunk)?;
                    }
                }
                _ => channel.send(&obj(vec![("result", Json::Null)])?)?,
            }
        }
    })();
    if let Err(refusal) = &served {
        let _ = channel.refuse(refusal);
    }
    for lease in leases {
        let _ = lease.release(meta);
    }
    served
}

/// The executor's `SourceInspection.from_native` facts: the source and its selected
/// components' tensors (geometry and identity, no payload locations).
fn inspection_value(view: &SourceView) -> Result<Json> {
    let fact = &view.fact;
    let source = obj(vec![
        ("alias", Json::Str(fact.alias.clone())),
        ("manifest_id", Json::Str(fact.manifest.id())),
        ("manifest_length", num(fact.manifest.length)),
        ("header_id", Json::Str(fact.header.id())),
        ("header_length", num(fact.header.length)),
        ("components", Json::Arr(fact.components.iter().cloned().map(Json::Str).collect())),
    ])?;
    let mut components = vec![];
    for (name, tensors) in &view.header.components {
        if !fact.components.contains(name) {
            continue;
        }
        let mut rows = vec![];
        for (key, tensor) in tensors {
            let parts = tensor
                .parts
                .iter()
                .map(|(role, part)| {
                    obj(vec![
                        ("role", Json::Str(role.clone())),
                        ("dtype", Json::Str(part.dtype.name().into())),
                        ("shape", Json::Arr(part.shape.iter().map(|&d| num(d)).collect())),
                    ])
                })
                .collect::<Result<_>>()?;
            rows.push(obj(vec![
                ("key", Json::Str(key.clone())),
                ("logical_dtype", Json::Str(tensor.dtype.name().into())),
                ("shape", Json::Arr(tensor.shape.iter().map(|&d| num(d)).collect())),
                ("encoding", Json::Str(tensor.encoding.clone())),
                ("identity", Json::Str(tensor.encoded_identity()?)),
                ("parts", Json::Arr(parts)),
            ])?);
        }
        components.push(obj(vec![("name", Json::Str(name.clone())), ("tensors", Json::Arr(rows))])?);
    }
    obj(vec![
        ("source", source),
        ("components", Json::Arr(components)),
        ("configs", Json::Arr(vec![])),
    ])
}

/// One bounded source-read window inside a writer, answered with its bytes.
fn read_source(writer: &Writer, channel: &mut Channel, request: &Json, hooks: &Hooks) -> Result<()> {
    let fields = request.only(&["op", "source", "component", "key", "role", "offset", "length"])?;
    let length = fields.uint("length")?;
    if length == 0 || length > READ_MAX {
        return refuse(Code::RANGE_BOUNDS, "invalid derived source range");
    }
    (hooks.check_current)()?;
    let mut target = vec![0; length as usize];
    writer.source_read_into(
        &fields.text("source")?,
        &fields.text("component")?,
        &fields.text("key")?,
        &fields.text("role")?,
        fields.uint("offset")?,
        &mut target,
        hooks.fetch,
    )?;
    (hooks.check_current)()?;
    channel.send(&obj(vec![("length", num(length))])?)?;
    for chunk in target.chunks(CHUNK) {
        channel.data(chunk)?;
    }
    Ok(())
}

/// A part's or config's bytes, pulled from the producer a bounded chunk at a time; the
/// producer may read its sources between chunks.
struct Pull<'c, 'w, 'h> {
    channel: &'c mut Channel,
    writer: &'w Writer,
    hooks: &'w Hooks<'h>,
    ended: bool,
}

impl Read for Pull<'_, '_, '_> {
    fn read(&mut self, into: &mut [u8]) -> io::Result<usize> {
        let pulled = (|| -> Result<usize> {
            (self.hooks.check_current)()?;
            if self.ended || into.is_empty() {
                return Ok(0);
            }
            let size = into.len().min(CHUNK);
            self.channel.send(&obj(vec![("pull", num(size as u64))])?)?;
            loop {
                match self.channel.receive()? {
                    None => return refuse(Code::TRANSFER_FAILED, "producer disconnected mid-stream"),
                    Some(Frame::Bytes(bytes)) => {
                        if bytes.len() > size {
                            return refuse(Code::LENGTH_MISMATCH, "producer exceeded native pull");
                        }
                        self.ended = bytes.is_empty();
                        into[..bytes.len()].copy_from_slice(&bytes);
                        return Ok(bytes.len());
                    }
                    // Only bounded source reads may nest inside a part.
                    Some(Frame::Json(request)) => read_source(self.writer, self.channel, &request, self.hooks)?,
                }
            }
        })();
        pulled.map_err(|refusal| io::Error::other(format!("{}: {}", refusal.code.as_str(), refusal.detail)))
    }
}

pub(crate) enum Frame {
    Json(Json),
    Bytes(Vec<u8>),
}

pub(crate) struct Channel(pub(crate) LocalStream);

impl Channel {
    fn write(&mut self, kind: u8, body: &[u8]) -> Result<()> {
        let mut head = [kind, 0, 0, 0, 0];
        head[1..].copy_from_slice(&(body.len() as u32).to_be_bytes());
        self.0
            .write_all(&head)
            .and_then(|()| self.0.write_all(body))
            .map_err(|e| Refusal {
                code: Code::TRANSFER_FAILED,
                detail: format!("weight channel write: {e}"),
            })
    }
    pub(crate) fn send(&mut self, value: &Json) -> Result<()> {
        let raw = jcs::write(value);
        if raw.len() > METADATA {
            return refuse(Code::SIZE_CAP, "derived metadata exceeds its bound");
        }
        self.write(JSON, &raw)
    }
    fn data(&mut self, bytes: &[u8]) -> Result<()> {
        self.write(BYTES, bytes)
    }
    fn refuse(&mut self, refusal: &Refusal) -> Result<()> {
        let detail: String = refusal.detail.chars().take(1024).collect();
        self.send(&obj(vec![("error", Json::Str(refusal.code.as_str().into())), ("detail", Json::Str(detail))])?)
    }
    /// The next frame, or None when the peer closed between frames.
    pub(crate) fn receive(&mut self) -> Result<Option<Frame>> {
        let mut head = [0u8; 5];
        match self.0.read_exact(&mut head) {
            Ok(()) => (),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return refuse(Code::TRANSFER_FAILED, format!("weight channel read: {e}")),
        }
        let length = u32::from_be_bytes(head[1..].try_into().unwrap()) as usize;
        let bound = match head[0] {
            JSON => METADATA,
            BYTES => CHUNK,
            _ => return refuse(Code::SIZE_CAP, "invalid derived frame"),
        };
        if length > bound {
            return refuse(Code::SIZE_CAP, "invalid derived frame");
        }
        let mut body = vec![0; length];
        self.0.read_exact(&mut body).map_err(|e| Refusal {
            code: Code::TRANSFER_FAILED,
            detail: format!("weight channel read: {e}"),
        })?;
        Ok(Some(match head[0] {
            JSON => match jcs::parse(&body, METADATA)? {
                value @ Json::Obj(_) => Frame::Json(value),
                _ => return refuse(Code::WRONG_TYPE, "derived frame must be an object"),
            },
            _ => Frame::Bytes(body),
        }))
    }
}

/// A request's fields, closed over the names its operation allows.
struct Fields<'a> {
    pairs: &'a [(String, Json)],
}

impl Fields<'_> {
    fn new<'a>(what: &str, value: &'a Json, allowed: &[&str]) -> Result<Fields<'a>> {
        let Json::Obj(pairs) = value else {
            return refuse(Code::WRONG_TYPE, format!("{what} is an object"));
        };
        if let Some((key, _)) = pairs.iter().find(|(key, _)| !allowed.contains(&key.as_str())) {
            return refuse(Code::UNKNOWN_FIELD, format!("{what}: unknown field {key:?}"));
        }
        Ok(Fields { pairs })
    }
    fn opt(&self, key: &str) -> Option<&Json> {
        self.pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    fn req(&self, key: &str) -> Result<&Json> {
        self.opt(key).ok_or_else(|| Refusal {
            code: Code::MISSING_FIELD,
            detail: format!("missing {key:?}"),
        })
    }
    fn text(&self, key: &str) -> Result<String> {
        text(key, self.req(key)?).map(String::from)
    }
    fn uint(&self, key: &str) -> Result<u64> {
        uint(key, self.req(key)?)
    }
}

trait Request {
    fn op(&self) -> Result<String>;
    fn only(&self, allowed: &[&str]) -> Result<Fields<'_>>;
}

impl Request for Json {
    fn op(&self) -> Result<String> {
        match self {
            Json::Obj(pairs) => match pairs.iter().find(|(k, _)| k == "op") {
                Some((_, Json::Str(op))) => Ok(op.clone()),
                _ => refuse(Code::MISSING_FIELD, "an operation names its op"),
            },
            _ => refuse(Code::WRONG_TYPE, "an operation is an object"),
        }
    }
    fn only(&self, allowed: &[&str]) -> Result<Fields<'_>> {
        let fields = Fields::new("operation", self, allowed)?;
        if fields.pairs.len() != allowed.len() {
            return refuse(Code::UNKNOWN_FIELD, "operation is outside this derived handle");
        }
        Ok(fields)
    }
}

fn object<'a>(what: &str, value: &'a Json) -> Result<&'a [(String, Json)]> {
    match value {
        Json::Obj(pairs) => Ok(pairs),
        _ => refuse(Code::WRONG_TYPE, format!("{what} is an object")),
    }
}

pub(crate) fn array<'a>(what: &str, value: &'a Json) -> Result<&'a [Json]> {
    match value {
        Json::Arr(items) => Ok(items),
        _ => refuse(Code::WRONG_TYPE, format!("{what} is an array")),
    }
}

pub(crate) fn text<'a>(what: &str, value: &'a Json) -> Result<&'a str> {
    match value {
        Json::Str(text) => Ok(text),
        _ => refuse(Code::WRONG_TYPE, format!("{what} is a string")),
    }
}

pub(crate) fn uint(what: &str, value: &Json) -> Result<u64> {
    match value {
        Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Ok(*n as u64),
        _ => refuse(Code::NUMBER_RANGE, format!("{what} is a nonnegative integer")),
    }
}

fn shape(value: &Json) -> Result<Vec<u64>> {
    array("shape", value)?.iter().map(|d| uint("shape", d)).collect()
}

pub(crate) fn reference(id: &str, length: u64) -> Result<ObjectRef> {
    let sha256 = id.strip_prefix("sha256:").unwrap_or(id).to_string();
    crate::ids::hex64("ObjectRef", &sha256)?;
    Ok(ObjectRef { sha256, length })
}

pub(crate) fn num(value: u64) -> Json {
    Json::Num(value as f64)
}

pub(crate) fn obj(pairs: Vec<(&str, Json)>) -> Result<Json> {
    Json::object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// A canonical document as channel JSON.
fn json(value: &Value) -> Json {
    match value {
        Value::Bool(b) => Json::Bool(*b),
        Value::Int(i) => Json::Num(*i as f64),
        Value::Str(s) => Json::Str(s.clone()),
        Value::Arr(items) => Json::Arr(items.iter().map(json).collect()),
        Value::Obj(pairs) => Json::Obj(pairs.iter().map(|(k, v)| (k.clone(), json(v))).collect()),
    }
}

#[cfg(test)]
mod source_channel_tests {
    use super::super::begin;
    use super::*;

    #[test]
    fn native_source_inspection_defaults_to_all_admitted_components_and_configs() {
        let root = std::env::temp_dir().join(format!(
            "native-source-channel-{}",
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let meta = Meta::open(&store).unwrap();
        let transaction = format!("sha256:{}", "91".repeat(32));
        let mut declaration = super::super::tests::created_declaration();
        declaration.configs.push(ConfigDeclaration::Add {
            target: "scheduler".into(),
        });
        declaration.max_new_bytes = 4096;
        let begun = super::super::begin(&store, &meta, &transaction, 1, declaration, None).unwrap();
        let data = vec![0u8; 2048];
        super::super::add_part(
            &store,
            &meta,
            &transaction,
            1,
            ("model", "weight", "value"),
            &mut data.as_slice(),
        )
        .unwrap();
        super::super::add_config(
            &meta,
            &transaction,
            1,
            "scheduler",
            &mut &b"{\"seed\":1}"[..],
        )
        .unwrap();
        let receipt = super::super::commit(&store, &meta, &transaction, 1).unwrap();
        drop(begun);
        let (client, server) = cozy_platform::ipc::pair().unwrap();
        std::thread::scope(|scope| {
            let held =
                scope.spawn(|| serve_source(&store, &meta, &receipt.manifest, server, &|| Ok(()), None));
            let mut client = Channel(client);
            client
                .send(
                    &jcs::parse(
                        br#"{"op":"inspect","components":[],"configs":[]}"#,
                        METADATA,
                    )
                    .unwrap(),
                )
                .unwrap();
            let Some(Frame::Json(answer)) = client.receive().unwrap() else {
                panic!("metadata reply absent")
            };
            let answer = String::from_utf8(jcs::write(&answer)).unwrap();
            assert!(!answer.contains("\"error\""), "{answer}");
            assert!(answer.contains("\"name\":\"model\""), "{answer}");
            assert!(answer.contains("\"name\":\"scheduler\""), "{answer}");
            let Some(Frame::Bytes(config)) = client.receive().unwrap() else {
                panic!("config reply absent")
            };
            assert_eq!(config, b"{\"seed\":1}");
            drop(client);
            held.join().unwrap().unwrap();
        });
        let _ = std::fs::remove_dir_all(root);
    }

    fn read_part(client: &mut Channel, source: &ObjectRef, key: &str) -> Vec<u8> {
        let request = format!(
            r#"{{"op":"read","manifest":"{}","manifest_length":{},"component":"model","key":"{key}","role":"value","offset":0,"length":2048}}"#,
            source.id(),
            source.length
        );
        client.send(&jcs::parse(request.as_bytes(), METADATA).unwrap()).unwrap();
        let Some(Frame::Json(answer)) = client.receive().unwrap() else { panic!("read reply absent") };
        let answer = String::from_utf8(jcs::write(&answer)).unwrap();
        assert!(!answer.contains("\"error\""), "{answer}");
        let mut data = vec![];
        while data.len() < 2048 {
            let Some(Frame::Bytes(chunk)) = client.receive().unwrap() else { panic!("bytes absent") };
            data.extend(chunk);
        }
        data
    }

    #[test]
    fn a_source_held_in_part_derives_by_reference_and_reads_through_its_origin() {
        let root = std::env::temp_dir().join(format!("native-source-origin-{}", crate::meta::now_nanos_unique()));
        let store = Store::init(&root).unwrap();
        let meta = Meta::open(&store).unwrap();
        let source_id = format!("sha256:{}", "93".repeat(32));
        let mut declaration = super::super::tests::created_declaration();
        let mut other = declaration.components[0].add[0].clone();
        other.key = "other".into();
        declaration.components[0].add.push(other);
        declaration.order.push(("model".into(), "other".into()));
        declaration.max_new_bytes = 4096;
        let writer = begin(&store, &meta, &source_id, 1, declaration, None).unwrap();
        let (weight, held_back) = (vec![0x21u8; 2048], vec![0x42u8; 2048]);
        for (key, bytes) in [("weight", &weight), ("other", &held_back)] {
            super::super::add_part(&store, &meta, &source_id, 1, ("model", key, "value"), &mut bytes.as_slice()).unwrap();
        }
        let source = commit(&store, &meta, &source_id, 1).unwrap().manifest;
        release_guards(&meta, Some(writer.writer_hold), writer.source_leases);

        // The origin holds `other`; this store has the rest.
        let absent = ObjectRef::of(&held_back);
        let walk = crate::checkpoint::walk(&store, &store.read_manifest(&source).unwrap()).unwrap();
        let mut objects: Vec<ObjectRef> = walk.distinct().into_iter().cloned().collect();
        objects.sort_by(|a, b| a.sha256.cmp(&b.sha256));
        let bodies = (objects.iter())
            .map(|o| (o.sha256.clone(), std::fs::read(store.object_path(&o.sha256)).unwrap()))
            .chain([(source.sha256.clone(), std::fs::read(store.manifest_path(&source.sha256)).unwrap())])
            .collect();
        let hub = crate::transport::tests::hub_origin(
            std::sync::Arc::new(crate::transport::tests::Checkpoint { bodies, manifest: source.clone(), objects }),
            |_, _| None,
        );
        std::fs::remove_file(store.object_path(&absent.sha256)).unwrap();
        let origin = crate::fetch::Origin {
            hub: hub.base.clone(),
            refspec: "acme/model@r1".into(),
            lane: "public".into(),
            manifest: source.clone(),
        };
        crate::origin::attest(&store, &source, &origin).unwrap();
        assert!(crate::ingest::custody::live(&root).unwrap().contains(&absent.sha256));

        // A derivation inherits it by reference, and its output names the same origin.
        let output_id = format!("sha256:{}", "94".repeat(32));
        let inherit = Declaration {
            files: Vec::new(),
            objects: Vec::new(),
            work_fingerprint: Some(format!("sha256:{}", "44".repeat(32))),
            sources: vec![Source { alias: "source".into(), manifest: source.clone() }],
            components: vec![ComponentDeclaration {
                target: "model".into(),
                source: Some("source".into()),
                source_component: Some("model".into()),
                drop: Vec::new(),
                add: Vec::new(),
            }],
            configs: Vec::new(),
            order: vec![("model".into(), "other".into()), ("model".into(), "weight".into())],
            max_new_bytes: 0,
        };
        let writer = begin(&store, &meta, &output_id, 1, inherit.clone(), None).unwrap();
        let output = commit(&store, &meta, &output_id, 1).unwrap().manifest;
        release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
        assert_eq!(crate::origin::of(&root, &output).unwrap(), Some(origin.clone()));
        let repo = crate::repository::RepositoryName::new("local", "output").unwrap();
        let mutation = crate::repository::Mutation::ReplaceLocal { repo, version: "94".repeat(32), manifest: output.clone() };
        store.apply_repository(None, &mutation, &crate::store::Fault::default()).unwrap();
        crate::gc::collect(&root, false).unwrap();
        assert!(!store.contains(&absent.sha256));

        // Reads fetch only what is absent, once: a writer's source reads, then a source channel's.
        let fetcher = Fetcher::new(&store, 8, std::time::Duration::from_millis(50)).unwrap();
        let stop = PullCancellation::default();
        let access: Arc<dyn Access> = Arc::new(crate::fetch::Plain {
            credential: crate::transport::Anonymous,
            policy: crate::transport::SourcePolicy::default(),
        });
        let fetch = Fetching { fetcher: &fetcher, access, stop: &stop };
        let gets = || hub.hits(&format!("/obj/{}", absent.sha256));
        let reread = Declaration { work_fingerprint: Some(format!("sha256:{}", "45".repeat(32))), ..inherit.clone() };
        let (reader_meta, id) = (std::sync::Arc::new(Meta::open(&store).unwrap()), format!("sha256:{}", "95".repeat(32)));
        let begun = begin(&store, &reader_meta, &id, 1, reread, None).unwrap();
        let writer = Writer::from_begin(&store, reader_meta, &id, 1, begun).unwrap();
        let mut copied = vec![0u8; 2048];
        writer.source_read_into("source", "model", "other", "value", 0, &mut copied, Some(&fetch)).unwrap();
        assert_eq!(copied, held_back);
        writer.source_read_into("source", "model", "other", "value", 0, &mut copied, Some(&fetch)).unwrap();
        assert_eq!(gets(), 1);
        drop(writer);
        std::fs::remove_file(store.object_path(&absent.sha256)).unwrap();
        let (client, server) = cozy_platform::ipc::pair().unwrap();
        std::thread::scope(|scope| {
            let served = scope.spawn(|| serve_source(&store, &meta, &output, server, &|| Ok(()), Some(&fetch)));
            let mut client = Channel(client);
            assert_eq!(read_part(&mut client, &output, "weight"), weight);
            assert_eq!(gets(), 1);
            assert_eq!(read_part(&mut client, &output, "other"), held_back);
            assert_eq!(read_part(&mut client, &output, "other"), held_back);
            assert_eq!(gets(), 2);
            drop(client);
            served.join().unwrap().unwrap();
        });
        assert!(crate::ingest::custody::live(&root).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(root);
    }
}
