//! THE CHECKPOINT CHAIN: what a conversion has already made durable, so a different pod can
//! resume it.
//!
//! A conversion of a real artifact runs for hours on rented hardware that can go away at any
//! instant (a MiniMax H3 ingest lost 210 GB of work when its pod was released). The conversion
//! journal (`ingest::journal`) says what the bytes MEAN and resumes a crashed run on the same
//! pod; this chain names which bytes a conversion checkpointed, and carries the journal by
//! reference (`Link::progress`) without parsing it. Its owner uploads the chain and advances a
//! durable pointer at the Hub; a resuming pod gets the objects back through the ordinary
//! verified door, and `transaction::convert` re-checks every journalled part, so nothing is
//! adopted on trust.
//!
//! **A stale chain is unadoptable, by arithmetic.** Every link carries the digest of the plan
//! it was written under (`ingest::journal::plan_digest`, from the source headers alone). A
//! resuming pod re-derives it and discards the whole chain on any difference.

use crate::canon::{as_arr, Fields, Value};
use crate::err::{refuse, Code, Result};
use crate::ids::{ascii_name, prefixed, Doc, ObjectRef};
use crate::limits;
use crate::store::Store;

/// How many links one chain may hold. A 4 GiB checkpoint interval over a 1 TiB artifact is
/// 256 links; the cap is three orders above the largest artifact TensorFS admits and exists
/// only so a walk over untrusted digests terminates.
pub const MAX_SEGMENTS: usize = 65_536;

/// A control exchange names at most this many objects. The chain, rather than a frame,
/// carries the rest of the operation's inventory.
pub const CHECKPOINT_REFS: usize = 128;

/// One link: the objects this conversion had checkpointed when it was written.
///
/// Sizing follows the header's: `objects` is bounded by `MAX_TOTAL_REFS` because a segment
/// cannot name more objects than a legal checkpoint can reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    /// The ingest operation this chain belongs to. Two conversions never share a chain.
    pub operation: String,
    /// `sha256:...` of the plan, re-derived from source headers alone. The staleness test.
    pub plan: String,
    /// The previous link, or `None` at the root.
    pub prev: Option<ObjectRef>,
    /// 0 at the root, +1 per link. Monotonic, and the value a hub pointer compare-and-sets
    /// against: a pointer only ever moves to a higher index.
    pub index: u64,
    /// Objects this segment adds to the durable set, sorted and unique by digest.
    pub blobs: Vec<ObjectRef>,
    /// Manifests this segment adds: a separate namespace, so a reader never guesses.
    pub manifests: Vec<ObjectRef>,
    /// Cumulative durable bytes through this link — the WATERMARK. It is a byte count, not
    /// an elapsed time, and every decision built on it (checkpoint here, this run is making
    /// progress, that run is not) is a decision about observed work.
    pub bytes: u64,
    /// The conversion journal as it stood when this link was written — the op → `Part`
    /// mapping `ingest::journal` owns, snapshotted into the CAS and published beside the
    /// objects it describes. Opaque here: carried, never parsed. `None` on a link taken
    /// before the first snapshot was itself durable.
    pub progress: Option<ObjectRef>,
}

impl Doc for Link {
    const FORMAT: &'static str = "tensorfs.ingest.durability/1";
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("Link", v)?;
        crate::ids::check_format("Link", f.req_str("format")?, Self::FORMAT)?;
        let blobs = refs("Link.blobs", f.req("blobs")?)?;
        let bytes = f.req_uint("bytes")?;
        let index = f.req_uint("index")?;
        let manifests = refs("Link.manifests", f.req("manifests")?)?;
        let operation = f.req_str("operation")?.to_string();
        let plan = prefixed("Link.plan", f.req_str("plan")?)?;
        let prev = match f.opt("prev") {
            Some(value) => Some(ObjectRef::from_value("Link.prev", value)?),
            None => None,
        };
        let progress = match f.opt("progress") {
            Some(value) => Some(ObjectRef::from_value("Link.progress", value)?),
            None => None,
        };
        f.done()?;
        ascii_name("Link.operation", &operation, limits::MAX_NAME_BYTES)?;
        if index == 0 && prev.is_some() {
            return refuse(
                Code::MISSING_FIELD,
                "the root link of a journal names no predecessor",
            );
        }
        if index > 0 && prev.is_none() {
            return refuse(
                Code::MISSING_FIELD,
                format!("journal link {index} names no predecessor; only the root may"),
            );
        }
        Ok(Link {
            operation,
            plan,
            prev,
            index,
            blobs,
            manifests,
            bytes,
            progress,
        })
    }

    fn to_value(&self) -> Value {
        let mut fields = vec![
            (
                "blobs",
                Value::arr(self.blobs.iter().map(ObjectRef::to_value).collect()),
            ),
            ("bytes", Value::uint(self.bytes)),
            ("format", Value::str(Self::FORMAT)),
            ("index", Value::uint(self.index)),
            (
                "manifests",
                Value::arr(self.manifests.iter().map(ObjectRef::to_value).collect()),
            ),
            ("operation", Value::str(self.operation.clone())),
            ("plan", Value::str(self.plan.clone())),
        ];
        if let Some(prev) = &self.prev {
            fields.push(("prev", prev.to_value()));
        }
        if let Some(progress) = &self.progress {
            fields.push(("progress", progress.to_value()));
        }
        Value::obj(fields)
    }
}

fn refs(what: &'static str, value: &Value) -> Result<Vec<ObjectRef>> {
    let mut out = Vec::new();
    for entry in as_arr(what, "refs", value)? {
        out.push(ObjectRef::from_value(what, entry)?);
    }
    if out.len() > limits::MAX_TOTAL_REFS {
        return refuse(
            Code::TOTAL_REFS_CAP,
            format!("{what}: {} refs exceeds the document cap", out.len()),
        );
    }
    if out.windows(2).any(|pair| pair[0].sha256 >= pair[1].sha256) {
        return refuse(
            Code::SORT_ORDER,
            format!("{what}: refs must be strictly sorted and unique by digest"),
        );
    }
    Ok(out)
}

/// Sort and dedupe a caller's object list into the form a segment stores it in.
pub fn canonical_refs(mut refs: Vec<ObjectRef>) -> Vec<ObjectRef> {
    refs.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    refs.dedup_by(|a, b| a.sha256 == b.sha256);
    refs
}

/// Read an admitted link from the local Store.
pub fn local_link(store: &Store, object: &ObjectRef) -> Result<Link> {
    Link::parse(&read_document(store, object, Link::MAX_BYTES)?)
}

/// Bound the actual pinned file before allocation, not only the untrusted claimed length.
/// Source and derived progress use the same exact ObjectRef admission contract as links.
pub(crate) fn read_document(store: &Store, object: &ObjectRef, cap: usize) -> Result<Vec<u8>> {
    if object.length > cap as u64 {
        return refuse(Code::COUNT_CAP, "checkpoint document exceeds its byte cap");
    }
    let mut file = store.open_verified(&object.sha256)?;
    if file.len() != object.length {
        return refuse(
            Code::LENGTH_MISMATCH,
            "checkpoint document differs from its ObjectRef length",
        );
    }
    let mut bytes = vec![0; object.length as usize];
    std::io::Read::read_exact(&mut file, &mut bytes).map_err(|error| crate::store::classify_io("read checkpoint document", error))?;
    if ObjectRef::of(&bytes) != *object {
        return refuse(
            Code::OBJECT_ID_MISMATCH,
            "checkpoint document differs from its ObjectRef",
        );
    }
    Ok(bytes)
}

/// Which namespace a link's object lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Blob,
    Manifest,
}

/// Each call reads one link and returns a bounded window, so a caller can fetch links and
/// their objects without first downloading the entire chain or building a model-sized frame.
pub struct Window {
    pub link: Link,
    pub objects: Vec<(Kind, ObjectRef)>,
    pub next: Option<usize>,
}

pub fn local_window(
    store: &Store,
    object: &ObjectRef,
    offset: usize,
    limit: usize,
) -> Result<Window> {
    if limit == 0 || limit > CHECKPOINT_REFS {
        return refuse(Code::COUNT_CAP, "checkpoint page limit is outside 1..=128");
    }
    let mut link = local_link(store, object)?;
    let count = link.blobs.len() + link.manifests.len();
    if offset > count {
        return refuse(
            Code::COUNT_CAP,
            "checkpoint page offset exceeds its object count",
        );
    }
    let objects = link
        .blobs
        .iter()
        .map(|r| (Kind::Blob, r.clone()))
        .chain(
            link.manifests
                .iter()
                .map(|r| (Kind::Manifest, r.clone())),
        )
        .skip(offset)
        .take(limit)
        .collect::<Vec<_>>();
    let end = offset + objects.len();
    link.blobs.clear();
    link.manifests.clear();
    Ok(Window {
        link,
        objects,
        next: (end < count).then_some(end),
    })
}

pub fn local_chain(store: &Store, head: &ObjectRef, plan: &str) -> Result<Set> {
    let mut links = Vec::new();
    let mut cursor = Some(head.clone());
    while let Some(object) = cursor {
        if links.len() >= MAX_SEGMENTS {
            return refuse(Code::COUNT_CAP, "checkpoint chain exceeds its link cap");
        }
        let link = local_link(store, &object)?;
        if let Some(newer) = links.last() {
            let newer: &Link = newer;
            if link.index.checked_add(1) != Some(newer.index) {
                return refuse(
                    Code::SORT_ORDER,
                    "checkpoint predecessor index does not decrease by one",
                );
            }
        }
        cursor = link.prev.clone();
        links.push(link);
    }
    links.reverse();
    if links.first().is_some_and(|link| link.index != 0) {
        return refuse(Code::SORT_ORDER, "checkpoint chain has no root");
    }
    fold(&links, plan)
}

/// A supplied chain this conversion can continue, or `None` when it was written under another
/// plan or slot or is no longer whole in this Store. Such a chain is discarded and the work
/// reconverts into a new chain: losing a checkpoint costs time, never a refusal.
pub fn continuable(
    store: &Store,
    head: &ObjectRef,
    plan: &str,
    chain: &str,
) -> Result<Option<Set>> {
    match local_chain(store, head, plan) {
        Ok(set) if set.operation == chain => Ok(Some(set)),
        Ok(_) => Ok(None),
        Err(error) if error.code == Code::IO_FAILED => Err(error),
        Err(_) => Ok(None),
    }
}

/// Export locally complete progress. This does not assert remote durability: the RecordOwner
/// publishes the bounded chain and advances its durable pointer only after upload acceptance.
pub fn checkpoint(
    store: &Store,
    policy: &Policy<'_>,
    previous: Option<&ObjectRef>,
    journal: &crate::ingest::journal::Journal,
) -> Result<Head> {
    let previous = checkpoint_predecessor(store, policy, journal.local_head()?.as_ref(), previous)?;
    let exported = checkpoint_bytes(
        store,
        policy,
        previous.as_ref(),
        &journal.snapshot()?,
        journal.objects(),
        Some(policy.operation),
    )?;
    crate::ingest::transaction::name_candidates(store.root(), policy.operation, &exported.roots)?;
    if let Some(head) = &exported.head.head {
        journal.save_head(head)?;
    }
    Ok(exported.head)
}

/// Extend the newest locally known chain while accepting delayed acknowledgments of its
/// ancestors. Two independently forked heads are a conflict, never a reason to rewind.
pub(crate) fn checkpoint_predecessor(
    store: &Store,
    policy: &Policy<'_>,
    local: Option<&ObjectRef>,
    supplied: Option<&ObjectRef>,
) -> Result<Option<ObjectRef>> {
    for head in [local, supplied].into_iter().flatten() {
        let link = local_link(store, head)?;
        if link.operation != policy.chain || link.plan != policy.plan {
            return refuse(
                Code::CROSS_SUBJECT_REPLAY,
                "checkpoint cursor differs from its operation or plan",
            );
        }
    }
    let (Some(local), Some(supplied)) = (local, supplied) else {
        return Ok(local.or(supplied).cloned());
    };
    if local == supplied {
        return Ok(Some(local.clone()));
    }
    let left = local_link(store, local)?;
    let right = local_link(store, supplied)?;
    let (newest, oldest, mut link, old_index) = if left.index >= right.index {
        (local, supplied, left, right.index)
    } else {
        (supplied, local, right, left.index)
    };
    let mut cursor = newest.clone();
    let mut walked = 0;
    while link.index > old_index {
        if walked >= MAX_SEGMENTS {
            return refuse(Code::COUNT_CAP, "checkpoint ancestry exceeds its link cap");
        }
        let previous = link.prev.ok_or_else(|| crate::err::Refusal {
            code: Code::CROSS_SUBJECT_REPLAY,
            detail: "checkpoint cursor has no expected ancestor".into(),
        })?;
        let parent = local_link(store, &previous)?;
        if parent.index.checked_add(1) != Some(link.index)
            || parent.plan != policy.plan
            || parent.operation != policy.chain
        {
            return refuse(
                Code::CROSS_SUBJECT_REPLAY,
                "checkpoint ancestry is inconsistent",
            );
        }
        cursor = previous;
        link = parent;
        walked += 1;
    }
    if &cursor != oldest {
        return refuse(
            Code::TRANSACTION_CONFLICT,
            "local and acknowledged checkpoint chains forked",
        );
    }
    Ok(Some(newest.clone()))
}

pub(crate) struct CheckpointExport {
    pub head: Head,
    /// Checkpoint documents to retain under the producing transaction's own GC root.
    pub roots: Vec<ObjectRef>,
}

/// Keep a verified derived recovery object across fetch/page/Ready/process-death gaps.
/// The live guard covers admission; the transaction's existing private root then owns
/// the bytes until begin installs complete writer custody or explicit abandonment wins.
pub fn restore_derived<T>(
    store: &Store,
    meta: &crate::meta::Meta,
    transaction: &str,
    epoch: u64,
    object: &ObjectRef,
    fetch: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let hold = meta.acquire_hold("derived-checkpoint-restore")?;
    let result = (|| {
        crate::derived::retain_restore(store, meta, transaction, epoch, None)?;
        let value = fetch()?;
        crate::derived::retain_restore(store, meta, transaction, epoch, Some(object))?;
        Ok(value)
    })();
    let released = hold.release(meta);
    result.and_then(|value| released.map(|()| value))
}

/// The shared byte-plane export. Its opaque progress is interpreted only by its producer;
/// source ingest and derived writers keep their existing journals and retention authorities.
pub(crate) fn checkpoint_bytes(
    store: &Store,
    policy: &Policy<'_>,
    previous: Option<&ObjectRef>,
    bytes: &[u8],
    objects: Vec<ObjectRef>,
    hold: Option<&str>,
) -> Result<CheckpointExport> {
    let prior = previous
        .map(|head| local_chain(store, head, policy.plan))
        .transpose()?;
    if prior
        .as_ref()
        .is_some_and(|set| set.operation != policy.chain)
    {
        return refuse(
            Code::CROSS_SUBJECT_REPLAY,
            "checkpoint belongs to another operation slot",
        );
    }
    if bytes.is_empty() {
        return Ok(CheckpointExport {
            head: Head::default(),
            roots: Vec::new(),
        });
    }
    let tip = Tip {
        head: Head {
            head: previous.cloned(),
            links: prior.as_ref().map_or(0, |set| set.links),
            bytes: prior.as_ref().map_or(0, |set| set.bytes),
        },
        held: prior
            .iter()
            .flat_map(|set| set.blobs.iter().map(|object| object.sha256.clone()))
            .collect(),
    };
    let mut roots = previous
        .map(|head| local_documents(store, head))
        .transpose()?
        .unwrap_or_default();
    let extended = extend(store, policy, &tip, bytes, objects, hold)?;
    roots.push(extended.progress);
    roots.extend(extended.links);
    Ok(CheckpointExport {
        head: extended.head,
        roots,
    })
}

/// A chain's tip and every blob digest the chain already names: all a writer needs to
/// extend it without walking it.
#[derive(Debug, Clone, Default)]
pub(crate) struct Tip {
    pub head: Head,
    pub held: std::collections::HashSet<String>,
}

pub(crate) struct Extended {
    pub head: Head,
    pub progress: ObjectRef,
    /// The progress document and every object the new links name, now held by the chain.
    pub named: Vec<ObjectRef>,
    pub links: Vec<ObjectRef>,
}

/// Write one progress document and the links naming only the objects `tip` does not yet
/// hold. Only the supplied objects are verified, so the cost is this checkpoint's own.
pub(crate) fn extend(
    store: &Store,
    policy: &Policy<'_>,
    tip: &Tip,
    bytes: &[u8],
    mut objects: Vec<ObjectRef>,
    hold: Option<&str>,
) -> Result<Extended> {
    if bytes.len() > limits::DOC_MAX_BYTES {
        return refuse(
            Code::COUNT_CAP,
            "checkpoint progress exceeds the document byte cap",
        );
    }
    let progress = store
        .put_stream_held(
            &mut &*bytes,
            Some(&ObjectRef::of(bytes)),
            &Default::default(),
            hold,
        )?
        .obj;
    for object in &objects {
        let record = match store.record_valid(&object.sha256) {
            Ok(record) => record,
            Err(_) => {
                store.verify(&object.sha256)?;
                store
                    .record_valid(&object.sha256)
                    .map_err(|detail| crate::err::Refusal {
                        code: Code::OBJECT_ABSENT,
                        detail,
                    })?
            }
        };
        if record.length != object.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                "checkpoint journal names the wrong object length",
            );
        }
    }
    objects.push(progress.clone());
    objects = canonical_refs(objects);
    objects.retain(|object| !tip.held.contains(&object.sha256));
    let mut head = tip.head.clone();
    let mut links = Vec::new();
    let pages = objects.chunks(CHECKPOINT_REFS);
    let page_count = pages.len();
    for (index, page) in pages.enumerate() {
        if head.links >= MAX_SEGMENTS as u64 {
            return refuse(Code::COUNT_CAP, "checkpoint chain exceeds its link cap");
        }
        head.bytes = page.iter().try_fold(head.bytes, |sum, object| {
            sum.checked_add(object.length)
                .ok_or_else(|| crate::err::Refusal {
                    code: Code::COUNT_CAP,
                    detail: "checkpoint byte watermark overflow".into(),
                })
        })?;
        let link = Link {
            operation: policy.chain.to_string(),
            plan: policy.plan.to_string(),
            prev: head.head.clone(),
            index: head.links,
            blobs: page.to_vec(),
            manifests: Vec::new(),
            bytes: head.bytes,
            progress: if index + 1 == page_count {
                Some(progress.clone())
            } else {
                None
            },
        };
        let bytes = link.canonical_bytes();
        let object = store
            .put_stream_held(
                &mut bytes.as_slice(),
                Some(&ObjectRef::of(&bytes)),
                &Default::default(),
                hold,
            )?
            .obj;
        links.push(object.clone());
        head.head = Some(object);
        head.links += 1;
    }
    Ok(Extended {
        head,
        progress,
        named: objects,
        links,
    })
}

/// Pin restored chain documents before their journal is installed. Tensor segments are
/// subsequently held by that journal; every immutable link/progress document is retained here.
pub fn retain_local_chain(store: &Store, head: &ObjectRef, operation: &str) -> Result<()> {
    crate::ingest::transaction::name_candidates(
        store.root(),
        operation,
        &local_documents(store, head)?,
    )
}

pub(crate) fn local_documents(store: &Store, head: &ObjectRef) -> Result<Vec<ObjectRef>> {
    let mut cursor = Some(head.clone());
    let mut documents = Vec::new();
    while let Some(object) = cursor {
        if documents.len() >= MAX_SEGMENTS * 2 {
            return refuse(Code::COUNT_CAP, "checkpoint chain exceeds its link cap");
        }
        let link = local_link(store, &object)?;
        documents.push(object);
        documents.extend(link.progress);
        cursor = link.prev;
    }
    Ok(documents)
}

/// The durable set a chain names, and the facts a resuming run needs about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Set {
    pub operation: String,
    pub plan: String,
    pub links: u64,
    pub bytes: u64,
    pub blobs: Vec<ObjectRef>,
    pub manifests: Vec<ObjectRef>,
    /// Every conversion-journal snapshot the chain referenced, root first. The LAST is the
    /// one a resume installs; the earlier ones are kept because a chain that named them is
    /// entitled to say so.
    pub progress: Vec<ObjectRef>,
}

/// Fold a chain into its durable set, refusing a chain that is not one conversion.
///
/// `expected_plan` is the digest the resuming run just re-derived from the source headers.
/// A chain produced under any other plan is DISCARDED WHOLE — not partially adopted, not
/// migrated. That is the property that makes a stale journal harmless: it cannot be half
/// believed.
pub fn fold(chain: &[Link], expected_plan: &str) -> Result<Set> {
    let Some(root) = chain.first() else {
        return refuse(
            Code::MISSING_FIELD,
            "an empty journal names nothing durable",
        );
    };
    let mut blobs = Vec::new();
    let mut manifests = Vec::new();
    let mut progress = Vec::new();
    let mut bytes = 0u64;
    for segment in chain {
        if segment.plan != expected_plan {
            return refuse(
                Code::JOURNAL_STALE,
                format!(
                    "journal link {} was written under plan {}, and the plan re-derived from \
                     these sources is {expected_plan}; the whole journal is discarded",
                    segment.index, segment.plan
                ),
            );
        }
        if segment.operation != root.operation {
            return refuse(
                Code::CROSS_SUBJECT_REPLAY,
                format!(
                    "journal link {} belongs to operation {:?}, the root to {:?}",
                    segment.index, segment.operation, root.operation
                ),
            );
        }
        if segment.bytes < bytes {
            return refuse(
                Code::SORT_ORDER,
                format!(
                    "journal link {} reports {} durable bytes, fewer than the {bytes} its \
                     predecessor already proved; the watermark only moves forward",
                    segment.index, segment.bytes
                ),
            );
        }
        bytes = segment.bytes;
        blobs.extend(segment.blobs.iter().cloned());
        manifests.extend(segment.manifests.iter().cloned());
        progress.extend(segment.progress.iter().cloned());
    }
    Ok(Set {
        operation: root.operation.clone(),
        plan: root.plan.clone(),
        links: chain.len() as u64,
        bytes,
        blobs: canonical_refs(blobs),
        manifests: canonical_refs(manifests),
        progress,
    })
}

// ---------------------------------------------------------------- taking checkpoints

/// The checkpoint interval, in DURABLE BYTES.
///
/// The unit is the whole point. A time-based checkpoint on a run whose throughput varies by
/// two orders of magnitude between an inline scalar and a 64 MiB segment either checkpoints
/// constantly or almost never, and neither answer is about the work. Four gibibytes is 64
/// segments on the current grid, and it puts roughly 52 links on the 210 GB conversion this
/// plane was built for — small enough that a lost pod costs minutes of re-conversion, large
/// enough that the links are a rounding error against the bytes they describe.
pub const INTERVAL_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// What a durable conversion is being run under.
pub struct Policy<'a> {
    /// The chain's NAME, written into every link and checked across the whole chain on
    /// resume. Two conversions never share one.
    pub chain: &'a str,
    /// The catalog operation the link objects are HELD under, which is a different fact from
    /// the chain's name and is why it is a different field.
    ///
    /// A link is an ordinary object in the local Store, and an object with no hold is an
    /// object the next `tfs gc` reclaims -- so it is admitted `_held`, and a hold row has a
    /// foreign key onto `tensorfs_operations`. `prepare_model_source` runs ONE catalog
    /// operation over SEVERAL slots and names a chain per slot (`<operation>/<slot>`),
    /// because a slot's chain must be validatable against that slot's plan alone. Holding
    /// under the chain name would insert a hold for an operation that does not exist: the
    /// insert fails, `link` returns, and the run publishes every object while producing no
    /// chain at all -- every byte durable and nothing able to find them. The hold belongs to
    /// the operation that owns the output; the chain name is a label inside the document.
    pub operation: &'a str,
    /// The digest of the plan, re-derivable from source headers alone.
    pub plan: &'a str,
    /// Durable bytes between links — see [`INTERVAL_BYTES`]. Zero means a link at every
    /// observed advance, a debugging shape.
    pub interval: u64,
}

/// Where the chain ends. `head` is the digest a caller hands the hub, and `bytes` is the
/// monotonic value the hub compare-and-sets on: a pointer only ever moves to a longer chain
/// over more durable bytes. TensorFS writes no mutable cell of its own.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Head {
    pub head: Option<ObjectRef>,
    pub links: u64,
    pub bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_documents_refuse_wrong_lengths_before_reading_into_memory() {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-document-bound-{}",
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let bytes = b"a verified resident object";
        let object = store
            .put_stream(
                &mut bytes.as_slice(),
                Some(&ObjectRef::of(bytes)),
                &Default::default(),
            )
            .unwrap()
            .obj;
        let mut short = object.clone();
        short.length = 1;
        assert_eq!(
            local_link(&store, &short).unwrap_err().code,
            Code::LENGTH_MISMATCH
        );
        assert_eq!(
            read_document(&store, &short, 1).unwrap_err().code,
            Code::LENGTH_MISMATCH
        );
        assert_eq!(
            read_document(&store, &object, 1).unwrap_err().code,
            Code::COUNT_CAP
        );
        assert_eq!(read_document(&store, &object, bytes.len()).unwrap(), bytes);
        std::fs::remove_dir_all(root).unwrap();
    }

    fn object(byte: &str, length: u64) -> ObjectRef {
        ObjectRef {
            sha256: byte.repeat(32),
            length,
        }
    }

    fn segment(index: u64, plan: &str, bytes: u64, prev: Option<ObjectRef>) -> Link {
        Link {
            operation: "op".into(),
            plan: plan.into(),
            prev,
            index,
            blobs: vec![object("aa", 1), object("bb", 2)],
            manifests: Vec::new(),
            bytes,
            progress: None,
        }
    }

    fn plan(byte: &str) -> String {
        format!("sha256:{}", byte.repeat(32))
    }

    #[test]
    fn a_link_round_trips_through_its_exact_canonical_bytes() {
        let root = segment(0, &plan("11"), 3, None);
        assert_eq!(
            Link::parse(&root.canonical_bytes()).unwrap(),
            root,
            "a link must re-emit the bytes it was stored as"
        );
        let mut next = segment(1, &plan("11"), 9, Some(root.object_ref()));
        // The converter's own progress document rides as an opaque reference: carried
        // through the canonical form and the fold, and parsed by nobody in this plane.
        next.progress = Some(object("dd", 77));
        assert_eq!(Link::parse(&next.canonical_bytes()).unwrap(), next);
        assert_eq!(
            fold(&[root, next], &plan("11")).unwrap().progress,
            vec![object("dd", 77)]
        );
    }

    #[test]
    fn the_chain_shape_is_enforced_at_the_document_and_not_by_a_walker() {
        let mut orphan = segment(1, &plan("11"), 3, None);
        orphan.index = 1;
        assert_eq!(
            Link::parse(&orphan.canonical_bytes()).unwrap_err().code,
            Code::MISSING_FIELD
        );
        let mut rooted = segment(0, &plan("11"), 3, Some(object("cc", 4)));
        rooted.index = 0;
        assert_eq!(
            Link::parse(&rooted.canonical_bytes()).unwrap_err().code,
            Code::MISSING_FIELD
        );
    }

    #[test]
    fn a_chain_written_under_another_plan_is_discarded_whole() {
        let root = segment(0, &plan("11"), 3, None);
        let next = segment(1, &plan("22"), 9, Some(root.object_ref()));
        // Not "the second link is dropped": the WHOLE chain refuses, including the link
        // whose plan does match, because a half-believed journal is the dangerous one.
        assert_eq!(
            fold(&[root.clone(), next], &plan("11")).unwrap_err().code,
            Code::JOURNAL_STALE
        );
        assert_eq!(fold(&[root], &plan("11")).unwrap().bytes, 3);
    }

    #[test]
    fn the_watermark_only_moves_forward_and_one_chain_is_one_operation() {
        let root = segment(0, &plan("11"), 9, None);
        let backwards = segment(1, &plan("11"), 3, Some(root.object_ref()));
        assert_eq!(
            fold(&[root.clone(), backwards], &plan("11"))
                .unwrap_err()
                .code,
            Code::SORT_ORDER
        );
        let mut foreign = segment(1, &plan("11"), 12, Some(root.object_ref()));
        foreign.operation = "another".into();
        assert_eq!(
            fold(&[root, foreign], &plan("11")).unwrap_err().code,
            Code::CROSS_SUBJECT_REPLAY
        );
    }

    #[test]
    fn the_durable_set_is_the_union_sorted_and_deduped() {
        let root = segment(0, &plan("11"), 3, None);
        let mut next = segment(1, &plan("11"), 9, Some(root.object_ref()));
        // The same object may be published under two links — two tensors sharing a byte-
        // identical segment is ordinary — and a restore must ask for it exactly once.
        next.blobs = vec![object("aa", 1), object("cc", 3)];
        let durable = fold(&[root, next], &plan("11")).unwrap();
        assert_eq!(
            durable.blobs,
            vec![object("aa", 1), object("bb", 2), object("cc", 3)]
        );
        assert_eq!(durable.bytes, 9);
        assert_eq!(durable.links, 2);
    }
}
