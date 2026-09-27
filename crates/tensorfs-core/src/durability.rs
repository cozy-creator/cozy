//! THE DURABILITY CHAIN — what a conversion has already put beyond the pod's own life.
//!
//! A conversion of a real artifact runs for hours on rented hardware, and the hardware can
//! go away at any instant for reasons the run has no part in. A MiniMax H3 ingest moved
//! 210 GB over 2h56m, reached 44 of 48 source files, and lost every byte when the pod was
//! released, because the only copy was the pod's local Store.
//!
//! **This plane and `ingest::journal` answer two halves of one question, and neither is the
//! other.** The conversion journal is the op → `Part` mapping: it says what the bytes MEAN,
//! it lives inside the ingest session root, and it lets a crashed run on THE SAME POD resume
//! instead of starting over. It cannot help a pod that no longer exists. This plane says
//! which bytes are already on the mounted repo-object cache, so a DIFFERENT pod can get them
//! back; it names them and nothing else, and it carries the conversion journal by reference
//! (`Link::progress`) without ever parsing it.
//!
//! Together they are a complete resume: the objects come back through the ordinary verified
//! door, the conversion journal comes back beside them, and `transaction::convert` then
//! applies its OWN law to every journalled part — `segments_still_stand`, a standing
//! verification record or exactly one rehash. So a part whose objects did not make it onto
//! the cache is simply re-converted. Nothing is adopted on trust at either layer, and the
//! two laws compose without either being weakened to fit.
//!
//! **The chain is immutable; the pointer is not, and the pointer is not here.** Each link
//! names its predecessor, so the whole durable set is reachable from one digest. That digest
//! is a MUTABLE CELL and it belongs at the hub, advanced by compare-and-set on a monotonic
//! watermark, exactly as a release pointer is. Writing it onto a volume that is one per
//! owner per datacenter would put a mutable cell on shared storage with no working lock —
//! the hazard the local-Store decision exists to refuse.
//!
//! **A stale chain is unadoptable, by arithmetic.** Every link carries the digest of the
//! plan it was written under — `ingest::journal::plan_digest`, the same identity the
//! conversion journal keys on, computed from the source headers with zero tensor bytes read.
//! A resuming pod re-derives it and discards the ENTIRE chain on any difference. There is no
//! partial trust and no version negotiation: either this is the same conversion, or it is
//! not one.

use crate::canon::{as_arr, Fields, Value};
use crate::err::{refuse, Code, Result};
use crate::fetch::FetchPlan;
use crate::ids::{ascii_name, prefixed, Doc, ObjectRef};
use crate::limits;
use crate::repo_cache::{CacheKind, CacheRead, RepoObjectCache};
use crate::store::Store;

/// How many links one chain may hold. A 4 GiB checkpoint interval over a 1 TiB artifact is
/// 256 links; the cap is three orders above the largest artifact TensorFS admits and exists
/// only so a walk over untrusted digests terminates.
pub const MAX_SEGMENTS: usize = 65_536;

/// A control exchange names at most this many objects. The chain, rather than a frame,
/// carries the rest of the operation's inventory.
pub const CHECKPOINT_REFS: usize = 128;

/// One link: the objects this conversion had published onto the cache when it was written.
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
    /// Manifests this segment adds. Separate because the cache has two namespaces and an
    /// admission must ask for the right one; folding them would make the reader guess.
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

/// Read an admitted link from the local Store. Unlike `chain`, this does not require an
/// NFS binding: the ordinary transfer plane may have obtained it from the hub.
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
    std::io::Read::read_exact(&mut file, &mut bytes).map_err(|error| crate::err::Refusal {
        code: Code::IO_FAILED,
        detail: format!("read checkpoint document: {error}"),
    })?;
    if ObjectRef::of(&bytes) != *object {
        return refuse(
            Code::OBJECT_ID_MISMATCH,
            "checkpoint document differs from its ObjectRef",
        );
    }
    Ok(bytes)
}

/// Each call reads one link and returns a bounded window, so a caller can fetch links and
/// their objects without first downloading the entire chain or building a model-sized frame.
pub struct Window {
    pub link: Link,
    pub objects: Vec<(CacheKind, ObjectRef)>,
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
        .map(|r| (CacheKind::Blob, r.clone()))
        .chain(
            link.manifests
                .iter()
                .map(|r| (CacheKind::Manifest, r.clone())),
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
/// NFS is not consulted and cannot delay this checkpoint.
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

/// Append one link and publish it. The segment goes into the local Store like any other
/// object and onto the cache like any other object; its digest is what the caller hands
/// the hub.
///
/// **The order matters and is the whole guarantee.** A link is published only after the
/// objects it names are already on the cache — the caller proves that with the mirror's
/// durable prefix before calling — so a chain never claims a durability the cache cannot
/// honour. The reverse order would produce a journal that reads clean and restores short.
pub fn append(
    store: &Store,
    cache: &RepoObjectCache,
    segment: &Link,
    operation: Option<&str>,
) -> Result<ObjectRef> {
    // The cache is still an argument HERE and only here: the journal thread already holds
    // the one the mirror is publishing through, and taking a second handle off the Store
    // would let a rebind mid-run split the chain across two mounts.
    let bytes = segment.canonical_bytes();
    let object = store
        .put_stream_held(
            &mut bytes.as_slice(),
            Some(&ObjectRef::of(&bytes)),
            &Default::default(),
            operation,
        )?
        .obj;
    match cache.backfill_store(store, CacheKind::Blob, &object)? {
        crate::repo_cache::CacheWrite::Unavailable => refuse(
            Code::DURABILITY_UNPROVEN,
            format!(
                "the journal link {} could not be published onto the cache; the objects it \
                 names may be durable but nothing can find them",
                object.id()
            ),
        ),
        _ => Ok(object),
    }
}

/// Walk a chain back from its head, reading each link off the CACHE. Nothing here touches
/// the network and nothing is trusted: every link is opened at its exact digest and length,
/// and a link whose bytes disagree is a corrupt cache, which is a miss.
///
/// Returns the links from ROOT to HEAD, so a caller reading them in order sees the durable
/// set grow the way it actually grew.
pub fn chain(cache: &RepoObjectCache, head: &ObjectRef) -> Result<Vec<Link>> {
    let mut chain = Vec::new();
    let mut cursor = Some(head.clone());
    while let Some(object) = cursor {
        if chain.len() >= MAX_SEGMENTS {
            return refuse(
                Code::COUNT_CAP,
                format!("the journal chain is longer than {MAX_SEGMENTS} links"),
            );
        }
        let bytes = cache.read_blob(&object)?;
        let segment = Link::parse(&bytes)?;
        cursor = segment.prev.clone();
        chain.push(segment);
    }
    chain.reverse();
    for (position, segment) in chain.iter().enumerate() {
        if segment.index != position as u64 {
            return refuse(
                Code::SORT_ORDER,
                format!(
                    "journal link at position {position} calls itself index {}",
                    segment.index
                ),
            );
        }
    }
    Ok(chain)
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
    /// entitled to say so, and because the newest may be the one the cache lost.
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

// ---------------------------------------------------------------- resume

/// What a restore found. `held` was already local — the ordinary answer on a warm pod, and
/// the reason a re-run costs nothing — `admitted` crossed the cache into the Store, and
/// `missing` is what the cache could not answer for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Restored {
    pub held: u64,
    pub admitted: u64,
    pub missing: u64,
    pub corrupt: u64,
    pub bytes_held: u64,
    pub bytes_admitted: u64,
    pub complete: bool,
}

/// The Store's binding, or a refusal. A RESTORE is the one place an absent cache is an
/// error rather than weather: "bring these bytes back off the cache" with no cache is a
/// caller mistake, not a cold mount, and answering it with an empty success would report a
/// resume that restored nothing as a resume that had nothing to restore.
fn bound_cache(store: &Store) -> Result<&RepoObjectCache> {
    store.repo_cache().ok_or_else(|| crate::err::Refusal {
        code: Code::MISSING_FIELD,
        detail: format!(
            "{} is bound to no repo cache; bind one with `tfs store ensure --repo-cache`",
            store.root().display()
        ),
    })
}

/// Bring a declared object set back from the cache into a local Store.
///
/// **This is literally a fetch, with the cache in the origin's place, and that is the
/// point.** The split into HELD and WANTED is [`FetchPlan`]'s, computed against the real
/// Store: an object with a valid verification record is not re-read, an object whose record
/// cannot answer costs exactly one rehash, and an object whose bytes disagree with its id is
/// REMOVED by that check and lands in `wanted` — so a corrupt cached object heals by being
/// fetched again rather than being skipped as present. Every byte that then moves crosses
/// [`crate::store::Store::put_stream`] at the digest that names it. There is no new trust
/// path here because there is no new admission door.
pub fn restore_objects(
    store: &Store,
    session: &str,
    blobs: &[ObjectRef],
    manifests: &[ObjectRef],
) -> Result<Restored> {
    let cache = bound_cache(store)?;
    let mut report = Restored::default();
    let mut wanted: Vec<(CacheKind, ObjectRef)> = Vec::new();
    if !blobs.is_empty() {
        let (plan, _) = FetchPlan::of_objects(store, session, blobs)?;
        for object in &plan.held {
            report.held += 1;
            report.bytes_held += object.length;
        }
        wanted.extend(plan.wanted.iter().map(|o| (CacheKind::Blob, o.clone())));
    }
    // Manifests are asked separately and not through the same plan, because a bare object
    // set is a set of BLOBS: `of_objects` would look for every Manifest in the blob
    // namespace, find none, and re-admit the whole lot on a Store that already held them.
    for object in manifests {
        if crate::fetch::manifest_presence(store, object)?.held() {
            report.held += 1;
            report.bytes_held += object.length;
        } else {
            wanted.push((CacheKind::Manifest, object.clone()));
        }
    }
    for (kind, object) in &wanted {
        match cache.admit(store, *kind, object)? {
            CacheRead::Hit => {
                report.admitted += 1;
                report.bytes_admitted += object.length;
            }
            CacheRead::Corrupt => report.corrupt += 1,
            CacheRead::Missing | CacheRead::Unavailable => report.missing += 1,
        }
    }
    report.complete = report.missing == 0 && report.corrupt == 0;
    Ok(report)
}

/// Bring a whole checkpoint back from the cache: admit the Manifest, walk its closure, and
/// admit everything the closure names that this Store does not already hold.
///
/// The completion proof at the end is the Store's own — a local walk over locally verified
/// bytes — so "restored" means the same thing here it means after a pull.
pub fn restore_checkpoint(store: &Store, session: &str, manifest: &ObjectRef) -> Result<Restored> {
    // Three rounds, and the order is forced by what names what. The Manifest names the
    // header; only the loaded header enumerates the tensor segments. Asking for the closure
    // before the header is local is asking the Store a question it cannot answer yet.
    let mut report = restore_objects(store, session, &[], std::slice::from_ref(manifest))?;
    let document = store.read_manifest(manifest)?;
    if let Some(header) = document.header() {
        add(
            &mut report,
            restore_objects(store, session, std::slice::from_ref(header), &[])?,
        );
    }
    let walk = crate::checkpoint::walk(store, &document)?;
    let declared: Vec<ObjectRef> = walk.distinct().into_iter().cloned().collect();
    add(
        &mut report,
        restore_objects(store, session, &canonical_refs(declared), &[])?,
    );
    report.complete = walk.require_resident(store).is_ok();
    Ok(report)
}

fn add(into: &mut Restored, one: Restored) {
    into.held += one.held;
    into.admitted += one.admitted;
    into.missing += one.missing;
    into.corrupt += one.corrupt;
    into.bytes_held += one.bytes_held;
    into.bytes_admitted += one.bytes_admitted;
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
    /// observed advance, which is a debugging shape and not a production one: the links are
    /// small but each is a separate mounted-filesystem publication.
    pub interval: u64,
}

/// Where the chain ends. `head` is the digest a caller hands the hub, and `bytes` is the
/// monotonic value the hub compare-and-sets on: a pointer only ever moves to a longer chain
/// over more durable bytes. TensorFS writes no mutable cell of its own — not here and not
/// on the cache, which is one volume per owner per datacenter and has no lock worth the
/// name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Head {
    pub head: Option<ObjectRef>,
    pub links: u64,
    pub bytes: u64,
}

/// What one durable run cost and proved.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    pub mirror: crate::repo_cache::MirrorReport,
    pub journal: Head,
}

/// Run `body` with write-through and a journal, and get back what survives the pod.
///
/// **The ordering is the guarantee and it is held here rather than at each caller.** The
/// journal thread wakes only when the producer says a unit of work is done; it appends a
/// link only over the mirror's DURABLE PREFIX, so a link never names an object the cache has
/// not taken; and the final link is written after the mirror has drained, so nothing the run
/// produced is left out of the chain. A caller that hand-assembled these steps could get any
/// of them backwards and the failure would look like a clean journal that restores short.
pub fn with<R>(
    store: &Store,
    policy: &Policy<'_>,
    body: impl FnOnce(Option<&crate::repo_cache::Mirror>) -> R,
) -> (R, Report) {
    // THE ONLY PLACE A MIRROR IS EVER STARTED, and it is an operation that starts it rather
    // than `Store::open`. A Store handle is made by `tfs get`, `tfs verify`, every transfer
    // and every read lease; starting publisher threads for those would spend threads and a
    // mounted-filesystem connection on runs that admit nothing. A `Mirror` also cannot exist
    // without the identity a `Policy` carries -- the chain is per (operation, plan) and a
    // link written under neither would name a durability nothing could adopt. So: the Store
    // knows WHERE the cache is, from open; an operation with durable output decides WHEN to
    // publish to it.
    let Some(cache) = store.repo_cache() else {
        return (body(None), Report::default());
    };
    let mirror = crate::repo_cache::Mirror::start(cache.clone(), store.root(), policy.interval);
    let head = std::sync::Mutex::new(Head::default());
    let out = std::thread::scope(|scope| {
        let chain = scope.spawn(|| keep_chain(&mirror, store, cache, policy, &head));
        // Drain BEFORE closing: every object offered has its answer, so the final link the
        // journalist takes covers the whole run and not the whole run minus what was still
        // in flight when the last op finished. Both happen through a guard rather than two
        // statements, because a producer that unwound is exactly the case where the bytes it
        // did produce most need naming — and two statements after `body` would run in
        // neither order at all.
        struct Close<'a>(&'a crate::repo_cache::Mirror);
        impl Drop for Close<'_> {
            fn drop(&mut self) {
                self.0.drain();
                self.0.close();
            }
        }
        let out = {
            let _close = Close(&mirror);
            body(Some(&mirror))
        };
        let _ = chain.join();
        out
    });
    let journal = head.into_inner().unwrap_or_default();
    (
        out,
        Report {
            mirror: mirror.report(),
            journal,
        },
    )
}

/// The journal thread. It owns no timer and asks no clock: it waits for the producer to
/// report work, reads the durable prefix, and appends a link when enough bytes have crossed.
fn keep_chain(
    mirror: &crate::repo_cache::Mirror,
    store: &Store,
    cache: &RepoObjectCache,
    policy: &Policy<'_>,
    head: &std::sync::Mutex<Head>,
) {
    let mut seen = 0u64;
    let mut written = 0u64;
    let mut position = 0u64;
    loop {
        let live = mirror.awaited(seen);
        let closed = live.is_none();
        seen = live.unwrap_or(seen);
        let durable = mirror.durable();
        let enough = durable.bytes.saturating_sub(written) >= policy.interval;
        // The last link is taken whatever the interval says: a run that ends 3 GiB into an
        // interval has those 3 GiB on the cache, and refusing to name them would throw away
        // durability that has already been paid for.
        if (enough || closed) && durable.objects > position {
            match link(
                mirror,
                store,
                cache,
                policy,
                head,
                position,
                durable.objects,
                durable.bytes,
            ) {
                Ok(()) => {
                    position = durable.objects;
                    written = durable.bytes;
                }
                // A link that cannot be published is the cache saying it is not working.
                // The conversion keeps running and keeps producing; it simply stops being
                // resumable, which is exactly what an absent cache means.
                Err(_) => return,
            }
        }
        if closed {
            return;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn link(
    mirror: &crate::repo_cache::Mirror,
    store: &Store,
    cache: &RepoObjectCache,
    policy: &Policy<'_>,
    head: &std::sync::Mutex<Head>,
    from: u64,
    to: u64,
    bytes: u64,
) -> Result<()> {
    let mut blobs = Vec::new();
    let mut manifests = Vec::new();
    for (kind, object) in mirror.offers(from, to) {
        match kind {
            CacheKind::Blob => blobs.push(object),
            CacheKind::Manifest => manifests.push(object),
        }
    }
    let mut current = head.lock().unwrap_or_else(|e| e.into_inner());
    let segment = Link {
        operation: policy.chain.to_string(),
        plan: policy.plan.to_string(),
        prev: current.head.clone(),
        index: current.links,
        blobs: canonical_refs(blobs),
        manifests: canonical_refs(manifests),
        bytes,
        // The newest conversion-journal snapshot that is itself inside the durable prefix.
        // Naming one the cache had not taken would send a resuming pod to an absent object.
        progress: mirror.progress_within(to),
    };
    let object = append(store, cache, &segment, Some(policy.operation))?;
    current.head = Some(object);
    current.links += 1;
    current.bytes = bytes;
    Ok(())
}

// ---------------------------------------------------------------- the conversion journal

/// Snapshot the conversion journal into the CAS and hand it to the write-through.
///
/// Called by the converter at checkpoint boundaries, not per op: the file is the whole
/// op → `Part` map and hashing it costs its length, so doing it 3,699 times over an H3 tree
/// would spend gigabytes of SHA-256 to say what one snapshot per 4 GiB already says.
///
/// It goes onto the cache as an ordinary blob and is named by a link only once it is inside
/// the durable prefix, exactly like every tensor object.
pub fn snapshot_conversion(
    store: &Store,
    mirror: &crate::repo_cache::Mirror,
    journal: &crate::ingest::journal::Journal,
    operation: &str,
) -> Result<ObjectRef> {
    let bytes = journal.snapshot()?;
    let object = store
        .put_stream_held(
            &mut bytes.as_slice(),
            Some(&ObjectRef::of(&bytes)),
            &Default::default(),
            Some(operation),
        )?
        .obj;
    mirror.offer_progress(&object);
    Ok(object)
}

/// Put the conversion journal back where `transaction::convert` will find it.
///
/// The bytes are admitted from the cache at their exact digest first, so what lands is the
/// snapshot the chain named and not something at that path. Installing it does NOT make the
/// parts it describes true: `convert` re-checks every one of them against the Store's own
/// admission law before reusing it, so a journal restored beside objects that did not survive
/// simply re-converts those ops.
pub fn restore_conversion(
    store: &Store,
    session: &str,
    tenant: &str,
    plan_digest: &str,
    snapshot: &ObjectRef,
) -> Result<()> {
    let cache = bound_cache(store)?;
    if cache.admit(store, CacheKind::Blob, snapshot)? != CacheRead::Hit {
        return refuse(
            Code::OBJECT_ABSENT,
            format!(
                "the cache cannot answer for conversion journal {}; the objects restore and \
                 the conversion re-derives what they mean",
                snapshot.id()
            ),
        );
    }
    let mut bytes = Vec::with_capacity(snapshot.length as usize);
    store.read_into(&snapshot.sha256, &mut bytes)?;
    // The session ROOT comes first and it is not a formality: it is the filesystem's own GC
    // hold protocol, and a directory holding a journal but no root is a directory the next
    // `open_root` refuses and the next `tfs gc` does not protect. `tenant` is stated by the
    // caller rather than read off the journal, so a resume onto the wrong tenant refuses
    // rather than silently adopting another tenant's work.
    crate::ingest::transaction::ensure_session_root(store.root(), session, tenant)?;
    crate::ingest::journal::install(
        &crate::ingest::transaction::candidate_dir(store.root(), session),
        plan_digest,
        &bytes,
    )
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
