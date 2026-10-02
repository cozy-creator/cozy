//! The fetch/admit plane (tfs-003 row 220, cr-016's `pull`). Bytes are ARRIVING, and
//! nothing about arriving bytes is known until the last one is hashed, so the authorization
//! is an EXPECTATION and the thing that enforces it is `Store::put_stream` — a streaming,
//! bounded-buffer, no-clobber door that commits at the digest it computed or commits
//! nothing. There is no second admission path and this module does not add one.
//!
//! **The digest is the only authority and `put_stream` is the only door** (tfs-048). A
//! `DeliveryGrant` carries a digest and a length and nothing else, so the same grant admits
//! bytes from tensorhub, from a mirror, from a foreign provider, from a pod-local file or
//! from a pipe, and the border cannot tell the difference: origin is indistinguishable and
//! irrelevant. This header used to claim "TensorFS never sees a presigned URL or a worker
//! credential" as the invariant; that conflated layering with security, and it bought three
//! transport implementations in three consumers, not a security property. The one transport
//! lives in `transport`, beside this plane, and what it moves still enters through exactly
//! this door.
//!
//! **Idempotency is a computed fact, not a client's guess.** `FetchPlan::of` splits a
//! declared object set into HELD and WANTED against a real store, and a caller that fetches
//! anything outside `wanted` is not resuming — it is re-downloading. The split asks the
//! store's own question (`record_valid`: may these bytes be trusted without hashing them
//! again?) and, only when the record cannot answer, pays one rehash. Bytes that fail that
//! rehash are REMOVED by `verify` and then land in `wanted`, which is why a corrupted
//! local object heals on the next pull instead of being skipped as "already present".

use crate::canon::{self, as_arr, Fields, Value};
use crate::err::{refuse, Code, Result};
use crate::ids::{ascii_name, hex64, Doc, ObjectRef};
use crate::limits;
use crate::store::{Store, Verdict};
use std::io::Read;

pub fn parse_refs_jsonl(bytes: &[u8]) -> Result<Vec<ObjectRef>> {
    let mut result = Vec::new();
    for line in bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
    {
        let value = canon::parse(line, 16 * 1024)?;
        let mut fields = Fields::new("FetchRef", &value)?;
        let length = fields.req_uint("length")?;
        let sha256 = hex64("FetchRef.sha256", fields.req_str("sha256")?)?;
        result.push(ObjectRef { sha256, length });
    }
    if result.is_empty() {
        return refuse(Code::MISSING_FIELD, "fetch refs are empty");
    }
    sorted_refs("fetch refs", result)
}

/// Digest order with identical repeats merged. One digest at two lengths still refuses.
fn sorted_refs(what: &str, mut refs: Vec<ObjectRef>) -> Result<Vec<ObjectRef>> {
    refs.sort_by(|a, b| a.sha256.cmp(&b.sha256).then(a.length.cmp(&b.length)));
    refs.dedup();
    if let Some(pair) = refs
        .windows(2)
        .find(|pair| pair[0].sha256 == pair[1].sha256)
    {
        return refuse(
            Code::LENGTH_MISMATCH,
            format!(
                "{what} name sha256:{} at {} B and at {} B",
                pair[0].sha256, pair[0].length, pair[1].length
            ),
        );
    }
    Ok(refs)
}

pub fn final_key(sha256: &str) -> Result<String> {
    crate::storage::blob_key(sha256)
}

// ---------------------------------------------------------------- the plan

/// What one object's presence turned out to be. Not stored — it is the reason an object is
/// in `held` or in `wanted`, and it is reported so a pull can say why it moved what it moved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Presence {
    /// Installed, and a valid verification record vouches for it. No bytes read.
    Recorded,
    /// Installed, the record could not vouch, and a rehash agreed with the id.
    Rehashed,
    /// Not installed.
    Absent,
    /// Installed and WRONG. `verify` removed it; the id is now free for
    /// the fetch to land at, so this is a repair, not a refusal.
    CorruptRemoved { why: String },
}

impl Presence {
    pub fn held(&self) -> bool {
        matches!(self, Presence::Recorded | Presence::Rehashed)
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Presence::Recorded => "recorded",
            Presence::Rehashed => "rehashed",
            Presence::Absent => "absent",
            Presence::CorruptRemoved { .. } => "corrupt_removed",
        }
    }
}

/// The exact set a pull session covers, split against a real store BEFORE a byte moves.
///
/// `declared` is what the remote catalog says the checkpoint reaches. It is a CLAIM: this
/// plane never treats it as proof, and the proof is `complete()` after the wanted set has
/// landed — a local walk over locally verified bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchPlan {
    pub session: String,
    /// The Manifest anchoring a checkpoint pull, or None for a bare object set — raw
    /// foreign-source members have no Manifest until preparation authors one (th-124).
    pub manifest: Option<ObjectRef>,
    /// Sorted unique by digest. Present and trustworthy: NOT fetched.
    pub held: Vec<ObjectRef>,
    /// Sorted unique by digest. The only objects a grant may be minted for.
    pub wanted: Vec<ObjectRef>,
}

impl FetchPlan {
    /// Split a declared closure against this store. One pass, no network, no tensor bytes
    /// read beyond a rehash the store's own record could not avoid.
    pub fn of(
        store: &Store,
        session: &str,
        manifest: &ObjectRef,
        declared: &[ObjectRef],
    ) -> Result<(FetchPlan, Vec<(ObjectRef, Presence)>)> {
        Self::plan(store, session, Some(manifest), declared)
    }

    /// The same split over a bare object set: no Manifest anchors it and none is fetched.
    /// `complete` then proves per-object residency instead of a closure walk.
    pub fn of_objects(
        store: &Store,
        session: &str,
        declared: &[ObjectRef],
    ) -> Result<(FetchPlan, Vec<(ObjectRef, Presence)>)> {
        Self::plan(store, session, None, declared)
    }

    fn plan(
        store: &Store,
        session: &str,
        manifest: Option<&ObjectRef>,
        declared: &[ObjectRef],
    ) -> Result<(FetchPlan, Vec<(ObjectRef, Presence)>)> {
        ascii_name("FetchPlan.session", session, limits::MAX_NAME_BYTES)?;
        if let Some(manifest) = manifest {
            validate_manifest_ref(manifest)?;
        }
        if declared.is_empty() {
            return refuse(
                Code::MISSING_FIELD,
                "a fetch plan declares no object — a checkpoint that reaches nothing is not \
                 something a worker can pull",
            );
        }
        let mut sorted = sorted_refs("the declared object set", declared.to_vec())?;
        if let Some(manifest) = manifest {
            if !sorted.iter().any(|object| object.sha256 == manifest.sha256) {
                sorted.push(manifest.clone());
                sorted.sort_by(|left, right| left.sha256.cmp(&right.sha256));
            }
        }
        crate::stats::presence_pass();
        let mut held = Vec::new();
        let mut wanted = Vec::new();
        let mut why = Vec::new();
        for o in sorted {
            hex64("FetchPlan.object", &o.sha256)?;
            let p = if manifest.is_some_and(|m| m.sha256 == o.sha256) {
                manifest_presence(store, &o)?
            } else {
                presence(store, &o)?
            };
            if p.held() {
                held.push(o.clone());
            } else {
                wanted.push(o.clone());
            }
            why.push((o, p));
        }
        Ok((
            FetchPlan {
                session: session.to_string(),
                manifest: manifest.cloned(),
                held,
                wanted,
            },
            why,
        ))
    }

    pub fn wants(&self, sha256: &str) -> bool {
        self.wanted.iter().any(|o| o.sha256 == sha256)
    }
    pub fn wanted_bytes(&self) -> u64 {
        self.wanted.iter().map(|o| o.length).sum()
    }
    pub fn held_bytes(&self) -> u64 {
        self.held.iter().map(|o| o.length).sum()
    }
    pub fn declared_bytes(&self) -> u64 {
        self.wanted_bytes() + self.held_bytes()
    }

    /// The proof half. Every declared object is now installed AND vouched for by this
    /// store's own record; anything else names the first object that is not.
    ///
    /// This is deliberately not "the transfer reported success": a 200 on every GET says the
    /// remote answered, and says nothing about what is on this disk.
    pub fn complete(&self, store: &Store) -> Result<()> {
        self.complete_with(store, crate::checkpoint::walk)
    }

    /// Prove completion of the selected CozyTensors runtime closure while ignoring ordinary
    /// snapshot siblings. The FetchPlan document is unchanged; only its declared object set and
    /// the authoritative walk it must equal differ.
    pub fn complete_cozytensors(&self, store: &Store) -> Result<()> {
        self.complete_with(store, crate::checkpoint::walk_cozytensors)
    }

    fn complete_with(
        &self,
        store: &Store,
        walk: fn(&Store, &crate::manifest::Manifest) -> Result<crate::checkpoint::Walk>,
    ) -> Result<()> {
        crate::stats::presence_pass();
        if let Some(manifest_ref) = &self.manifest {
            let manifest = store.read_manifest(manifest_ref)?;
            let reached = walk(store, &manifest)?;
            let mut declared: Vec<ObjectRef> =
                self.held.iter().chain(&self.wanted).cloned().collect();
            declared.sort_by(|left, right| {
                (&left.sha256, left.length).cmp(&(&right.sha256, right.length))
            });
            let mut actual: Vec<ObjectRef> = reached.distinct().into_iter().cloned().collect();
            actual.push(manifest_ref.clone());
            actual.sort_by(|left, right| {
                (&left.sha256, left.length).cmp(&(&right.sha256, right.length))
            });
            if declared != actual {
                return refuse(
                    Code::NOT_CONTAINED,
                    "fetch plan blob set differs from the admitted manifest closure",
                );
            }
        }
        for o in self.held.iter().chain(self.wanted.iter()) {
            let resident = if self.manifest.as_ref().is_some_and(|m| m.sha256 == o.sha256) {
                manifest_presence(store, o)?
            } else {
                presence(store, o)?
            };
            if !resident.held() {
                return refuse(
                    Code::DURABILITY_UNPROVEN,
                    format!(
                        "{}: declared by the fetch plan and not verifiably resident — the pull \
                         is INCOMPLETE, and a transfer's own success report is not residency",
                        o.id()
                    ),
                );
            }
        }
        Ok(())
    }
}

/// The Manifest namespace's presence question — the same one [`FetchPlan`] asks about the
/// Manifest it anchors on, asked about a Manifest that anchors nothing. `of_objects` cannot
/// answer it, because a bare object set is a set of BLOBS and `presence` looks in the blob
/// namespace; a caller restoring Manifests from a cache would get "absent" for every one it
/// already holds. Public for that caller, and for no other reason.
pub fn manifest_presence(store: &Store, object: &ObjectRef) -> Result<Presence> {
    match store.read_manifest(object) {
        Ok(_) => Ok(Presence::Recorded),
        Err(error) if error.code == Code::OBJECT_ABSENT => Ok(Presence::Absent),
        Err(error) if error.code == Code::LENGTH_MISMATCH => Err(error),
        Err(error) => {
            let why = error.to_string();
            store.remove_corrupt_manifest(&object.sha256)?;
            Ok(Presence::CorruptRemoved { why })
        }
    }
}

/// The store's own presence question, and the repair that follows a bad answer.
fn presence(store: &Store, o: &ObjectRef) -> Result<Presence> {
    if !store.contains(&o.sha256) {
        return Ok(Presence::Absent);
    }
    let (record, presence) = match store.record_valid(&o.sha256) {
        Ok(record) => (record, Presence::Recorded),
        Err(_) => match store.verify(&o.sha256)? {
            Verdict::CorruptRemoved { why } => return Ok(Presence::CorruptRemoved { why }),
            Verdict::Verified { .. } | Verdict::Invalidated { .. } => {
                let record = store
                    .record_valid(&o.sha256)
                    .map_err(|why| crate::err::Refusal {
                        code: Code::OBJECT_CORRUPT,
                        detail: format!("object verification changed while planning: {why}"),
                    })?;
                (record, Presence::Rehashed)
            }
        },
    };
    if record.length != o.length {
        return refuse(
            Code::LENGTH_MISMATCH,
            "declared object length differs from verified bytes",
        );
    }
    Ok(presence)
}

impl FetchPlan {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Self::from_value(&canon::parse(bytes, limits::DOC_MAX_BYTES)?)
    }
    pub fn canonical_bytes(&self) -> Vec<u8> {
        canon::write(&self.to_value())
    }

    pub fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("FetchPlan", v)?;
        let mut held = Vec::new();
        for o in as_arr("FetchPlan", "held", f.req("held")?)? {
            held.push(ObjectRef::from_value("FetchPlan.held", o)?);
        }
        let session = f.req_str("session")?.to_string();
        let manifest = match f.opt("manifest") {
            Some(value) => {
                let parsed = ObjectRef::from_value("FetchPlan.manifest", value)?;
                validate_manifest_ref(&parsed)?;
                Some(parsed)
            }
            None => None,
        };
        let mut wanted = Vec::new();
        for o in as_arr("FetchPlan", "wanted", f.req("wanted")?)? {
            wanted.push(ObjectRef::from_value("FetchPlan.wanted", o)?);
        }
        ascii_name("FetchPlan.session", &session, limits::MAX_NAME_BYTES)?;
        let held = sorted_refs("FetchPlan.held", held)?;
        let wanted = sorted_refs("FetchPlan.wanted", wanted)?;
        if held.is_empty() && wanted.is_empty() {
            return refuse(Code::MISSING_FIELD, "a fetch plan covers no object");
        }
        Ok(FetchPlan {
            session,
            manifest,
            held,
            wanted,
        })
    }

    pub fn to_value(&self) -> Value {
        let mut fields = vec![
            (
                "held",
                Value::arr(self.held.iter().map(|o| o.to_value()).collect()),
            ),
            ("session", Value::str(self.session.clone())),
            (
                "wanted",
                Value::arr(self.wanted.iter().map(|o| o.to_value()).collect()),
            ),
        ];
        if let Some(manifest) = &self.manifest {
            fields.push(("manifest", manifest.to_value()));
        }
        Value::obj(fields)
    }
}

fn validate_manifest_ref(manifest: &ObjectRef) -> Result<()> {
    if manifest.length == 0 || manifest.length > crate::manifest::Manifest::MAX_BYTES as u64 {
        return refuse(
            Code::SIZE_CAP,
            format!(
                "manifest length {} is outside 1..={}",
                manifest.length,
                crate::manifest::Manifest::MAX_BYTES
            ),
        );
    }
    usize::try_from(manifest.length).map_err(|_| crate::err::Refusal {
        code: Code::SIZE_CAP,
        detail: "manifest length exceeds this platform's address space".into(),
    })?;
    Ok(())
}

// ---------------------------------------------------------------- the grant

/// An authorization to ADMIT one object's bytes, and the expectation the border enforces on
/// them. Like `PublishGrant` there is no constructor that omits the binding: a grant is
/// minted only against a plan's wanted set, so it can never name an object the session did
/// not declare, and never one the store already holds.
///
/// It carries no URL, no token, no expiry and no method. Those are the transport adapter's,
/// and a grant that carried them would be a credential this plane has no business holding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveryGrant {
    pub session: String,
    pub object: ObjectRef,
    pub destination: Destination,
}

/// Where the bytes this grant authorizes come to rest.
///
/// The grant already carried this bit — it was `manifest: bool` — so a third destination is
/// the existing shape rather than a new concept: the downloader commits a hashed temp file
/// where the grant says, so the staging path shares the CAS path's transport.
///
/// `Blob` and `Manifest` are store objects. `Staging` is not: it is a foreign source
/// carrier the store will read once and throw away, and it has no catalog record, no census
/// membership and no `FetchPlan` that answers HELD or WANTED about it. See
/// [`crate::staging`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Destination {
    Blob,
    Manifest,
    Staging,
}

impl Destination {
    pub fn as_str(&self) -> &'static str {
        match self {
            Destination::Blob => "blob",
            Destination::Manifest => "manifest",
            Destination::Staging => "staging",
        }
    }
}

impl DeliveryGrant {
    /// The only way to get one.
    pub fn mint(plan: &FetchPlan, object: &ObjectRef) -> Result<Self> {
        hex64("DeliveryGrant.object", &object.sha256)?;
        match plan.wanted.iter().find(|o| o.sha256 == object.sha256) {
            None if plan.held.iter().any(|o| o.sha256 == object.sha256) => refuse(
                Code::NOT_CONTAINED,
                format!(
                    "{}: already HELD by this store — a pull that re-fetches a verified object \
                     is not idempotent, it is a second download",
                    object.id()
                ),
            ),
            None => refuse(
                Code::NOT_CONTAINED,
                format!(
                    "{}: not in fetch session {:?}'s wanted set — the grant set is a subset of \
                     a plan made before a byte moved, never a superset",
                    object.id(),
                    plan.session
                ),
            ),
            Some(o) if o.length != object.length => refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "{}: the plan declares {} B and the grant asks for {} B — an ObjectRef is \
                     length-bearing and the two halves may not disagree",
                    object.id(),
                    o.length,
                    object.length
                ),
            ),
            Some(o) => Ok(DeliveryGrant {
                session: plan.session.clone(),
                object: o.clone(),
                destination: match plan.manifest.as_ref() {
                    Some(m) if m.sha256 == o.sha256 => Destination::Manifest,
                    _ => Destination::Blob,
                },
            }),
        }
    }

    /// A grant to STAGE one carrier, with no `FetchPlan` behind it.
    ///
    /// There is no closure for this to be a subset of: a carrier is not declared by a
    /// remote catalog and is not reached by any manifest. The binding it carries is the
    /// object id itself, which [`DeliveryGrant::admit`] enforces by digest through the same
    /// hashing core `put_stream` uses — so the authorization is exactly as strong as a
    /// minted one, and narrower, because a staging grant can never land in `blobs/`.
    pub fn stage(session: &str, object: &ObjectRef) -> Result<Self> {
        ascii_name("DeliveryGrant.session", session, limits::MAX_NAME_BYTES)?;
        hex64("DeliveryGrant.object", &object.sha256)?;
        if object.length == 0 {
            return refuse(
                Code::LENGTH_MISMATCH,
                "a staged carrier is a file the converter opens; zero bytes is not one",
            );
        }
        Ok(DeliveryGrant {
            session: session.to_string(),
            object: object.clone(),
            destination: Destination::Staging,
        })
    }

    /// DERIVED, not stored — the same key publish wrote to. A staging grant's key is not a
    /// store key at all, and saying so in the same shape is the point.
    pub fn key(&self) -> String {
        match self.destination {
            Destination::Manifest => crate::storage::manifest_key(&self.object.sha256)
                .expect("DeliveryGrant carries a validated digest"),
            Destination::Blob => {
                final_key(&self.object.sha256).expect("DeliveryGrant carries a validated digest")
            }
            Destination::Staging => format!("staging/{}", self.object.sha256),
        }
    }

    /// THE DOOR. Every byte this system pulls from anywhere crosses exactly this call:
    /// streamed in a bounded buffer, hashed as it goes, length- and digest-checked against
    /// the grant, committed under no-clobber admission at the digest that was COMPUTED — or
    /// committed nowhere at all, with the partial temp file removed.
    ///
    /// `r` is whatever the transport adapter opened. This plane neither knows nor asks.
    pub fn admit<R: Read>(&self, store: &Store, r: &mut R) -> Result<Admitted> {
        if self.destination == Destination::Staging {
            // Not an admission at all: the bytes cross the same hashing core and land
            // OUTSIDE every namespace a census walks. Nothing is recorded about them,
            // because there is nothing to be atomic about — see `crate::staging`.
            let carrier = crate::staging::stage(store, r, &self.object)?;
            return Ok(Admitted {
                object: carrier.object,
                first_writer: carrier.fetched,
            });
        }
        let put = if self.destination == Destination::Manifest {
            validate_manifest_ref(&self.object)?;
            let capacity =
                usize::try_from(self.object.length).map_err(|_| crate::err::Refusal {
                    code: Code::SIZE_CAP,
                    detail: "manifest length exceeds this platform's address space".into(),
                })?;
            let mut bytes = Vec::with_capacity(capacity);
            r.take(self.object.length.saturating_add(1))
                .read_to_end(&mut bytes)
                .map_err(|error| crate::err::Refusal {
                    code: Code::IO_FAILED,
                    detail: format!("read manifest body: {error}"),
                })?;
            if ObjectRef::of(&bytes) != self.object {
                return refuse(
                    Code::OBJECT_ID_MISMATCH,
                    "fetched manifest bytes disagree with the plan",
                );
            }
            let manifest = crate::manifest::Manifest::parse(&bytes)?;
            store.put_manifest(&manifest)?
        } else {
            store.put_stream(r, Some(&self.object), &Default::default())?
        };
        Ok(Admitted {
            object: put.obj,
            first_writer: put.admitted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::header::{Asset, Body, Closure, Header, Part, Tensor};
    use crate::manifest::{Draft, Entry, Manifest};
    use crate::meta::Meta;
    use crate::registry;
    use crate::store::Fault;
    use crate::{checkpoint, read};
    use std::fs;

    fn temporary(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "tensorfs-fetch-{name}-{}-{}",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ))
    }

    fn fixture(root: &std::path::Path) -> (Store, ObjectRef, ObjectRef, Vec<u8>) {
        let store = Store::init(root).unwrap();
        let spec = registry::seeds()
            .into_iter()
            .find(|seed| seed.alias == "plain/1")
            .unwrap()
            .spec;
        let header = Header {
            configs: Vec::new(),
            assets: Vec::new(),
            encodings: vec![spec.clone()],
            components: vec![(
                "model".into(),
                vec![(
                    "weight".into(),
                    Tensor {
                        dtype: Dtype::F32,
                        shape: vec![1],
                        encoding: spec.object_id(),
                        parts: vec![("value".into(), Part::plan(Dtype::F32, vec![1], &[0; 4]))],
                    },
                )],
            )],
        };
        header.validate(&Closure::default()).unwrap();
        let header_bytes = header.canonical_bytes().unwrap();
        let header_ref = store
            .put_stream(
                &mut header_bytes.as_slice(),
                Some(&ObjectRef::of(&header_bytes)),
                &Fault::default(),
            )
            .unwrap()
            .obj;
        let manifest = Draft {
            entries: vec![(
                "model.cozytensors".into(),
                Entry::CozyTensors(header_ref.clone()),
            )],
        }
        .seal()
        .unwrap();
        let manifest_bytes = manifest.canonical_bytes();
        let manifest_ref = store.put_manifest(&manifest).unwrap().obj;
        (store, manifest_ref, header_ref, manifest_bytes)
    }

    #[test]
    fn manifest_length_cap_refuses_before_allocation() {
        let root = temporary("cap");
        let store = Store::init(&root).unwrap();
        let manifest = ObjectRef {
            sha256: "11".repeat(32),
            length: Manifest::MAX_BYTES as u64 + 1,
        };
        assert_eq!(
            FetchPlan::of(&store, "pull", &manifest, &[ObjectRef::of(b"blob")],)
                .unwrap_err()
                .code,
            Code::SIZE_CAP
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn wrong_reference_lengths_refuse_without_removing_verified_bytes() {
        let root = temporary("wrong-reference-length");
        let (store, manifest, header, manifest_bytes) = fixture(&root);
        let wrong_manifest = ObjectRef {
            length: manifest.length + 1,
            ..manifest.clone()
        };
        assert_eq!(
            FetchPlan::of(
                &store,
                "wrong-manifest",
                &wrong_manifest,
                std::slice::from_ref(&header)
            )
            .unwrap_err()
            .code,
            Code::LENGTH_MISMATCH
        );
        assert_eq!(
            fs::read(store.manifest_path(&manifest.sha256)).unwrap(),
            manifest_bytes
        );
        let wrong_blob = ObjectRef {
            length: header.length + 1,
            ..header.clone()
        };
        assert_eq!(
            FetchPlan::of_objects(&store, "wrong-blob", &[wrong_blob])
                .unwrap_err()
                .code,
            Code::LENGTH_MISMATCH
        );
        assert!(store.record_valid(&header.sha256).is_ok());
        let (plan, _) = FetchPlan::of(&store, "still-valid", &manifest, &[header]).unwrap();
        assert!(plan.wanted.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn corrupt_manifest_is_removed_and_readmitted_without_touching_blobs() {
        let root = temporary("repair");
        let (store, manifest, header, manifest_bytes) = fixture(&root);
        let path = store.manifest_path(&manifest.sha256);
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o644);
        fs::set_permissions(&path, permissions).unwrap();
        fs::write(&path, vec![b'x'; manifest.length as usize]).unwrap();

        let (plan, why) =
            FetchPlan::of(&store, "repair", &manifest, std::slice::from_ref(&header)).unwrap();
        assert!(matches!(
            why.iter()
                .find(|(object, _)| object == &manifest)
                .unwrap()
                .1,
            Presence::CorruptRemoved { .. }
        ));
        assert!(!path.exists());
        assert!(store.blob_path(&header.sha256).exists());
        let grant = DeliveryGrant::mint(&plan, &manifest).unwrap();
        grant.admit(&store, &mut manifest_bytes.as_slice()).unwrap();
        plan.complete(&store).unwrap();
        assert!(path.exists());

        let second = temporary("oversized-body");
        let destination = Store::init(&second).unwrap();
        let (plan, _) = FetchPlan::of(
            &destination,
            "oversized",
            &manifest,
            std::slice::from_ref(&header),
        )
        .unwrap();
        let grant = DeliveryGrant::mint(&plan, &manifest).unwrap();
        let mut oversized = manifest_bytes;
        oversized.push(b'x');
        assert_eq!(
            grant
                .admit(&destination, &mut oversized.as_slice())
                .unwrap_err()
                .code,
            Code::OBJECT_ID_MISMATCH
        );
        assert!(!destination.manifest_path(&manifest.sha256).exists());
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(second);
    }

    #[test]
    fn runtime_closure_ignores_missing_snapshot_siblings() {
        let root = temporary("runtime-closure");
        let store = Store::init(&root).unwrap();
        let spec = registry::seeds()
            .into_iter()
            .find(|seed| seed.alias == "plain/1")
            .unwrap()
            .spec;
        let config_bytes = b"{}";
        let config = ObjectRef::of(config_bytes);
        store
            .put_stream(
                &mut config_bytes.as_slice(),
                Some(&config),
                &Fault::default(),
            )
            .unwrap();
        let tokenizer_bytes = b"tokenizer vocabulary";
        let tokenizer = ObjectRef::of(tokenizer_bytes);
        store
            .put_stream(
                &mut tokenizer_bytes.as_slice(),
                Some(&tokenizer),
                &Fault::default(),
            )
            .unwrap();
        let part_bytes = vec![0u8; 260];
        let part = ObjectRef::of(&part_bytes);
        store
            .put_stream(&mut part_bytes.as_slice(), Some(&part), &Fault::default())
            .unwrap();
        let header_document = Header {
            configs: vec![("model".into(), config_bytes.to_vec())],
            assets: vec![(
                "tokenizer/vocab.txt".into(),
                Asset {
                    logical_sha256: tokenizer.sha256.clone(),
                    logical_length: tokenizer.length,
                    media_type: "text/plain".into(),
                    segments: vec![tokenizer.clone()],
                },
            )],
            encodings: vec![spec.clone()],
            components: vec![(
                "model".into(),
                vec![(
                    "weight".into(),
                    Tensor {
                        dtype: Dtype::F32,
                        shape: vec![65],
                        encoding: spec.object_id(),
                        parts: vec![(
                            "value".into(),
                            Part {
                                dtype: Dtype::F32,
                                shape: vec![65],
                                body: Body::Segments(vec![part.clone()]),
                            },
                        )],
                    },
                )],
            )],
        };
        header_document.validate(&Closure::default()).unwrap();
        let header_bytes = header_document.canonical_bytes().unwrap();
        let header = ObjectRef::of(&header_bytes);
        store
            .put_stream(
                &mut header_bytes.as_slice(),
                Some(&header),
                &Fault::default(),
            )
            .unwrap();
        let readme = ObjectRef::of(b"checkpoint notes");
        let sample = ObjectRef::of(b"sample image bytes");
        let manifest = Draft {
            entries: vec![
                ("README.txt".into(), Entry::File(readme.clone())),
                (
                    "model.cozytensors".into(),
                    Entry::CozyTensors(header.clone()),
                ),
                ("model/config.json".into(), Entry::File(config.clone())),
                ("samples/example.png".into(), Entry::File(sample.clone())),
            ],
        }
        .seal()
        .unwrap();
        let manifest_ref = store.put_manifest(&manifest).unwrap().obj;

        let runtime = checkpoint::walk_cozytensors(&store, &manifest).unwrap();
        assert_eq!(runtime.count("header"), 1);
        assert_eq!(runtime.count("config"), 0);
        assert_eq!(runtime.count("model_asset"), 1);
        assert_eq!(runtime.count("part"), 1);
        assert_eq!(runtime.distinct().len(), 3);
        assert!(runtime.siblings.is_empty());
        runtime.require_resident(&store).unwrap();

        let snapshot = checkpoint::walk(&store, &manifest).unwrap();
        assert_eq!(
            snapshot.siblings,
            vec![
                "README.txt".to_string(),
                "model/config.json".to_string(),
                "samples/example.png".to_string(),
            ]
        );
        assert!(snapshot.objects.iter().any(|reached| reached.obj == readme));
        assert!(snapshot.objects.iter().any(|reached| reached.obj == sample));
        assert_eq!(
            snapshot.require_resident(&store).unwrap_err().code,
            Code::OBJECT_ABSENT
        );

        let declared = vec![header, tokenizer, part];
        let (plan, _) = FetchPlan::of(&store, "runtime", &manifest_ref, &declared).unwrap();
        plan.complete_cozytensors(&store).unwrap();
        assert_eq!(plan.complete(&store).unwrap_err().code, Code::NOT_CONTAINED);

        let meta = Meta::open(&store).unwrap();
        let (lease, _) = read::acquire_cozytensors(&store, &meta, &manifest).unwrap();
        assert_eq!(lease.objects().len(), 3);
        lease.release(&meta).unwrap();
        assert_eq!(
            read::acquire_manifest(&store, &meta, &manifest)
                .err()
                .unwrap()
                .code,
            Code::OBJECT_ABSENT
        );

        let _ = fs::remove_dir_all(root);
    }
}

/// What crossing the door produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admitted {
    pub object: ObjectRef,
    /// This process won the no-clobber admission, rather than finding the id already
    /// installed by a concurrent puller. Both are success; only one moved the file.
    pub first_writer: bool,
}

impl DeliveryGrant {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Self::from_value(&canon::parse_canonical(bytes, limits::SPEC_MAX_BYTES)?)
    }
    pub fn canonical_bytes(&self) -> Vec<u8> {
        canon::write(&self.to_value())
    }

    pub fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("DeliveryGrant", v)?;
        let kind = f.req_str("kind")?;
        let destination = match kind {
            "blob" => Destination::Blob,
            "manifest" => Destination::Manifest,
            "staging" => Destination::Staging,
            _ => {
                return refuse(
                    Code::UNKNOWN_FIELD,
                    "grant kind must be blob, manifest or staging",
                )
            }
        };
        let object = ObjectRef::from_value("DeliveryGrant.object", f.req("object")?)?;
        let session = f.req_str("session")?.to_string();
        ascii_name("DeliveryGrant.session", &session, limits::MAX_NAME_BYTES)?;
        Ok(DeliveryGrant {
            session,
            object,
            destination,
        })
    }

    pub fn to_value(&self) -> Value {
        // `key` is deliberately ABSENT for the same reason `PublishGrant` omits it: a
        // derivable fact stored beside its source is a second authority waiting to disagree.
        Value::obj(vec![
            ("kind", Value::str(self.destination.as_str())),
            ("object", self.object.to_value()),
            ("session", Value::str(self.session.clone())),
        ])
    }
}

#[cfg(test)]
mod object_set_tests {
    use super::*;
    use crate::store::Fault;
    use std::fs;

    fn temporary(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "tensorfs-fetch-objects-{name}-{}-{}",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ))
    }

    #[test]
    fn bare_object_set_splits_admits_and_completes() {
        let root = temporary("split");
        let store = Store::init(&root).unwrap();
        let held_bytes = b"a member the store already holds".to_vec();
        let held_ref = ObjectRef::of(&held_bytes);
        store
            .put_stream(
                &mut held_bytes.as_slice(),
                Some(&held_ref),
                &Fault::default(),
            )
            .unwrap();
        let wanted_bytes = b"a member the store does not hold".to_vec();
        let wanted_ref = ObjectRef::of(&wanted_bytes);

        let declared = vec![held_ref.clone(), wanted_ref.clone()];
        let (plan, why) = FetchPlan::of_objects(&store, "members", &declared).unwrap();
        assert_eq!(plan.manifest, None);
        assert_eq!(plan.held, vec![held_ref.clone()]);
        assert_eq!(plan.wanted, vec![wanted_ref.clone()]);
        assert!(why
            .iter()
            .any(|(o, p)| o == &wanted_ref && *p == Presence::Absent));

        // The document round-trips without a manifest field.
        let parsed = FetchPlan::parse(&plan.canonical_bytes()).unwrap();
        assert_eq!(parsed, plan);

        // Incomplete until the wanted member lands; a held member cannot be re-granted.
        assert_eq!(
            plan.complete(&store).unwrap_err().code,
            Code::DURABILITY_UNPROVEN
        );
        assert_eq!(
            DeliveryGrant::mint(&plan, &held_ref).unwrap_err().code,
            Code::NOT_CONTAINED
        );
        let grant = DeliveryGrant::mint(&plan, &wanted_ref).unwrap();
        assert_eq!(grant.destination, Destination::Blob);
        grant.admit(&store, &mut wanted_bytes.as_slice()).unwrap();
        plan.complete(&store).unwrap();

        // The warm replay: the same declaration now wants NOTHING.
        let (warm, _) = FetchPlan::of_objects(&store, "members", &declared).unwrap();
        assert_eq!(warm.wanted, Vec::new());
        assert_eq!(warm.held.len(), 2);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn corrupt_member_heals_into_wanted() {
        let root = temporary("heal");
        let store = Store::init(&root).unwrap();
        let bytes = b"bytes that will rot on disk".to_vec();
        let object = ObjectRef::of(&bytes);
        store
            .put_stream(&mut bytes.as_slice(), Some(&object), &Fault::default())
            .unwrap();
        let path = store.blob_path(&object.sha256);
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o644);
        fs::set_permissions(&path, permissions).unwrap();
        fs::write(&path, vec![b'x'; bytes.len()]).unwrap();

        let (plan, why) =
            FetchPlan::of_objects(&store, "heal", std::slice::from_ref(&object)).unwrap();
        assert!(matches!(why[0].1, Presence::CorruptRemoved { .. }));
        assert_eq!(plan.wanted, vec![object.clone()]);
        assert!(!path.exists());
        let grant = DeliveryGrant::mint(&plan, &object).unwrap();
        grant.admit(&store, &mut bytes.as_slice()).unwrap();
        plan.complete(&store).unwrap();
        let _ = fs::remove_dir_all(root);
    }
}
