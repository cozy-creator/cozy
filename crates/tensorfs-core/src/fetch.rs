//! The fetch/admit plane (tfs-003 row 220, cr-016's `pull`) — cloud.rs run backwards.
//!
//! `cloud.rs` is bytes LEAVING: the store already holds the truth, so a grant only has to
//! bind the upload to it. This is bytes ARRIVING, and the asymmetry is the whole design:
//! nothing about arriving bytes is known until the last one is hashed, so the authorization
//! is an EXPECTATION and the thing that enforces it is `Store::put_stream` — a streaming,
//! bounded-buffer, no-clobber door that commits at the digest it computed or commits
//! nothing. There is no second admission path and this module does not add one.
//!
//! **TensorFS never sees a presigned URL or a worker credential** (boundaries.md §2). A
//! `DeliveryGrant` carries a digest and a length and nothing else. The caller's transport
//! adapter — `HTTP/R2 fetch with attempt authority`, boundaries.md §9's placement row, owned
//! by cozy-runtime — holds the URL, opens the connection, and hands the border a byte
//! stream. So the same grant admits bytes from tensorhub, from a mirror, from a pod-local
//! file or from a pipe, and the border cannot tell the difference, which is the point: the
//! digest is the only authority.
//!
//! **Idempotency is a computed fact, not a client's guess.** `FetchPlan::of` splits a
//! declared object set into HELD and WANTED against a real store, and a caller that fetches
//! anything outside `wanted` is not resuming — it is re-downloading. The split asks the
//! store's own question (`record_valid`: may these bytes be trusted without hashing them
//! again?) and, only when the record cannot answer, pays one rehash. Bytes that fail that
//! rehash are QUARANTINED by `verify` and then land in `wanted`, which is why a corrupted
//! local object heals on the next pull instead of being skipped as "already present".

use crate::canon::{self, as_arr, Fields, Value};
use crate::err::{refuse, Code, Result};
use crate::ids::{ascii_name, hex64, ObjectRef};
use crate::limits;
use crate::store::{Store, Verdict};
use std::io::Read;

/// Where an object lives in a content-addressed backend. Deliberately the SAME derivation
/// `cloud::final_key` publishes at: if pull and publish ever computed different keys, one of
/// them would be writing where nobody reads.
pub use crate::cloud::final_key;

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
    /// Installed and WRONG. `verify` renamed it into `quarantine/`; the id is now free for
    /// the fetch to land at, so this is a repair, not a refusal.
    Quarantined { why: String },
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
            Presence::Quarantined { .. } => "quarantined",
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
    pub snapshot: ObjectRef,
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
        snapshot: &ObjectRef,
        declared: &[ObjectRef],
    ) -> Result<(FetchPlan, Vec<(ObjectRef, Presence)>)> {
        ascii_name("FetchPlan.session", session, limits::MAX_NAME_BYTES)?;
        if declared.is_empty() {
            return refuse(
                Code::MISSING_FIELD,
                "a fetch plan declares no object — a checkpoint that reaches nothing is not \
                 something a worker can pull",
            );
        }
        let mut sorted: Vec<ObjectRef> = declared.to_vec();
        sorted.sort_by(|a, b| a.sha256.cmp(&b.sha256));
        sorted.dedup_by(|a, b| a.sha256 == b.sha256);
        if sorted.len() != declared.len() {
            return refuse(
                Code::SORT_ORDER,
                "the declared object set lists one digest twice — two opinions about how many \
                 bytes this pull moves",
            );
        }
        // THE ROOT IS PART OF THE PULL. A publish closure lists what the snapshot REACHES
        // and carries the snapshot beside it, because the publisher already holds the
        // manifest. A puller does not: landing every reachable object without the manifest
        // produces a store full of bytes and no artifact, and `snapshot verify` in the
        // destination says OBJECT_ABSENT for the very digest that was asked for. Observed on
        // the first live pull, and fixed here rather than in each caller — a rule every
        // transport adapter has to remember separately is a rule one of them will forget.
        if !sorted.iter().any(|o| o.sha256 == snapshot.sha256) {
            sorted.push(snapshot.clone());
            sorted.sort_by(|a, b| a.sha256.cmp(&b.sha256));
        }

        let mut held = Vec::new();
        let mut wanted = Vec::new();
        let mut why = Vec::new();
        for o in sorted {
            hex64("FetchPlan.object", &o.sha256)?;
            let p = presence(store, &o)?;
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
                snapshot: snapshot.clone(),
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
        for o in self.held.iter().chain(self.wanted.iter()) {
            if !presence(store, o)?.held() {
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

/// The store's own presence question, and the repair that follows a bad answer.
fn presence(store: &Store, o: &ObjectRef) -> Result<Presence> {
    if !store.contains(&o.sha256) {
        return Ok(Presence::Absent);
    }
    if store.record_valid(&o.sha256).is_ok() {
        return Ok(Presence::Recorded);
    }
    // The record could not vouch. One rehash decides, and a disagreement is quarantined by
    // `verify` itself — which is what frees the id for the re-fetch below.
    match store.verify(&o.sha256)? {
        Verdict::Verified { .. } | Verdict::Invalidated { .. } => Ok(Presence::Rehashed),
        Verdict::Quarantined { why, .. } => Ok(Presence::Quarantined { why }),
    }
}

impl FetchPlan {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Self::from_value(&canon::parse_canonical(bytes, limits::DOC_MAX_BYTES)?)
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
        let snapshot = ObjectRef::from_value("FetchPlan.snapshot", f.req("snapshot")?)?;
        let mut wanted = Vec::new();
        for o in as_arr("FetchPlan", "wanted", f.req("wanted")?)? {
            wanted.push(ObjectRef::from_value("FetchPlan.wanted", o)?);
        }
        f.done()?;
        ascii_name("FetchPlan.session", &session, limits::MAX_NAME_BYTES)?;
        for (what, set) in [("held", &held), ("wanted", &wanted)] {
            let mut ids: Vec<&str> = set.iter().map(|o| o.sha256.as_str()).collect();
            let given = ids.clone();
            ids.sort_unstable();
            ids.dedup();
            if ids != given {
                return refuse(
                    Code::SORT_ORDER,
                    format!("FetchPlan.{what} must be sorted and unique by digest"),
                );
            }
        }
        if held.is_empty() && wanted.is_empty() {
            return refuse(Code::MISSING_FIELD, "a fetch plan covers no object");
        }
        Ok(FetchPlan {
            session,
            snapshot,
            held,
            wanted,
        })
    }

    pub fn to_value(&self) -> Value {
        Value::obj(vec![
            (
                "held",
                Value::arr(self.held.iter().map(|o| o.to_value()).collect()),
            ),
            ("session", Value::str(self.session.clone())),
            ("snapshot", self.snapshot.to_value()),
            (
                "wanted",
                Value::arr(self.wanted.iter().map(|o| o.to_value()).collect()),
            ),
        ])
    }
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
            }),
        }
    }

    /// DERIVED, not stored — the same key publish wrote to.
    pub fn key(&self) -> String {
        final_key(&self.object.sha256)
    }

    /// THE DOOR. Every byte this system pulls from anywhere crosses exactly this call:
    /// streamed in a bounded buffer, hashed as it goes, length- and digest-checked against
    /// the grant, committed under no-clobber admission at the digest that was COMPUTED — or
    /// committed nowhere at all, with the partial temp file removed.
    ///
    /// `r` is whatever the transport adapter opened. This plane neither knows nor asks.
    pub fn admit<R: Read>(&self, store: &Store, r: &mut R) -> Result<Admitted> {
        let put = store.put_stream(r, Some(&self.object), &Default::default())?;
        Ok(Admitted {
            object: put.obj,
            first_writer: put.admitted,
        })
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
        let object = ObjectRef::from_value("DeliveryGrant.object", f.req("object")?)?;
        let session = f.req_str("session")?.to_string();
        f.done()?;
        ascii_name("DeliveryGrant.session", &session, limits::MAX_NAME_BYTES)?;
        Ok(DeliveryGrant { session, object })
    }

    pub fn to_value(&self) -> Value {
        // `key` is deliberately ABSENT for the same reason `PublishGrant` omits it: a
        // derivable fact stored beside its source is a second authority waiting to disagree.
        Value::obj(vec![
            ("object", self.object.to_value()),
            ("session", Value::str(self.session.clone())),
        ])
    }
}
