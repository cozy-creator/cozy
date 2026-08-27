//! The cloud publish protocol (tfs-017), GREEN branch.
//!
//! The invariant is no-corrupt/no-clobber. HOW it is enforced was a qualification outcome,
//! not an assumption, and the gate came back GREEN against live R2 (`scripts/r2-qualify.py`,
//! 10/10 arms): a SigV4-signed `x-amz-checksum-sha256` binds the body to its digest (a wrong
//! body is `400 BadDigest`, a stripped header is `403 SignatureDoesNotMatch`) and
//! `If-None-Match: *` refuses replacement (`412 PreconditionFailed`; eight concurrent
//! same-key writers produced exactly one `200`). So publish is
//!
//!   declare closure -> conditional checksummed PUTs AT FINAL KEYS
//!   -> verify-resident-on-conflict -> VerifiedObject facts
//!   -> atomic catalog-root publication
//!
//! and **no staging namespace, staged-verifier pass or promotion copy exists anywhere in
//! this file**. The staged protocol was the fallback for a RED gate; it is not built.
//!
//! What lives here is the DOCUMENT plane. TensorFS core links no HTTP, no cloud SDK and no
//! credentials, by fence — the coordinator executes the requests. What TensorFS owns is the
//! shape of what may be authorized and the facts that come back, and the shape is where the
//! invariant is actually held: `PublishGrant` cannot be constructed without both conditions,
//! so an unconditional or unchecksummed final-key grant is not a thing a caller can express.
//!
//! Possession stays a DOOR (law 13): a grant is session authorization, never proof the
//! caller holds bytes that already exist server-side. No product policy here mints or
//! consumes a possession receipt, so no range challenge is implemented.

use crate::b64;
use crate::canon::{as_arr, as_str, Fields, Value};
use crate::err::{refuse, Code, Result};
use crate::ids::{ascii_name, check_format, hex64, prefixed, Doc, ObjectRef};
use crate::limits;

/// The final key of an object, DERIVED from its digest and never stored. Storing it would
/// make the key a second authority that could disagree with the identity it names.
pub fn final_key(sha256: &str) -> String {
    format!(
        "objects/sha256/{}/{}/{}",
        &sha256[..2],
        &sha256[2..4],
        sha256
    )
}

// ---------------------------------------------------------------- declare-first

/// The exact object set a publish session covers, declared BEFORE any byte moves. The hub
/// issues grants only for what is in here and only for what it does not already hold, so a
/// rename-only dialect of an already-mirrored base uploads nearly nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishClosure {
    pub session: String,
    pub snapshot: ObjectRef,
    /// Sorted unique by digest. A closure that lists one object twice has two opinions
    /// about how many bytes it is publishing.
    pub objects: Vec<ObjectRef>,
}

/// A first-party build's exact upload set. It is deliberately smaller than a
/// `PublishClosure`: release meaning lives in Tensorhub's canonical build documents, while
/// this document authorizes only these exact bytes to enter the shared object store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactUploadSet {
    pub session: String,
    /// Sorted unique by digest, for the same declare-first reason as `PublishClosure`.
    pub objects: Vec<ObjectRef>,
}

/// The one input shape from which a no-clobber upload grant may be minted. Model publishes
/// and first-party build artifacts differ in their higher-level completion transaction, not
/// in how an exact object is authorized at the storage boundary.
pub trait DeclaredObjectSet {
    fn session(&self) -> &str;
    fn objects(&self) -> &[ObjectRef];

    fn contains(&self, sha256: &str) -> bool {
        self.objects().iter().any(|object| object.sha256 == sha256)
    }
}

impl PublishClosure {
    pub fn bytes(&self) -> u64 {
        self.objects.iter().map(|o| o.length).sum()
    }
}

impl DeclaredObjectSet for PublishClosure {
    fn session(&self) -> &str {
        &self.session
    }

    fn objects(&self) -> &[ObjectRef] {
        &self.objects
    }
}

impl DeclaredObjectSet for ArtifactUploadSet {
    fn session(&self) -> &str {
        &self.session
    }

    fn objects(&self) -> &[ObjectRef] {
        &self.objects
    }
}

fn validate_declared_objects(what: &str, session: &str, objects: &[ObjectRef]) -> Result<()> {
    ascii_name(&format!("{what}.session"), session, limits::MAX_NAME_BYTES)?;
    if objects.is_empty() {
        return refuse(Code::MISSING_FIELD, format!("a {what} declares no object"));
    }
    let mut ids: Vec<&str> = objects
        .iter()
        .map(|object| object.sha256.as_str())
        .collect();
    let given = ids.clone();
    ids.sort_unstable();
    ids.dedup();
    if ids != given {
        return refuse(
            Code::SORT_ORDER,
            format!("{what}.objects must be sorted and unique by digest"),
        );
    }
    Ok(())
}

impl Doc for PublishClosure {
    const FORMAT: &'static str = "tensorfs.publish_closure/1";
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("PublishClosure", v)?;
        check_format(
            "PublishClosure",
            as_str("PublishClosure", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;
        let mut objects = Vec::new();
        for o in as_arr("PublishClosure", "objects", f.req("objects")?)? {
            objects.push(ObjectRef::from_value("PublishClosure.object", o)?);
        }
        let session = f.req_str("session")?.to_string();
        let snapshot = ObjectRef::from_value("PublishClosure.snapshot", f.req("snapshot")?)?;
        f.done()?;
        validate_declared_objects("PublishClosure", &session, &objects)?;
        Ok(PublishClosure {
            session,
            snapshot,
            objects,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("format", Value::str(Self::FORMAT)),
            (
                "objects",
                Value::arr(self.objects.iter().map(|o| o.to_value()).collect()),
            ),
            ("session", Value::str(self.session.clone())),
            ("snapshot", self.snapshot.to_value()),
        ])
    }
}

impl Doc for ArtifactUploadSet {
    const FORMAT: &'static str = "tensorfs.artifact_upload_set/1";
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("ArtifactUploadSet", v)?;
        check_format(
            "ArtifactUploadSet",
            as_str("ArtifactUploadSet", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;
        let mut objects = Vec::new();
        for object in as_arr("ArtifactUploadSet", "objects", f.req("objects")?)? {
            objects.push(ObjectRef::from_value("ArtifactUploadSet.object", object)?);
        }
        let session = f.req_str("session")?.to_string();
        f.done()?;
        validate_declared_objects("ArtifactUploadSet", &session, &objects)?;
        Ok(ArtifactUploadSet { session, objects })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("format", Value::str(Self::FORMAT)),
            (
                "objects",
                Value::arr(
                    self.objects
                        .iter()
                        .map(|object| object.to_value())
                        .collect(),
                ),
            ),
            ("session", Value::str(self.session.clone())),
        ])
    }
}

// ---------------------------------------------------------------- the grant

/// An authorization to write ONE object at ITS final key, under both conditions.
///
/// There is no constructor that omits a condition and no field that can turn one off. The
/// conditions are not stored as data at all — they are DERIVED, so the only representable
/// grant is the safe one. That is the whole mechanism: the v1 ruling this reverses did not
/// fail because someone forgot a flag, it failed because a flag existed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishGrant {
    pub session: String,
    pub object: ObjectRef,
    /// Session authority window. An expiry, never a byte-liveness clock; an expired grant
    /// is a typed REPLAN, never a retry.
    pub expires_at: u64,
}

impl PublishGrant {
    /// The only way to get one. It is checked against the declared closure, so a grant can
    /// never name an object the session never declared.
    pub fn mint<T: DeclaredObjectSet + ?Sized>(
        declared: &T,
        object: &ObjectRef,
        expires_at: u64,
    ) -> Result<Self> {
        hex64("PublishGrant.object", &object.sha256)?;
        if !declared.contains(&object.sha256) {
            return refuse(
                Code::NOT_CONTAINED,
                format!(
                    "{}: not in upload session {:?}'s declared object set — declare-first means \
                     the grant set is a subset of the declaration, never a superset",
                    object.id(),
                    declared.session()
                ),
            );
        }
        Ok(PublishGrant {
            session: declared.session().to_string(),
            object: object.clone(),
            expires_at,
        })
    }

    /// DERIVED, not stored. The key is the digest path.
    pub fn key(&self) -> String {
        final_key(&self.object.sha256)
    }

    /// The base64 sha256 the signature must bind. Derived from the object's own digest, so
    /// a grant cannot advertise a checksum that disagrees with the key it writes to.
    pub fn checksum_sha256(&self) -> Result<String> {
        required_checksum(&self.object.sha256)
    }

    /// Every header the coordinator MUST sign into the presigned URL. Not a suggestion and
    /// not a default: this is the complete set, and the qualification gate proved that a
    /// client which strips any of them gets `SignatureDoesNotMatch`.
    pub fn signed_headers(&self) -> Result<Vec<(String, String)>> {
        Ok(vec![
            ("if-none-match".to_string(), "*".to_string()),
            ("x-amz-checksum-sha256".to_string(), self.checksum_sha256()?),
        ])
    }
}

impl Doc for PublishGrant {
    const FORMAT: &'static str = "tensorfs.publish_grant/1";
    const MAX_BYTES: usize = limits::SPEC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("PublishGrant", v)?;
        check_format(
            "PublishGrant",
            as_str("PublishGrant", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;
        let expires_at = f.req_uint("expires_at")?;
        let object = ObjectRef::from_value("PublishGrant.object", f.req("object")?)?;
        let session = f.req_str("session")?.to_string();
        f.done()?;
        ascii_name("PublishGrant.session", &session, limits::MAX_NAME_BYTES)?;
        Ok(PublishGrant {
            session,
            object,
            expires_at,
        })
    }

    fn to_value(&self) -> Value {
        // `key`, `if_none_match` and `x-amz-checksum-sha256` are deliberately ABSENT. They
        // are derivable facts, and a derivable fact stored beside its source is a second
        // authority waiting to disagree with it.
        Value::obj(vec![
            ("expires_at", Value::uint(self.expires_at)),
            ("format", Value::str(Self::FORMAT)),
            ("object", self.object.to_value()),
            ("session", Value::str(self.session.clone())),
        ])
    }
}

// ---------------------------------------------------------------- the fact

/// One exact cloud backend version was verified at one time. That is the whole claim.
///
/// It never says who uploaded the bytes and never grants perpetual possession; a digest is
/// not read authority and not a cross-tenant disclosure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedObject {
    pub object: ObjectRef,
    /// The backend's own version handle for the exact bytes verified.
    pub backend_version: String,
    pub etag: String,
    /// base64 sha256 as the backend reports it.
    pub checksum_sha256: String,
    pub verifier_build: String,
    pub verified_at: u64,
}

/// The base64 sha256 an object's own digest REQUIRES. One arithmetic, two callers: the
/// grant that signs it and the fact that is only a fact if the backend reported it.
///
/// REFUSES, never panics, on anything but 64 lowercase hex: `ObjectRef` has pub fields, so
/// a hand-built ref with short hex reaches this — and slicing it aborted the process where
/// `MALFORMED_DIGEST` is the answer (the same class `Store::shard` already fixed).
fn required_checksum(sha256_hex: &str) -> Result<String> {
    crate::ids::hex64("ObjectRef.sha256", sha256_hex)?;
    let mut raw = [0u8; 32];
    for (i, b) in raw.iter_mut().enumerate() {
        *b = u8::from_str_radix(&sha256_hex[i * 2..i * 2 + 2], 16).unwrap_or(0);
    }
    Ok(b64::encode(&raw))
}

/// The one adjudication, wherever a `VerifiedObject` comes into existence: the checksum the
/// backend reported against the checksum the object's own id requires.
fn adjudicate_checksum(where_: &str, object: &ObjectRef, reported: &str) -> Result<()> {
    let want = required_checksum(&object.sha256)?;
    if reported != want {
        return refuse(
            Code::OBJECT_CORRUPT,
            format!(
                "{}: {where_} reports checksum {reported}, the digest requires {want} — \
                 QUARANTINE and repair. There is no branch here that accepts a final-key \
                 mismatch and continues.",
                object.id()
            ),
        );
    }
    Ok(())
}

impl VerifiedObject {
    /// The fact is minted from what the backend REPORTED, checked against what we asked
    /// for. A mismatch is a quarantine/repair event, never overwrite-and-continue.
    pub fn of(
        object: &ObjectRef,
        backend_version: &str,
        etag: &str,
        checksum_sha256: &str,
        verified_at: u64,
    ) -> Result<Self> {
        adjudicate_checksum("the backend", object, checksum_sha256)?;
        Ok(VerifiedObject {
            object: object.clone(),
            backend_version: backend_version.to_string(),
            etag: etag.to_string(),
            checksum_sha256: checksum_sha256.to_string(),
            verifier_build: format!("tensorfs-core/{}", env!("CARGO_PKG_VERSION")),
            verified_at,
        })
    }
}

impl Doc for VerifiedObject {
    const FORMAT: &'static str = "tensorfs.verified_object/1";
    const MAX_BYTES: usize = limits::SPEC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("VerifiedObject", v)?;
        check_format(
            "VerifiedObject",
            as_str("VerifiedObject", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;
        let backend_version = f.req_str("backend_version")?.to_string();
        let checksum_sha256 = f.req_str("checksum_sha256")?.to_string();
        let etag = f.req_str("etag")?.to_string();
        let object = ObjectRef::from_value("VerifiedObject.object", f.req("object")?)?;
        let verified_at = f.req_uint("verified_at")?;
        let verifier_build = f.req_str("verifier_build")?.to_string();
        f.done()?;
        for (what, s) in [
            ("backend_version", &backend_version),
            ("etag", &etag),
            ("verifier_build", &verifier_build),
        ] {
            if s.is_empty() || s.len() > limits::MAX_NAME_BYTES {
                return refuse(
                    Code::KEY_GRAMMAR,
                    format!("VerifiedObject.{what} out of bounds"),
                );
            }
        }
        // DESERIALIZATION DOES NOT MINT VERIFICATION (#553b). v1 read `checksum_sha256`
        // straight off the document, so a hand-written JSON file dropped into a facts
        // directory became a fact — and `CatalogRoot::publish` authorizes a root over
        // exactly those. The adjudication `of()` performs at mint time is arithmetic over
        // the object's own id, so it costs nothing to perform it again here, and a fact
        // that cannot survive it was never one.
        adjudicate_checksum("this document", &object, &checksum_sha256)?;
        Ok(VerifiedObject {
            object,
            backend_version,
            etag,
            checksum_sha256,
            verifier_build,
            verified_at,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("backend_version", Value::str(self.backend_version.clone())),
            ("checksum_sha256", Value::str(self.checksum_sha256.clone())),
            ("etag", Value::str(self.etag.clone())),
            ("format", Value::str(Self::FORMAT)),
            ("object", self.object.to_value()),
            ("verified_at", Value::uint(self.verified_at)),
            ("verifier_build", Value::str(self.verifier_build.clone())),
        ])
    }
}

// ---------------------------------------------------------------- conflict resolution

/// What a `412 PreconditionFailed` means once the resident object has been asked about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Conflict {
    /// The resident bytes ARE ours. An honest duplicate: nothing to upload, the fact stands.
    HonestDuplicate,
    /// The resident bytes are NOT what this key promises. Quarantine and repair; there is
    /// no path from here that overwrites.
    Quarantine { resident: String, wanted: String },
}

/// The verify-resident-on-conflict step, which is why losing a race costs one HEAD rather
/// than a re-upload. The gate proved the backend answers it: a HEAD with checksum mode
/// enabled returns the resident object's sha256.
pub fn resolve_conflict(grant: &PublishGrant, resident_checksum: &str) -> Result<Conflict> {
    let wanted = grant.checksum_sha256()?;
    Ok(if resident_checksum == wanted {
        Conflict::HonestDuplicate
    } else {
        Conflict::Quarantine {
            resident: resident_checksum.to_string(),
            wanted,
        }
    })
}

// ---------------------------------------------------------------- the catalog root

/// The atomic publication. It names the snapshot and the exact backend versions that were
/// verified — nothing else may be referenced, which is what makes the root's existence a
/// statement about bytes rather than about intentions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogRoot {
    pub snapshot: ObjectRef,
    /// digest -> the backend version verified for it, sorted unique by digest.
    pub verified: Vec<(String, String)>,
    pub published_at: u64,
}

impl CatalogRoot {
    /// A root may be published only over facts. Every object in the closure needs a
    /// VerifiedObject whose digest and backend version both match; anything else refuses.
    pub fn publish(
        closure: &PublishClosure,
        facts: &[VerifiedObject],
        published_at: u64,
    ) -> Result<CatalogRoot> {
        let mut verified = Vec::new();
        for o in &closure.objects {
            match facts.iter().find(|f| f.object.sha256 == o.sha256) {
                Some(f) if f.object.length == o.length => {
                    verified.push((o.sha256.clone(), f.backend_version.clone()))
                }
                Some(f) => {
                    return refuse(
                        Code::LENGTH_MISMATCH,
                        format!(
                            "{}: the verified fact covers {} B and the closure declares {} B",
                            o.id(),
                            f.object.length,
                            o.length
                        ),
                    )
                }
                None => {
                    return refuse(
                        Code::DURABILITY_UNPROVEN,
                        format!(
                            "{}: no VerifiedObject — a catalog root references only verified \
                             backend versions, and a PUT's 200 is not one",
                            o.id()
                        ),
                    )
                }
            }
        }
        verified.sort();
        Ok(CatalogRoot {
            snapshot: closure.snapshot.clone(),
            verified,
            published_at,
        })
    }
}

impl Doc for CatalogRoot {
    const FORMAT: &'static str = "tensorfs.catalog_root/1";
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("CatalogRoot", v)?;
        check_format(
            "CatalogRoot",
            as_str("CatalogRoot", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;
        let published_at = f.req_uint("published_at")?;
        let snapshot = ObjectRef::from_value("CatalogRoot.snapshot", f.req("snapshot")?)?;
        let mut verified = Vec::new();
        for e in as_arr("CatalogRoot", "verified", f.req("verified")?)? {
            let mut ef = Fields::new("CatalogRoot.verified", e)?;
            let backend_version = ef.req_str("backend_version")?.to_string();
            let sha256 = hex64("CatalogRoot.verified", ef.req_str("sha256")?)?;
            ef.done()?;
            verified.push((sha256, backend_version));
        }
        f.done()?;
        if verified.is_empty() {
            return refuse(Code::MISSING_FIELD, "a catalog root verifies nothing");
        }
        let mut sorted = verified.clone();
        sorted.sort();
        sorted.dedup_by(|a, b| a.0 == b.0);
        if sorted != verified {
            return refuse(
                Code::SORT_ORDER,
                "CatalogRoot.verified must be sorted and unique by digest",
            );
        }
        Ok(CatalogRoot {
            snapshot,
            verified,
            published_at,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("format", Value::str(Self::FORMAT)),
            ("published_at", Value::uint(self.published_at)),
            ("snapshot", self.snapshot.to_value()),
            (
                "verified",
                Value::arr(
                    self.verified
                        .iter()
                        .map(|(d, v)| {
                            Value::obj(vec![
                                ("backend_version", Value::str(v.clone())),
                                ("sha256", Value::str(d.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

/// The one place a prefixed digest becomes an ObjectRef for this plane.
pub fn object_ref(id: &str, length: u64) -> Result<ObjectRef> {
    let p = prefixed("object", id)?;
    Ok(ObjectRef {
        sha256: p.trim_start_matches("sha256:").to_string(),
        length,
    })
}
