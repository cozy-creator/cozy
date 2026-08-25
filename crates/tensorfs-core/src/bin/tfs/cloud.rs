//! `tfs cloud` (tfs-017) — the publish protocol's document plane, driven from the CLI.
//!
//! Core links no HTTP and holds no credential, by fence, so this emits what a coordinator
//! must sign and consumes what the backend reported. `scripts/r2-qualify.py` is the half
//! that speaks to R2, and it drives these verbs.

use std::fs;
use std::path::Path;
use std::process::ExitCode;

use tensorfs_core::cloud::{
    self, CatalogRoot, Conflict, PublishClosure, PublishGrant, VerifiedObject,
};
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::store::Store;

use crate::{bail, flag, flag_num, Flags};

macro_rules! ok {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return bail(e),
        }
    };
}

fn rd(p: &Path) -> Result<Vec<u8>, ExitCode> {
    fs::read(p).map_err(|e| {
        eprintln!("REFUSED IO_FAILED: read {}: {e}", p.display());
        ExitCode::FAILURE
    })
}

macro_rules! read {
    ($p:expr) => {
        match rd($p) {
            Ok(b) => b,
            Err(c) => return c,
        }
    };
}

fn emit<D: Doc>(d: &D, flags: &Flags) {
    if let Some(p) = flag(flags, "out") {
        if let Err(e) = fs::write(p, d.canonical_bytes()) {
            eprintln!("REFUSED IO_FAILED: write {p}: {e}");
        }
    }
}

/// DECLARE FIRST. The closure is every object the snapshot reaches, so the grant set is a
/// subset of a declaration made before a byte moved.
pub fn cmd_closure(root: &Path, hex: &str, session: &str, flags: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let length = match fs::metadata(store.object_path(hex)) {
        Ok(m) => m.len(),
        Err(e) => {
            eprintln!("REFUSED OBJECT_ABSENT: {hex}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let snapshot = ObjectRef {
        sha256: hex.to_string(),
        length,
    };
    let manifest = ok!(tensorfs_core::checkpoint::load_manifest(&store, &snapshot));
    let walk = ok!(tensorfs_core::checkpoint::walk(&store, &manifest));
    let mut objects: Vec<ObjectRef> = walk.distinct().into_iter().cloned().collect();
    objects.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    objects.dedup_by(|a, b| a.sha256 == b.sha256);
    let closure = PublishClosure {
        session: session.to_string(),
        snapshot,
        objects,
    };
    emit(&closure, flags);
    println!("closure      {}", closure.object_id());
    println!("session      {}", closure.session);
    println!("objects      {}", closure.objects.len());
    println!("bytes        {}", closure.bytes());
    ExitCode::SUCCESS
}

/// MINT A GRANT. There is no flag that produces an unconditional or unchecksummed one:
/// both conditions are derived from the digest, so the unsafe grant is unrepresentable.
pub fn cmd_grant(closure_p: &Path, id: &str, flags: &Flags) -> ExitCode {
    let closure = ok!(PublishClosure::parse(&read!(closure_p)));
    let hex = id.trim_start_matches("sha256:");
    let object = match closure.objects.iter().find(|o| o.sha256 == hex) {
        Some(o) => o.clone(),
        None => ok!(cloud::object_ref(id, flag_num(flags, "length", 0))),
    };
    let grant = ok!(PublishGrant::mint(
        &closure,
        &object,
        flag_num(flags, "expires-at", now() + 900)
    ));
    emit(&grant, flags);
    println!("grant        {}", grant.object_id());
    println!("key          {}", grant.key());
    for (k, v) in grant.signed_headers() {
        println!("sign         {k}: {v}");
    }
    println!("expires_at   {}", grant.expires_at);
    ExitCode::SUCCESS
}

/// Turn what the backend REPORTED into a fact, or refuse. A 200 and an ETag are not proof.
pub fn cmd_verified(
    grant_p: &Path,
    version: &str,
    etag: &str,
    checksum: &str,
    flags: &Flags,
) -> ExitCode {
    let grant = ok!(PublishGrant::parse(&read!(grant_p)));
    let v = ok!(VerifiedObject::of(
        &grant.object,
        version,
        etag,
        checksum,
        now()
    ));
    emit(&v, flags);
    println!("verified     {}", v.object.id());
    println!("backend      {} etag {}", v.backend_version, v.etag);
    println!("fact         {}", v.object_id());
    ExitCode::SUCCESS
}

/// The 412 answer: an honest duplicate, or a quarantine. Never an overwrite.
pub fn cmd_conflict(grant_p: &Path, resident: &str) -> ExitCode {
    let grant = ok!(PublishGrant::parse(&read!(grant_p)));
    match cloud::resolve_conflict(&grant, resident) {
        Conflict::HonestDuplicate => {
            println!("conflict     HONEST DUPLICATE — the resident bytes are ours");
            println!("             zero bytes re-uploaded; the fact stands");
            ExitCode::SUCCESS
        }
        Conflict::Quarantine { resident, wanted } => {
            eprintln!(
                "REFUSED OBJECT_CORRUPT: final-key mismatch at {}: resident {resident}, this \
                 key promises {wanted} — quarantine and repair, never overwrite-and-continue",
                grant.key()
            );
            ExitCode::FAILURE
        }
    }
}

/// The atomic publication, over facts only.
pub fn cmd_publish(closure_p: &Path, facts_dir: &Path, flags: &Flags) -> ExitCode {
    let closure = ok!(PublishClosure::parse(&read!(closure_p)));
    let mut facts = Vec::new();
    if let Ok(rd) = fs::read_dir(facts_dir) {
        for e in rd.flatten() {
            if e.path().extension().map(|x| x == "json").unwrap_or(false) {
                if let Ok(b) = fs::read(e.path()) {
                    if let Ok(v) = VerifiedObject::parse(&b) {
                        facts.push(v);
                    }
                }
            }
        }
    }
    let root = ok!(CatalogRoot::publish(&closure, &facts, now()));
    emit(&root, flags);
    println!("published    {}", root.object_id());
    println!("snapshot     {}", root.snapshot.id());
    println!(
        "verified     {} object(s), every one a fact",
        root.verified.len()
    );
    ExitCode::SUCCESS
}

fn now() -> u64 {
    tensorfs_core::ingest::transaction::now_secs()
}
