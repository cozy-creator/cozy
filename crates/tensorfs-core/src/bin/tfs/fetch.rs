//! `tfs fetch` (tfs-003 row 220) — the pull plane's document half, driven from the CLI.
//!
//! Core links no HTTP and holds no credential, by fence, so this emits what a transport
//! adapter must move and CONSUMES the bytes it moved. The adapter — cozy-runtime's, per
//! boundaries.md §9 — holds the URL and the token; `fetch admit` reads the body on stdin.
//! That is the whole seam: `curl "$url" | tfs fetch admit <store> <sha256> <length>` and a
//! worker's Python transport calling `Store.put_reader` are the SAME door, because both end
//! inside `Store::put_stream`.

use std::fs;
use std::io;
use std::path::Path;
use std::process::ExitCode;

use tensorfs_core::fetch::{DeliveryGrant, FetchPlan};
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::store::Store;

use crate::{bail, flag, Flags};

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

/// PLAN FIRST, against a real store. The declaration is the remote catalog's claim about
/// what the checkpoint reaches; the split into held/wanted is this machine's own answer, and
/// it is where idempotency lives — an object already installed and vouched for is HELD and
/// no grant can be minted for it.
///
/// `declared` is a canonical `tensorfs.publish_closure/1` (what the hub's manifest reach and
/// the publisher's own closure both spell), so the pull side reads the publish side's
/// document rather than a second description of the same set.
pub fn cmd_plan(root: &Path, declared_p: &Path, session: &str, flags: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let closure = ok!(tensorfs_core::cloud::PublishClosure::parse(&read!(
        declared_p
    )));
    let (plan, why) = ok!(FetchPlan::of(
        &store,
        session,
        &closure.snapshot,
        &closure.objects
    ));
    emit(&plan, flags);
    println!("plan         {}", plan.object_id());
    println!("session      {}", plan.session);
    println!("snapshot     {}", plan.snapshot.id());
    println!(
        "held         {} object(s), {} B — NOT fetched",
        plan.held.len(),
        plan.held_bytes()
    );
    println!(
        "wanted       {} object(s), {} B",
        plan.wanted.len(),
        plan.wanted_bytes()
    );
    if crate::flag_on(flags, "why") {
        for (o, p) in &why {
            println!("             {} {}", p.as_str(), o.id());
        }
    }
    ExitCode::SUCCESS
}

/// THE DOOR. Bytes arrive on stdin and leave as an installed object or as nothing.
///
/// There is no `--force`, no `--skip-verify` and no path that writes at an id it did not
/// compute: `put_stream` hashes what it reads and refuses at the first disagreement with the
/// grant, having already removed the partial temp file.
pub fn cmd_admit(root: &Path, plan_p: &Path, id: &str, flags: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let plan = ok!(FetchPlan::parse(&read!(plan_p)));
    let hex = id.trim_start_matches("sha256:");
    let length = match plan.wanted.iter().find(|o| o.sha256 == hex) {
        Some(o) => o.length,
        // Not in the wanted set: mint anyway so `mint` produces the exact typed refusal
        // (already-held vs never-declared) rather than this function inventing one.
        None => crate::flag_num(flags, "length", 0u64),
    };
    let object = ObjectRef {
        sha256: hex.to_string(),
        length,
    };
    let grant = ok!(DeliveryGrant::mint(&plan, &object));
    let mut stdin = io::stdin().lock();
    let admitted = ok!(grant.admit(&store, &mut stdin));
    println!("admitted     {}", admitted.object.id());
    println!("length       {} B", admitted.object.length);
    println!("key          {}", grant.key());
    println!(
        "writer       {}",
        if admitted.first_writer {
            "this process won the no-clobber admission"
        } else {
            "already installed by a concurrent puller; bytes verified either way"
        }
    );
    ExitCode::SUCCESS
}

/// The PROOF. Every object the plan declared is installed and this store's own record
/// vouches for it. A transfer's success report is not residency and does not appear here.
pub fn cmd_complete(root: &Path, plan_p: &Path) -> ExitCode {
    let store = ok!(Store::open(root));
    let plan = ok!(FetchPlan::parse(&read!(plan_p)));
    ok!(plan.complete(&store));
    println!("complete     {}", plan.snapshot.id());
    println!(
        "resident     {} object(s), {} B, every one verifiably local",
        plan.held.len() + plan.wanted.len(),
        plan.declared_bytes()
    );
    ExitCode::SUCCESS
}
