//! Roots, holds and GC on the command line (tfs-006). Every verb is a real transaction
//! against the real authority; nothing here is simulated.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use tensorfs_core::err::{Code, Refusal};
use tensorfs_core::gc;
use tensorfs_core::meta::Meta;
use tensorfs_core::store::Store;

use super::{bail, flag, flag_num, flag_on, Flags};

macro_rules! ok {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return bail(e),
        }
    };
}

fn open(root: &Path) -> Result<(Store, Meta), Refusal> {
    let s = Store::open(root)?;
    let m = Meta::open(&s)?;
    Ok((s, m))
}

// ---------------------------------------------------------------- roots

pub fn cmd_root(root: &Path, verb: &str, args: &[&str], f: &Flags) -> ExitCode {
    let (s, meta) = ok!(open(root));
    match verb {
        "register" => {
            let expected: Option<u64> = flag(f, "expect-generation").and_then(|v| v.parse().ok());
            let mut t = ok!(meta.txn());
            ok!(t.register_root(
                args[0],
                args[1],
                flag(f, "kind").unwrap_or("workspace"),
                expected,
                &[]
            ));
            if flag_on(f, "hub-published") {
                ok!(t.note_snapshot(args[1], true, flag(f, "source-url").unwrap_or("")));
            } else if let Some(u) = flag(f, "source-url") {
                ok!(t.note_snapshot(args[1], false, u));
            }
            let g = ok!(t.commit());
            println!("root {:?} -> sha256:{} at generation {g}", args[0], args[1]);
        }
        // Durability is a per-snapshot FACT, recorded without pinning anything: rooting a
        // snapshot in order to mark it published would protect the very bytes the record is
        // supposed to make evictable.
        "note" => {
            let mut t = ok!(meta.txn());
            ok!(t.note_snapshot(
                args[0],
                flag_on(f, "hub-published"),
                flag(f, "source-url").unwrap_or("")
            ));
            ok!(t.commit());
            println!(
                "snapshot sha256:{} hub_published={} source_url={:?}",
                args[0],
                flag_on(f, "hub-published"),
                flag(f, "source-url").unwrap_or("")
            );
        }
        "release" => {
            let mut t = ok!(meta.txn());
            let snap = ok!(t.release_root(args[0]));
            let g = ok!(t.commit());
            println!("root {:?} released (was sha256:{snap}) at generation {g}", args[0]);
            println!("bytes are NOT freed by this — a later complete mark decides that");
        }
        "open" => {
            let (snap, hold) = ok!(meta.open_root(args[0]));
            println!("root {:?} resolved ONCE to sha256:{snap}", args[0]);
            println!("hold {} now protects the reader, not the name", hold.id());
            let secs: u64 = flag_num(f, "hold-secs", 0);
            if secs > 0 {
                if let Some(p) = flag(f, "ready") {
                    let _ = std::fs::write(p, format!("{}\n", hold.id()));
                }
                println!("HOLDING pid={}", std::process::id());
                std::thread::sleep(std::time::Duration::from_secs(secs));
            }
            ok!(hold.release(&meta));
            println!("hold released");
        }
        "reconcile" => {
            // `name=snapshot` pairs are the coordinator's binding set.
            let bindings: Vec<(String, String)> = args
                .iter()
                .filter_map(|a| a.split_once('='))
                .map(|(n, s)| (n.to_string(), s.to_string()))
                .collect();
            let objects_before = ok!(s.objects()).len();
            let (added, removed) = ok!(gc::reconcile_bindings(&meta, &bindings, "binding"));
            println!(
                "bindings {} -> roots {} (added {added:?}, removed {removed:?})",
                bindings.len(),
                ok!(meta.read()).roots.len()
            );
            println!(
                "objects {} -> {} — binding churn moves ROOTS, never bytes",
                objects_before,
                ok!(s.objects()).len()
            );
        }
        _ => {
            let st = ok!(meta.read());
            println!("generation: {}", st.generation);
            for r in &st.roots {
                println!("  root  {:<24} {:<10} sha256:{}", r.name, r.kind, r.snapshot);
            }
            for h in &st.holds {
                println!(
                    "  hold  {:<24} {:<10} pid {} boot {} snapshot sha256:{}",
                    h.id,
                    h.kind,
                    h.owner_pid,
                    h.boot_epoch,
                    if h.snapshot.is_empty() { "-".into() } else { h.snapshot.clone() }
                );
            }
            for q in &st.quarantine {
                println!("  claim {} at generation {}", q.object, q.claim_generation);
            }
            for sn in &st.snapshots {
                println!(
                    "  snap  sha256:{} hub_published={} source_url={:?}",
                    sn.snapshot, sn.hub_published, sn.source_url
                );
            }
        }
    }
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------- holds

pub fn cmd_hold(root: &Path, verb: &str, args: &[&str], f: &Flags) -> ExitCode {
    let (_, meta) = ok!(open(root));
    match verb {
        "take" => {
            let hold = ok!(meta.acquire_hold(
                flag(f, "kind").unwrap_or("fill"),
                args.first().copied().unwrap_or(""),
                Vec::new()
            ));
            println!("hold {}", hold.id());
            if let Some(p) = flag(f, "ready") {
                let _ = std::fs::write(p, format!("{}\n", hold.id()));
            }
            let secs: u64 = flag_num(f, "hold-secs", 0);
            if secs > 0 {
                println!("HOLDING pid={} for {secs}s", std::process::id());
                std::thread::sleep(std::time::Duration::from_secs(secs));
            }
            ok!(hold.release(&meta));
            println!("released");
        }
        "reap" => {
            let t0 = Instant::now();
            let reaped = ok!(meta.reap_holds());
            println!(
                "reaped {} holds in {} ms — proof is TAKING the owner's lock, never elapsed time",
                reaped.len(),
                t0.elapsed().as_millis()
            );
            for r in &reaped {
                println!("  {r}");
            }
        }
        "steal" => {
            ok!(meta.steal_hold(args[0]));
            println!("hold {:?} removed WITHOUT the owner-liveness proof (dev-only red arm)", args[0]);
        }
        _ => return ExitCode::from(2),
    }
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------- gc

pub fn cmd_gc(root: &Path, verb: &str, f: &Flags) -> ExitCode {
    let (s, meta) = ok!(open(root));
    match verb {
        "mark" => {
            let st = ok!(meta.read());
            let t0 = Instant::now();
            let c = ok!(gc::mark(&s, &st));
            println!(
                "roots {} holds {} ingest-roots {} -> {} live objects, {} B in {} ms",
                c.roots,
                c.holds,
                c.ingest_roots,
                c.distinct(),
                c.bytes(),
                t0.elapsed().as_millis()
            );
            for kind in ["snapshot", "header", "config", "spec", "vectors", "part", "asset", "blob", "evidence", "hold-object", "ingest-candidate"] {
                let n = c.live.iter().filter(|l| l.kind == kind).count();
                if n > 0 {
                    println!("  {kind:<18} {n}");
                }
            }
            println!("store holds {} objects total", ok!(s.objects()).len());
        }
        "usage" => {
            let st = ok!(meta.read());
            let c = ok!(gc::mark(&s, &st));
            for u in gc::usage(&s, &c) {
                println!("snapshot sha256:{}", u.snapshot);
                println!("  logical   {:>14} B", u.logical);
                println!("  resident  {:>14} B", u.resident);
                println!(
                    "  exclusive {:>14} B  <- deleting this root must reclaim EXACTLY this",
                    u.exclusive
                );
            }
        }
        "sweep" => {
            let t0 = Instant::now();
            let (sw, c) = ok!(gc::sweep_at(
                &s,
                &meta,
                flag_num(f, "limit", 0),
                flag_on(f, "dry-run"),
                flag_on(f, "allow-empty"),
                flag(f, "fault")
            ));
            println!(
                "scanned {} live {} collected {} ({} B) deferred {} absent {} in {} ms",
                sw.scanned,
                sw.live,
                sw.collected,
                sw.bytes,
                sw.deferred,
                sw.absent,
                t0.elapsed().as_millis()
            );
            println!("claim generation {}", sw.claim_generation);
            println!("census reached {} objects via {} roots / {} holds", c.distinct(), c.roots, c.holds);
            if flag_on(f, "dry-run") {
                println!("(dry run — nothing was removed)");
            }
        }
        "reap" => {
            let (n, b) = ok!(gc::reap_removed(&s));
            println!("finished {n} interrupted removals ({b} B) — already unreachable by name");
        }
        "refs" => {
            let st = ok!(meta.read());
            let d = ok!(gc::render_refs(&s, &st));
            println!("rendered {} refs to {} — CACHE, zero authority", st.roots.len(), d.display());
        }
        "pressure" => {
            let st = ok!(meta.read());
            let c = ok!(gc::mark(&s, &st));
            let u = gc::usage(&s, &c);
            let free: u64 = flag_num(f, "free", 0);
            let floor: u64 = flag_num(f, "reserve", 0);
            let p = gc::pressure_plan(&st, &u, free, floor);
            println!("free {free} B, reserve floor {floor} B -> need {} B", p.need);
            for e in &p.plan {
                println!("  EVICT sha256:{} {} B (last used {})", e.snapshot, e.bytes, e.last_used);
            }
            for (snap, why) in &p.protected {
                println!("  KEEP  sha256:{} — {why}", snap);
            }
            println!("evictable total {} B", p.freed);
            match gc::admission_under_pressure(&p) {
                Ok(()) => println!("new admission: allowed"),
                Err(e) => println!("new admission: REFUSED {}", e),
            }
        }
        _ => return ExitCode::from(2),
    }
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------- red arms

pub fn cmd_arms(root: &Path) -> ExitCode {
    let (s, meta) = ok!(open(root));
    let (mut fired, mut missed) = (0usize, 0usize);
    let mut arm = |name: &str, want: Code, r: Result<(), Refusal>| match r {
        Err(e) if e.code == want => {
            fired += 1;
            println!("  FIRED  {name}: {}", e.code.as_str());
        }
        Err(e) => {
            missed += 1;
            println!("  WRONG  {name}: wanted {}, got {e}", want.as_str());
        }
        Ok(()) => {
            missed += 1;
            println!("  MISSED {name}: the call SUCCEEDED");
        }
    };

    println!("roots/GC red arms, planted live:");

    // 1. a root swap against a stale expected generation
    let st = ok!(meta.read());
    let stale = st.generation.wrapping_sub(1);
    arm(
        "root swap at a stale expected generation",
        Code::ROOT_GENERATION,
        (|| {
            let mut t = meta.txn()?;
            t.register_root("stale-swap", &"e".repeat(64), "workspace", Some(stale), &[])?;
            t.commit()?;
            Ok(())
        })(),
    );

    // 2. releasing a root that was never registered
    arm(
        "release a root the authority never knew",
        Code::ROOT_ABSENT,
        (|| {
            let mut t = meta.txn()?;
            t.release_root("never-existed")?;
            t.commit()?;
            Ok(())
        })(),
    );

    // 3. a hold pinning a NAME rather than an immutable id
    arm(
        "hold pinning a mutable name",
        Code::MALFORMED_DIGEST,
        meta.acquire_hold("read", "latest", Vec::new()).map(|_| ()),
    );

    // 4. an address prefix outside the sweep set: unreachable AND uncollectable
    let alien = s.root().join("objects/blake3/aa/bb");
    let _ = std::fs::create_dir_all(&alien);
    let _ = std::fs::write(alien.join("f".repeat(64)), b"an address this sweep cannot reach");
    arm(
        "an unswept address prefix",
        Code::SWEEP_SET_INCOMPLETE,
        gc::sweep(&s, &meta, 0, true, false).map(|_| ()),
    );
    let _ = std::fs::remove_dir_all(s.root().join("objects/blake3"));

    // 5. a rooted snapshot the store cannot walk — the census is incomplete, so nothing goes
    let mut t = ok!(meta.txn());
    ok!(t.register_root("dangling", &"a".repeat(64), "workspace", None, &[]));
    ok!(t.commit());
    arm(
        "sweeping with a rooted snapshot that cannot be read",
        Code::CENSUS_INCOMPLETE,
        gc::sweep(&s, &meta, 0, false, false).map(|_| ()),
    );
    let mut t = ok!(meta.txn());
    ok!(t.release_root("dangling"));
    ok!(t.commit());

    // 6. capacity: only protected bytes left under pressure
    let st = ok!(meta.read());
    let c = ok!(gc::mark(&s, &st));
    let u = gc::usage(&s, &c);
    let p = gc::pressure_plan(&st, &u, 0, 1 << 40);
    arm(
        "new admission when only protected bytes remain",
        Code::CAPACITY_EXHAUSTED,
        gc::admission_under_pressure(&p),
    );

    println!("\n{fired} fired, {missed} missed");
    if missed > 0 {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
