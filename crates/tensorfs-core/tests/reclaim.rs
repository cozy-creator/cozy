//! Reclaim accounting proofs against a REAL store built through the production API — real
//! blobs through `put_stream`, real manifests through `put_manifest`, real repository
//! documents through `apply_repository`. No fixture is hand-written into the tree.
//!
//! Every expected byte figure is derived from the fixture's OWN declared object lengths and
//! then cross-checked against a literal, because th-152 shipped an arm that computed its
//! expected value from the constants it was checking — it asserted the code equalled itself
//! and passed review twice (proto-038, decisions.md 698).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use tensorfs_core::canon::Value;
use tensorfs_core::err::Code;
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::manifest::{Draft, Entry};
use tensorfs_core::reclaim;
use tensorfs_core::repository::{Mutation, RepositoryName};
use tensorfs_core::storage::{Census, HeldKey};
use tensorfs_core::store::{Fault, Store};

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-reclaim-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

/// A blob whose bytes are decided by its label, so the same label is the same object in
/// every repository that names it — which is the whole subject of this file.
fn blob(store: &Store, label: &str, length: usize) -> ObjectRef {
    let mut body = vec![0u8; length];
    for (at, byte) in body.iter_mut().enumerate() {
        *byte = (label.as_bytes()[at % label.len()] as usize).wrapping_add(at) as u8;
    }
    let want = ObjectRef::of(&body);
    store
        .put_stream(&mut body.as_slice(), Some(&want), &Fault::default())
        .unwrap();
    want
}

/// One retention root over the named objects, through the ordinary repository mutation.
fn model(store: &Store, org: &str, name: &str, objects: &[(&str, ObjectRef)]) {
    let mut entries: Vec<(String, Entry)> = objects
        .iter()
        .map(|(path, object)| ((*path).to_string(), Entry::File(object.clone())))
        .collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let manifest = Draft { entries }.seal().unwrap();
    let put = store.put_manifest(&manifest).unwrap();
    let repo = RepositoryName::new(org, name).unwrap();
    store
        .apply_repository(
            None,
            &Mutation::PutCheckpoint {
                repo,
                manifest: put.obj,
            },
            &Fault::default(),
        )
        .unwrap();
}

fn reclaimable(root: &Path, drop: &[(&str, &str)]) -> u64 {
    let drop: Vec<RepositoryName> = drop
        .iter()
        .map(|(org, name)| RepositoryName::new(*org, *name).unwrap())
        .collect();
    Census::open(root)
        .unwrap()
        .reclaim_plan(&drop, &[])
        .unwrap()
        .iter()
        .map(|row| row.length)
        .sum()
}

const A: usize = 1_000;
const B: usize = 2_000;
const C: usize = 4_000;

#[test]
fn the_reclaim_of_a_set_is_more_than_the_sum_of_its_parts() {
    // alpha = {A, B}, beta = {B, C}. B is reachable from both and from nothing else, so it
    // is unique to NEITHER — and dropping both frees it. This is the error that understates
    // without bound, and it is the one a `sum(bytes_unique)` implementation makes.
    let root = temporary("superadditive");
    let store = Store::init(&root).unwrap();
    let a = blob(&store, "aaaa", A);
    let b = blob(&store, "bbbb", B);
    let c = blob(&store, "cccc", C);
    model(
        &store,
        "acme",
        "alpha",
        &[("a.bin", a), ("b.bin", b.clone())],
    );
    model(&store, "acme", "beta", &[("b.bin", b), ("c.bin", c)]);

    let solo_alpha = reclaimable(&root, &[("acme", "alpha")]);
    let solo_beta = reclaimable(&root, &[("acme", "beta")]);
    let both = reclaimable(&root, &[("acme", "alpha"), ("acme", "beta")]);

    // Manifests are collected too, so each figure carries its own root's manifest bytes.
    // The SHAPE is what is pinned: B belongs to the pair and to neither singleton.
    assert_eq!(
        both - solo_alpha - solo_beta,
        B as u64,
        "the shared object must be reclaimable by the pair and by neither alone \
         (solo_alpha={solo_alpha} solo_beta={solo_beta} both={both})"
    );
    assert!(
        both > solo_alpha + solo_beta,
        "summing per-root figures understated the set"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn the_chooser_takes_the_root_that_frees_the_most_not_the_one_that_is_biggest() {
    // `hoard` has by far the largest closure, and almost all of it is shared with the root
    // being KEPT, so dropping it frees only its own unique object. `lean` is much smaller
    // and entirely unique. A chooser ranking by closure size picks `hoard` and frees
    // almost nothing — which is exactly the owner's stated worry.
    let root = temporary("marginal-not-size");
    let store = Store::init(&root).unwrap();
    let shared_a = blob(&store, "shared-a", 40_000);
    let shared_b = blob(&store, "shared-b", 40_000);
    let hoard_only = blob(&store, "hoard-only", 1_000);
    let lean_only = blob(&store, "lean-only", 9_000);

    model(
        &store,
        "acme",
        "keeper",
        &[("a.bin", shared_a.clone()), ("b.bin", shared_b.clone())],
    );
    model(
        &store,
        "acme",
        "hoard",
        &[
            ("a.bin", shared_a),
            ("b.bin", shared_b),
            ("own.bin", hoard_only),
        ],
    );
    model(&store, "acme", "lean", &[("own.bin", lean_only)]);

    let keep = [RepositoryName::new("acme", "keeper").unwrap()];
    let plan = reclaim::plan(&store, 5_000, &keep, &[]).unwrap();
    assert_eq!(
        plan.victims.first().map(|victim| victim.name.as_str()),
        Some("lean"),
        "the chooser picked by closure size, not by what dropping actually frees: {:?}",
        plan.victims
    );
    let lean = &plan.victims[0];
    // The gap between these two numbers on `hoard` is what makes the arm non-vacuous.
    assert!(
        lean.marginal_bytes >= 9_000,
        "lean freed {} B",
        lean.marginal_bytes
    );
    let hoard = reclaimable(&root, &[("acme", "hoard")]);
    assert!(
        hoard < 2_000,
        "the fixture does not actually pit size against benefit: hoard frees {hoard} B"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_root_that_frees_nothing_is_never_proposed_at_any_need() {
    // `ghost` reaches only objects the kept root also reaches. Dropping it costs a model
    // and frees zero bytes, so it must be unreachable from any plan rather than merely
    // ranked last.
    let root = temporary("zero-benefit");
    let store = Store::init(&root).unwrap();
    let shared = blob(&store, "shared", 30_000);
    let own = blob(&store, "own", 30_000);
    model(&store, "acme", "keeper", &[("s.bin", shared.clone())]);
    model(&store, "acme", "ghost", &[("s.bin", shared)]);
    model(&store, "acme", "real", &[("o.bin", own)]);

    let keep = [RepositoryName::new("acme", "keeper").unwrap()];
    let plan = reclaim::plan(&store, 30_000, &keep, &[]).unwrap();
    assert!(
        plan.victims.iter().all(|victim| victim.name != "ghost"),
        "a zero-benefit root was proposed: {:?}",
        plan.victims
    );

    // And it stays unreachable at the largest need the store can satisfy, which is the
    // half a "ranked last" implementation would fail.
    let everything = reclaim::plan(&store, 30_000, &keep, &[])
        .unwrap()
        .reclaimable_bytes;
    let plan = reclaim::plan(&store, everything, &keep, &[]).unwrap();
    assert!(plan.victims.iter().all(|victim| victim.name != "ghost"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn asking_for_more_than_exists_refuses_instead_of_emptying_the_store() {
    let root = temporary("insufficient");
    let store = Store::init(&root).unwrap();
    let one = blob(&store, "one", 10_000);
    let two = blob(&store, "two", 10_000);
    model(&store, "acme", "one", &[("o.bin", one)]);
    model(&store, "acme", "two", &[("t.bin", two)]);

    let all = reclaim::plan(&store, 20_000, &[], &[]).unwrap();
    let ceiling = all.reclaimable_bytes;
    assert_eq!(all.victims.len(), 2);

    // One byte below the ceiling plans; one byte above refuses. A refusal that fires for
    // everything and one that fires for nothing both pass a one-sided test.
    reclaim::plan(&store, ceiling, &[], &[]).expect("the exact ceiling must plan");
    let refusal = reclaim::plan(&store, ceiling + 1, &[], &[]).unwrap_err();
    assert_eq!(refusal.code, Code::RECLAIM_INSUFFICIENT, "{refusal}");
    assert!(
        refusal.detail.contains("1 B short"),
        "the refusal must name the shortfall: {refusal}"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_held_object_is_never_promised_and_releasing_the_hold_promises_it() {
    // The hold document is the exact one `gc::session_holds` writes — same canonical line,
    // same parser, same seam the CLI feeds `reclaim` from.
    let root = temporary("holds");
    let store = Store::init(&root).unwrap();
    let held = blob(&store, "held", 25_000);
    let free = blob(&store, "free", 25_000);
    model(
        &store,
        "acme",
        "one",
        &[("h.bin", held.clone()), ("f.bin", free)],
    );

    let line = tensorfs_core::canon::write(&Value::obj(vec![
        (
            "key",
            Value::str(tensorfs_core::storage::blob_key(&held.sha256).unwrap()),
        ),
        ("kind", Value::str("blob")),
        ("length", Value::uint(held.length)),
    ]));
    let hold = HeldKey::parse_line(&line).unwrap();

    let without = reclaim::plan(&store, 1, &[], &[])
        .unwrap()
        .reclaimable_bytes;
    let with = reclaim::plan(&store, 1, &[], std::slice::from_ref(&hold))
        .unwrap()
        .reclaimable_bytes;
    assert_eq!(
        without - with,
        held.length,
        "the hold did not protect its object"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn the_index_and_the_oracle_agree_on_a_store_with_three_way_sharing() {
    // The greedy search runs on the holder index; the reported number is re-derived through
    // `reclaim_plan`. This pins them together over every subset of a store where one object
    // is reached by three roots, one by two, and one by one — the shape where an index that
    // is nearly right is still wrong.
    let root = temporary("index-vs-oracle");
    let store = Store::init(&root).unwrap();
    let three = blob(&store, "three-way", A);
    let two = blob(&store, "two-way", B);
    let one = blob(&store, "one-way", C);
    model(
        &store,
        "acme",
        "x",
        &[("3.bin", three.clone()), ("2.bin", two.clone())],
    );
    model(
        &store,
        "acme",
        "y",
        &[("3.bin", three.clone()), ("2.bin", two), ("1.bin", one)],
    );
    model(&store, "acme", "z", &[("3.bin", three)]);

    let census = Census::open(&root).unwrap();
    let index = census.holder_index(&[]).unwrap();
    let names = ["x", "y", "z"];
    for mask in 0u8..8 {
        let mut positions = BTreeSet::new();
        let mut drop = Vec::new();
        for (bit, name) in names.iter().enumerate() {
            if mask & (1 << bit) != 0 {
                let repo = RepositoryName::new("acme", *name).unwrap();
                positions.insert(index.position(&repo).unwrap());
                drop.push(repo);
            }
        }
        let oracle: u64 = census
            .reclaim_plan(&drop, &[])
            .unwrap()
            .iter()
            .map(|row| row.length)
            .sum();
        assert_eq!(
            index.reclaimable(&positions),
            oracle,
            "index and census disagree for {drop:?}"
        );
    }
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn unreferenced_bytes_are_counted_before_any_model_is_proposed() {
    // A collection frees these without dropping anything. Proposing an eviction for bytes
    // `tfs gc` would have handed back for free is the same class of error as proposing one
    // that frees nothing.
    let root = temporary("unreferenced-first");
    let store = Store::init(&root).unwrap();
    let orphan = blob(&store, "orphan", 50_000);
    let owned = blob(&store, "owned", 10_000);
    model(&store, "acme", "one", &[("o.bin", owned)]);

    let plan = reclaim::plan(&store, 50_000, &[], &[]).unwrap();
    assert_eq!(plan.unreferenced_bytes, orphan.length);
    assert!(
        plan.victims.is_empty(),
        "a model was evicted for bytes a plain collection already frees: {:?}",
        plan.victims
    );
    let _ = std::fs::remove_dir_all(root);
}

/// Age a root's document so recency is deterministic rather than whatever the clock did.
fn age(store: &Store, org: &str, name: &str, seconds_ago: u64) {
    let repo = RepositoryName::new(org, name).unwrap();
    let when = std::time::SystemTime::UNIX_EPOCH
        + std::time::Duration::from_secs(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                - seconds_ago,
        );
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(store.repository_path(&repo))
        .unwrap();
    file.set_times(std::fs::FileTimes::new().set_modified(when))
        .unwrap();
}

#[test]
fn two_roots_that_free_nothing_apart_are_planned_together_and_the_useless_third_is_pruned() {
    // THE CASE A ZERO-MARGINAL SKIP GETS WRONG, found by planting.
    //
    // Every root here has a TWIN — the same entry set, so byte-identical manifest bytes,
    // so literally the same manifest object. That is what makes each root's marginal
    // genuinely zero: with distinct manifests every root frees at least its own manifest
    // and the pathological branch is never reached, which is exactly how the first draft of
    // this arm managed to be vacuous.
    //
    //   keeper / ghost -> {S}   ghost frees nothing, ever: the keeper holds all of it
    //   alpha  / beta  -> {T}   neither frees a byte ALONE; the PAIR frees all of T
    //
    // An earlier draft skipped every zero-marginal root, so the search had no first move
    // and reported RECLAIM_INSUFFICIENT over a store where dropping two models frees T.
    let root = temporary("pair-unlock");
    let store = Store::init(&root).unwrap();
    let s = blob(&store, "shared-with-keeper", 20_000);
    let t = blob(&store, "shared-by-the-pair", 30_000);
    model(&store, "acme", "keeper", &[("x.bin", s.clone())]);
    model(&store, "acme", "ghost", &[("x.bin", s)]);
    model(&store, "acme", "alpha", &[("y.bin", t.clone())]);
    model(&store, "acme", "beta", &[("y.bin", t.clone())]);
    // Coldest by a day, so the search's first move is provably the useless root and the
    // prune is provably the thing that removes it.
    age(&store, "acme", "ghost", 86_400);

    let keep = [RepositoryName::new("acme", "keeper").unwrap()];
    // Every candidate's marginal is zero at the start. If that is read as "nothing can be
    // freed" the plan refuses here.
    let index = Census::open(&root).unwrap().holder_index(&[]).unwrap();
    for name in ["ghost", "alpha", "beta"] {
        let repo = RepositoryName::new("acme", name).unwrap();
        let position = index.position(&repo).unwrap();
        assert_eq!(
            index.reclaimable(&BTreeSet::from([position])),
            0,
            "{name} frees bytes on its own; the arm never reaches the branch it is for"
        );
    }

    let plan = reclaim::plan(&store, t.length, &keep, &[]).unwrap();
    let named: Vec<&str> = plan.victims.iter().map(|v| v.name.as_str()).collect();
    assert!(
        named.contains(&"alpha") && named.contains(&"beta"),
        "the pair that frees T was not planned: {named:?}"
    );
    assert!(
        !named.contains(&"ghost"),
        "a root that frees nothing survived into the plan: {named:?}"
    );
    // T plus the pair's shared manifest, and nothing else: `ghost` contributed no bytes and
    // is not in the total, which is the pruning showing up in the arithmetic.
    assert!(plan.reclaimable_bytes >= t.length);
    assert!(plan.reclaimable_bytes < t.length + 1_000);
    assert_eq!(
        plan.victims.iter().map(|v| v.marginal_bytes).sum::<u64>() + plan.unreferenced_bytes,
        plan.reclaimable_bytes
    );
    // And the plan is MINIMAL: no victim can be dropped from it and still meet the need.
    let positions: BTreeSet<usize> = plan
        .victims
        .iter()
        .map(|victim| {
            index
                .position(&RepositoryName::new(victim.org.clone(), victim.name.clone()).unwrap())
                .unwrap()
        })
        .collect();
    for position in &positions {
        let mut without = positions.clone();
        without.remove(position);
        assert!(
            index.reclaimable(&without) < t.length,
            "victim {position} was not needed and should have been pruned"
        );
    }
    let _ = std::fs::remove_dir_all(root);
}
