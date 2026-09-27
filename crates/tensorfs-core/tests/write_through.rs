//! THE DURABILITY PROOF, end to end, against the real `tfs` binary.
//!
//! The failure this exists to stop is not hypothetical. A MiniMax H3 ingest moved 210 GB
//! onto a rented pod over 2h56m, reached 44 of 48 source files, and lost EVERY BYTE when the
//! pod was released — because the TensorFS Store is pod-local by design and nothing else had
//! a copy. So the test destroys the Store the same way the pod was destroyed, with the same
//! totality: `remove_dir_all`, no salvage, no second copy on the box. Whatever a fresh Store
//! can be rebuilt from afterwards is exactly what would have survived.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-write-through-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn tfs(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(args)
        .output()
        .unwrap()
}

fn ok(what: &str, output: &Output) -> String {
    assert!(
        output.status.success(),
        "{what} failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).to_string()
}

/// One safetensors carrier whose tensors are big enough to become real CAS objects. The
/// inline threshold is 256 B, so a 4 KiB tensor is the smallest thing that proves anything
/// about object publication at all — a carrier of scalars would write nothing to a cache and
/// the whole test would pass vacuously.
const TENSOR_BYTES: usize = 4096;

fn carrier(path: &Path, keys: &[&str], salt: u8) {
    let mut fields = Vec::new();
    for (index, key) in keys.iter().enumerate() {
        let begin = index * TENSOR_BYTES;
        fields.push(format!(
            "{key:?}:{{\"data_offsets\":[{begin},{}],\"dtype\":\"F32\",\"shape\":[{}]}}",
            begin + TENSOR_BYTES,
            TENSOR_BYTES / 4
        ));
    }
    let header = format!("{{{}}}", fields.join(","));
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    // A distinct byte per (tensor, offset) so two tensors never share a digest and a
    // reassembly that returned the wrong one could not pass.
    for index in 0..keys.len() {
        for offset in 0..TENSOR_BYTES {
            bytes.push((index as u8).wrapping_mul(31).wrapping_add(offset as u8) ^ salt);
        }
    }
    fs::write(path, bytes).unwrap();
}

fn order(path: &Path, rows: &[(&str, &str)]) {
    let body = format!(
        "[{}]",
        rows.iter()
            .map(|(component, key)| format!("[{component:?},{key:?}]"))
            .collect::<Vec<_>>()
            .join(",")
    );
    fs::write(path, body).unwrap();
}

fn bank(registry: &Path, sources: &[(&str, &Path)], order_path: &Path) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tfs"));
    command.args([
        "ingest",
        "bank",
        "safetensors.diffusers",
        "diffusers.identity/1",
        "--registry",
    ]);
    command.arg(registry);
    command.args(["--source-profile", "write-through", "--order"]);
    command.arg(order_path);
    for (component, path) in sources {
        command.args(["--source", &format!("{component}={}", path.display())]);
        command.args(["--target", &format!("{component}=plain/1")]);
    }
    ok("bank", &command.output().unwrap());
}

fn source_plan(registry: &Path, carriers: &[&Path], out: &Path) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tfs"));
    command.args(["ingest", "source-plan", "--registry"]);
    command.arg(registry);
    command.args(["--out", out.to_str().unwrap()]);
    for carrier in carriers {
        command.args(["--carrier", carrier.to_str().unwrap()]);
    }
    ok("source-plan", &command.output().unwrap());
}

/// The value after a leading label on one of `tfs`'s aligned report lines.
fn field(text: &str, label: &str, index: usize) -> String {
    text.lines()
        .find(|line| line.starts_with(label))
        .unwrap_or_else(|| panic!("no {label:?} line in:\n{text}"))
        .split_whitespace()
        .nth(index)
        .unwrap_or_else(|| panic!("{label:?} line has no field {index}"))
        .to_string()
}

fn hex_of(id: &str) -> String {
    id.trim_start_matches("sha256:").to_string()
}

/// The cache's own documented layout — `<blobs|manifests>/<first 2>/<next 2>/<digest>` — is
/// a cross-language contract, so reading a published object's length straight off it is
/// legitimate here AND is a check that the layout is what the contract says. It is also the
/// only place these lengths can come from once the Store is gone, which is the point.
fn cached_length(cache: &Path, namespace: &str, hex: &str) -> u64 {
    let mut path = cache
        .join(namespace)
        .join(&hex[..2])
        .join(&hex[2..4])
        .join(hex);
    if namespace == "manifests" {
        path.set_extension("json");
    }
    fs::metadata(&path)
        .unwrap_or_else(|error| panic!("{} is not on the cache: {error}", path.display()))
        .len()
}

struct Converted {
    session: String,
    manifest: String,
    manifest_length: u64,
    chain: String,
    chain_length: u64,
    plan: String,
    stdout: String,
}

/// Convert two carriers into one checkpoint, writing through to `cache` as it runs.
fn convert(root: &Path, store: &Path, cache: &Path, interval: &str) -> Converted {
    fs::create_dir_all(root).unwrap();
    let first = root.join("encoder.safetensors");
    let second = root.join("decoder.safetensors");
    let registry = root.join("registry.json");
    let order_path = root.join("order.json");
    let plan_path = root.join("source-plan.json");
    carrier(&first, &["a", "b", "c", "d"], 0x11);
    carrier(&second, &["e", "f", "g", "h"], 0x77);
    order(
        &order_path,
        &[
            ("encoder", "a"),
            ("encoder", "b"),
            ("encoder", "c"),
            ("encoder", "d"),
            ("decoder", "e"),
            ("decoder", "f"),
            ("decoder", "g"),
            ("decoder", "h"),
        ],
    );
    bank(
        &registry,
        &[("encoder", &first), ("decoder", &second)],
        &order_path,
    );
    source_plan(&registry, &[&first, &second], &plan_path);
    // THE CACHE IS NAMED ONCE, AT SETUP, AND NEVER AGAIN. Every command below — the
    // conversion, the restore, the resume — finds it on the Store.
    ok(
        "store init --repo-cache",
        &tfs(&[
            "store",
            "init",
            store.to_str().unwrap(),
            "--repo-cache",
            cache.to_str().unwrap(),
        ]),
    );

    let stdout = ok(
        "ingest run",
        &tfs(&[
            "ingest",
            "run",
            store.to_str().unwrap(),
            "--source-plan",
            plan_path.to_str().unwrap(),
            "--checkpoint-bytes",
            interval,
        ]),
    );
    let manifest = hex_of(&field(&stdout, "candidate    manifest", 2));
    let chain = hex_of(&field(&stdout, "chain        sha256:", 1));
    Converted {
        manifest_length: cached_length(cache, "manifests", &manifest),
        chain_length: cached_length(cache, "blobs", &chain),
        manifest,
        chain,
        plan: field(&stdout, "plan         ", 1),
        session: field(&stdout, "session      ", 1),
        stdout,
    }
}

/// What one tensor role reassembles to, read back through the ordinary lease path. This is
/// the end-to-end claim: not "the object is present" but "the tensor reads back".
fn span(store: &Path, manifest: &str, component: &str, key: &str) -> String {
    let text = ok(
        "read span",
        &tfs(&[
            "read",
            "span",
            store.to_str().unwrap(),
            manifest,
            component,
            key,
            "value",
            "0",
            &TENSOR_BYTES.to_string(),
        ]),
    );
    field(&text, "sha256:", 1)
}

#[test]
fn converted_objects_survive_the_loss_of_the_store_and_are_re_admitted_verified() {
    let root = temporary("survives");
    let store = root.join("store");
    let cache = root.join("repo-cache");
    // 8 KiB against 32 KiB of tensor bytes: several checkpoints, so the run is proved
    // resumable at more than one point and not only at its end.
    let converted = convert(&root, &store, &cache, "8192");
    assert!(
        converted.stdout.contains("write-through"),
        "the run reported no write-through:\n{}",
        converted.stdout
    );
    let before: Vec<String> = ["a", "d"]
        .iter()
        .map(|key| span(&store, &converted.manifest, "encoder", key))
        .collect();

    // THE POD IS RELEASED. Not stopped, not unmounted — deleted, exactly as a reclaimed
    // rental deletes it. Everything that outlives this line is on the cache and nowhere else.
    fs::remove_dir_all(&store).unwrap();
    assert!(!store.exists());

    let fresh = root.join("fresh");
    ok(
        "store init --repo-cache",
        &tfs(&[
            "store",
            "init",
            fresh.to_str().unwrap(),
            "--repo-cache",
            cache.to_str().unwrap(),
        ]),
    );
    let restored = ok(
        "repo-cache restore",
        &tfs(&[
            "repo-cache",
            "restore",
            fresh.to_str().unwrap(),
            &converted.manifest,
            &converted.manifest_length.to_string(),
        ]),
    );
    assert!(
        restored.contains("complete     every declared object is resident and verified"),
        "the restore proved nothing:\n{restored}"
    );
    assert!(
        !restored.contains("restored     0 object(s)"),
        "the restore moved nothing, so the cache held nothing:\n{restored}"
    );

    let after: Vec<String> = ["a", "d"]
        .iter()
        .map(|key| span(&fresh, &converted.manifest, "encoder", key))
        .collect();
    assert_eq!(
        before, after,
        "the checkpoint did not reassemble to the same bytes on the new Store"
    );

    // A SECOND restore onto the now-warm Store moves nothing. The Manifest is the case that
    // catches this: a bare object set is a set of blobs, so asking for a Manifest through the
    // blob namespace would report it absent and re-admit it forever on a Store that holds it.
    let again = ok(
        "repo-cache restore, warm",
        &tfs(&[
            "repo-cache",
            "restore",
            fresh.to_str().unwrap(),
            &converted.manifest,
            &converted.manifest_length.to_string(),
        ]),
    );
    assert!(
        again.contains("restored     0 object(s), 0 B"),
        "a warm restore moved bytes it already held:\n{again}"
    );
    assert!(again.contains("complete     every declared object is resident and verified"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn the_journal_restores_what_was_durable_and_a_stale_one_is_discarded_whole() {
    let root = temporary("journal");
    let store = root.join("store");
    let cache = root.join("repo-cache");
    let converted = convert(&root, &store, &cache, "8192");
    let before: Vec<String> = ["e", "h"]
        .iter()
        .map(|key| span(&store, &converted.manifest, "decoder", key))
        .collect();

    fs::remove_dir_all(&store).unwrap();
    let fresh = root.join("fresh");
    ok(
        "store init --repo-cache",
        &tfs(&[
            "store",
            "init",
            fresh.to_str().unwrap(),
            "--repo-cache",
            cache.to_str().unwrap(),
        ]),
    );

    // A journal written under a different plan is not partially adopted, migrated or
    // reconciled. It is discarded whole, restores nothing, and the run converts afresh.
    let stale = ok(
        "repo-cache resume (stale plan)",
        &tfs(&[
            "repo-cache",
            "resume",
            fresh.to_str().unwrap(),
            &converted.chain,
            &converted.chain_length.to_string(),
            "--plan",
            &format!("sha256:{}", "ab".repeat(32)),
        ]),
    );
    assert!(
        stale.contains("DISCARDED JOURNAL_STALE"),
        "a journal from another plan was not discarded as stale:\n{stale}"
    );

    // THE INTERMEDIATE LINK IS A REAL RESUME POINT, and this is where that is proved. The
    // chain is read off the cache and its ROOT is presented instead of its head — exactly
    // what a pod reclaimed one checkpoint into the run would have left behind. It must
    // restore strictly less than the head does, and everything it does restore must be
    // verified, because a watermark that could not be resumed from partway is not a
    // watermark, it is a completion flag.
    let listing = ok(
        "repo-cache chain",
        &tfs(&[
            "repo-cache",
            "chain",
            cache.to_str().unwrap(),
            &converted.chain,
            &converted.chain_length.to_string(),
        ]),
    );
    let links: Vec<&str> = listing
        .lines()
        .filter(|line| line.starts_with(char::is_numeric))
        .collect();
    assert!(
        links.len() > 1,
        "an 8 KiB interval over 32 KiB produced one link, so nothing partial can be shown:\n{listing}"
    );
    let root_link = links[0].split_whitespace().collect::<Vec<_>>();
    let partial = ok(
        "resume from the root link",
        &tfs(&[
            "repo-cache",
            "resume",
            fresh.to_str().unwrap(),
            &hex_of(root_link[1]),
            root_link.last().unwrap(),
            "--plan",
            &converted.plan,
        ]),
    );
    let partial_objects: u64 = field(&partial, "restored     ", 1).parse().unwrap();
    assert!(
        partial_objects > 0,
        "the first checkpoint made nothing durable:\n{partial}"
    );

    let resumed = ok(
        "repo-cache resume",
        &tfs(&[
            "repo-cache",
            "resume",
            fresh.to_str().unwrap(),
            &converted.chain,
            &converted.chain_length.to_string(),
            "--plan",
            &converted.plan,
        ]),
    );
    assert!(
        resumed.contains("complete     every declared object is resident and verified"),
        "the journal's own set did not restore:\n{resumed}"
    );
    // The head's chain covers the whole conversion, so the checkpoint reassembles from the
    // journal alone — the same bytes the pod that is now gone produced.
    let after: Vec<String> = ["e", "h"]
        .iter()
        .map(|key| span(&fresh, &converted.manifest, "decoder", key))
        .collect();
    assert_eq!(before, after, "the journal restored different bytes");
    let _ = fs::remove_dir_all(root);
}

#[test]
fn a_cache_that_cannot_be_written_costs_the_conversion_nothing() {
    let root = temporary("weather");
    let store = root.join("store");
    // A regular FILE where the cache root should be: every directory creation under it
    // fails, which is the shape of an unmounted volume, a full one and a read-only one at
    // once. The conversion must not notice.
    fs::create_dir_all(&root).unwrap();
    let cache = root.join("not-a-directory");
    fs::write(&cache, b"this is not a mount point").unwrap();

    let first = root.join("encoder.safetensors");
    let second = root.join("decoder.safetensors");
    let registry = root.join("registry.json");
    let order_path = root.join("order.json");
    let plan_path = root.join("source-plan.json");
    carrier(&first, &["a", "b"], 0x11);
    carrier(&second, &["c", "d"], 0x77);
    order(
        &order_path,
        &[
            ("encoder", "a"),
            ("encoder", "b"),
            ("decoder", "c"),
            ("decoder", "d"),
        ],
    );
    bank(
        &registry,
        &[("encoder", &first), ("decoder", &second)],
        &order_path,
    );
    source_plan(&registry, &[&first, &second], &plan_path);
    ok(
        "store init --repo-cache",
        &tfs(&[
            "store",
            "init",
            store.to_str().unwrap(),
            "--repo-cache",
            cache.to_str().unwrap(),
        ]),
    );

    let stdout = ok(
        "ingest run over a broken cache",
        &tfs(&[
            "ingest",
            "run",
            store.to_str().unwrap(),
            "--source-plan",
            plan_path.to_str().unwrap(),
        ]),
    );
    assert!(
        stdout.contains("candidate    manifest sha256:"),
        "a broken cache stopped the conversion:\n{stdout}"
    );
    assert!(
        stdout.contains("the cache stopped answering; this run is no longer resumable"),
        "a broken cache was not reported as broken:\n{stdout}"
    );
    assert!(
        !stdout.contains("chain        sha256:"),
        "a cache that took nothing still produced a resume pointer:\n{stdout}"
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn a_resumed_pod_reuses_the_restored_conversion_journal_instead_of_re_converting() {
    // THE WHOLE POINT, END TO END. The two planes are separately correct and this is the
    // property only their composition has: the objects come back from the cache, the
    // conversion journal comes back beside them, and the run that follows converts NOTHING
    // — not because it was told the work was done, but because `convert` re-checked every
    // journalled part against the Store's own admission law and found the objects standing.
    let root = temporary("resumed-work");
    let store = root.join("store");
    let cache = root.join("repo-cache");
    let converted = convert(&root, &store, &cache, "8192");
    assert!(
        converted.stdout.contains("roles_converted"),
        "the first run reported no conversion work:\n{}",
        converted.stdout
    );
    let first_converted = observation(&converted.stdout, "roles_converted");
    let first_resumed = observation(&converted.stdout, "roles_resumed");
    assert!(first_converted > 0 && first_resumed == 0);

    // The pod is released.
    fs::remove_dir_all(&store).unwrap();
    let fresh = root.join("fresh");
    ok(
        "store init --repo-cache",
        &tfs(&[
            "store",
            "init",
            fresh.to_str().unwrap(),
            "--repo-cache",
            cache.to_str().unwrap(),
        ]),
    );

    let resumed = ok(
        "repo-cache resume",
        &tfs(&[
            "repo-cache",
            "resume",
            fresh.to_str().unwrap(),
            &converted.chain,
            &converted.chain_length.to_string(),
            "--plan",
            &converted.plan,
            "--session",
            &converted.session,
        ]),
    );
    assert!(
        resumed.contains("installed under session"),
        "no conversion journal came back with the bytes:\n{resumed}"
    );

    // The same conversion, on the new pod, with the carriers still present. It must do no
    // conversion work at all.
    let again = ok(
        "ingest run on the resumed store",
        &tfs(&[
            "ingest",
            "run",
            fresh.to_str().unwrap(),
            "--source-plan",
            root.join("source-plan.json").to_str().unwrap(),
        ]),
    );
    assert_eq!(
        observation(&again, "roles_converted"),
        0,
        "the resumed run re-converted work the journal already described:\n{again}"
    );
    assert_eq!(
        observation(&again, "roles_resumed"),
        first_converted,
        "the resumed run did not reuse every op the first one journalled:\n{again}"
    );
    assert_eq!(
        observation(&again, "bytes_hashed"),
        0,
        "the resumed run re-hashed carrier bytes:\n{again}"
    );
    // And it is the same artifact, not merely a cheap one.
    assert!(again.contains(&format!(
        "candidate    manifest sha256:{}",
        converted.manifest
    )));
    let _ = fs::remove_dir_all(root);
}

/// One counter off the worker's aligned observation block (`  <name>   <value>`).
fn observation(text: &str, name: &str) -> i64 {
    text.lines()
        .map(str::trim)
        .find(|line| line.starts_with(name))
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or_else(|| panic!("no {name:?} observation in:\n{text}"))
        .parse()
        .unwrap()
}
