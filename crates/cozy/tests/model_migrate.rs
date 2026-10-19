//! `cozy model migrate` end to end (xs-045 U12; dev builds, feature `migrate`), in tensorhub's
//! dev-build `cozy_ref` job: its Hub at release 1 with U12b's rows read, on MinIO and PostgreSQL
//! (TENSORHUB_DATABASE_URL). A model is seeded as a Hub from before the format cut held it: its
//! objects in the repo CAS and its rows in the read-only v1 tables, one checkpoint released in
//! 1.0.0 and in the yanked 0.9.0, another unreleased. A machine running this build migrates it;
//! the job's own TensorD on another machine pulls both releases' lane bit-identical to the old
//! bytes; and Hub release 2's migration 0102, which refuses while any release lane names an old
//! checkpoint, then passes.
#![cfg(all(unix, feature = "migrate"))]
mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::{Home, Out};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use tensorfs_core::format::{Body, ObjectRef};
use tensorfs_core::store2::Store;

/// tensorhub `migrations/postgres/0102_drop_tensorfs_v1.up.sql` at 250e29ef (#1203), verbatim.
const RELEASE_TWO: &str = include_str!("fixtures/migrate/0102_drop_tensorfs_v1.up.sql");

/// The old format's encoding specs, as real headers carry them (their digests are their names).
const PLAIN: &str = "848c030b0201000405080706090a8081836576616c756581008100818258204766f931bbb70ed431e34b34e4f9fc5dda8174e96f43a8acd79d8799efcd43d2190204";
const FP8_ROWWISE: &str = "848303020181028283646461746182018104810083657363616c658201810182018182000081825820e7a1264e4b60ac9f800736b23b3e3c50bb3e6ed506bdc30a2769abfc850ef2ba1917c1";
/// The old format's dtype codes.
const F32: u64 = 1;
const BF16: u64 = 3;
const F8_E4M3: u64 = 4;
const COMPONENT: &str = "migrate_test";

fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Deterministic bytes.
fn noise(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

/// A CBOR item, the subset old headers use.
enum Cbor {
    U(u64),
    B(Vec<u8>),
    T(&'static str),
    A(Vec<Cbor>),
    Raw(Vec<u8>),
}

fn head(major: u8, n: u64, out: &mut Vec<u8>) {
    let m = major << 5;
    match n {
        0..=23 => out.push(m | n as u8),
        24..=0xff => out.extend([m | 24, n as u8]),
        0x100..=0xffff => {
            out.push(m | 25);
            out.extend((n as u16).to_be_bytes());
        }
        0x1_0000..=0xffff_ffff => {
            out.push(m | 26);
            out.extend((n as u32).to_be_bytes());
        }
        _ => {
            out.push(m | 27);
            out.extend(n.to_be_bytes());
        }
    }
}

fn encode(c: &Cbor, out: &mut Vec<u8>) {
    match c {
        Cbor::U(n) => head(0, *n, out),
        Cbor::B(b) => {
            head(2, b.len() as u64, out);
            out.extend(b);
        }
        Cbor::T(t) => {
            head(3, t.len() as u64, out);
            out.extend(t.as_bytes());
        }
        Cbor::A(items) => {
            head(4, items.len() as u64, out);
            items.iter().for_each(|i| encode(i, out));
        }
        Cbor::Raw(bytes) => out.extend(bytes),
    }
}

/// One old tensor: its key, logical dtype and shape, spec, and parts.
struct Tensor {
    key: &'static str,
    dtype: u64,
    shape: Vec<u64>,
    spec: u64,
    parts: Vec<Part>,
}

/// One old part, and whether the header holds its bytes inline.
struct Part {
    role: &'static str,
    dtype: u64,
    shape: Vec<u64>,
    bytes: Vec<u8>,
    inline: bool,
}

/// A checkpoint from before the cut: its objects by digest, its header and two manifests naming
/// it (byte-different, so two checkpoints: one released, one not).
struct Old {
    tensors: Vec<Tensor>,
    objects: Vec<(String, Vec<u8>)>,
    header: (String, u64),
    manifests: [(String, Vec<u8>); 2],
}

fn old() -> Old {
    let plain = |key, shape: Vec<u64>, seed, inline| {
        let n = shape.iter().product::<u64>() as usize * 2;
        let part = Part { role: "value", dtype: BF16, shape: shape.clone(), bytes: noise(seed, n), inline };
        Tensor { key, dtype: BF16, shape, spec: 0, parts: vec![part] }
    };
    let tensors = vec![
        // 16 MiB: one object on the old grid, which the new checkpoint reuses.
        plain("blocks.0.big.weight", vec![4096, 2048], 1, false),
        plain("blocks.0.small.weight", vec![64, 64], 2, false),
        plain("blocks.0.small.bias", vec![64], 3, true),
        Tensor {
            key: "blocks.1.proj.weight",
            dtype: BF16,
            shape: vec![32, 256],
            spec: 1,
            parts: vec![
                Part { role: "data", dtype: F8_E4M3, shape: vec![32, 256], bytes: noise(4, 32 * 256), inline: false },
                Part { role: "scale", dtype: F32, shape: vec![32], bytes: noise(5, 32 * 4), inline: true },
            ],
        },
    ];
    let mut objects = vec![];
    let items = (tensors.iter())
        .map(|t| {
            let parts = (t.parts.iter())
                .map(|p| {
                    let body = match p.inline {
                        true => Cbor::B(p.bytes.clone()),
                        false => {
                            objects.push((sha(&p.bytes), p.bytes.clone()));
                            let digest = hex::decode(sha(&p.bytes)).unwrap();
                            Cbor::A(vec![Cbor::A(vec![Cbor::B(digest), Cbor::U(p.bytes.len() as u64)])])
                        }
                    };
                    let shape = p.shape.iter().map(|n| Cbor::U(*n)).collect();
                    Cbor::A(vec![Cbor::T(p.role), Cbor::U(p.dtype), Cbor::A(shape), body])
                })
                .collect();
            let shape = t.shape.iter().map(|n| Cbor::U(*n)).collect();
            Cbor::A(vec![Cbor::T(t.key), Cbor::U(t.dtype), Cbor::A(shape), Cbor::U(t.spec), Cbor::A(parts)])
        })
        .collect();
    let header = Cbor::A(vec![
        Cbor::T("cozytensors/1"),
        Cbor::A(vec![Cbor::A(vec![Cbor::T(COMPONENT), Cbor::B(br#"{"_class_name":"MigrateTest"}"#.to_vec())])]),
        Cbor::A(vec![]),
        Cbor::A(vec![Cbor::Raw(hex::decode(PLAIN).unwrap()), Cbor::Raw(hex::decode(FP8_ROWWISE).unwrap())]),
        Cbor::A(vec![Cbor::A(vec![Cbor::T(COMPONENT), Cbor::A(items)])]),
    ]);
    let mut bytes = vec![];
    encode(&header, &mut bytes);
    let header = (sha(&bytes), bytes.len() as u64);
    objects.push((header.0.clone(), bytes));
    let manifest = format!(
        r#"{{"entries":[{{"blob":{{"length":{},"sha256":"{}"}},"kind":"cozytensors","path":"model.cozytensors"}}]}}"#,
        header.1, header.0
    );
    let manifests = [manifest.clone(), format!("{manifest}\n")].map(|m| (sha(m.as_bytes()), m.into_bytes()));
    Old { tensors, objects, header, manifests }
}

fn run(program: &str, args: &[&str], env: &[(&str, &str)]) -> Out {
    let out = Command::new(program).args(args).envs(env.iter().copied()).output().unwrap();
    common::out(out)
}

/// Hub release 2's migration, applied and rolled back: whether it passes, and why not.
fn release_two(database: &str, dir: &Path) -> Out {
    let file = dir.join("0102.sql");
    std::fs::write(&file, RELEASE_TWO).unwrap();
    let file = file.to_string_lossy().into_owned();
    run("psql", &[database, "-v", "ON_ERROR_STOP=1", "-c", "BEGIN", "-f", &file, "-c", "ROLLBACK"], &[])
}

fn psql(database: &str, sql: &str) -> String {
    let out = run("psql", &[database, "-v", "ON_ERROR_STOP=1", "-tAc", sql], &[]);
    assert_eq!(out.code, 0, "{sql}: {out:?}");
    out.stdout.trim().to_owned()
}

/// The old model in the job's repo CAS (MinIO, as live-hub-client.sh runs it) and in the Hub's
/// read-only v1 rows, as migration 0101 indexed them: manifest 0 in 1.0.0 (revision 3) and in the
/// yanked 0.9.0 (revision 2), lane bf16; manifest 1 unreleased.
fn seed(old: &Old, org: &str, name: &str, database: &str, dir: &Path) {
    let tree = dir.join("cas");
    let place = |key: String, bytes: &[u8]| {
        let path = tree.join(key);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    };
    for (digest, bytes) in &old.objects {
        place(format!("blobs/{}/{}/{digest}", &digest[..2], &digest[2..4]), bytes);
    }
    for (digest, bytes) in &old.manifests {
        place(format!("manifests/{}/{}/{digest}.json", &digest[..2], &digest[2..4]), bytes);
    }
    let tree = tree.to_string_lossy().into_owned();
    let aws = [("AWS_ACCESS_KEY_ID", "proof"), ("AWS_SECRET_ACCESS_KEY", "proof-secret")];
    let copied = run("aws", &["--endpoint-url", "http://127.0.0.1:9000", "--region", "us-east-1", "s3", "cp", "--recursive", "--quiet", &tree, "s3://repo/"], &aws);
    assert_eq!(copied.code, 0, "{copied:?}");

    let parts: Vec<&(String, Vec<u8>)> = old.objects.iter().filter(|o| o.0 != old.header.0).collect();
    let bytes: u64 = old.objects.iter().map(|o| o.1.len() as u64).sum();
    let count = old.objects.len();
    let (h, hl) = &old.header;
    let mut sql = vec![
        format!("INSERT INTO hub_v2.repos(org,name,kind) VALUES('{org}','{name}','model')"),
        format!("INSERT INTO hub_v2.tensorfs_repos(storage_domain,org,name,document_sha256) VALUES('repo_cas','{org}','{name}',repeat('0',64))"),
        format!("INSERT INTO hub_v2.model_repos(repo_id,visibility,etag) SELECT id,'public','' FROM hub_v2.repos WHERE org='{org}' AND name='{name}'"),
    ];
    for (digest, body) in &old.objects {
        sql.push(format!("INSERT INTO hub_v2.verified_objects(object_id,length,bucket,key,backend_version,etag,checksum_sha256,storage_domain,custody_scope,public) \
            VALUES('sha256:{digest}',{},'repo','blobs/{}/{}/{digest}','','','','repo_cas','shared',true)", body.len(), &digest[..2], &digest[2..4]));
    }
    let lanes: [&[(&str, u64, bool)]; 2] = [&[("1.0.0", 3, false), ("0.9.0", 2, true)], &[("", 0, false)]];
    for ((m, body), lanes) in old.manifests.iter().zip(lanes) {
        let ml = body.len();
        let published = !lanes[0].0.is_empty();
        sql.push(format!("INSERT INTO hub_v2.tensorfs_checkpoints(storage_domain,org,name,manifest_sha256,manifest_length,published,\
            header_sha256,header_length,object_count,object_bytes) VALUES('repo_cas','{org}','{name}','{m}',{ml},{published},'{h}',{hl},{count},{bytes})"));
        sql.push(format!("INSERT INTO hub_v2.tensorfs_checkpoint_objects(storage_domain,org,name,manifest_sha256,object_kind,sha256,length) \
            VALUES('repo_cas','{org}','{name}','{m}','header','{h}',{hl})"));
        for (digest, part) in &parts {
            sql.push(format!("INSERT INTO hub_v2.tensorfs_checkpoint_objects(storage_domain,org,name,manifest_sha256,object_kind,sha256,length) \
                VALUES('repo_cas','{org}','{name}','{m}','part','{digest}',{})", part.len()));
        }
        sql.push(format!("INSERT INTO hub_v2.tensorfs_checkpoint_components(storage_domain,org,name,manifest_sha256,component,bytes) \
            VALUES('repo_cas','{org}','{name}','{m}','{COMPONENT}',{bytes})"));
        for (release, revision, yanked) in lanes {
            let lane = if release.is_empty() { "" } else { "bf16" };
            if !release.is_empty() {
                sql.push(format!("INSERT INTO hub_v2.tensorfs_releases(storage_domain,org,name,version,release_revision,release_yanked,lane,\
                    manifest_sha256,manifest_length) VALUES('repo_cas','{org}','{name}','{release}',{revision},{yanked},'{lane}','{m}',{ml})"));
            }
            sql.push(format!("INSERT INTO hub_v2.model_lanes(repo_id,release,lane,revision,yanked,manifest_sha256,manifest_length,\
                header_sha256,objects,bytes,encodings,components,v1) SELECT id,'{release}','{lane}',{revision},{yanked},'{m}',{ml},'{h}',\
                {count},{bytes},'{{}}',jsonb_build_object('{COMPONENT}',{bytes}::bigint),true FROM hub_v2.repos WHERE org='{org}' AND name='{name}'"));
        }
    }
    psql(database, &format!("BEGIN; {}; COMMIT", sql.join("; ")));
}

/// `cozy args`, its stdout echoed so a failure shows the run's events.
fn cozy(home: &Home, args: &[&str]) -> Out {
    let out = home.cozy(args);
    println!("cozy {}: {}\n{}{}", args.join(" "), out.code, out.stdout, out.stderr);
    out
}

/// The last line a `--json` verb printed: its document (a followed run's terminal event, or
/// the migration's summary).
fn last(out: &Out) -> Value {
    serde_json::from_str(out.stdout.lines().last().unwrap_or_default()).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}

/// What the machine `on` made present of `model`: its manifest.
fn download(home: &Home, model: &str, on: &str) -> String {
    let out = cozy(home, &["--json", "model", "download", model, "--on", on]);
    let ended = last(&out);
    assert_eq!(ended["type"], "run.completed", "{out:?}");
    ended["run"]["result"]["models"][0]["manifest"].as_str().unwrap().to_owned()
}

/// Every object `manifest` reaches in the store at `root`, with its bytes.
fn closure(root: &Path, manifest: &str) -> Vec<(String, Vec<u8>)> {
    let store = Store::open(root).unwrap();
    let sha256 = manifest.trim_start_matches("sha256:").to_string();
    let length = std::fs::metadata(store.path(&ObjectRef { sha256: sha256.clone(), length: 0 }).unwrap()).unwrap().len();
    let mut objects: Vec<_> = (store.closure(&ObjectRef { sha256, length }).unwrap().into_iter())
        .map(|object| (object.sha256.clone(), store.read(&object).unwrap()))
        .collect();
    objects.sort();
    objects
}

/// Every old tensor, read from the new checkpoint in the store at `root`: byte for byte the same.
fn same_tensors(old: &Old, root: &Path, manifest: &str) {
    let store = Store::open(root).unwrap();
    let sha256 = manifest.trim_start_matches("sha256:").to_string();
    let length = std::fs::metadata(store.path(&ObjectRef { sha256: sha256.clone(), length: 0 }).unwrap()).unwrap().len();
    let ck = store.checkpoint(&ObjectRef { sha256, length }).unwrap();
    for t in &old.tensors {
        let stored = ck.tensor(COMPONENT, t.key).unwrap();
        let at: Vec<(String, tensorfs_core::format::Location)> = match &stored.body {
            Body::Data(at) => vec![(String::new(), *at)],
            Body::Encoded { parts, .. } => parts.iter().map(|p| (p.role.clone(), p.at())).collect(),
        };
        assert_eq!(at.len(), t.parts.len(), "{}", t.key);
        for ((_, location), part) in at.iter().zip(&t.parts) {
            let mut read = vec![0; location.length as usize];
            ck.bytes_into(location, &mut read).unwrap();
            assert!(read == part.bytes, "{} part {} differs from the old bytes", t.key, part.role);
        }
    }
}

#[test]
fn an_old_model_migrates_by_its_own_rows_and_pulls_bit_identical() {
    let Some(home) = Home::with_hub() else { return };
    let Some(database) = common::needs("TENSORHUB_DATABASE_URL", "the Hub's database: tensorhub's cozy_ref job names it") else {
        return;
    };
    let database = database.to_string_lossy().into_owned();
    let Some(tensord) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else { return };
    let account = home.sign_in("migrator@cozy.test");
    let (dir, old) = (tempfile::tempdir().unwrap(), old());
    let model = format!("{account}/legacy");
    seed(&old, &account, "legacy", &database, dir.path());
    let refused = release_two(&database, dir.path());
    assert!(refused.code != 0 && refused.stderr.contains(&format!("{model}@0.9.0 bf16, {model}@1.0.0 bf16")), "{refused:?}");

    // The machine that migrates runs this build (`cozy machine serve`): the job's TensorD has no
    // migration runs. SAFETY: this file's one test; no other thread reads the environment.
    let wrapper = dir.path().join("tensord");
    let cozy_binary = env!("CARGO_BIN_EXE_cozy");
    std::fs::write(&wrapper, format!("#!/bin/sh\n[ \"$1\" = run ] && shift\nexec '{cozy_binary}' machine serve \"$@\"\n")).unwrap();
    std::fs::set_permissions(&wrapper, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    unsafe { std::env::set_var("COZY_TEST_TENSORD", &wrapper) };
    let migrator = common::tensord::added(&home, "m1");
    unsafe { std::env::set_var("COZY_TEST_TENSORD", &tensord) };

    let audit: PathBuf = dir.path().join("audit");
    let audit_arg = audit.to_string_lossy().into_owned();
    let out = cozy(&home, &["--json", "model", "migrate", &model, "--on", "m1", "--audit", &audit_arg]);
    assert_eq!(out.code, 0, "{out:?}");
    let summary = last(&out);
    let checkpoints = summary["models"][0]["checkpoints"].as_array().unwrap().clone();
    assert_eq!(checkpoints.len(), 2, "{summary}");
    let moved = |m: usize| {
        let old_id = format!("sha256:{}", old.manifests[m].0);
        let row = checkpoints.iter().find(|c| c["old"] == old_id.as_str()).unwrap_or_else(|| panic!("{summary}"));
        assert_eq!(row["removed"], true, "{row}");
        assert!(row["migration"]["verified"]["tensors"] == 4, "{row}");
        assert!(audit.join(&account).join("legacy").join(format!("{}.json", old.manifests[m].0)).is_file());
        (row["new"].as_str().unwrap().to_owned(), row.clone())
    };
    let ((released, row), (unreleased, _)) = (moved(0), moved(1));
    assert_ne!(released, unreleased);
    let repointed = row["repointed"].as_array().unwrap();
    for (release, revision, yanked) in [("1.0.0", 4, false), ("0.9.0", 3, true)] {
        let found = repointed.iter().find(|r| r["release"] == release).unwrap_or_else(|| panic!("{row}"));
        assert_eq!((&found["revision"], &found["yanked"], &found["lanes"]["bf16"]), (&revision.into(), &yanked.into(), &released.as_str().into()), "{row}");
    }

    // The model's repo file names no old checkpoint: its rows hold none, and release 2 passes.
    let rows = format!("FROM hub_v2.model_lanes l JOIN hub_v2.repos r ON r.id=l.repo_id WHERE r.org='{account}' AND r.name='legacy'");
    assert_eq!(psql(&database, &format!("SELECT count(*) FILTER (WHERE l.v1)||' '||count(*) {rows}")), "0 3");
    let passed = release_two(&database, dir.path());
    assert_eq!(passed.code, 0, "{passed:?}");

    // Another machine, on the job's TensorD, pulls both releases' lane, the yanked one by name.
    let _puller = common::tensord::added(&home, "m2");
    let puller = &_puller;
    for release in ["1.0.0", "0.9.0"] {
        assert_eq!(download(&home, &format!("{model}@{release}/bf16"), "m2"), released);
    }
    same_tensors(&old, &puller.store(), &released);
    assert!(closure(&puller.store(), &released) == closure(&migrator.store(), &released), "the pull differs from what migrated");

    // Again: nothing is old any more.
    let again = cozy(&home, &["--json", "model", "migrate", &model, "--on", "m1", "--audit", &audit_arg]);
    assert_eq!(again.code, 0, "{again:?}");
    assert_eq!(last(&again)["models"][0]["checkpoints"], serde_json::json!([]), "{again:?}");

    // --local: a pre-cut store's local/ model into the machine's store on the same filesystem.
    // The 16 MiB object it reuses moves in by hard link: one inode in both stores.
    let precut = dir.path().join("precut");
    let (m, body) = &old.manifests[0];
    let place = |key: String, bytes: &[u8]| {
        std::fs::create_dir_all(precut.join(&key).parent().unwrap()).unwrap();
        std::fs::write(precut.join(key), bytes).unwrap();
    };
    for (digest, bytes) in &old.objects {
        place(format!("blobs/{}/{}/{digest}", &digest[..2], &digest[2..4]), bytes);
    }
    place(format!("manifests/{}/{}/{m}.json", &m[..2], &m[2..4]), body);
    let repo = format!(r#"{{"checkpoints":[{{"manifest":{{"length":{},"sha256":"{m}"}}}}],"releases":[],"repo":{{"name":"tiny","org":"local"}}}}"#, body.len());
    place("repos/local/tiny.json".into(), repo.as_bytes());
    // A fresh machine on this build, whose store holds none of it yet. SAFETY: as above.
    unsafe { std::env::set_var("COZY_TEST_TENSORD", &wrapper) };
    let laptop = common::tensord::added(&home, "m3");
    unsafe { std::env::set_var("COZY_TEST_TENSORD", &tensord) };
    let local = precut.to_string_lossy().into_owned();
    let out = cozy(&home, &["--json", "model", "migrate", "--local", &local, "--on", "m3", "--audit", &audit_arg]);
    assert_eq!(out.code, 0, "{out:?}");
    let row = last(&out)["models"][0].clone();
    assert_eq!((&row["repository"], &row["linked"]["bytes"], &row["copied"]["bytes"]), (&"local/tiny".into(), &(16u64 << 20).into(), &0.into()), "{row}");
    let made = row["manifest"].as_str().unwrap();
    same_tensors(&old, &laptop.store(), made);
    let big = &old.objects.iter().find(|o| o.1.len() == 16 << 20).unwrap().0;
    let blob = |root: &Path| std::fs::metadata(root.join("blobs").join(&big[..2]).join(&big[2..4]).join(big)).unwrap();
    use std::os::unix::fs::MetadataExt as _;
    assert_eq!(blob(&precut).ino(), blob(&laptop.store()).ino(), "the reused object was copied, not linked");
    let store = Store::open(laptop.store()).unwrap();
    assert_eq!(store.resolve("local/tiny", None).unwrap().manifest.id(), made);
}
