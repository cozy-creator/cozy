//! Local code on a real machine: `cozy run ./script.py` and `cozy run ./dir/fn`. TensorD
//! (COZY_TEST_TENSORD) installs the code it is sent, answers its interface, and runs it; an
//! edited script runs as edited. Nothing Python runs on this computer.
#![cfg(unix)]
mod common;

use std::path::Path;

use common::{Home, Out};
use serde_json::Value;

const SCRIPT: &str = r#"# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime"]
# [tool.cozy]
# accelerator = false
# ///
def main() -> str:
    return "hello from a script"
"#;

fn cozy(home: &Home, dir: &Path, args: &[&str]) -> Out {
    common::out(home.command().current_dir(dir).args(args).output().unwrap())
}

/// A followed `--json` run's end, which must be a completion.
fn completed(out: &Out) -> Value {
    assert_eq!(out.code, 0, "{out:?}");
    let end = out
        .stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .rfind(|e| e["type"] == "run.completed");
    end.unwrap_or_else(|| panic!("no completion: {out:?}"))["run"].clone()
}

fn copy(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        match entry.file_type().unwrap().is_dir() {
            true => copy(&entry.path(), &to.join(entry.file_name())),
            false => drop(std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap()),
        }
    }
}

#[test]
fn local_code_runs_on_a_real_machine_and_runs_again_as_edited() {
    let Some(home) = Home::new() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    std::fs::create_dir_all(&work).unwrap();

    // A script: the machine installs its generated project, answers `main`'s interface, runs it.
    std::fs::write(work.join("greet.py"), SCRIPT).unwrap();
    let described = cozy(
        &home,
        &work,
        &["run", "./greet.py", "--describe", "--on", "m1"],
    );
    assert_eq!(described.code, 0, "{described:?}");
    assert!(
        described.stdout.contains("request: none") && described.stdout.contains("runs on: CPU"),
        "{described:?}"
    );
    let run = completed(&cozy(
        &home,
        &work,
        &["--json", "run", "greet.py", "--on", "m1"],
    ));
    assert_eq!(run["machine"], "m1");
    assert_eq!(run["result"]["value"], "hello from a script", "{run}");
    assert!(
        run["target"].as_str().unwrap().starts_with("local/"),
        "{run}"
    );

    // Edited, it runs as edited: only the changed file travels.
    std::fs::write(
        work.join("greet.py"),
        SCRIPT.replace("hello from a script", "edited, and run again"),
    )
    .unwrap();
    let again = completed(&cozy(
        &home,
        &work,
        &["--json", "run", "greet.py", "--on", "m1"],
    ));
    assert_eq!(again["result"]["value"], "edited, and run again", "{again}");

    // A GPU function sent to a machine with no supported GPU: it says so, and how to rent one;
    // asked for 8 bits, that its models take their 16-bit lanes there instead.
    std::fs::write(work.join("gpu.py"), SCRIPT.replace("# accelerator = false\n", "")).unwrap();
    let gpu = cozy(&home, &work, &["run", "gpu.py", "--on", "m1", "--detach"]);
    assert!(
        gpu.stderr.contains("m1 has no supported GPU") && gpu.stderr.contains("--on new"),
        "{gpu:?}"
    );
    assert!(!gpu.stderr.contains("16-bit"), "{gpu:?}");
    let eight = cozy(&home, &work, &["run", "greet.py", "--bits", "8", "--on", "m1", "--detach"]);
    assert!(
        eight.stderr.contains("its models take their 16-bit lanes (--bits 8 needs a GPU)")
            && !eight.stderr.contains("slowly"),
        "{eight:?}"
    );

    // A project directory: its functions, then one of them.
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny"),
        &work.join("tiny"),
    );
    let listed = cozy(&home, &work, &["--json", "run", "./tiny", "--on", "m1"]);
    assert_eq!(listed.code, 0, "{listed:?}");
    let functions: Value = serde_json::from_str(&listed.stdout).unwrap();
    assert_eq!(functions["functions"][0]["function"], "echo", "{functions}");
    let echoed = completed(&cozy(
        &home,
        &work,
        &[
            "--json",
            "run",
            "./tiny/echo",
            "from a directory",
            "--on",
            "m1",
        ],
    ));
    assert_eq!(echoed["result"]["text"], "from a directory", "{echoed}");
}

/// Public CLI -> Calcifer -> TensorD -> Runtime Writer -> checkpoint_item -> Calcifer.
/// The configured shared store holds the exact bytes; Calcifer persists references only.
#[test]
fn runtime_checkpoint_writes_replay_and_survive_calcifer_restart() {
    let Some(home) = Home::new() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else { return };
    let store_path = home.path().join("shared-tensorfs");
    let store_name = store_path.to_str().unwrap();
    let _machine = common::tensord::added_with(&home, "m1", &[("COZY_TENSORFS_ROOT", store_name)]);
    let work = home.path().join("work");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../tensord/tests/fixtures/cpu_weights"),
        &work.join("cpu_weights"),
    );
    let run = |size: &str| completed(&cozy(&home, &work,
        &["--json", "run", "./cpu_weights/table", &format!("size={size}"), "--on", "m1"]));
    let checkpoint = |run: &Value, size: usize| {
        let output = run["output"].as_array().unwrap().iter()
            .find(|output| output["name"] == "model").unwrap_or_else(|| panic!("no model output: {run}"));
        assert_eq!(output["status"], "completed", "{output}");
        for field in ["path", "sha256", "export_error"] {
            assert!(output.get(field).is_none_or(Value::is_null), "checkpoint has byte-export metadata: {output}");
        }
        let reference = output["checkpoint"].clone();
        assert_eq!(reference["repo_id"], "local/derived", "{reference}");
        let store = tensorfs::store::Store::open(&store_path).unwrap();
        let revision = reference["revision"].as_str().unwrap();
        let mut tree = tensorfs::format::ObjectRef {
            sha256: revision.strip_prefix("sha256:").unwrap().into(), length: 0,
        };
        tree.length = std::fs::metadata(store.path(&tree).unwrap()).unwrap().len();
        let variant = reference["variant"].as_str().unwrap();
        let (_, header) = tensorfs::fetch::variant_of_tree(&store.tree(&tree).unwrap(),
            (!variant.is_empty()).then_some(variant)).unwrap();
        assert_eq!(header.id(), run["result"]["checkpoint"].as_str().unwrap(), "tree and produced header differ: {run}");
        let expected: Vec<u8> = (0u8..=255).cycle().take(size * 4).collect();
        assert_eq!(store.checkpoint(&header).unwrap().read("model", "table").unwrap(), expected);
        reference
    };
    let first = run("64");
    let original = checkpoint(&first, 64);
    let again = run("64");
    assert_eq!(checkpoint(&again, 64), original);
    assert_eq!(first["result"]["transaction"], again["result"]["transaction"]);
    assert_ne!(first["result"]["producer_request_id"], again["result"]["producer_request_id"],
        "a writer executes again rather than returning the first run's result");
    let wider = run("128");
    assert_ne!(checkpoint(&wider, 128), original);
    assert_ne!(first["result"]["transaction"], wider["result"]["transaction"]);

    let number = first["number"].as_u64().unwrap().to_string();
    let down = home.cozy(["down"]);
    assert_eq!(down.code, 0, "{down:?}");
    let shown = home.json(["--json", "run", "show", &number]);
    assert_eq!(checkpoint(&shown, 64), original, "Calcifer persisted the typed reference");
    let replayed = run("64");
    assert_eq!(checkpoint(&replayed, 64), original);
    assert_eq!(first["result"]["transaction"], replayed["result"]["transaction"]);
}

/// Public CLI conversion -> ephemeral ordinary owner Bearer -> Runtime Writer -> real Hub.
/// A private Hub source starts cold; Runtime reads it, writes and uploads a new output.
/// A failed upload retains a typed local tree, retried without executing the model again.
#[test]
fn runtime_checkpoint_uploads_with_the_ordinary_owner_credential() {
    use std::sync::Arc;
    use tensorfs::{
        fetch::{
            Client, Config, Endpoint, Fetcher, ObjectStore, Origin, SourcePolicy, Want, bearer,
            upload_tree,
        },
        format::{Dtype, Tree},
        store::Store,
        writer::{Data, Source},
    };

    let Some(home) = Home::with_hub() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    let account = home.sign_in("cli-writer-upload@cozy.test");
    let destination = format!("{account}/writer-output");
    let store_path = home.path().join("shared-tensorfs");
    let origin = std::env::var("TENSORHUB_ORIGIN").unwrap();
    let ca = std::fs::read(std::env::var("TENSORHUB_CA").unwrap()).unwrap();
    let client = Client::with_roots(
        SourcePolicy {
            any_host: true,
            allow_local: true,
            max_redirects: 5,
            ..Default::default()
        },
        &ca,
    )
    .unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let source_store = Store::open(home.path().join("seed-tensorfs")).unwrap();
    let (seed, source_tree) = {
        let mut writer = source_store.writer(None, None).unwrap();
        writer
            .get("model")
            .put(
                "seed",
                &Dtype::F32,
                &[1],
                Data::Plain(Source::Read(&mut &[0u8; 4][..])),
            )
            .unwrap();
        let header = writer.commit().unwrap();
        let mut tree = Tree::default();
        assert_eq!(
            source_store.checkpoint(&header).unwrap().header().chunks,
            vec![tensorfs::format::ObjectRef::of(&[0u8; 4])]
        );
        tree.set_variant("f32", header.clone());
        (header, source_store.put_tree(&tree).unwrap())
    };
    let source_repo = format!("{account}/writer-source");
    runtime.block_on(async {
        let endpoint = Endpoint::new(&client, &origin, Some(bearer(home.token()))).unwrap();
        upload_tree(&endpoint, &source_repo, &source_tree, &source_store, "main")
            .await
            .unwrap();
    });
    let source_ref = format!("{source_repo}#{}", source_tree.id());
    assert!(
        !store_path.exists(),
        "the machine must start without source objects"
    );
    let _machine = common::tensord::added_with(
        &home,
        "m1",
        &[("COZY_TENSORFS_ROOT", store_path.to_str().unwrap())],
    );
    assert!(
        !Store::open(&store_path).unwrap().has(&seed),
        "source is warm before the call"
    );
    let work = home.path().join("work");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../tensord/tests/fixtures/cpu_weights"),
        &work.join("cpu_weights"),
    );
    let run = || {
        completed(&cozy(
            &home,
            &work,
            &[
                "--json",
                "run",
                "./cpu_weights/convert",
                &source_ref,
                &destination,
                "size=64",
                "--on",
                "m1",
            ],
        ))
    };
    let reference = |run: &Value| {
        let output = run["output"]
            .as_array()
            .unwrap()
            .iter()
            .find(|output| output["name"] == "model")
            .unwrap();
        assert_eq!(output["status"], "completed", "{output}");
        for field in ["path", "sha256", "export_error"] {
            assert!(output.get(field).is_none_or(Value::is_null), "{output}");
        }
        let reference = output["checkpoint"].clone();
        assert_eq!(reference["repo_id"], destination, "{reference}");
        assert!(
            reference["revision"]
                .as_str()
                .unwrap()
                .starts_with("sha256:")
        );
        reference
    };
    let first = run();
    let remote = reference(&first);
    let again = run();
    assert_eq!(
        reference(&again),
        remote,
        "repeated work republishes the same tree"
    );
    assert_eq!(
        first["result"]["transaction"],
        again["result"]["transaction"]
    );

    let expected: Vec<u8> = (0u8..=255).collect();
    let local = Store::open(&store_path).unwrap();
    assert_eq!(
        local
            .checkpoint(&seed)
            .unwrap()
            .read("model", "seed")
            .unwrap(),
        [0u8; 4]
    );
    let header =
        tensord::storage::held(&local, first["result"]["checkpoint"].as_str().unwrap()).unwrap();
    assert_eq!(
        local
            .checkpoint(&header)
            .unwrap()
            .read("model", "table")
            .unwrap(),
        expected
    );

    let cold = tempfile::tempdir().unwrap();
    let downloaded = Store::open(cold.path()).unwrap();
    runtime.block_on(async {
        let anonymous = Endpoint::new(&client, &origin, None).unwrap().anonymous();
        assert!(
            anonymous.repo(&destination).await.unwrap().is_none(),
            "private output is public"
        );
        let endpoint =
            Arc::new(Endpoint::new(&client, &origin, Some(bearer(home.token()))).unwrap());
        let (repo, _) = endpoint.repo(&destination).await.unwrap().unwrap();
        let published = repo
            .roots()
            .find(|tree| tree.id() == remote["revision"].as_str().unwrap())
            .unwrap();
        let endpoint: Arc<dyn Origin> = endpoint;
        let fetcher = Fetcher::new(
            Arc::new(downloaded.clone()) as Arc<dyn ObjectStore>,
            client.clone(),
            None,
            Config::default(),
        );
        let pulled = fetcher
            .pull_tree(&endpoint, published, None, &Want::default())
            .await
            .unwrap();
        fetcher.settled().await;
        assert_eq!(pulled.header, header);
        assert_eq!(
            pulled.tree.as_ref().unwrap().id(),
            remote["revision"].as_str().unwrap()
        );
        assert_eq!(
            downloaded
                .checkpoint(&pulled.header)
                .unwrap()
                .read("model", "table")
                .unwrap(),
            expected
        );
    });

    // Refusal comes from the real ordinary-owner boundary, after Runtime commits locally.
    let rival = Home::with_hub().unwrap();
    let rival_account = rival.sign_in("cli-writer-rival@cozy.test");
    let refused_destination = format!("{rival_account}/private-output");
    let rival_endpoint = Endpoint::new(&client, &origin, Some(bearer(rival.token()))).unwrap();
    let rival_before = runtime.block_on(async {
        upload_tree(
            &rival_endpoint,
            &refused_destination,
            &source_tree,
            &source_store,
            "main",
        )
        .await
        .unwrap();
        rival_endpoint.repo(&refused_destination).await.unwrap()
    });
    // The denial case reads an already held local alias: its unrelated publish permission
    // cannot be refused by a private-source token trade before the Writer commits.
    tensord::storage::set_local(&local, "seed", &seed, &Default::default()).unwrap();
    let refused = cozy(
        &home,
        &work,
        &[
            "--json",
            "run",
            "./cpu_weights/convert",
            "local/seed",
            &refused_destination,
            "size=128",
            "count=true",
            "--on",
            "m1",
        ],
    );
    assert_eq!(refused.code, 1, "{refused:?}");
    let failed = refused
        .stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .rfind(|event| event["type"] == "run.failed")
        .unwrap()["run"]
        .clone();
    assert!(
        failed["reason"].to_string().contains("upload_failed"),
        "{failed}"
    );
    assert!(
        failed["reason"].to_string().contains("404")
            && failed["reason"].to_string().contains("not visible"),
        "expected the ordinary permission refusal: {failed}"
    );
    let retained = failed["output"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["name"] == "model")
        .unwrap()["checkpoint"]
        .clone();
    assert_eq!(retained["repo_id"], "local/derived", "{retained}");
    let retained_tree =
        tensord::storage::held(&local, retained["revision"].as_str().unwrap()).unwrap();
    let variant = retained["variant"].as_str().unwrap();
    let (_, retained_header) = tensorfs::fetch::variant_of_tree(
        &local.tree(&retained_tree).unwrap(),
        (!variant.is_empty()).then_some(variant),
    )
    .unwrap();
    let retained_bytes: Vec<u8> = (0u8..=255).cycle().take(512).collect();
    assert_eq!(
        local
            .checkpoint(&retained_header)
            .unwrap()
            .read("model", "table")
            .unwrap(),
        retained_bytes
    );
    let counter = store_path.join("test-convert-invocations");
    assert_eq!(std::fs::read(&counter).unwrap(), b"invoked\n");
    assert_eq!(
        runtime
            .block_on(rival_endpoint.repo(&refused_destination))
            .unwrap(),
        rival_before,
        "the refused upload changed another account's repository"
    );

    let retry_destination = format!("{account}/writer-retry");
    let retry = completed(&home.cozy([
        "--json",
        "run",
        "upload",
        &format!("{}#model", failed["number"].as_u64().unwrap()),
        &retry_destination,
    ]));
    assert_eq!(
        retry["result"]["models"][0]["published"]["tree"], retained["revision"],
        "{retry}"
    );
    assert_eq!(
        retry["result"]["models"][0]["published"]["checkpoint"],
        retained_header.id(),
        "{retry}"
    );
    assert_eq!(
        std::fs::read(&counter).unwrap(),
        b"invoked\n",
        "a retained upload reran model code"
    );
    let retry_cache = tempfile::tempdir().unwrap();
    let retry_store = Store::open(retry_cache.path()).unwrap();
    runtime.block_on(async {
        let endpoint =
            Arc::new(Endpoint::new(&client, &origin, Some(bearer(home.token()))).unwrap());
        let (repo, _) = endpoint.repo(&retry_destination).await.unwrap().unwrap();
        assert!(
            repo.roots().any(|commit| commit == &retained_tree),
            "the retry did not publish its retained tree"
        );
        let endpoint: Arc<dyn Origin> = endpoint;
        let fetcher = Fetcher::new(
            Arc::new(retry_store.clone()) as Arc<dyn ObjectStore>,
            client.clone(),
            None,
            Config::default(),
        );
        let pulled = fetcher
            .pull_tree(
                &endpoint,
                &retained_tree,
                (!variant.is_empty()).then_some(variant),
                &Want::default(),
            )
            .await
            .unwrap();
        fetcher.settled().await;
        assert_eq!(pulled.tree.as_ref().unwrap(), &retained_tree);
        assert_eq!(pulled.variant, variant);
        assert_eq!(pulled.header, retained_header);
        assert_eq!(
            retry_store
                .checkpoint(&pulled.header)
                .unwrap()
                .read("model", "table")
                .unwrap(),
            retained_bytes
        );
    });

    let down = home.cozy(["down"]);
    assert_eq!(down.code, 0, "{down:?}");
    let db_path = home.root().join("calcifer/state.db");
    let db = calcifer::store::Store::open(&db_path).unwrap();
    for run in [&first, &again, &failed, &retry] {
        let stored = db.run(run["id"].as_str().unwrap()).unwrap().unwrap();
        assert!(
            stored.spec.tensorfs.is_none(),
            "send-time credential entered stored spec"
        );
        assert!(
            stored.spec.hub.is_none(),
            "send-time legacy access entered stored spec"
        );
    }
    drop(db);
    let token = home.token();
    let bytes = std::fs::read(db_path).unwrap();
    assert!(
        !bytes
            .windows(token.len())
            .any(|part| part == token.as_bytes()),
        "owner token entered SQLite"
    );
    let shown = home.json([
        "--json",
        "run",
        "show",
        &first["number"].as_u64().unwrap().to_string(),
    ]);
    assert_eq!(
        reference(&shown),
        remote,
        "remote reference survives Calcifer restart"
    );
    let failed_shown = home.json([
        "--json",
        "run",
        "show",
        &failed["number"].as_u64().unwrap().to_string(),
    ]);
    assert_eq!(
        failed_shown["output"][0]["checkpoint"], retained,
        "failed run lost its retained reference"
    );
    assert!(
        !serde_json::to_string(&shown).unwrap().contains(&token),
        "owner token entered public history"
    );
}

/// Public local model conversion needs neither a Hub connection nor a sign-in.
#[test]
fn runtime_checkpoint_uploads_local_alias_on_disk() {
    use tensorfs::{
        format::Dtype,
        store::Store,
        writer::{Data, Source},
    };
    let Some(home) = Home::new() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    home.write_config("hub = \"offline\"\n[hubs]\noffline = \"http://127.0.0.1:1\"\n");
    let store_path = home.path().join("local-tensorfs");
    let seed = {
        let store = Store::open(&store_path).unwrap();
        let mut writer = store.writer(None, None).unwrap();
        writer
            .get("model")
            .put(
                "seed",
                &Dtype::F32,
                &[1],
                Data::Plain(Source::Read(&mut &[0u8; 4][..])),
            )
            .unwrap();
        let header = writer.commit().unwrap();
        assert_eq!(
            store.checkpoint(&header).unwrap().header().chunks,
            vec![tensorfs::format::ObjectRef::of(&[0u8; 4])]
        );
        tensord::storage::set_local(&store, "seed", &header, &Default::default()).unwrap();
        header
    };
    let _machine = common::tensord::added_with(
        &home,
        "m1",
        &[("COZY_TENSORFS_ROOT", store_path.to_str().unwrap())],
    );
    let work = home.path().join("work");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../tensord/tests/fixtures/cpu_weights"),
        &work.join("cpu_weights"),
    );
    let run = completed(&cozy(
        &home,
        &work,
        &[
            "--json",
            "run",
            "./cpu_weights/convert",
            "local/seed",
            "local/output",
            "size=64",
            "--on",
            "m1",
        ],
    ));
    let output = run["output"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["name"] == "model")
        .unwrap();
    let reference = &output["checkpoint"];
    assert_eq!(reference["repo_id"], "local/output", "{run}");
    for field in ["path", "sha256", "export_error"] {
        assert!(output.get(field).is_none_or(Value::is_null), "{output}");
    }
    let store = Store::open(&store_path).unwrap();
    assert_eq!(
        store
            .checkpoint(&seed)
            .unwrap()
            .read("model", "seed")
            .unwrap(),
        [0u8; 4]
    );
    let tree = tensord::storage::held(&store, reference["revision"].as_str().unwrap()).unwrap();
    let variant = reference["variant"].as_str().unwrap();
    let (_, header) = tensorfs::fetch::variant_of_tree(
        &store.tree(&tree).unwrap(),
        (!variant.is_empty()).then_some(variant),
    )
    .unwrap();
    assert_eq!(header.id(), run["result"]["checkpoint"].as_str().unwrap());
    assert_eq!(
        store
            .checkpoint(&header)
            .unwrap()
            .read("model", "table")
            .unwrap(),
        (0u8..=255).collect::<Vec<_>>()
    );
    let repo = store.get_ref("local/output").unwrap().unwrap();
    assert!(
        repo.roots().any(|commit| commit == &tree),
        "local repository lost its checkpoint"
    );
    let chosen = store
        .variant(&tree, (!variant.is_empty()).then_some(variant))
        .unwrap();
    assert_eq!(chosen.id(), &header);
    assert!(
        !home.device_key("http://127.0.0.1:1").exists(),
        "a local conversion created sign-in credentials"
    );
}

/// A wheel no index serves (an unpublished build, as cozy-kernels before its release), named by a
/// path source: the machine installs it as a carried wheel, for a script (resolved on the machine)
/// and for a locked project (`uv sync` leaves it to the carried file), and the code imports it.
#[test]
fn a_local_wheel_a_path_source_names_is_carried_to_the_machine() {
    let Some(home) = Home::new() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    let write = |path: &str, text: &str| {
        let path = work.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    };
    let uv = |dir: &Path, args: &[&str]| {
        let out = std::process::Command::new("uv")
            .current_dir(dir)
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "uv {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    let backend = "[build-system]\nrequires = [\"uv_build>=0.9,<1\"]\nbuild-backend = \"uv_build\"\n\n\
                   [tool.uv.build-backend]\nmodule-root = \"\"\n";
    write(
        "probe/pyproject.toml",
        &format!(
            "[project]\nname = \"probe-kernels\"\nversion = \"0.1.0\"\nrequires-python = \">=3.12\"\n\n{backend}"
        ),
    );
    write(
        "probe/probe_kernels/__init__.py",
        "VALUE = \"from a carried wheel\"\n",
    );
    uv(&work, &["build", "--wheel", "--out-dir", "dist", "probe"]);
    let wheel = "dist/probe_kernels-0.1.0-py3-none-any.whl";
    assert!(work.join(wheel).is_file());

    // A script, which nothing locks.
    write(
        "carried.py",
        &format!(
            "# /// script\n# requires-python = \">=3.12\"\n# dependencies = [\"cozy-runtime\", \"probe-kernels\"]\n\
         # [tool.uv.sources]\n# probe-kernels = {{ path = \"{wheel}\" }}\n# [tool.cozy]\n# accelerator = false\n# ///\n\
         import probe_kernels\n\n\ndef main() -> str:\n    return probe_kernels.VALUE\n"
        ),
    );
    let run = completed(&cozy(
        &home,
        &work,
        &["--json", "run", "carried.py", "--on", "m1"],
    ));
    assert_eq!(run["result"]["value"], "from a carried wheel", "{run}");

    // A locked project.
    write(
        "locked/pyproject.toml",
        &format!(
            "[project]\nname = \"locked-probe\"\nversion = \"0.1.0\"\nrequires-python = \">=3.12\"\n\
         dependencies = [\"cozy-runtime\", \"msgspec>=0.19\", \"probe-kernels\"]\n\n\
         [project.entry-points.\"cozy.application\"]\ndefault = \"locked_probe:app\"\n\n\
         [tool.uv.sources]\nprobe-kernels = {{ path = \"../{wheel}\" }}\n\n{backend}"
        ),
    );
    write(
        "locked/package.toml",
        "[application]\nobject = \"locked_probe:app\"\n",
    );
    write(
        "locked/locked_probe/__init__.py",
        "import msgspec\nimport probe_kernels\nfrom cozy_runtime.author import App\n\n\
        app = App()\n\n\nclass ProbeInput(msgspec.Struct):\n    text: str = \"\"\n\n\nclass ProbeOutput(msgspec.Struct):\n    text: str\n\n\n\
        @app.entrypoint(accelerator=False)\ndef probe(payload: ProbeInput) -> ProbeOutput:\n    \
        \"\"\"Answers the carried wheel's value.\"\"\"\n    return ProbeOutput(probe_kernels.VALUE)\n",
    );
    uv(&work.join("locked"), &["lock", "--quiet"]);
    let run = completed(&cozy(
        &home,
        &work,
        &["--json", "run", "./locked/probe", "x", "--on", "m1"],
    ));
    assert_eq!(run["result"]["text"], "from a carried wheel", "{run}");
}

/// Run 10297: `cozy run --json` exited at `run.completed` while its 11 MB video was still being
/// saved, and its caller ended the machine. A followed run returns only once each output's last
/// revision landed here, checked against its digest: a 128 MiB output still exporting when the
/// run completed is whole at its path when `cozy run` exits, its landing before the final record.
#[test]
fn a_followed_run_returns_only_once_its_outputs_landed() {
    use sha2::{Digest as _, Sha256};
    let Some(home) = Home::new() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    let script = SCRIPT
        .replace(
            "def main() -> str:\n    return \"hello from a script\"",
            "from cozy_runtime.author import FileAsset, Outputs\n\n\
             def main(*, out: Outputs) -> FileAsset:\n    \
             return out.save_bytes(bytes(range(256)) * (512 << 10), media_type=\"application/octet-stream\")",
        );
    std::fs::write(work.join("big.py"), script).unwrap();
    let out = cozy(
        &home,
        &work,
        &["--json", "run", "big.py", "--on", "m1", "--out", "saved"],
    );
    assert_eq!(out.code, 0, "{out:?}");
    let events: Vec<Value> = out
        .stdout
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let last = events.last().expect("events");
    assert_eq!(last["type"], "run.completed", "{last}");
    let item = &last["run"]["output"][0];
    let path = Path::new(
        item["path"]
            .as_str()
            .expect("the record names the landed file"),
    );
    let expected: Vec<u8> = (0..=255u8).cycle().take(128 << 20).collect();
    let digest = format!("sha256:{}", hex::encode(Sha256::digest(&expected)));
    assert_eq!(item["sha256"], digest.as_str(), "{item}");
    let held = std::fs::read(path).unwrap();
    assert!(
        held.len() == expected.len() && held == expected,
        "{} holds other bytes",
        path.display()
    );
    assert!(!Path::new(&format!("{}.part", path.display())).exists());
    let landed = events
        .iter()
        .find(|e| e["type"] == "output_item.landed" && e["item"]["rev"] == item["rev"])
        .expect("its landing comes before the final record");
    // It was still being saved when the run completed: the follower waited for it.
    assert!(
        landed["sequence_number"].as_u64() > last["sequence_number"].as_u64(),
        "landed {landed} before completing {last}"
    );
}

/// A run that names no directory puts its outputs in its package's folder, as the Go CLI did
/// (`outputs/<org>-<name>/`, here a script's `local-cozy-script-<hex>`); `--out` names another.
#[test]
fn outputs_land_in_their_packages_folder_unless_the_run_names_one() {
    let Some(home) = Home::new() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    let script = SCRIPT.replace(
        "def main() -> str:\n    return \"hello from a script\"",
        "from cozy_runtime.author import FileAsset, Outputs\n\n\
         def main(*, out: Outputs) -> FileAsset:\n    \
         return out.save_bytes(b\"a small file\", media_type=\"text/plain\")",
    );
    std::fs::write(work.join("small.py"), script).unwrap();
    let landed = |args: &[&str]| -> std::path::PathBuf {
        let run = completed(&cozy(&home, &work, args));
        let path = run["output"][0]["path"].as_str().expect("a landed path");
        assert_eq!(std::fs::read(path).unwrap(), b"a small file");
        Path::new(path).parent().unwrap().to_owned()
    };
    let folder = landed(&["--json", "run", "small.py", "--on", "m1"]);
    let name = folder.file_name().unwrap().to_str().unwrap().to_owned();
    assert_eq!(folder.parent().unwrap(), home.root().join("outputs"));
    assert!(
        name.starts_with("local-cozy-script-") && name.len() == "local-cozy-script-".len() + 24,
        "{}",
        folder.display()
    );
    assert_eq!(
        landed(&["--json", "run", "small.py", "--on", "m1", "--out", "mine"]),
        work.join("mine")
    );
}

/// The `--json` follow's stages (`run.progress`).
fn stages(out: &Out) -> Vec<String> {
    out.stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|e| e["type"] == "run.progress")
        .filter_map(|e| e["progress"]["stage"].as_str().map(str::to_owned))
        .collect()
}

/// The targets of the home's warm runs, oldest first.
fn warm_targets(home: &Home, dir: &Path) -> Vec<String> {
    let listed = cozy(home, dir, &["--json", "run", "list"]);
    let listed: Value = serde_json::from_str(&listed.stdout).unwrap();
    let mut runs: Vec<&Value> = listed["runs"].as_array().unwrap().iter().collect();
    runs.sort_by_key(|r| r["number"].as_u64());
    runs.iter()
        .filter(|r| r["kind"] == "warm")
        .map(|r| r["target"].as_str().unwrap().to_owned())
        .collect()
}

/// Local code sent again unchanged is called at once: no install run, its call accepted within a
/// second of its creation. Its source edited, it is synced, not installed; its
/// lock changed, installed. Owner report: each run of unchanged local/minimax-h3 spent 20 s on
/// an install run and on asking the machine for its 37 files one round trip at a time.
#[test]
fn unchanged_local_code_is_called_at_once_and_an_edit_only_syncs() {
    let Some(home) = Home::new() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny"),
        &work.join("tiny"),
    );
    // As many files as minimax-h3 sends: each one a round trip to ask the machine whether it holds it.
    let data = work.join("tiny/tiny/data");
    std::fs::create_dir_all(&data).unwrap();
    for i in 0..36 {
        std::fs::write(data.join(format!("part{i}.txt")), format!("part {i}\n")).unwrap();
    }
    let call = |text: &str| {
        let out = cozy(
            &home,
            &work,
            &["--json", "run", "./tiny/echo", text, "--on", "m1"],
        );
        let run = completed(&out);
        assert_eq!(run["result"]["text"], text, "{run}");
        let installing: Vec<String> = stages(&out)
            .into_iter()
            .filter(|s| s.contains("install") || s.contains("sync"))
            .collect();
        (run, installing)
    };
    call("first");
    assert_eq!(warm_targets(&home, &work), ["install local/tiny"]);

    let began = std::time::Instant::now();
    let (run, installing) = call("again");
    assert!(installing.is_empty(), "{installing:?}");
    let started = run["accepted_at_ms"].as_i64().unwrap() - run["created_at_ms"].as_i64().unwrap();
    eprintln!(
        "unchanged resubmit: accepted {started} ms after creation, completed in {:?}",
        began.elapsed()
    );
    assert_eq!(
        warm_targets(&home, &work),
        ["install local/tiny"],
        "no install run"
    );
    assert!(started < 1000, "accepted {started} ms after creation");

    let module = work.join("tiny/tiny/__init__.py");
    let source = std::fs::read_to_string(&module).unwrap();
    std::fs::write(
        &module,
        source.replace("Answers its input.", "Answers its input, edited."),
    )
    .unwrap();
    call("edited");
    assert_eq!(
        warm_targets(&home, &work),
        ["install local/tiny", "sync local/tiny (source changed)"]
    );

    let pyproject = work.join("tiny/pyproject.toml");
    let metadata = std::fs::read_to_string(&pyproject).unwrap();
    std::fs::write(
        &pyproject,
        metadata.replace("\"msgspec>=0.19\"", "\"msgspec>=0.19.0\""),
    )
    .unwrap();
    call("relocked");
    assert_eq!(
        warm_targets(&home, &work),
        [
            "install local/tiny",
            "sync local/tiny (source changed)",
            "install local/tiny"
        ]
    );
}

/// Run 10375 showed a bare "queued" while it uploaded its clip. Two runs, each with a 60 MiB
/// file, go to one machine at once: each follower's line says why it waits with what `cozy run
/// list` reads (its upload and its progress, the send, another run's upload it waits for, by
/// number), never a bare `queued`; `--json` carries the same text; and a run past its wait shows
/// no reason in `cozy run show`.
#[test]
fn a_waiting_run_says_why_and_stops_saying_it_once_it_no_longer_waits() {
    let Some(home) = Home::new() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny"),
        &work.join("tiny"),
    );
    let code = work.join("tiny/tiny/__init__.py");
    let measure = "\n\nimport time\n\nfrom cozy_runtime.author import FileAsset\n\n\n\
                   class MeasureInput(msgspec.Struct):\n    document: FileAsset\n\n\n\
                   class MeasureOutput(msgspec.Struct):\n    length: int\n\n\n\
                   @app.entrypoint(accelerator=False)\n\
                   def measure(payload: MeasureInput) -> MeasureOutput:\n    \
                   time.sleep(4)\n    \
                   return MeasureOutput(len(payload.document.read_bytes()))\n";
    let mut text = std::fs::read_to_string(&code).unwrap();
    text.push_str(measure);
    std::fs::write(&code, text).unwrap();
    for (name, seed) in [("a.bin", 1u32), ("b.bin", 2)] {
        let bytes: Vec<u8> = (0..60u32 << 20)
            .map(|i| (i.wrapping_add(seed).wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        std::fs::write(work.join(name), bytes).unwrap();
    }
    let follow = |file: &str, json: bool| {
        let mut args = vec!["run", "./tiny/measure", file, "--on", "m1"];
        if json {
            args.insert(0, "--json");
        }
        home.command()
            .current_dir(&work)
            .args(args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap()
    };
    let (mut person, mut machine) = (follow("./a.bin", false), follow("./b.bin", true));
    // While one runs past its wait, `run show` says no reason.
    let (mut held_shown, mut seen) = (false, Value::Null);
    let ended = |child: &mut std::process::Child| child.try_wait().unwrap().is_some();
    while !held_shown && !(ended(&mut person) && ended(&mut machine)) {
        seen = home.json(["run", "list", "--json"]);
        for run in seen["runs"].as_array().into_iter().flatten() {
            let measuring = run["target"]
                .as_str()
                .is_some_and(|t| t.ends_with("/measure"));
            let running = run["status"] == "in_progress"
                && run["machine_state"]
                    .as_str()
                    .is_some_and(|s| s != "queued" && s != "preparing");
            if measuring && running {
                let number = run["number"].to_string();
                let shown = cozy(&home, &work, &["run", "show", &number]);
                assert!(!shown.stdout.contains("waiting"), "{shown:?}");
                held_shown = true;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let person = common::out(person.wait_with_output().unwrap());
    let machine = common::out(machine.wait_with_output().unwrap());
    assert!(
        held_shown,
        "no run was seen running: {seen}\n{person:?}\n{machine:?}"
    );
    assert_eq!(
        (person.code, machine.code),
        (0, 0),
        "{person:?}\n{machine:?}"
    );
    let said: Vec<&str> = person.stderr.lines().map(str::trim).collect();
    let reasons: Vec<String> = machine
        .stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|e| e["type"] == "run.waiting")
        .filter_map(|e| e["reason"].as_str().map(str::to_owned))
        .collect();
    // Each upload says so with its progress, in the person's line and in the events alike.
    let uploading =
        |line: &str| line.starts_with("uploading document: ") && line.contains("/60.0 MiB");
    assert!(said.iter().any(|l| uploading(l)), "{said:?}");
    assert!(reasons.iter().any(|r| uploading(r)), "{reasons:?}");
    assert!(!said.contains(&"queued"), "a bare queued: {said:?}");
}

/// What a run's executor served other than as sent reaches its record: the machine keeps the
/// executor's rows on the outcome, Calcifer on the run, and `cozy run` prints each, `--json`
/// carries them (Runtime 0.27.0 sends each row's kind and source).
#[test]
fn a_runs_adjustments_reach_its_record() {
    let Some(home) = Home::new() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny-adjusted"),
        &work.join("tiny-adjusted"),
    );

    let run = completed(&cozy(
        &home,
        &work,
        &["--json", "run", "./tiny-adjusted/steps", "--on", "m1"],
    ));
    assert_eq!(run["result"]["steps"], 30, "{run}");
    assert_eq!(
        run["adjustments"],
        serde_json::json!([
            {"field": "steps", "kind": "clamp", "requested": "50", "applied": "30",
             "reason": "at most 30 steps", "source": "handler"},
            {"field": "seed", "kind": "warn", "requested": "", "applied": "",
             "reason": "none given; the result is not reproducible", "source": "handler"},
        ]),
        "{run}"
    );

    let shown = cozy(
        &home,
        &work,
        &["run", "./tiny-adjusted/steps", "--on", "m1"],
    );
    assert_eq!(shown.code, 0, "{shown:?}");
    eprintln!("{}", shown.stdout);
    for line in [
        "clamp steps: 50 -> 30 (at most 30 steps) [handler]",
        "warn seed: none given; the result is not reproducible [handler]",
    ] {
        assert!(shown.stdout.contains(line), "{line:?} not in {shown:?}");
    }
}

/// `--attention-kernel` reaches the executor: the run records the pin it was sent, `run show`
/// says it, and a pin no call could apply (this function has no attention) warns in its log. A
/// malformed pin is refused before anything is sent.
#[test]
fn an_attention_kernel_pin_reaches_the_executor_and_run_show() {
    let Some(home) = Home::new() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny-adjusted"),
        &work.join("tiny-adjusted"),
    );
    let pin = "model/unet=sdpa";
    let steps = [
        "run",
        "./tiny-adjusted/steps",
        "--on",
        "m1",
        "--attention-kernel",
    ];

    let run = completed(&cozy(
        &home,
        &work,
        &[&["--json"], &steps[..], &[pin]].concat(),
    ));
    let id = run["id"].as_str().unwrap();
    let full = cozy(&home, &work, &["--json", "run", "show", id]);
    let full: Value = serde_json::from_str(&full.stdout).unwrap_or_else(|_| panic!("{full:?}"));
    // A Runtime before 0.28 names its GPUs `ranks`.
    let execution = &full["measurements"]["execution"];
    let first = execution["gpus"]
        .get(0)
        .or_else(|| execution["ranks"].get(0));
    assert_eq!(
        first.unwrap_or_else(|| panic!("{full}"))["attention"]["requested"],
        pin,
        "{full}"
    );

    let shown = cozy(&home, &work, &["run", "show", id]);
    eprintln!("{}", shown.stdout);
    let said = |line: &str| {
        shown
            .stdout
            .lines()
            .any(|l| l.contains(line) && l.contains(pin))
    };
    assert!(said("attention kernel"), "{shown:?}");
    assert!(said("was never applied"), "{shown:?}");

    let refused = cozy(&home, &work, &[&steps[..], &["unet=a=b"]].concat());
    assert_eq!(refused.code, 2, "{refused:?}");
    assert!(
        refused.stderr.contains("[SLOT/]COMPONENT=KERNEL"),
        "{refused:?}"
    );
}

/// `--asset-fidelity` puts its hint on the attached asset its label or index names, and the
/// package reads it from the input its machine received; one naming no attached asset is refused
/// before the run is submitted.
#[test]
fn asset_fidelity_reaches_the_package_by_label_or_index() {
    let Some(home) = Home::new() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny"),
        &work.join("tiny"),
    );
    let code = work.join("tiny/tiny/__init__.py");
    let fidelity = "\n\nfrom cozy_runtime.author import Assets, ImageAsset\n\n\n\
                    class FidelityInput(msgspec.Struct):\n    note: str = \"\"\n\n\n\
                    class FidelityOutput(msgspec.Struct):\n    fidelity: dict[str, str]\n\n\n\
                    @app.entrypoint(accelerator=False)\n\
                    def fidelity(payload: FidelityInput, assets: Assets[ImageAsset]) -> FidelityOutput:\n    \
                    infos = [assets.info(i) for i in range(len(assets))]\n    \
                    return FidelityOutput({i.label or str(i.position): i.fidelity for i in infos})\n";
    let mut text = std::fs::read_to_string(&code).unwrap();
    text.push_str(fidelity);
    std::fs::write(&code, text).unwrap();
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/media/azul.png"),
        work.join("azul.png"),
    )
    .unwrap();
    let run = |second: &str| {
        let args = [
            "--json",
            "run",
            "./tiny/fidelity",
            "--asset",
            "woman=./azul.png",
            "--asset",
            "./azul.png",
            "--asset",
            "./azul.png",
            "--asset-fidelity",
            "woman=high",
            "--asset-fidelity",
            second,
            "--on",
            "m1",
        ];
        cozy(&home, &work, &args)
    };

    let done = completed(&run("1=low"));
    assert_eq!(
        done["result"]["fidelity"],
        serde_json::json!({"woman": "high", "1": "low", "2": "auto"}),
        "{done}"
    );

    let refused = run("3=low");
    assert_eq!(refused.code, 2, "{refused:?}");
    assert!(
        refused.stdout.contains("asset_fidelity_unmatched"),
        "{refused:?}"
    );
}

/// A local directory given to a Tree input reaches the package whole: the CLI sends a TensorFS
/// manifest naming its files with them, and the machine materializes the tree from it. Sent again,
/// the machine already holds it.
#[test]
fn a_local_directory_reaches_a_tree_input_whole() {
    let Some(home) = Home::new() else { return };
    if std::env::var_os("COZY_TEST_TENSORD").is_none() {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return;
    }
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny"),
        &work.join("tiny"),
    );
    let code = work.join("tiny/tiny/__init__.py");
    let tree = "\n\nfrom cozy_runtime.author import Tree\n\n\n\
                class TreeInput(msgspec.Struct):\n    constants: Tree\n\n\n\
                class TreeOutput(msgspec.Struct):\n    files: dict[str, str]\n\n\n\
                @app.entrypoint(accelerator=False)\n\
                def tree(payload: TreeInput) -> TreeOutput:\n    \
                root = payload.constants.path\n    \
                found = sorted(p for p in root.rglob(\"*\") if p.is_file())\n    \
                return TreeOutput({p.relative_to(root).as_posix(): p.read_text() for p in found})\n";
    let mut text = std::fs::read_to_string(&code).unwrap();
    text.push_str(tree);
    std::fs::write(&code, text).unwrap();
    std::fs::create_dir_all(work.join("constants/sub")).unwrap();
    std::fs::write(work.join("constants/pos_emb.pt"), "positive").unwrap();
    std::fs::write(work.join("constants/sub/neg_emb.pt"), "negative").unwrap();
    std::fs::write(work.join("constants/sub/empty.pt"), "").unwrap();

    let args = [
        "--json",
        "run",
        "./tiny/tree",
        "constants=./constants",
        "--on",
        "m1",
    ];
    let first = cozy(&home, &work, &args);
    let run = completed(&first);
    let files = serde_json::json!({
        "pos_emb.pt": "positive", "sub/empty.pt": "", "sub/neg_emb.pt": "negative",
    });
    assert_eq!(run["result"]["files"], files, "{run}");

    let again = cozy(&home, &work, &args);
    assert_eq!(completed(&again)["result"]["files"], files, "{again:?}");
    let reused = again
        .stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .any(|e| {
            e["type"] == "run.log"
                && e["text"]
                    .as_str()
                    .is_some_and(|t| t.starts_with("reused constants"))
        });
    assert!(reused, "{again:?}");
}

/// Local code whose subpackage has empty `__init__.py` files runs on another machine: zero-byte
/// files are sent like any other (run 11762's seedvr2 install lacked its ten).
#[test]
fn local_code_with_empty_files_runs_on_another_machine() {
    let Some(home) = Home::new() else { return };
    if std::env::var_os("COZY_TEST_TENSORD").is_none() {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return;
    }
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny"),
        &work.join("tiny"),
    );
    let code = work.join("tiny/tiny");
    let inner = code.join("upstream/inner");
    std::fs::create_dir_all(&inner).unwrap();
    std::fs::write(code.join("upstream/__init__.py"), "").unwrap();
    std::fs::write(inner.join("__init__.py"), "").unwrap();
    std::fs::write(inner.join("value.py"), "VALUE = \"deep\"\n").unwrap();
    let mut text = std::fs::read_to_string(code.join("__init__.py")).unwrap();
    text.push_str(
        "\n\nfrom .upstream.inner.value import VALUE  # noqa: E402\n\n\n\
         @app.entrypoint(accelerator=False)\n\
         def deep(payload: EchoInput) -> EchoOutput:\n    \
         return EchoOutput(VALUE)\n",
    );
    std::fs::write(code.join("__init__.py"), text).unwrap();

    let run = completed(&cozy(
        &home,
        &work,
        &["--json", "run", "./tiny/deep", "--on", "m1"],
    ));
    assert_eq!(run["result"]["text"], "deep", "{run}");
}

/// Counts the bytes sent to `target` through it: a TCP relay a machine is added behind.
struct Relay {
    address: String,
    sent: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl Relay {
    fn to(target: &str) -> Self {
        use std::{
            io::{Read as _, Write as _},
            net::{Shutdown, TcpListener, TcpStream},
            sync::{
                Arc,
                atomic::{AtomicU64, Ordering},
            },
        };
        fn pump(mut from: TcpStream, mut to: TcpStream, count: Option<Arc<AtomicU64>>) {
            let mut buffer = vec![0; 64 << 10];
            while let Ok(read @ 1..) = from.read(&mut buffer) {
                if to.write_all(&buffer[..read]).is_err() {
                    break;
                }
                if let Some(count) = &count {
                    count.fetch_add(read as u64, Ordering::Relaxed);
                }
            }
            let _ = to.shutdown(Shutdown::Write);
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let sent = Arc::new(AtomicU64::new(0));
        let (target, counted) = (target.to_owned(), sent.clone());
        std::thread::spawn(move || {
            for client in listener.incoming().flatten() {
                let Ok(machine) = TcpStream::connect(&target) else {
                    continue;
                };
                let (back, forth) = (machine.try_clone().unwrap(), client.try_clone().unwrap());
                let counted = counted.clone();
                std::thread::spawn(move || pump(client, machine, Some(counted)));
                std::thread::spawn(move || pump(back, forth, None));
            }
        });
        Self { address, sent }
    }

    fn sent(&self) -> u64 {
        self.sent.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Owner: "inputs that were already uploaded shouldn't be uploaded again by the client". A second
/// run with the same video and image to the same machine sends none of their bytes (counted on
/// the way to the machine) and says it reused them; the client read each file once, its digest
/// remembered by the file's path, length and time.
#[test]
fn a_second_run_reuses_the_video_and_image_its_machine_holds() {
    let Some(home) = Home::new() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut machine =
        common::tensord::Tensord::start_on(Some(port), &[common::tensord::stranger()], &[]);
    let relay = Relay::to(&machine.address);
    let fingerprint = machine.fingerprint.clone();
    let add = || {
        home.cozy([
            "machine",
            "add",
            "m1",
            &relay.address,
            "--fingerprint",
            &fingerprint,
            "--json",
        ])
    };
    machine.restart(&[common::tensord::key_to_authorize(&add())]);
    let added = add();
    assert_eq!(added.code, 0, "{added:?}");
    common::tensord::on_named_runtime(&home, "m1");

    let work = home.path().join("work");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny"),
        &work.join("tiny"),
    );
    let code = work.join("tiny/tiny/__init__.py");
    let reuse = "\n\nfrom cozy_runtime.author import FileAsset\n\n\n\
                 class ReuseInput(msgspec.Struct):\n    video: FileAsset\n    image: FileAsset\n\n\n\
                 class ReuseOutput(msgspec.Struct):\n    video: int\n    image: int\n\n\n\
                 @app.entrypoint(accelerator=False)\n\
                 def reuse(payload: ReuseInput) -> ReuseOutput:\n    \
                 return ReuseOutput(len(payload.video.read_bytes()), len(payload.image.read_bytes()))\n";
    let mut text = std::fs::read_to_string(&code).unwrap();
    text.push_str(reuse);
    std::fs::write(&code, text).unwrap();
    let video: Vec<u8> = (0..16u32 << 20)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    std::fs::write(work.join("clip.mp4"), &video).unwrap();
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/media/azul.png"),
        work.join("azul.png"),
    )
    .unwrap();

    let run = || {
        let before = relay.sent();
        let args = [
            "--json",
            "run",
            "./tiny/reuse",
            "--asset",
            "video=./clip.mp4",
            "--asset",
            "image=./azul.png",
            "--on",
            "m1",
        ];
        let out = cozy(&home, &work, &args);
        let run = completed(&out);
        assert_eq!(run["result"]["video"], 16 << 20, "{run}");
        let said: Vec<String> = out
            .stdout
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter_map(|e| match e["type"].as_str() {
                Some("run.waiting") => e["reason"].as_str().map(str::to_owned),
                Some("run.log") => {
                    Some(format!("{}: {}", e["level"].as_str()?, e["text"].as_str()?))
                }
                _ => None,
            })
            .collect();
        (run["number"].to_string(), said, relay.sent() - before)
    };
    let (_, first, sent) = run();
    assert!(
        sent > 16 << 20,
        "the first run sent {sent} bytes: {first:?}"
    );

    let (number, second, sent) = run();
    assert!(
        sent < 1 << 20,
        "the second run sent {sent} bytes, its files among them: {second:?}"
    );
    assert!(
        !second.iter().any(|r| r.starts_with("uploading ")),
        "{second:?}"
    );
    // The run's log keeps it: its follower and `run show` say it, however late they look.
    let reused = "inputs: reused image, video (16.0 MiB): m1 already holds them";
    assert!(second.iter().any(|r| r == reused), "{second:?}");
    let shown = cozy(&home, &work, &["run", "show", &number]);
    assert!(shown.stdout.contains(reused), "{shown:?}");
    // Each file was read once on this computer: its digest is remembered.
    let remembered = std::fs::read_dir(home.root().join("cache/digests"))
        .unwrap()
        .count();
    assert_eq!(remembered, 2);
}
