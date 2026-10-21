//! Runs that complete on a real machine: the job's Tensorhub (Tensorhub's dev-build `cozy_ref`
//! job: TENSORHUB_URL, TENSORHUB_LOG) holds just-published torch-free packages, and TensorD from
//! the cozy-runtime wheel (COZY_TEST_TENSORD) installs and runs them. Elsewhere it skips.
#![cfg(unix)]
mod common;

use std::{
    io::BufRead,
    path::{Path, PathBuf},
    process::Stdio,
};

use common::Home;
use serde_json::Value;
use sha2::{Digest as _, Sha256};

/// Locks and publishes a fixture package from a copy in the home, as its author would.
fn publish(home: &Home, fixture: &str) {
    let from = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/package")
        .join(fixture);
    let project = home.path().join("src").join(fixture);
    copy(&from, &project);
    for verb in ["lock", "publish"] {
        let out = common::out(
            home.command()
                .current_dir(&project)
                .args(["--json", "package", verb])
                .output()
                .unwrap(),
        );
        assert_eq!(out.code, 0, "package {verb} {fixture}: {out:?}");
    }
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

/// The events a followed `--json` run printed.
fn events(out: &common::Out) -> Vec<Value> {
    out.stdout
        .lines()
        .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}")))
        .collect()
}

/// Reads NDJSON lines until one satisfies `want`.
fn until(reader: &mut impl BufRead, want: impl Fn(&Value) -> bool) -> Value {
    loop {
        let mut line = String::new();
        assert!(reader.read_line(&mut line).unwrap() > 0, "stdout closed");
        let Ok(event) = serde_json::from_str::<Value>(&line) else {
            let mut rest = String::new();
            let _ = reader.read_to_string(&mut rest);
            panic!("not an event: {line}{rest}");
        };
        if want(&event) {
            return event;
        }
    }
}

/// Every file under `dir`.
fn files(dir: &Path) -> Vec<PathBuf> {
    let mut out = vec![];
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        match entry.file_type().unwrap().is_dir() {
            true => out.extend(files(&entry.path())),
            false => out.push(entry.path()),
        }
    }
    out
}

/// One account (the Hub paces sign-ups), two packages:
/// - a published function runs to completion on its machine and its result comes back; a field
///   its interface lacks is dropped with a warning, so the package's strict struct never sees it;
/// - an output the machine writes lands once in `--out`, its bytes the digest the run records,
///   though Calcifer stopped while the run went on.
#[test]
fn published_functions_run_to_completion_on_a_real_machine() {
    let Some(home) = Home::with_hub() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    let account = home.sign_in("runner@cozy.test");
    publish(&home, "tiny");
    publish(&home, "tiny-output");
    let _machine = common::tensord::added(&home, "m1");

    let echo = format!("{account}/tiny/echo");
    let out = home.cozy([
        "--json",
        "run",
        &echo,
        "hi there",
        "mood=calm",
        "--on",
        "m1",
    ]);
    assert_eq!(out.code, 0, "{out:?}");
    assert!(out.stderr.contains("ignored unknown field mood"), "{out:?}");
    let seen = events(&out);
    let end = seen
        .iter()
        .rev()
        .find(|e| e["type"] == "run.completed")
        .unwrap_or_else(|| panic!("{seen:?}"));
    let run = &end["run"];
    assert_eq!(
        (run["machine"].as_str(), run["result"]["text"].as_str()),
        (Some("m1"), Some("hi there")),
        "{run}"
    );
    let number = run["number"].as_u64().unwrap().to_string();
    let human = home.cozy(["run", &echo, "again", "--on", "m1"]);
    assert_eq!(human.code, 0, "{human:?}");
    assert!(
        human.stdout.contains("completed") && human.stdout.contains("again"),
        "{human:?}"
    );
    let shown = home.json(["run", "show", &number, "--json"]);
    assert_eq!(
        (
            shown["status"].as_str(),
            shown["warnings"][0]["code"].as_str()
        ),
        (Some("completed"), Some("request_fields_ignored"))
    );

    // The writer waits on its machine while Calcifer stops; the follower starts it again and
    // reattaches, and the file lands once.
    let landing = home.path().join("landing");
    let write = format!("{account}/tiny-output/write");
    let text = "a file the machine wrote";
    let mut follower = home
        .command()
        .args([
            "--json",
            "run",
            &write,
            &format!("text={text}"),
            "seconds=8",
            "--on",
            "m1",
            "--out",
        ])
        .arg(&landing)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = std::io::BufReader::new(follower.stdout.take().unwrap());
    let accepted = until(&mut stdout, |e| e["type"] == "run.in_progress");
    let number = accepted["run"]["number"].as_u64().unwrap().to_string();
    let down = home.cozy(["down"]);
    assert_eq!(down.code, 0, "{down:?}");
    let ended = until(&mut stdout, |e| {
        e.get("type").and_then(Value::as_str).is_some_and(|t| {
            t.starts_with("run.") && ["run.completed", "run.failed", "run.canceled"].contains(&t)
        })
    });
    assert_eq!(ended["type"], "run.completed", "{ended}");
    assert_eq!(follower.wait().unwrap().code(), Some(0));

    let shown = home.json(["run", "show", &number, "--json"]);
    let item = &shown["output"][0];
    assert_eq!(
        (item["name"].as_str(), item["status"].as_str()),
        (Some("file"), Some("completed")),
        "{shown}"
    );
    let landed = files(&landing);
    assert_eq!(landed.len(), 1, "the output landed once: {landed:?}");
    assert_eq!(
        item["path"].as_str().map(PathBuf::from),
        Some(landed[0].clone()),
        "{shown}"
    );
    let bytes = std::fs::read(&landed[0]).unwrap();
    assert_eq!(bytes, text.as_bytes());
    assert_eq!(
        item["sha256"].as_str(),
        Some(format!("sha256:{}", hex::encode(Sha256::digest(&bytes))).as_str()),
        "{shown}"
    );
}

/// Canonical metadata reaches the real Hub through an ordinary owner's Endpoint. This case
/// needs no TensorD, worker, catalog, package, or rental; its only objects are this tiny tree.
#[test]
fn canonical_tags_round_trip_at_a_real_hub_without_a_machine() {
    use std::sync::Arc;
    use tensorfs::{
        fetch::{
            Client, Config, Endpoint, Fetcher, Fresh, ObjectStore, Origin, SourcePolicy, Want,
            bearer, upload_tree, with_tag,
        },
        format::{Dtype, Tree},
        store::Store,
        writer::{Data, Source},
    };

    let Some(home) = Home::with_hub() else { return };
    let account = home.sign_in("canonical-tags@cozy.test");
    let name = format!("{account}/canonical");
    let url = std::env::var("TENSORHUB_ORIGIN").unwrap();
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
    let owner = Arc::new(Endpoint::new(&client, &url, Some(bearer(home.token()))).unwrap());
    let anonymous = Arc::new(Endpoint::new(&client, &url, None).unwrap().anonymous());
    let source = Store::open(home.path().join("canonical-source")).unwrap();
    let bytes: Vec<u8> = (0..300_000u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 11) as u8)
        .collect();
    let mut writer = source.writer(None, None).unwrap();
    let mut model = writer
        .component("model", "conformance.Fixture", Default::default(), None)
        .unwrap();
    model
        .put(
            "w",
            &Dtype::U8,
            &[bytes.len() as u64],
            Data::Plain(Source::Read(&mut &bytes[..])),
        )
        .unwrap();
    let header = writer.commit().unwrap();
    let mut tree = Tree::default();
    tree.set_variant("int8", header.clone());
    let first = source.put_tree(&tree).unwrap();
    tree.files.insert(
        "README.md".into(),
        source.put_bytes(b"a second commit").unwrap(),
    );
    let second = source.put_tree(&tree).unwrap();
    let replay = hex::encode(Sha256::digest(b"a stable private replay identity"));

    tokio::runtime::Runtime::new().unwrap().block_on(async {
        upload_tree(&owner, &name, &first, &source, "main")
            .await
            .unwrap();
        let (private, etag) = owner.repo(&name).await.unwrap().unwrap();
        assert_eq!(private.tags.len(), 1);
        let main = private.tag("main").unwrap();
        assert_eq!(main.commit, first);
        assert!(!main.public && !main.yanked);
        assert!(
            anonymous.repo(&name).await.unwrap().is_none(),
            "private refs stay private"
        );
        assert_eq!(
            owner.repo_since(&name, Some(&etag)).await.unwrap(),
            Fresh::Unchanged
        );

        owner
            .update_repo(&name, |old| Ok(Some(with_tag(old, &name, &replay, &first))))
            .await
            .unwrap();
        let (named, named_etag) = owner.repo(&name).await.unwrap().unwrap();
        assert_eq!(named.tag(&replay).unwrap().commit, first);
        assert!(!named.tag(&replay).unwrap().public);
        owner.update_repo(&name, Ok).await.unwrap();
        assert_eq!(
            owner.repo(&name).await.unwrap().unwrap(),
            (named.clone(), named_etag.clone()),
            "an unchanged CAS preserves its ETag"
        );
        assert!(
            !owner
                .put_repo_bytes(&name, private.to_json().unwrap(), Some(&etag))
                .await
                .unwrap(),
            "a stale ETag cannot erase the replay tag"
        );

        upload_tree(&owner, &name, &second, &source, "1.0")
            .await
            .unwrap();
        owner
            .update_repo(&name, |old| {
                let mut repo = old.unwrap();
                repo.tag_mut("1.0").unwrap().public = true;
                repo.set_tag("2.0rc1", first.clone());
                repo.tag_mut("2.0rc1").unwrap().public = true;
                repo.set_tag("3.0", first.clone());
                let yanked = repo.tag_mut("3.0").unwrap();
                yanked.public = true;
                yanked.yanked = true;
                Ok(Some(repo))
            })
            .await
            .unwrap();
        let (public, _) = anonymous.repo(&name).await.unwrap().unwrap();
        assert!(public.tags.iter().all(|tag| tag.public));
        assert!(public.tag("main").is_none() && public.tag(&replay).is_none());
        assert_eq!(
            public.default_tag().unwrap().name,
            "1.0",
            "final versions precede prereleases; yanked tags are excluded"
        );
        let (owned, _) = owner.repo(&name).await.unwrap().unwrap();
        assert_eq!(
            owned.tag(&replay),
            named.tag(&replay),
            "publication preserves private replay identity and flags"
        );

        let into = Store::open(home.path().join("canonical-pulled")).unwrap();
        let fetcher = Fetcher::new(
            Arc::new(into.clone()) as Arc<dyn ObjectStore>,
            client.clone(),
            None,
            Config::default(),
        );
        let public_origin: Arc<dyn Origin> = anonymous;
        for revision in [None, Some(second.id()), Some(header.id())] {
            let pulled = fetcher
                .pull(
                    &public_origin,
                    &name,
                    revision.as_deref(),
                    Some("int8"),
                    &Want::default(),
                )
                .await
                .unwrap();
            assert_eq!(pulled.tree.as_ref(), Some(&second));
            assert_eq!(pulled.header, header);
            assert_eq!(pulled.variant, "int8");
        }
        fetcher.settled().await;
        assert_eq!(
            into.checkpoint(&header)
                .unwrap()
                .read("model", "w")
                .unwrap(),
            bytes
        );
        let owner_origin: Arc<dyn Origin> = owner;
        let private = fetcher
            .pull(
                &owner_origin,
                &name,
                Some(&replay),
                Some("int8"),
                &Want::default(),
            )
            .await
            .unwrap();
        assert_eq!(private.tree, Some(first));
        assert_eq!(private.header, header);
    });
}
