//! `cozy package lock|publish|yank` through the real binary against a real Tensorhub: Tensorhub's
//! dev-build `cozy_ref` job starts one (TENSORHUB_ORIGIN, TENSORHUB_LOG) with MinIO, and uv is on the
//! runner. Elsewhere it skips. One test, two accounts: the Hub allows a sign-up a minute.
mod common;

use std::path::{Path, PathBuf};

use common::{Home, Out};
use serde_json::Value;

/// A home signed in to the CI Hub as `email`, and its account.
fn publisher(email: &str) -> Option<(Home, String)> {
    let home = Home::with_hub()?;
    let account = home.sign_in(email);
    Some((home, account))
}

/// A fixture project copied into the home, where `cozy` runs on it.
fn project(home: &Home, fixture: &str) -> PathBuf {
    let from = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/package")
        .join(fixture);
    let to = home.path().join("src").join(fixture);
    copy(&from, &to);
    to
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

fn cozy(home: &Home, dir: &Path, args: &[&str]) -> Out {
    common::out(home.command().current_dir(dir).args(args).output().unwrap())
}

fn json(home: &Home, dir: &Path, args: &[&str]) -> Value {
    let out = cozy(home, dir, &[&["--json"], args].concat());
    assert_eq!(out.code, 0, "{args:?}: {out:?}");
    serde_json::from_str(&out.stdout).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}

fn refused(home: &Home, dir: &Path, args: &[&str]) -> Value {
    let out = cozy(home, dir, &[&["--json"], args].concat());
    assert_eq!(out.code, 1, "{args:?}: {out:?}");
    out.refusal()
}

fn read(path: PathBuf) -> String {
    std::fs::read_to_string(path).unwrap()
}

/// Points `home` at its Hub's listen address (`TENSORHUB_URL`, plain HTTP on loopback) instead of
/// its public origin, keeping the device key it signed in with. Answers that address.
fn at_listen_address(home: &Home) -> String {
    let listen = std::env::var("TENSORHUB_URL").expect("the Tensorhub's listen address");
    let paths = cozy_platform::paths::Home::at(home.root());
    let hub = |url: &str| cozy_proto::config::Hub {
        name: Some("ci".into()),
        origin: cozy_proto::config::origin(url).unwrap(),
    };
    let public = std::env::var("TENSORHUB_ORIGIN").unwrap();
    std::fs::copy(
        hub(&public).device_key(&paths),
        hub(&listen).device_key(&paths),
    )
    .unwrap();
    home.write_config(&format!("hub = \"ci\"\n[hubs]\nci = \"{listen}\"\n"));
    listen
}

/// The uv.lock the Hub holds for `account/name@release`, as a machine's install plan carries it.
fn published_lock(url: &str, account: &str, name: &str, release: &str) -> String {
    use base64::Engine as _;
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let plan = runtime.block_on(async {
        let hub = hub_client::Client::connect(
            url,
            reqwest::Client::new(),
            hub_client::Credential::Anonymous,
        );
        let empty = hub_client::types::Empty(Default::default());
        hub.get_package_download_plan(account, name, None, Some(release), &empty)
            .await
    });
    let plan = plan.expect("the release's install plan").into_inner();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&plan.uv_lock.canonical_bytes)
        .unwrap();
    String::from_utf8(bytes).unwrap()
}

/// - What must never ship is refused, and so is a project that declares the account index itself.
/// - A release publishes once: the second publish uploads nothing.
/// - A package naming its author's packages through `{ index = "tensorhub" }` locks against that
///   account's index on this Hub without its authored files changing, and publishes.
/// - Another account publishes the same tree: its lock is rebound to that account's index, every
///   version kept, a newer release there included (minimax-h3 1.41.2 was refused when uv
///   re-resolved its describer to the newest); a pin that account does not publish is refused
///   by name.
/// - Addressed where it listens rather than at its public origin, the Hub's index is still the
///   public one: `lock` binds it, and `publish` relocks a lock made against the listen address.
/// - A yank is idempotent and shows in `info`.
#[test]
fn packages_lock_publish_and_yank_on_a_real_hub() {
    let Some((home, account)) = publisher("publisher@cozy.test") else {
        return;
    };
    let url = std::env::var("TENSORHUB_ORIGIN").unwrap();
    let tiny = project(&home, "tiny");
    let user = project(&home, "tiny-user");

    std::fs::write(tiny.join(".env"), "TOKEN=x\n").unwrap();
    assert_eq!(
        refused(&home, &tiny, &["package", "publish"])["code"],
        "package_source_refused"
    );
    std::fs::remove_file(tiny.join(".env")).unwrap();
    let authored = read(user.join("pyproject.toml"));
    let declared = format!(
        "{authored}\n[[tool.uv.index]]\nname = \"tensorhub\"\nurl = \"https://x.invalid/simple/\"\n"
    );
    std::fs::write(user.join("pyproject.toml"), declared).unwrap();
    assert_eq!(
        refused(&home, &user, &["package", "lock"])["code"],
        "account_index_declared"
    );
    std::fs::write(user.join("pyproject.toml"), &authored).unwrap();

    let locked = json(&home, &tiny, &["package", "lock"]);
    assert_eq!(locked["changed"], true, "{locked}");
    assert!(locked["account_index"].is_null(), "{locked}");
    assert_eq!(json(&home, &tiny, &["package", "lock"])["changed"], false);
    let published = json(&home, &tiny, &["package", "publish"]);
    assert_eq!(
        published["package"],
        format!("{account}/tiny"),
        "{published}"
    );
    // Bytes another release already stored (here, another test's identical `tiny`) never upload
    // again, so a first publish may send nothing.
    assert_eq!(published["status"], "published", "{published}");
    let again = json(&home, &tiny, &["package", "publish"]);
    assert_eq!(again["status"], "already published", "{again}");
    assert_eq!(again["uploaded_bytes"], 0, "{again}");

    let locked = json(&home, &user, &["package", "lock"]);
    let index = format!("{}/v1/index/{account}/simple/", url.trim_end_matches('/'));
    assert_eq!(locked["account_index"], index.as_str(), "{locked}");
    assert!(read(user.join("uv.lock")).contains(&index));
    assert_eq!(read(user.join("pyproject.toml")), authored);
    let published = json(&home, &user, &["package", "publish"]);
    assert_eq!(
        published["package"],
        format!("{account}/tiny-user"),
        "{published}"
    );
    let info = json(
        &home,
        &user,
        &["package", "info", &format!("{account}/Tiny-User")],
    );
    assert_eq!(info["releases"][0]["release"], "0.1.0", "{info}");

    // The same trees, locked for the first account, published by a second.
    let Some((second, other)) = publisher("second@cozy.test") else {
        return;
    };
    let first_lock = |fixture: &str| home.path().join("src").join(fixture).join("uv.lock");
    let user_two = project(&second, "tiny-user");
    std::fs::copy(first_lock("tiny-user"), user_two.join("uv.lock")).unwrap();
    let unmet = refused(&second, &user_two, &["package", "publish"]);
    assert_eq!(unmet["code"], "account_index_pin_unpublished", "{unmet}");
    assert!(
        unmet["message"]
            .as_str()
            .unwrap_or_default()
            .contains(&format!("{other}/tiny 0.1.0")),
        "{unmet}"
    );
    let tiny_two = project(&second, "tiny");
    std::fs::copy(first_lock("tiny"), tiny_two.join("uv.lock")).unwrap();
    let published = json(&second, &tiny_two, &["package", "publish"]);
    assert_eq!(published["package"], format!("{other}/tiny"), "{published}");
    assert_eq!(published["status"], "published", "{published}");
    let newer =
        read(tiny_two.join("pyproject.toml")).replace("version = \"0.1.0\"", "version = \"0.2.0\"");
    std::fs::write(tiny_two.join("pyproject.toml"), newer).unwrap();
    json(&second, &tiny_two, &["package", "lock"]);
    assert_eq!(
        json(&second, &tiny_two, &["package", "publish"])["status"],
        "published"
    );
    let published = json(&second, &user_two, &["package", "publish"]);
    assert_eq!(
        published["package"],
        format!("{other}/tiny-user"),
        "{published}"
    );
    assert_eq!(published["status"], "published", "{published}");
    let newer = format!("{other}/tiny 0.2.0 is newer than the 0.1.0 this lock pins");
    assert!(
        published["warnings"]
            .as_array()
            .is_some_and(|w| w.iter().any(|w| w == newer.as_str())),
        "{published}"
    );
    let lock = published_lock(&url, &other, "tiny-user", "0.1.0");
    assert!(
        lock.contains("/tiny/0.1.0/") && !lock.contains("/tiny/0.2.0/"),
        "{lock}"
    );

    // The first account addressing its Hub where the Hub listens (TENSORHUB_URL), as a laptop names
    // its local Hub: the index it locks against is still the one at the Hub's public origin, and
    // a lock made against the listen address is relocked to it on publish.
    let listen = at_listen_address(&home);
    let pyproject = authored.replace("version = \"0.1.0\"", "version = \"0.2.0\"");
    std::fs::write(user.join("pyproject.toml"), &pyproject).unwrap();
    let locked = json(&home, &user, &["package", "lock"]);
    assert_eq!(locked["account_index"], index.as_str(), "{locked}");
    let public = url.trim_end_matches('/');
    let elsewhere = read(user.join("uv.lock")).replace(public, listen.trim_end_matches('/'));
    assert!(elsewhere.contains(&format!("{listen}/v1/index/{account}/simple/")));
    std::fs::write(user.join("uv.lock"), &elsewhere).unwrap();
    let published = json(&home, &user, &["package", "publish"]);
    assert_eq!(published["status"], "published", "{published}");
    let lock = published_lock(&url, &account, "tiny-user", "0.2.0");
    assert!(
        lock.contains(&index) && !lock.contains(listen.as_str()),
        "{lock}"
    );

    let release = format!("{account}/tiny@0.1.0");
    assert_eq!(
        json(&home, &tiny, &["package", "yank", &release])["changed"],
        true
    );
    assert_eq!(
        json(&home, &tiny, &["package", "yank", &release])["changed"],
        false
    );
    let info = json(
        &home,
        &tiny,
        &["package", "info", &format!("{account}/tiny")],
    );
    assert_eq!(info["releases"][0]["yanked"], true, "{info}");
}
