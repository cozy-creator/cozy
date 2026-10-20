//! Release cards this computer keeps. A release never changes, so its card is read from the Hub
//! once and kept forever, by release; which release a run binds and runs is resolved before
//! (`super::resolve`), never guessed from what was read last.
use std::path::{Path, PathBuf};

use platform::{durable, paths::Home};
use protocol::config::Hub;
use protocol::error::{Error, Kind};
use hub_client::types;
use progenitor_client::{ClientHooks as _, ClientInfo as _, OperationInfo};
use reqwest::{StatusCode, header};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::interface::Interface;
use crate::{
    Context,
    context::{hub_answer, hub_error},
};

#[derive(Debug)]
pub struct Card {
    pub release: String,
    pub interface: Interface,
    /// The card as the Hub answered it.
    document: Value,
}

impl Card {
    /// What the package does in a line: its pyproject description, as the Hub read it from the
    /// release's wheel ("" for a release committed before the Hub kept it).
    pub fn summary(&self) -> &str {
        self.document["summary"].as_str().unwrap_or_default()
    }

    /// A callable's request struct as its card spells it, with `accelerator: false` for a CPU
    /// function: what `--describe --json` prints.
    pub fn request(&self, name: &str) -> Value {
        let entrypoints = self.document["package_interface"]["entrypoints"].as_array();
        let Some(entry) = entrypoints
            .into_iter()
            .flatten()
            .find(|e| e["name"] == name)
        else {
            return Value::Null;
        };
        let mut request = entry["request"].clone();
        if entry["accelerator"] == Value::Bool(false) {
            request["accelerator"] = Value::Bool(false);
        }
        request
    }
}

/// Where `(hub, package)`'s cards live: one file per release, and `latest.json`, the Hub's latest
/// release as last read with its ETag.
fn dir(home: &Home, hub: &str, package: &str) -> PathBuf {
    let host = hub.split_once("://").map_or(hub, |(_, rest)| rest);
    let host: String = host
        .trim_end_matches('/')
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let mut dir = home.root().join("packages").join(host);
    dir.extend(package.split('/'));
    dir
}

/// The card of `release`: the kept one, else the Hub's, which is then kept.
pub async fn card(ctx: &Context, hub: &Hub, package: &str, release: &str) -> Result<Card, Error> {
    if let Some(card) = kept(&ctx.home, &hub.origin, package, release) {
        return Ok(card);
    }
    let (org, name) = package.split_once('/').expect("a package is org/name");
    let answer = ctx
        .hub_client(hub)?
        .get_package_release(org, name, release)
        .await;
    let document = hub_answer(answer).map_err(|e| absent(e, &format!("{package}@{release}")))?;
    let document = serde_json::to_value(&document).expect("a card serializes");
    keep(&ctx.home, &hub.origin, package, &document)
}

/// The Hub's newest published release of `package`: one small read of its package, conditional
/// on the ETag kept from the last one when the Hub sent one (304: unchanged).
pub async fn latest(ctx: &Context, hub: &Hub, package: &str) -> Result<String, Error> {
    let (org, name) = package.split_once('/').expect("a package is org/name");
    let pointer = dir(&ctx.home, &hub.origin, package).join("latest.json");
    let kept: Option<Latest> = std::fs::read(&pointer)
        .ok()
        .and_then(|raw| serde_json::from_slice(&raw).ok());
    let client = ctx.hub_client(hub)?;
    let info = OperationInfo {
        operation_id: "get_package",
    };
    let url = format!("{}/v1/packages/{org}/{name}", client.baseurl());
    let mut request = client
        .client()
        .get(url)
        .header(header::ACCEPT, "application/json")
        .header("api-version", hub_client::Client::api_version());
    if let Some(kept) = &kept {
        request = request.header(header::IF_NONE_MATCH, &kept.etag);
    }
    let mut request = request
        .build()
        .map_err(|e| hub_error(hub_client::Error::Call(e.to_string())))?;
    client
        .pre::<()>(&mut request, &info)
        .await
        .map_err(hub_error)?;
    let answer = client
        .exec(request, &info)
        .await
        .map_err(|e| hub_error(hub_client::Error::Call(e.to_string())))?;
    let status = answer.status();
    if status == StatusCode::NOT_MODIFIED
        && let Some(kept) = kept
    {
        return Ok(kept.release);
    }
    let etag = answer
        .headers()
        .get(header::ETAG)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let body = answer
        .bytes()
        .await
        .map_err(|e| hub_error(hub_client::Error::Call(e.to_string())))?;
    if !status.is_success() {
        return Err(absent(refusal(status.as_u16(), &body), package));
    }
    let card: types::PackageCard = serde_json::from_slice(&body)
        .map_err(|e| hub_error(hub_client::Error::Answer(e.to_string())))?;
    let release = card.package.latest_release.ok_or_else(|| {
        Error::new(
            Kind::NotFound,
            "package_unreleased",
            format!("{package} has no release to run"),
        )
        .remedy("its owner publishes one with `cozy package publish`")
    })?;
    // Kept only with the Hub's ETag: without one there is nothing to ask conditionally.
    match etag {
        Some(etag) => {
            let latest = Latest {
                release: release.clone(),
                etag,
            };
            let latest = serde_json::to_vec(&latest).expect("serializes");
            let dir = pointer.parent().expect("a package dir");
            let _ =
                durable::private_dir(dir).and_then(|()| durable::replace(&pointer, &latest, 0o600));
        }
        None => drop(std::fs::remove_file(&pointer)),
    }
    Ok(release)
}

/// The Hub's latest release as last read, with the ETag it came with.
#[derive(Deserialize, Serialize)]
struct Latest {
    release: String,
    etag: String,
}

/// A Hub refusal's typed envelope, else the status alone.
fn refusal(status: u16, body: &[u8]) -> Error {
    let doc: Value = serde_json::from_slice(body).unwrap_or_default();
    let field = |name: &str| doc["error"][name].as_str().unwrap_or_default().to_owned();
    let code = match field("code") {
        code if code.is_empty() => format!("http_{status}"),
        code => code,
    };
    hub_error(hub_client::Error::Hub {
        status,
        code,
        message: field("message"),
        remedy: field("remedy"),
    })
}

/// The Hub's 404 names what it lacks, and where to look.
fn absent(error: Error, what: &str) -> Error {
    if error.kind != Kind::NotFound {
        return error;
    }
    let name = what.split('@').next().unwrap_or(what);
    Error::new(
        Kind::NotFound,
        "package_not_found",
        format!("the Hub has no {what}"),
    )
    .cause(error.code)
    .next(format!(
        "cozy package search {}",
        name.rsplit('/').next().unwrap_or(name)
    ))
}

/// The kept card of `release`.
pub fn kept(home: &Home, hub: &str, package: &str, release: &str) -> Option<Card> {
    let path = file(&dir(home, hub, package), release);
    let document: Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    read(&document).ok()
}

/// Keeps a release card as the Hub answered it.
pub fn keep(home: &Home, hub: &str, package: &str, document: &Value) -> Result<Card, Error> {
    let card = read(document)?;
    let dir = dir(home, hub, package);
    let kept = durable::private_dir(&dir).and_then(|()| {
        durable::replace(
            &file(&dir, &card.release),
            document.to_string().as_bytes(),
            0o600,
        )
    });
    // A card that cannot be kept is read again next time; the run goes on.
    drop(kept);
    Ok(card)
}

fn file(dir: &Path, release: &str) -> PathBuf {
    dir.join(format!("{}.json", release.replace(['/', '\\'], "_")))
}

/// A release card: `{release: {release, …}, package_interface, …}`.
pub fn read(document: &Value) -> Result<Card, Error> {
    let release = document["release"]["release"].as_str().unwrap_or_default();
    if release.is_empty() {
        return Err(Error::new(
            Kind::Unavailable,
            "hub.release_card_invalid",
            "the Hub's release card names no release",
        ));
    }
    let interface = Interface::parse(&document["package_interface"]).map_err(|why| {
        Error::new(
            Kind::Unavailable,
            "hub.package_interface_invalid",
            format!("the Hub's package interface is unreadable: {why}"),
        )
    })?;
    Ok(Card {
        release: release.to_owned(),
        interface,
        document: document.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::interface::tests::card;
    use super::*;

    #[test]
    fn a_kept_card_answers_by_its_release_only() {
        let home = tempfile::tempdir().unwrap();
        let home = Home::at(home.path());
        let hub = "http://127.0.0.1:8819";
        assert!(kept(&home, hub, "fidika/sdxl", "2.10.0").is_none());
        keep(&home, hub, "fidika/sdxl", &card("sdxl-2.10.0")).unwrap();
        let kept_card = kept(&home, hub, "fidika/sdxl", "2.10.0").unwrap();
        assert_eq!(kept_card.release, "2.10.0");
        assert!(kept(&home, hub, "fidika/sdxl", "2.9.0").is_none());
        assert!(kept(&home, "https://tensorhub.com", "fidika/sdxl", "2.10.0").is_none());
        assert!(
            home.root()
                .join("packages/127.0.0.1_8819/fidika/sdxl/2.10.0.json")
                .is_file()
        );
    }
}
