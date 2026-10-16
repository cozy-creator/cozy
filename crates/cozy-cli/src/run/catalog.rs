//! Release cards this computer keeps. A release never changes, so its card is read from the Hub
//! once and kept forever; a warm run reads no Hub. Machines resolve which release runs: the card
//! here only binds terms.
use std::path::{Path, PathBuf};

use cozy_platform::{durable, paths::Home};
use cozy_proto::config::Hub;
use cozy_proto::error::{Error, Kind};
use serde_json::Value;

use super::interface::Interface;
use crate::{Context, context::hub_answer};

#[derive(Debug)]
pub struct Card {
    pub release: String,
    pub interface: Interface,
    /// The card as the Hub answered it.
    document: Value,
}

impl Card {
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

/// Where `(hub, package)`'s cards live: one file per release and `newest`, the newest one read.
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

/// The card of `release`, else of the newest release: the kept one, else the Hub's, which is then
/// kept. `fresh` reads the Hub's newest release even when one is kept. Answers whether it read
/// the Hub.
pub async fn card(
    ctx: &Context,
    hub: &Hub,
    package: &str,
    release: Option<&str>,
    fresh: bool,
) -> Result<(Card, bool), Error> {
    if !(fresh && release.is_none())
        && let Some(card) = kept(&ctx.home, &hub.origin, package, release)
    {
        return Ok((card, false));
    }
    let (org, name) = package.split_once('/').expect("a package is org/name");
    let client = ctx.hub_client(hub)?;
    let newest = release.is_none();
    let release = match release {
        Some(release) => release.to_owned(),
        None => hub_answer(client.get_package(org, name).await)
            .map_err(|e| absent(e, package))?
            .package
            .latest_release
            .ok_or_else(|| {
                Error::new(
                    Kind::NotFound,
                    "package_unreleased",
                    format!("{package} has no release to run"),
                )
                .remedy("its owner publishes one with `cozy package publish`")
            })?,
    };
    let document = hub_answer(client.get_package_release(org, name, &release).await)
        .map_err(|e| absent(e, &format!("{package}@{release}")))?;
    let document = serde_json::to_value(&document).expect("a card serializes");
    Ok((
        keep(&ctx.home, &hub.origin, package, &document, newest)?,
        true,
    ))
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

/// The kept card of `release`, else of the newest release read.
pub fn kept(home: &Home, hub: &str, package: &str, release: Option<&str>) -> Option<Card> {
    let dir = dir(home, hub, package);
    let release = match release {
        Some(release) => release.to_owned(),
        None => std::fs::read_to_string(dir.join("newest"))
            .ok()?
            .trim()
            .to_owned(),
    };
    let document: Value =
        serde_json::from_slice(&std::fs::read(file(&dir, &release)).ok()?).ok()?;
    read(&document).ok()
}

/// Keeps a release card as the Hub answered it; `newest` marks it the newest release read.
pub fn keep(
    home: &Home,
    hub: &str,
    package: &str,
    document: &Value,
    newest: bool,
) -> Result<Card, Error> {
    let card = read(document)?;
    let dir = dir(home, hub, package);
    let kept = durable::private_dir(&dir)
        .and_then(|()| {
            durable::replace(
                &file(&dir, &card.release),
                document.to_string().as_bytes(),
                0o600,
            )
        })
        .and_then(|()| match newest {
            true => durable::replace(&dir.join("newest"), card.release.as_bytes(), 0o600),
            false => Ok(()),
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
    fn a_kept_card_answers_by_release_and_as_the_newest() {
        let home = tempfile::tempdir().unwrap();
        let home = Home::at(home.path());
        let hub = "http://127.0.0.1:8819";
        assert!(kept(&home, hub, "fidika/sdxl", None).is_none());
        keep(&home, hub, "fidika/sdxl", &card("sdxl-2.10.0"), true).unwrap();
        assert_eq!(
            kept(&home, hub, "fidika/sdxl", None).unwrap().release,
            "2.10.0"
        );
        assert_eq!(
            kept(&home, hub, "fidika/sdxl", Some("2.10.0"))
                .unwrap()
                .release,
            "2.10.0"
        );
        assert!(kept(&home, hub, "fidika/sdxl", Some("2.9.0")).is_none());
        assert!(kept(&home, "https://tensorhub.com", "fidika/sdxl", None).is_none());
        assert!(
            home.root()
                .join("packages/127.0.0.1_8819/fidika/sdxl/2.10.0.json")
                .is_file()
        );
    }
}
