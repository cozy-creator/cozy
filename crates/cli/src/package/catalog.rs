//! `cozy package search|info`: the Hub's catalog, as this computer's account sees it (anonymous when signed out).
use std::process::ExitCode;

use protocol::{
    config::Hub,
    error::{Error, Kind},
};
use serde::Serialize;

use super::{PackageRef, release_order, stamp};
use crate::{
    Context,
    context::hub_error,
    output::{self, Table},
};

#[derive(Serialize)]
struct Found {
    package: String,
    latest_release: Option<String>,
    created_at: String,
}

#[derive(Serialize)]
struct Search {
    packages: Vec<Found>,
    total: i64,
    omitted: i64,
}

pub async fn search(ctx: &Context, query: &str, limit: u32) -> Result<ExitCode, Error> {
    let hub = ctx.hub()?;
    let query = query.trim();
    let answer = ctx
        .hub_client(&hub)?
        .search_packages(Some(limit.into()), (!query.is_empty()).then_some(query))
        .await
        .map_err(hub_error)?
        .into_inner();
    let packages: Vec<Found> = answer
        .packages
        .into_iter()
        .map(|p| Found {
            package: format!("{}/{}", p.org, p.name),
            latest_release: p.latest_release,
            created_at: stamp(&p.created_at),
        })
        .collect();
    let omitted = (answer.search.total - packages.len() as i64).max(0);
    let result = Search {
        packages,
        total: answer.search.total,
        omitted,
    };
    if ctx.json {
        output::json(&result);
        return Ok(ExitCode::SUCCESS);
    }
    if result.packages.is_empty() {
        println!("No packages on {} match.", hub.label());
        let next = if query.is_empty() {
            "cozy package publish --help"
        } else {
            "cozy package search"
        };
        output::next(ctx, &[next]);
        return Ok(ExitCode::SUCCESS);
    }
    let mut table = Table::new(ctx, ["PACKAGE", "LATEST", "CREATED"]);
    for p in &result.packages {
        let latest = p.latest_release.as_deref().unwrap_or("-");
        table = table.row([p.package.as_str(), latest, p.created_at.as_str()]);
    }
    table.print();
    if omitted > 0 {
        let shown = result.total.min(200);
        println!("{omitted} more not shown. Use --limit {shown} to show them.");
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Serialize)]
struct Release {
    release: String,
    published: Option<String>,
    yanked: bool,
}

#[derive(Serialize)]
struct Info {
    package: String,
    latest_release: Option<String>,
    releases: Vec<Release>,
    /// The named release's publish warnings, which the Hub tells its owner.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
}

pub async fn info(ctx: &Context, package: &PackageRef) -> Result<ExitCode, Error> {
    let hub = ctx.hub()?;
    let card = ctx
        .hub_client(&hub)?
        .get_package(&package.org, &package.name, None)
        .await
        .map_err(|e| missing(hub_error(e), package, hub.label()))?
        .into_inner();
    let mut releases: Vec<Release> = card
        .releases
        .into_iter()
        .map(|r| Release {
            yanked: r.yanked.unwrap_or(false) || r.yanked_at.is_some(),
            published: r.cut_at.as_deref().map(stamp),
            release: r.release,
        })
        .collect();
    releases.sort_by_key(|r| std::cmp::Reverse(release_order(&r.release)));
    // A named release: its row alone, and what its publish warned of.
    let mut warned = vec![];
    if let Some(release) = &package.release {
        releases.retain(|r| r.release == *release);
        if releases.is_empty() {
            return Err(Error::new(
                Kind::NotFound,
                "release_not_found",
                format!(
                    "{} has no release {release} on {}",
                    package.package(),
                    hub.label()
                ),
            )
            .next(format!("cozy package info {}", package.package())));
        }
        warned = warnings(ctx, &hub, package, release).await?;
    }
    let info = Info {
        package: package.package(),
        latest_release: card.package.latest_release,
        releases,
        warnings: warned,
    };
    if ctx.json {
        output::json(&info);
        return Ok(ExitCode::SUCCESS);
    }
    println!(
        "{} on {}, latest {}",
        info.package,
        hub.label(),
        info.latest_release.as_deref().unwrap_or("none")
    );
    let mut table = Table::new(ctx, ["RELEASE", "PUBLISHED", "YANKED"]);
    for r in &info.releases {
        let yanked = if r.yanked { "yes" } else { "no" };
        table = table.row([
            r.release.as_str(),
            r.published.as_deref().unwrap_or("-"),
            yanked,
        ]);
    }
    for warning in &info.warnings {
        table = table.note(format!("warning: {warning}"));
    }
    table.print();
    Ok(ExitCode::SUCCESS)
}

/// A release's publish warnings (a pin of its org's own package since yanked, or with a newer
/// release), as the Hub's publish status tells the package's owner; none for anyone else.
async fn warnings(
    ctx: &Context,
    hub: &Hub,
    package: &PackageRef,
    release: &str,
) -> Result<Vec<String>, Error> {
    if !ctx.signed_in(hub) {
        return Ok(vec![]);
    }
    let status = ctx
        .hub_signed_in(hub)?
        .get_package_finalization(&package.org, &package.name, release)
        .await;
    match status.map_err(hub_error) {
        Ok(status) => Ok(status.into_inner().warnings),
        Err(e) if matches!(e.kind, Kind::Forbidden | Kind::Credential | Kind::NotFound) => {
            Ok(vec![])
        }
        Err(e) => Err(e),
    }
}

/// A package this Hub does not have, named with the Hub searched and where to look instead.
fn missing(error: Error, package: &PackageRef, hub: &str) -> Error {
    if error.kind != Kind::NotFound {
        return error;
    }
    let cause = error.cause.clone().unwrap_or(error.code);
    Error::new(
        Kind::NotFound,
        "package_not_found",
        format!("no package {} on {hub}", package.package()),
    )
    .remedy("check the name, and the Hub with `cozy hub list`")
    .next(format!("cozy package search {}", package.name))
    .next("cozy hub list")
    .cause(cause)
}
