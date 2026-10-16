//! `cozy model search|info|family`: the Hub's model catalog. Reads are anonymous.
use std::collections::{BTreeMap, BTreeSet};
use std::process::ExitCode;

use cozy_proto::error::{Error, Kind};
use hub_client::types::{ModelCard, ModelCatalogRelease, ModelFamilyUpdate};
use serde::Serialize;

use super::refs::{Repo, label};
use super::{Family, Info, Search};
use crate::Context;
use crate::context::hub;
use crate::output::{self, Record, Table};

/// One search row: the model's first available release in the Hub's order, and its lanes.
#[derive(Serialize)]
struct Found {
    model: String,
    family: Option<String>,
    release: Option<String>,
    lanes: Vec<String>,
}

#[derive(Serialize)]
struct Searched {
    models: Vec<Found>,
    /// How many models matched, shown or not.
    total: i64,
}

fn available(release: &ModelCatalogRelease) -> bool {
    !release.release.is_empty() && release.yanked != Some(true) && release.yanked_at.is_none()
}

fn found(card: &ModelCard) -> Found {
    let release = card.releases.iter().find(|r| available(r));
    Found {
        model: format!("{}/{}", card.model.org, card.model.name),
        family: card.model.family.clone().filter(|f| !f.is_empty()),
        release: release.map(|r| r.release.clone()),
        lanes: release
            .map(|r| r.lanes.iter().map(|l| l.lane.clone()).collect::<BTreeSet<_>>())
            .unwrap_or_default()
            .into_iter()
            .collect(),
    }
}

pub(crate) async fn search(ctx: &Context, args: Search) -> Result<ExitCode, Error> {
    let at = ctx.hub()?;
    let client = ctx.hub_anonymous(&at)?;
    let exact = match (&args.family, args.query.as_slice()) {
        (None, [one]) => Repo::parse(one).ok(),
        _ => None,
    };
    let (cards, total) = match exact {
        Some(repo) => (vec![hub(client.get_model(&repo.org, &repo.name).await)?], 1),
        None => {
            let query = args.query.join(" ");
            let answer = hub(
                client
                    .search_models(
                        args.family.as_deref(),
                        Some(args.limit.into()),
                        Some(query.as_str()).filter(|q| !q.is_empty()),
                    )
                    .await,
            )?;
            let mut cards = tokio::task::JoinSet::new();
            for (i, model) in answer.models.into_iter().take(args.limit as usize).enumerate() {
                let client = client.clone();
                cards.spawn(async move { (i, hub(client.get_model(&model.org, &model.name).await)) });
            }
            let mut ordered = BTreeMap::new();
            while let Some(joined) = cards.join_next().await {
                let (i, card) = joined.map_err(|e| Error::new(Kind::Internal, "join", e.to_string()))?;
                ordered.insert(i, card?);
            }
            (ordered.into_values().collect(), answer.search.total)
        }
    };
    let view = Searched {
        models: cards.iter().map(found).collect(),
        total,
    };
    if ctx.json {
        return output::json(&view);
    }
    let mut table = Table::new(ctx, &["model", "family", "release", "lanes"]);
    for row in &view.models {
        table = table.row(vec![
            row.model.clone(),
            row.family.clone().unwrap_or_default(),
            row.release.clone().unwrap_or_default(),
            row.lanes.join(", "),
        ]);
    }
    table.print();
    let shown = view.models.len() as i64;
    if total > shown {
        output::note(ctx, &format!("{shown} of {total} matching models; raise --limit or narrow the search"));
    }
    match view.models.first() {
        Some(first) => output::next(ctx, &[format!("cozy model info {}", first.model)]),
        None => output::next(ctx, &["cozy model search".to_owned()]),
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Serialize)]
struct Lane {
    lane: String,
    bytes: i64,
    components: Vec<String>,
    component_bytes: BTreeMap<String, i64>,
    checkpoint: String,
}

#[derive(Serialize)]
struct Release {
    release: String,
    revision: Option<i64>,
    created_at: Option<String>,
    yanked: bool,
    lanes: Vec<Lane>,
}

#[derive(Serialize)]
struct Detail {
    model: String,
    family: Option<String>,
    created_at: String,
    releases: Vec<Release>,
}

pub(crate) async fn info(ctx: &Context, args: Info) -> Result<ExitCode, Error> {
    let (name, selected) = match args.model.split_once('@') {
        Some((name, release)) => (name, Some(label("release", release)?)),
        None => (args.model.as_str(), None),
    };
    let repo = Repo::parse(name)?;
    let at = ctx.hub()?;
    let card = hub(ctx.hub_anonymous(&at)?.get_model(&repo.org, &repo.name).await)?;
    let releases: Vec<Release> = card
        .releases
        .iter()
        .filter(|r| selected.as_ref().is_none_or(|s| *s == r.release))
        .map(|r| Release {
            release: r.release.clone(),
            revision: r.revision,
            created_at: r.cut_at.map(|t| t.to_rfc3339()),
            yanked: !available(r),
            lanes: r
                .lanes
                .iter()
                .map(|l| Lane {
                    lane: l.lane.clone(),
                    bytes: l.bytes,
                    components: l.components.clone().unwrap_or_default(),
                    component_bytes: l.component_bytes.clone().into_iter().collect(),
                    checkpoint: format!("{repo}#{}", l.manifest_id),
                })
                .collect(),
        })
        .collect();
    if let Some(release) = selected.filter(|_| releases.is_empty()) {
        return Err(Error::new(
            Kind::NotFound,
            "model_release_not_found",
            format!("{repo} has no release {release:?}"),
        )
        .next(format!("cozy model info {repo}")));
    }
    let view = Detail {
        model: repo.to_string(),
        family: card.model.family.filter(|f| !f.is_empty()),
        created_at: card.model.created_at.to_rfc3339(),
        releases,
    };
    if ctx.json {
        return output::json(&view);
    }
    Record::new()
        .field("model", &view.model)
        .field("family", view.family.as_deref().unwrap_or(""))
        .field("created", &view.created_at)
        .print(ctx);
    for release in &view.releases {
        let mut table = Table::new(ctx, &["lane", "bytes", "checkpoint"]);
        for lane in &release.lanes {
            table = table.row(vec![lane.lane.clone(), output::bytes(lane.bytes as u64), lane.checkpoint.clone()]);
        }
        let state = if release.yanked { " (yanked)" } else { "" };
        output::note(ctx, &format!("{repo}@{}{state}", release.release));
        table.print();
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Serialize)]
struct FamilySet {
    model: String,
    family: Option<String>,
}

pub(crate) async fn family(ctx: &Context, args: Family) -> Result<ExitCode, Error> {
    let repo = Repo::hub(&args.model)?;
    let family = if args.clear { String::new() } else { args.family.unwrap_or_default() };
    let at = ctx.hub()?;
    let answer = hub(
        ctx.hub_signed_in(&at)?
            .set_model_family(&repo.org, &repo.name, &ModelFamilyUpdate { family })
            .await,
    )?;
    let view = FamilySet {
        model: repo.to_string(),
        family: answer.model.family.filter(|f| !f.is_empty()),
    };
    if ctx.json {
        return output::json(&view);
    }
    Record::new()
        .field("model", &view.model)
        .field("family", view.family.as_deref().unwrap_or("(none)"))
        .print(ctx);
    Ok(ExitCode::SUCCESS)
}
