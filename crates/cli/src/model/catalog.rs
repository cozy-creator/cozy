//! `cozy model search|info|family`: the Hub's model catalog. Reads are anonymous.
use std::collections::{BTreeMap, BTreeSet};
use std::process::ExitCode;

use protocol::error::{Error, Kind};
use hub_client::types::{ModelCard, ModelCatalogRelease, ModelFamilyUpdate};
use serde::Serialize;

use super::refs::{ModelRef, Repo};
use super::{Family, Info, Search};
use crate::Context;
use crate::context::hub_answer;
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
    !release.release.is_empty() && release.yanked != Some(true) && release.yanked_at.as_deref().is_none_or(str::is_empty)
}

fn found(card: &ModelCard) -> Found {
    let release = tensorfs::format::newest_by(card.releases.iter().filter(|r| available(r)), |r| r.release.as_str());
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
    let client = ctx.hub_anonymous(&ctx.hub()?);
    let exact = match (&args.family, args.query.as_slice()) {
        (None, [one]) => Repo::parse(one).ok(),
        _ => None,
    };
    let (cards, total) = match exact {
        Some(repo) => (vec![hub_answer(client.get_model(&repo.org, &repo.name).await)?], 1),
        None => {
            let query = args.query.join(" ");
            let q = Some(query.as_str()).filter(|q| !q.is_empty());
            let answer = hub_answer(client.search_models(args.family.as_deref(), Some(args.limit.into()), q).await)?;
            // One card read per model, all at once.
            let mut reads = tokio::task::JoinSet::new();
            for (i, model) in answer.models.into_iter().take(args.limit as usize).enumerate() {
                let client = client.clone();
                reads.spawn(async move { (i, client.get_model(&model.org, &model.name).await) });
            }
            let mut cards = BTreeMap::new();
            while let Some(read) = reads.join_next().await {
                let (i, card) = read.map_err(|e| Error::new(Kind::Internal, "card_read", e.to_string()))?;
                cards.insert(i, hub_answer(card)?);
            }
            (cards.into_values().collect(), answer.search.total)
        }
    };
    let view = Searched {
        models: cards.iter().map(found).collect(),
        total,
    };
    if ctx.json {
        output::json(&view);
        return Ok(ExitCode::SUCCESS);
    }
    let mut table = Table::new(ctx, ["model", "family", "release", "lanes"]);
    for row in &view.models {
        let lanes = row.lanes.join(", ");
        let family = row.family.as_deref().unwrap_or("-");
        table = table.row([row.model.as_str(), family, row.release.as_deref().unwrap_or("-"), &lanes]);
    }
    let shown = view.models.len() as i64;
    if total > shown {
        table = table.note(format!("{shown} of {total} matching models: raise --limit or narrow the search"));
    }
    table.print();
    match view.models.first() {
        Some(first) => output::next(ctx, &[&format!("cozy model info {}", first.model)]),
        None => output::next(ctx, &["cozy model search"]),
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
    let (repo, selected) = match ModelRef::parse(&args.model)? {
        ModelRef::Hub { repo, release, lane: None } => (repo, release),
        _ => {
            return Err(Error::new(Kind::Usage, "model_ref_invalid", format!("{:?}: info reads org/name[@release]", args.model)));
        }
    };
    let client = ctx.hub_anonymous(&ctx.hub()?);
    let card = hub_answer(client.get_model(&repo.org, &repo.name).await)?;
    let releases: Vec<Release> = card
        .releases
        .iter()
        .filter(|r| selected.as_ref().is_none_or(|s| *s == r.release))
        .map(|r| Release {
            release: r.release.clone(),
            revision: r.revision,
            created_at: r.cut_at.clone().filter(|t| !t.is_empty()),
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
        return Err(Error::new(Kind::NotFound, "model_release_not_found", format!("{repo} has no release {release:?}"))
            .next(format!("cozy model info {repo}")));
    }
    let view = Detail {
        model: repo.to_string(),
        family: card.model.family.filter(|f| !f.is_empty()),
        created_at: card.model.created_at,
        releases,
    };
    if ctx.json {
        output::json(&view);
        return Ok(ExitCode::SUCCESS);
    }
    let mut record = Record::new().field("model", &view.model).field("family", view.family.as_deref().unwrap_or("-"));
    record = record.field("created", &view.created_at);
    record.print(ctx);
    for release in &view.releases {
        let yanked = if release.yanked { ", yanked" } else { "" };
        let revision = release.revision.map(|r| format!("revision {r}")).unwrap_or_default();
        anstream::println!("\n{repo}@{} ({revision}{yanked})", release.release);
        let mut table = Table::new(ctx, ["lane", "bytes", "checkpoint"]);
        for lane in &release.lanes {
            table = table.row([lane.lane.clone(), output::bytes(lane.bytes as u64), lane.checkpoint.clone()]);
        }
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
    let client = ctx.hub_signed_in(&ctx.hub()?)?;
    let answer = hub_answer(client.set_model_family(&repo.org, &repo.name, &ModelFamilyUpdate { family }).await)?;
    let view = FamilySet {
        model: repo.to_string(),
        family: answer.model.family.filter(|f| !f.is_empty()),
    };
    if ctx.json {
        output::json(&view);
    } else {
        Record::of(&view).print(ctx);
    }
    Ok(ExitCode::SUCCESS)
}
