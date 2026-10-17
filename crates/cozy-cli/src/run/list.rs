//! `cozy run list`: runs on every Hub, newest first: a live view on a terminal, else a table or
//! JSON.
use std::process::ExitCode;

use cozy_proto::{
    calcifer::{Kind as RunKind, Run, RunList, Status},
    error::Error,
};
use serde::Serialize;

use super::{follow, live};
use crate::{Context, output, tui};

pub(crate) const HEADERS: [&str; 7] = [
    "#", "TARGET", "MACHINE", "STATUS", "PROGRESS", "TIME", "REASON",
];
/// The newest runs a list shows unless `--limit` says otherwise.
const SHOWN: u64 = 50;
/// A reason is one table cell; `cozy run show` has it whole.
const REASON: usize = 72;

/// `GET /v1/runs`' filters.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Query {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
}

pub async fn exec(ctx: &Context, mut query: Query, once: bool) -> Result<ExitCode, Error> {
    let limit = query.limit.unwrap_or(SHOWN);
    query.limit = (limit > 0).then_some(limit);
    if !once && !ctx.json && ctx.term.stdout {
        let view = live::Runs::open(ctx, query, limit).await?;
        tui::show(ctx, tui::Screen::Runs(view)).await?;
        return Ok(ExitCode::SUCCESS);
    }
    let list = runs(ctx, &query).await?;
    if ctx.json {
        output::json(&list);
        return Ok(ExitCode::SUCCESS);
    }
    let mut table = output::Table::new(ctx, HEADERS);
    for run in &list.runs {
        table = table.row(row(run));
    }
    if list.older > 0 {
        table = table.note(format!(
            "{} older runs: cozy run list --limit 0",
            list.older
        ));
    }
    table.print();
    Ok(ExitCode::SUCCESS)
}

/// The runs `query` asks for, page by page; with `--tensorhub`, only that Hub's.
pub(crate) async fn runs(ctx: &Context, query: &Query) -> Result<RunList, Error> {
    let calcifer = ctx.calcifer().await?;
    let hub = ctx.narrowed()?;
    let mut out = RunList {
        runs: vec![],
        older: 0,
    };
    let mut page = Query {
        status: query.status.clone(),
        package: query.package.clone(),
        before: query.before,
        limit: Some(query.limit.unwrap_or(500).min(500)),
    };
    loop {
        let list = calcifer.runs(&page).await?;
        let last = list.runs.last().map(|r| r.number);
        let more = list.older > 0;
        out.older = list.older;
        out.runs.extend(
            list.runs
                .into_iter()
                .filter(|r| hub.as_ref().is_none_or(|h| h.origin == r.hub)),
        );
        let wanted = query
            .limit
            .is_none_or(|limit| (out.runs.len() as u64) < limit);
        match last {
            Some(number) if more && wanted => page.before = Some(number),
            _ => return Ok(out),
        }
    }
}

/// Whether `run` is one `query` lists: its state, package and (with `--tensorhub`) Hub.
pub(crate) fn matches(ctx: &Context, query: &Query, run: &Run) -> Result<bool, Error> {
    let status = serde_json::to_value(run.status).ok();
    let state = query
        .status
        .as_ref()
        .is_none_or(|s| status.as_ref().and_then(|v| v.as_str()) == Some(s.as_str()));
    let package = query
        .package
        .as_ref()
        .is_none_or(|p| run.target.starts_with(p.as_str()));
    let hub = ctx.narrowed()?.is_none_or(|h| h.origin == run.hub);
    Ok(state && package && hub)
}

/// A run as one row of the table and the live view.
pub(crate) fn row(run: &Run) -> Vec<String> {
    let mut target = run.target.clone();
    if run.kind != RunKind::Call {
        target = format!("{} {target}", run.kind.as_str());
    }
    let progress = match (&run.status, &run.progress) {
        (Status::InProgress, Some(p)) => {
            let fraction = p
                .fraction
                .filter(|f| *f >= 0.0)
                .or_else(|| (p.total > 0).then(|| p.completed as f64 / p.total as f64));
            match fraction {
                Some(f) => format!("{} {:.0}%", p.stage, f * 100.0),
                None => p.stage.clone(),
            }
        }
        (Status::InProgress, None) => run.machine_state.clone().unwrap_or_default(),
        _ => String::new(),
    };
    let time = match (run.execution_ms, run.accepted_at_ms) {
        (Some(ms), _) => seconds(ms),
        (None, Some(at)) if run.status == Status::InProgress => {
            seconds(now_ms().saturating_sub(at.max(0) as u64))
        }
        _ => "—".into(),
    };
    let reason = match run.status {
        // Why it waits, until it settles: sending, uploading, a GPU another run holds.
        Status::Queued | Status::InProgress => run.waiting.clone().unwrap_or_default(),
        Status::Failed | Status::Canceled => run
            .reason
            .as_ref()
            .map(|r| r.message.clone())
            .unwrap_or_default(),
        _ => String::new(),
    };
    let reason = reason.split_whitespace().collect::<Vec<_>>().join(" ");
    let reason = match reason.chars().count() > REASON {
        true => format!("{}…", reason.chars().take(REASON - 1).collect::<String>()),
        false => reason,
    };
    vec![
        run.number.to_string(),
        target,
        run.machine.clone().unwrap_or_else(|| "—".into()),
        follow::status(run),
        progress,
        time,
        reason,
    ]
}

fn seconds(ms: u64) -> String {
    format!("{:.1}s", ms as f64 / 1000.0)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}
