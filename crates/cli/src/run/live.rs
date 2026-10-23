//! `cozy run list` as a live view (`crate::tui`), fed by Calcifer's events.
use protocol::{
    calcifer::{Run, RunsEvent},
    error::Error,
};

use super::list;
use crate::{Context, calcifer_client::Watch};

/// The runs `cozy run list` shows, kept as Calcifer says they change.
pub(crate) struct Runs {
    query: list::Query,
    limit: u64,
    runs: Vec<Run>,
    watch: Watch<RunsEvent>,
}

impl Runs {
    /// Subscribes, then lists: a run that changes in between is in the stream. Changes match
    /// `--on` by the machine's name, which an alias or rental id resolved to.
    pub(crate) async fn open(ctx: &Context, query: list::Query, limit: u64) -> Result<Self, Error> {
        let watch = ctx.calcifer().await?.watch("/v1/runs/events").await?;
        let list = list::runs(ctx, &query).await?;
        Ok(Self {
            query: list::Query {
                machine: list.machine.or(query.machine),
                ..query
            },
            limit,
            runs: list.runs,
            watch,
        })
    }

    pub(crate) async fn changed(&mut self, ctx: &Context) -> Result<(), Error> {
        loop {
            match self.watch.next().await {
                Some(RunsEvent::Updated { run }) => {
                    if self.apply(ctx, run)? {
                        return Ok(());
                    }
                }
                // A stream that ended (Calcifer restarted, or this view fell behind) may have
                // missed a change: subscribe and list again.
                None => {
                    *self = Self::open(ctx, self.query.clone(), self.limit).await?;
                    return Ok(());
                }
            }
        }
    }

    /// Takes in `run` as it is now; true when the view changed.
    fn apply(&mut self, ctx: &Context, run: Run) -> Result<bool, Error> {
        let fits = list::matches(ctx, &self.query, &run)?;
        match self.runs.iter().position(|r| r.id == run.id) {
            Some(i) if fits => self.runs[i] = run,
            Some(i) => drop(self.runs.remove(i)),
            None if !fits => return Ok(false),
            None => {
                let at = self.runs.partition_point(|r| r.number > run.number);
                self.runs.insert(at, run);
                if self.limit > 0 {
                    self.runs.truncate(self.limit as usize);
                }
            }
        }
        Ok(true)
    }

    pub(crate) fn title(&self) -> String {
        match self.limit {
            0 => "runs".into(),
            limit => format!("runs, newest {limit}"),
        }
    }

    pub(crate) fn headers(&self) -> Vec<&'static str> {
        list::HEADERS.to_vec()
    }

    pub(crate) fn rows(&self) -> Vec<Vec<String>> {
        list::rows(&self.runs)
    }

    pub(crate) fn id(&self, selected: usize) -> Option<String> {
        self.runs.get(selected).map(|r| r.id.to_string())
    }
}
