//! `cozy run list` and `cozy run show` as live views (`crate::tui`), fed by Calcifer's events.
use std::collections::VecDeque;

use protocol::{
    calcifer::{Event, OutputItem, Run, RunsEvent},
    error::Error,
};

use super::{follow, list};
use crate::{
    Context,
    calcifer_client::{Events, Watch},
};

/// How many of a run's newest log lines its view keeps.
const LOG: usize = 200;

/// The runs `cozy run list` shows, kept as Calcifer says they change.
pub(crate) struct Runs {
    query: list::Query,
    limit: u64,
    runs: Vec<Run>,
    watch: Watch<RunsEvent>,
}

impl Runs {
    /// Subscribes, then lists: a run that changes in between is in the stream.
    pub(crate) async fn open(ctx: &Context, query: list::Query, limit: u64) -> Result<Self, Error> {
        let watch = ctx.calcifer().await?.watch("/v1/runs/events").await?;
        let runs = list::runs(ctx, &query).await?.runs;
        Ok(Self {
            query,
            limit,
            runs,
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
        self.runs.iter().map(list::row).collect()
    }

    pub(crate) fn id(&self, selected: usize) -> Option<String> {
        self.runs.get(selected).map(|r| r.id.to_string())
    }
}

/// One run, kept as its events say it changes, and its newest log lines.
pub(crate) struct Live {
    run: Run,
    events: Events,
    logs: VecDeque<String>,
    ended: bool,
}

impl Live {
    pub(crate) async fn open(ctx: &Context, id: &str) -> Result<Self, Error> {
        let calcifer = ctx.calcifer().await?;
        let run = calcifer.run(id).await?;
        let events = calcifer.events(run.id.as_str(), 0);
        Ok(Self {
            run,
            events,
            logs: VecDeque::new(),
            ended: false,
        })
    }

    pub(crate) async fn changed(&mut self) -> Result<(), Error> {
        if self.ended {
            // The run's stream is over: nothing more will change.
            return std::future::pending().await;
        }
        match self.events.next().await {
            Some(Ok(event)) => self.apply(event),
            Some(Err(e)) => return Err(e),
            None => self.ended = true,
        }
        Ok(())
    }

    fn apply(&mut self, event: Event) {
        match event {
            Event::Created { run, .. }
            | Event::Queued { run, .. }
            | Event::InProgress { run, .. }
            | Event::Completed { run, .. }
            | Event::Failed { run, .. }
            | Event::Canceled { run, .. } => self.run = run,
            Event::Progress { progress } => self.run.progress = Some(progress),
            // An empty reason: it no longer waits.
            Event::Waiting { reason } => self.run.waiting = Some(reason).filter(|r| !r.is_empty()),
            Event::OutputAdded { item, .. } | Event::OutputDone { item, .. } => self.output(item),
            Event::OutputDelta {
                item_id,
                rev,
                checkpoint: Some(checkpoint),
                ..
            } => {
                if let Some(item) = self.run.output.iter_mut().find(|item| item.id == item_id) {
                    item.rev = rev;
                    item.checkpoint = Some(checkpoint);
                }
            }
            Event::Log { text, .. } => {
                self.logs.push_back(text);
                if self.logs.len() > LOG {
                    self.logs.pop_front();
                }
            }
            _ => {}
        }
    }

    fn output(&mut self, item: OutputItem) {
        match self.run.output.iter_mut().find(|o| o.id == item.id) {
            Some(held) => *held = item,
            None => self.run.output.push(item),
        }
    }

    pub(crate) fn title(&self) -> String {
        format!("run {}", self.run.number)
    }

    /// `cozy run show`'s record, then the newest log lines.
    pub(crate) fn lines(&self, ctx: &Context) -> Vec<String> {
        let mut lines = follow::fields(ctx, &self.run).lines();
        if !self.logs.is_empty() {
            lines.push(String::new());
            lines.extend(self.logs.iter().cloned());
        }
        lines
    }
}
