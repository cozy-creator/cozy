//! Following a run: live progress on stderr, each output named as it lands, and the run's record
//! on stdout at the end; NDJSON of Calcifer's events with `--json`. Ctrl-C stops following and
//! never cancels: the run is Calcifer's, and `cozy run watch` reattaches.
use std::{collections::HashSet, fmt::Write as _, process::ExitCode, time::Duration};

use cozy_proto::{
    calcifer::{Event, OutputItem, Progress, Run, Status, SubmitRun, Submitted},
    error::Error,
};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::{Context, output};

/// Submits under `key`, or a fresh one: one key names one submission forever.
pub(crate) async fn submit(
    ctx: &Context,
    run: SubmitRun,
    key: Option<&str>,
) -> Result<Submitted, Error> {
    let key = key.map_or_else(mint_key, str::to_owned);
    let submitted = ctx.calcifer().await?.submit(&key, &run).await?;
    if submitted.replay {
        output::say(
            ctx,
            &format!(
                "this key names run {}, submitted before",
                submitted.run.number
            ),
        );
    }
    Ok(submitted)
}

/// A fresh idempotency key.
pub(crate) fn mint_key() -> String {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("the OS has randomness");
    format!("idem-{}", hex::encode(bytes))
}

/// 0 for a run that completed or goes on (detached), 1 for one that failed or was canceled.
pub(crate) fn exit_code(run: &Run) -> ExitCode {
    match run.status {
        Status::Failed | Status::Canceled => ExitCode::FAILURE,
        _ => ExitCode::SUCCESS,
    }
}

/// Follows `run` to its end, or prints it at once with `detach`; answers the run as it ended,
/// or as it was when following stopped.
pub(crate) async fn follow(ctx: &Context, run: Run, detach: bool) -> Result<Run, Error> {
    if detach {
        record(ctx, &run);
        return Ok(run);
    }
    let calcifer = ctx.calcifer().await?;
    let mut events = calcifer.events(run.id.as_str(), 0);
    let mut view = View::new(ctx, &run);
    let interrupt = tokio::signal::ctrl_c();
    tokio::pin!(interrupt);
    let id = run.id.clone();
    loop {
        tokio::select! {
            _ = &mut interrupt => {
                view.done();
                let run = calcifer.run(id.as_str()).await?;
                output::say(ctx, &format!(
                    "detached: run {n} keeps running; `cozy run watch {n}` follows it again, `cozy run cancel {n}` cancels it",
                    n = run.number
                ));
                return Ok(run);
            }
            event = events.next() => match event {
                None => break,
                Some(Err(error)) => {
                    view.done();
                    return Err(error);
                }
                Some(Ok(event)) if ctx.json => output::ndjson(&event),
                Some(Ok(event)) => view.on(&event),
            }
        }
    }
    view.done();
    let run = calcifer.run(id.as_str()).await?;
    if !ctx.json {
        record(ctx, &run);
    }
    Ok(run)
}

pub(crate) fn settled(run: &Run) -> bool {
    matches!(
        run.status,
        Status::Completed | Status::Failed | Status::Canceled
    )
}

/// The run's record: number, status, machine, outputs, the result's scalars, and why it ended.
pub(crate) fn record(ctx: &Context, run: &Run) {
    if ctx.json {
        output::json(run);
        return;
    }
    let mut record = output::Record::new()
        .field("run", run.number)
        .field("target", &run.target)
        .field("status", status(run))
        .field("machine", run.machine.as_deref().unwrap_or("—"));
    if let Some(waiting) = run.waiting.as_deref().filter(|_| !settled(run)) {
        record = record.field("waiting", waiting);
    }
    if let Some(ms) = run.execution_ms {
        record = record.field("execution", format!("{:.1}s", ms as f64 / 1000.0));
    }
    if let Some(reason) = &run.reason {
        record = record.field("reason", format!("{} ({})", reason.message, reason.code));
    }
    for item in &run.output {
        let shown = match (&item.path, &item.export_error) {
            (Some(path), _) => output::link(ctx, path),
            (None, Some(error)) => format!("not saved: {error}"),
            (None, None) => output::bytes(item.length),
        };
        record = record.field(&item_name(item), shown);
    }
    if let Some(serde_json::Value::Object(result)) = &run.result {
        for (key, value) in result {
            let named = run.output.iter().any(|item| &item.name == key);
            if named || value.is_object() || value.is_array() || value.is_null() {
                continue;
            }
            record = record.field(
                key,
                value
                    .as_str()
                    .map_or_else(|| value.to_string(), str::to_owned),
            );
        }
    }
    for warning in &run.warnings {
        record = record.note(format!("warning: {}", warning.message));
    }
    record.print(ctx);
    let watch = format!("cozy run watch {}", run.number);
    let show = format!("cozy run show {}", run.number);
    match run.status {
        Status::Completed => {}
        Status::Failed | Status::Canceled => output::next(ctx, &[&show]),
        _ => output::next(ctx, &[&watch]),
    }
}

/// The status as people read it: `canceling` once a cancel is asked, `⚠` when warned.
fn status(run: &Run) -> String {
    let mut status = match run.status {
        Status::Queued => "queued",
        Status::InProgress => "in_progress",
        Status::Completed => "completed",
        Status::Failed => "failed",
        Status::Canceled => "canceled",
    }
    .to_owned();
    if run.cancel_requested && !settled(run) {
        status = "canceling".into();
    }
    if !run.warnings.is_empty() {
        status.push_str(" ⚠");
    }
    status
}

fn item_name(item: &OutputItem) -> String {
    match item.index {
        0 => item.name.clone(),
        index => format!("{} {index}", item.name),
    }
}

/// The live view: one line redrawn in place on a terminal; sparse appended lines elsewhere.
struct View {
    bar: Option<ProgressBar>,
    /// Off a terminal, what was last written: the status or stage, and the tenth reached.
    shown: (String, i64),
    machine: Option<String>,
    /// The output items already named.
    named: HashSet<String>,
}

impl View {
    fn new(ctx: &Context, run: &Run) -> Self {
        output::say(ctx, &format!("run {} · {}", run.number, run.target));
        let bar = (!ctx.json && ctx.term.stderr).then(|| {
            let bar = ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr());
            bar.set_style(
                ProgressStyle::with_template("{spinner} {msg} {wide_bar} {elapsed}")
                    .expect("a template")
                    .progress_chars("━╸ "),
            );
            bar.enable_steady_tick(Duration::from_millis(120));
            bar
        });
        let mut view = Self {
            bar,
            shown: (String::new(), -1),
            machine: None,
            named: HashSet::new(),
        };
        view.lifecycle(run);
        view
    }

    fn line(&self, text: &str) {
        match &self.bar {
            Some(bar) => bar.println(text),
            None => anstream::eprintln!("{text}"),
        }
    }

    fn on(&mut self, event: &Event) {
        match event {
            Event::Created { run, .. }
            | Event::Queued { run, .. }
            | Event::InProgress { run, .. } => self.lifecycle(run),
            Event::Waiting { reason } => self.status(&format!("waiting: {reason}")),
            Event::Progress { progress } => self.progress(progress),
            // Each output is named once, when its file first lands; a person's terminal shows
            // where to look, the log off a terminal each revision after.
            Event::OutputLanded { item, .. } => {
                match (item.path.as_ref(), self.named.insert(item.id.clone())) {
                    (Some(path), true) => {
                        self.line(&format!("  {}  {}", item_name(item), path.display()))
                    }
                    (Some(_), false) if self.bar.is_none() => {
                        self.line(&format!(
                            "  {}  r{}  {}",
                            item_name(item),
                            item.rev,
                            output::bytes(item.length)
                        ));
                    }
                    _ => {}
                }
            }
            Event::OutputDone { item, .. } => {
                if let Some(error) = &item.export_error {
                    self.line(&format!("  {}  not saved: {error}", item_name(item)));
                }
            }
            Event::Log { level, text, .. } if !matches!(level.as_str(), "info" | "debug") => {
                self.line(&format!("  {level}: {text}"));
            }
            Event::Call { call, .. } => {
                let mut line = format!("  call {}", call.function);
                if !call.label.is_empty() {
                    let _ = write!(line, " ({})", call.label);
                }
                let _ = write!(line, ": {}", call.status);
                if let Some(reason) = &call.reason {
                    let _ = write!(line, ", {}", reason.message);
                }
                self.line(&line);
            }
            _ => {}
        }
    }

    fn lifecycle(&mut self, run: &Run) {
        if run.machine.is_some() && run.machine != self.machine {
            self.machine = run.machine.clone();
            self.line(&format!(
                "  on {}",
                run.machine.as_deref().unwrap_or_default()
            ));
        }
        match run.status {
            Status::Queued => self.status(
                run.waiting
                    .as_deref()
                    .map_or("queued".into(), |w| format!("waiting: {w}"))
                    .as_str(),
            ),
            Status::InProgress => self.status(run.machine_state.as_deref().unwrap_or("running")),
            _ => {}
        }
    }

    fn status(&mut self, text: &str) {
        match &self.bar {
            Some(bar) => {
                bar.set_length(0);
                bar.set_message(text.to_owned());
            }
            None if self.shown.0 != text => {
                self.shown = (text.to_owned(), -1);
                anstream::eprintln!("  {text}");
            }
            None => {}
        }
    }

    fn progress(&mut self, progress: &Progress) {
        let fraction = progress.fraction.filter(|f| *f >= 0.0).or_else(|| {
            (progress.total > 0).then(|| progress.completed as f64 / progress.total as f64)
        });
        let mut text = progress.stage.clone();
        if progress.total > 0 {
            let _ = write!(text, " {}/{}", progress.completed, progress.total);
        }
        if progress.bytes_total > 0 {
            let _ = write!(
                text,
                " {}/{}",
                output::bytes(progress.bytes_done),
                output::bytes(progress.bytes_total)
            );
        }
        if progress.step_ms > 0.0 {
            let _ = write!(text, " {:.2}s/step", progress.step_ms / 1000.0);
        }
        match &self.bar {
            Some(bar) => {
                bar.set_message(text);
                if let Some(fraction) = fraction {
                    bar.set_length(1000);
                    bar.set_position((fraction.clamp(0.0, 1.0) * 1000.0) as u64);
                }
            }
            None => {
                let tenth = fraction.map_or(-1, |f| (f * 10.0) as i64);
                if self.shown.0 != progress.stage || tenth > self.shown.1 {
                    self.shown = (progress.stage.clone(), tenth);
                    anstream::eprintln!("  {text}");
                }
            }
        }
    }

    fn done(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
    }
}
