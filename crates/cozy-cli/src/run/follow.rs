//! Following a run: live progress on stderr, each output named as it lands, and the run's record
//! on stdout at the end; NDJSON of Calcifer's events with `--json`, its terminal event last. A
//! followed run is over once it ended and each output's last revision landed here (checked
//! against its digest) or failed; an output not saved fails the follow. Ctrl-C stops following
//! and never cancels: the run is Calcifer's, and `cozy run watch` reattaches.
use std::{collections::HashSet, fmt::Write as _, process::ExitCode, time::Duration};

use cozy_proto::{
    calcifer::{Event, OutputItem, Progress, Run, Status, SubmitRun, Submitted},
    error::{Error, Kind},
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

/// Follows `run` to its end and its outputs to this computer, or prints it at once with
/// `detach`; answers the run as it ended, or as it was when following stopped. An output that
/// was not saved is a refusal naming it, after the record.
pub(crate) async fn follow(ctx: &Context, run: Run, detach: bool) -> Result<Run, Error> {
    if detach {
        record(ctx, &run);
        return Ok(run);
    }
    let (run, interrupted) = watch(ctx, run, ctx.json).await?;
    if !ctx.json {
        record(ctx, &run);
    }
    match interrupted {
        true => Ok(run),
        false => saved(run),
    }
}

/// An output still owed to this computer: its last revision neither landed nor failed. A
/// checkpoint stays on its machine and is owed nowhere.
fn owed(run: &Run) -> bool {
    run.output
        .iter()
        .any(|item| !item.is_checkpoint() && item.path.is_none() && item.export_error.is_none())
}

/// `run`, once each of its outputs' last revision landed; else a refusal naming those that did not.
fn saved(run: Run) -> Result<Run, Error> {
    let lost: Vec<String> = run
        .output
        .iter()
        .filter(|item| !item.is_checkpoint() && item.path.is_none())
        .map(|item| match &item.export_error {
            Some(error) => format!("{} ({error})", item_name(item)),
            None => format!("{} (still saving when following ended)", item_name(item)),
        })
        .collect();
    if lost.is_empty() || !settled(&run) {
        return Ok(run);
    }
    let show = format!("cozy run show {}", run.number);
    Err(Error::new(
        Kind::Unavailable,
        "output_not_saved",
        format!(
            "run {} ended {}, but this computer did not save its output {}",
            run.number,
            status_word(run.status),
            lost.join(", ")
        ),
    )
    .remedy(format!(
        "`cozy run watch {}` saves what its machine still holds",
        run.number
    ))
    .next(&show))
}

/// Follows a run another one waits on (local code's install): its progress on a terminal, no
/// record and no events of its own.
pub(super) async fn settle(ctx: &Context, run: Run) -> Result<Run, Error> {
    Ok(watch(ctx, run, false).await?.0)
}

/// Renders `run`'s events until it ended and its outputs landed or failed (NDJSON with `events`,
/// its terminal event held back to come last, as the run then stands); Ctrl-C stops early and
/// says so.
async fn watch(ctx: &Context, run: Run, events_out: bool) -> Result<(Run, bool), Error> {
    let calcifer = ctx.calcifer().await?;
    let mut events = calcifer.events_until_landed(run.id.as_str(), 0);
    let mut view = View::new(ctx, &run);
    let interrupt = tokio::signal::ctrl_c();
    tokio::pin!(interrupt);
    let id = run.id.clone();
    let mut terminal = None;
    loop {
        tokio::select! {
            _ = &mut interrupt => {
                view.done();
                let run = calcifer.run(id.as_str()).await?;
                let n = run.number;
                output::say(ctx, &match settled(&run) {
                    true => format!("detached: run {n} ended and its outputs are still being saved; `cozy run watch {n}` follows them"),
                    false => format!("detached: run {n} keeps running; `cozy run watch {n}` follows it again, `cozy run cancel {n}` cancels it"),
                });
                return Ok((run, true));
            }
            event = events.next() => match event {
                None => {
                    let run = calcifer.run(id.as_str()).await?;
                    // Its Calcifer stopped with an output still owed: wait on the next one.
                    if settled(&run) && owed(&run) {
                        events.again();
                        continue;
                    }
                    view.done();
                    if let Some(sequence_number) = terminal {
                        output::ndjson(&Event::lifecycle(sequence_number, run.clone(), false));
                    }
                    return Ok((run, false));
                }
                Some(Err(error)) => {
                    view.done();
                    return Err(error);
                }
                Some(Ok(event)) if events_out && event.is_terminal() => {
                    terminal = event.sequence_number();
                }
                Some(Ok(event)) if events_out => output::ndjson(&event),
                Some(Ok(_)) if ctx.json => {}
                Some(Ok(event)) => view.on(&event),
            }
        }
    }
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
    fields(ctx, run).print(ctx);
    next(ctx, run);
}

/// A checkpoint as a record shows it: its digest, the machine it stays on, and how to publish it.
fn checkpoint(run: &Run, item: &OutputItem) -> String {
    let digest = item
        .sha256
        .map_or_else(|| "not final yet".into(), |d| d.to_string());
    let on = run.machine.as_deref().unwrap_or("its machine");
    format!(
        "checkpoint {digest} on {on} (`cozy run upload {}#{} <org/model>`)",
        run.number, item.name
    )
}

/// The record's fields, for a person.
pub(super) fn fields(ctx: &Context, run: &Run) -> output::Record {
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
            _ if item.is_checkpoint() => checkpoint(run, item),
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
    record
}

/// What to do next with `run`: watch it while it goes on, show it when it did not succeed.
pub(super) fn next(ctx: &Context, run: &Run) {
    let watch = format!("cozy run watch {}", run.number);
    let show = format!("cozy run show {}", run.number);
    match run.status {
        Status::Completed => {}
        Status::Failed | Status::Canceled => output::next(ctx, &[&show]),
        _ => output::next(ctx, &[&watch]),
    }
}

/// A status's word.
pub(super) fn status_word(status: Status) -> &'static str {
    match status {
        Status::Queued => "queued",
        Status::InProgress => "in_progress",
        Status::Completed => "completed",
        Status::Failed => "failed",
        Status::Canceled => "canceled",
    }
}

/// The status as people read it: `canceling` once a cancel is asked, `⚠` when warned.
pub(super) fn status(run: &Run) -> String {
    let mut status = status_word(run.status).to_owned();
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
            Event::Completed { run, .. }
            | Event::Failed { run, .. }
            | Event::Canceled { run, .. }
                if owed(run) =>
            {
                self.status("saving its outputs to this computer")
            }
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
                } else if item.is_checkpoint() && self.named.insert(item.id.clone()) {
                    let on = self.machine.as_deref().unwrap_or("its machine");
                    let digest = item.sha256.map(|d| d.to_string()).unwrap_or_default();
                    self.line(&format!(
                        "  {}  checkpoint {digest} on {on}",
                        item_name(item)
                    ));
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

#[cfg(test)]
mod tests {
    use cozy_proto::{
        calcifer::{CHECKPOINT, ItemStatus},
        ids::RunId,
    };

    use super::*;

    /// A settled run with one output of each kind: (name, media type, landed path, error).
    fn run(outputs: &[(&str, &str, Option<&str>, Option<&str>)]) -> Run {
        let mut run = serde_json::from_value::<Run>(serde_json::json!({
            "id": RunId::mint([7; 12]), "number": 7, "kind": "call", "target": "fidika/flux/fp8",
            "hub": "https://tensorhub.com", "machine": "m1", "status": Status::Completed,
            "created_at_ms": 1,
        }))
        .unwrap();
        run.output = (outputs.iter())
            .map(|(name, media_type, path, error)| OutputItem {
                id: format!("7/{name}"),
                name: (*name).to_owned(),
                index: 0,
                rev: 1,
                length: 1,
                media_type: (*media_type).to_owned(),
                label: String::new(),
                status: ItemStatus::Completed,
                sha256: None,
                path: path.map(Into::into),
                export_error: error.map(Into::into),
            })
            .collect();
        run
    }

    /// Each output kind's terminal state: a landed file, a lost one and a checkpoint end a
    /// follow (and only a lost file fails it); a file still on its way keeps it going.
    #[test]
    fn a_follow_ends_once_each_output_is_landed_lost_or_a_checkpoint() {
        let image = "image/webp";
        let cases = [
            (("image", image, Some("/o/7-image.webp"), None), false, true),
            (
                ("image", image, None, Some("its machine ended")),
                false,
                false,
            ),
            (("fp8", CHECKPOINT, None, None), false, true),
            (("image", image, None, None), true, false),
        ];
        for (output, owes, saves) in cases {
            let run = run(&[output]);
            assert_eq!(owed(&run), owes, "{output:?}");
            assert_eq!(saved(run).is_ok(), saves, "{output:?}");
        }
    }
}
