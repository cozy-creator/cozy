//! Following a run: live progress on stderr, each output named as it lands, and the run's record
//! on stdout at the end; NDJSON of Calcifer's events with `--json`, its terminal event last. A
//! followed run is over once it ended and each output's last revision landed here (checked
//! against its digest) or failed; an output not saved fails the follow. Ctrl-C (Esc or q in
//! `cozy run list`) stops following and never cancels: the run is Calcifer's, and `cozy run
//! watch` reattaches.
use std::{
    collections::{HashSet, VecDeque},
    fmt::Write as _,
    process::ExitCode,
    time::{Duration, Instant},
};

use protocol::names::shown;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use protocol::{
    calcifer::{
        Adjustment, Event, Machine, MachineState, On, OutputItem, Progress, Run, RunQuote, Status,
        SubmitRun, Submitted,
    },
    error::{Error, Kind},
};

use crate::{
    Context,
    calcifer_client::Client,
    machine::{Following, Notices, route},
    output,
};

/// Submits under `key`, or a fresh one: one key names one submission forever. A run that rents
/// carries the quote its renter agreed to (`machine::agree`).
pub(crate) async fn submit(
    ctx: &Context,
    mut run: SubmitRun,
    key: Option<&str>,
) -> Result<Submitted, Error> {
    let key = key.map_or_else(mint_key, str::to_owned);
    let calcifer = ctx.calcifer().await?;
    if matches!(run.on, On::New(_)) && run.agreed.is_none() {
        let quoted: RunQuote = calcifer.post("/v1/run-quotes", &run, None).await?;
        let said = crate::machine::placed(&quoted);
        if !crate::machine::agree(ctx, &said)? {
            return Err(Error::new(Kind::Usage, "quote_declined", "nothing was rented and the run was not sent"));
        }
        run.agreed = Some(quoted.quote.agreed());
    }
    let submitted = calcifer.submit(&key, &run).await?;
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
    follow_until(ctx, run, interrupt()).await
}

/// `cozy run watch` until `stop`, which never cancels the run: the live view, then the record.
pub(crate) async fn follow_until(
    ctx: &Context,
    run: Run,
    stop: impl Future<Output = ()>,
) -> Result<Run, Error> {
    let (run, interrupted) = watch(ctx, run, ctx.json, stop).await?;
    if !ctx.json {
        record(ctx, &run);
    }
    match interrupted {
        true => Ok(run),
        false => saved(run),
    }
}

/// Ctrl-C in a shell.
async fn interrupt() {
    let _ = tokio::signal::ctrl_c().await;
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
    Ok(watch(ctx, run, false, interrupt()).await?.0)
}

/// Renders `run`'s events until it ended and its outputs landed or failed (NDJSON with `events`,
/// its terminal event held back to come last, as the run then stands); `stop` ends it early and
/// says so.
async fn watch(
    ctx: &Context,
    run: Run,
    events_out: bool,
    stop: impl Future<Output = ()>,
) -> Result<(Run, bool), Error> {
    let calcifer = ctx.calcifer().await?;
    let mut events = calcifer.events_until_landed(run.id.as_str(), 0);
    let mut view = View::new(ctx, &run);
    let mut interrupt = std::pin::pin!(stop);
    let id = run.id.clone();
    let mut terminal = None;
    // The rental the run was placed on, followed until it is known whether its Hub placed it far
    // away: the Hub decides after it accepts the order.
    let mut placing: Option<Following> = None;
    let mut notices = Notices::default();
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    loop {
        tokio::select! {
            _ = tick.tick() => view.tick(),
            next = next_of(&mut placing, &calcifer), if placing.is_some() => match next {
                Ok(Some(machine)) if !placed(&view, &machine, &mut notices) => {}
                _ => placing = None,
            },
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
                Some(Ok(event)) => {
                    let before = view.machine.clone();
                    view.on(&event);
                    if let Some(name) = view.machine.as_deref().filter(|_| view.machine != before) {
                        placing = follow_placement(&calcifer, &view, name, &mut notices).await;
                    }
                }
            }
        }
    }
}

/// Follows the machine a run was placed on while it boots, saying the Hub's notices of it as they
/// appear; a machine already up needs no following.
async fn follow_placement(
    calcifer: &Client,
    view: &View,
    name: &str,
    notices: &mut Notices,
) -> Option<Following> {
    let machine: Machine = calcifer.get(&route(name).ok()?).await.ok()?;
    if placed(view, &machine, notices) {
        return None;
    }
    let (following, machine) = Following::open(calcifer, name).await.ok()?;
    (!placed(view, &machine, notices)).then_some(following)
}

async fn next_of(
    placing: &mut Option<Following>,
    calcifer: &Client,
) -> Result<Option<Machine>, Error> {
    match placing {
        Some(following) => following.next(calcifer).await,
        None => std::future::pending().await,
    }
}

/// Says each notice the Hub gives of the run's rental as it appears: true once the machine is done
/// booting, when nothing more is followed.
fn placed(view: &View, machine: &Machine, notices: &mut Notices) -> bool {
    for note in notices.fresh(machine) {
        view.line(&format!("  {note}"));
    }
    machine.state != MachineState::Booting
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

/// A checkpoint's TensorFS reference, or the location of its legacy manifest bytes.
fn checkpoint(reference: &protocol::machine::CheckpointRef) -> String {
    let mut shown = format!("checkpoint {}@{}", reference.repo_id, reference.revision);
    if !reference.variant.is_empty() {
        let _ = write!(shown, " (variant {})", reference.variant);
    }
    shown
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
    if let Some(progress) = run.progress.as_ref().filter(|_| !settled(run)) {
        record = record.field("progress", said(progress));
    }
    if let Some(ms) = run.execution_ms {
        record = record.field("execution", format!("{:.1}s", ms as f64 / 1000.0));
    }
    if let Some(reason) = &run.reason {
        record = record.field("reason", format!("{} ({})", reason.message, reason.code));
        if !reason.detail.is_empty() {
            record = record.field("traceback", &reason.detail);
        }
    }
    for item in &run.output {
        let shown = match (&item.path, &item.export_error) {
            _ if let Some(reference) = &item.checkpoint => checkpoint(reference),
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
    for row in &run.adjustments {
        record = record.note(adjustment(row));
    }
    for warning in &run.warnings {
        record = record.note(format!("warning: {}", warning.message));
    }
    record
}

/// One adjustment as a line: `adjusted assets.2.asset: 21.4 -> 15 (video truncated to 15 s)
/// [policy]`; a warning reads `warn field: reason [source]`.
fn adjustment(row: &Adjustment) -> String {
    let mut line = format!("{} {}:", row.kind, row.field);
    if row.kind != "warn" {
        let _ = write!(line, " {} -> {}", row.requested, row.applied);
    }
    let _ = match (row.kind.as_str(), row.reason.as_str()) {
        (_, "") => Ok(()),
        ("warn", reason) => write!(line, " {reason}"),
        (_, reason) => write!(line, " ({reason})"),
    };
    if !row.source.is_empty() {
        let _ = write!(line, " [{}]", row.source);
    }
    line
}

/// The Runtime's warnings a local compile would answer (K's codes): the toolchain is offered,
/// never installed on its own.
const TOOLCHAIN_WOULD_HELP: [&str; 4] = [
    "kernel_prebuilt_missing",
    "kernel_verify_failed",
    "extension_needs_toolchain",
    "triton_no_compiler",
];

/// What to do next with `run`: watch it while it goes on, show it when it did not succeed, and
/// the machine's toolchain when a warning says a local compile would have served it.
pub(super) fn next(ctx: &Context, run: &Run) {
    let watch = format!("cozy run watch {}", run.number);
    let show = format!("cozy run show {}", run.number);
    match run.status {
        Status::Completed => {}
        Status::Failed | Status::Canceled => output::next(ctx, &[&show]),
        _ => output::next(ctx, &[&watch]),
    }
    toolchain_offer(ctx, run);
}

/// `cozy machine toolchain install` when a warning of `run` says a local compile would have
/// served it.
pub(super) fn toolchain_offer(ctx: &Context, run: &Run) {
    let compile = run
        .warnings
        .iter()
        .any(|w| TOOLCHAIN_WOULD_HELP.contains(&w.code.as_str()));
    if let (true, Some(machine)) = (compile, &run.machine) {
        output::next(
            ctx,
            &[&format!("cozy machine toolchain install --on {machine}")],
        );
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
    /// The run's own word for where it is (`queued`, the machine's state), and why it waits now
    /// (Calcifer's reason, as `cozy run list` shows it): the status line says the reason first.
    state: String,
    waiting: Option<String>,
    status: Option<Status>,
    /// The output items already named.
    named: HashSet<String>,
    /// The stage moving bytes, and its last few seconds of (when, bytes moved): its rate.
    moving: (String, VecDeque<(Instant, u64)>),
    /// The progress last shown, and when its step count last moved: a step running far past its
    /// stage's pace says so.
    last: Option<(Progress, Instant)>,
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
            state: String::new(),
            waiting: None,
            status: None,
            named: HashSet::new(),
            moving: (String::new(), VecDeque::new()),
            last: None,
        };
        view.lifecycle(run);
        view
    }

    fn line(&self, text: &str) {
        match &self.bar {
            Some(bar) => bar.println(shown(text)),
            None => anstream::eprintln!("{}", shown(text)),
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
            Event::Waiting { reason } => {
                self.waiting = Some(reason.clone()).filter(|r| !r.is_empty());
                self.show();
            }
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
                } else if let Some(reference) = &item.checkpoint
                    && self.named.insert(item.id.clone())
                {
                    self.line(&format!("  {}  {}", item_name(item), checkpoint(reference)));
                }
            }
            // A stage the run entered shows in its progress line as it goes.
            Event::Log { level, text, .. } if !matches!(level.as_str(), "info" | "debug" | "stage") => {
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
        self.state = match (run.status, run.machine_state.as_deref(), &run.machine) {
            (Status::Queued, ..) => "queued".into(),
            (Status::InProgress, Some("queued"), Some(machine)) => format!("queued on {machine}"),
            (Status::InProgress, state, _) => state.unwrap_or("running").into(),
            _ => return,
        };
        // A replayed event of the same status knows no reason; a new status says its own.
        if run.waiting.is_some() || self.status != Some(run.status) {
            self.waiting.clone_from(&run.waiting);
        }
        self.status = Some(run.status);
        self.show();
    }

    /// The status line: why the run waits, else where it is.
    fn show(&mut self) {
        let text = self.waiting.clone().unwrap_or_else(|| self.state.clone());
        self.status(&text);
    }

    fn status(&mut self, text: &str) {
        self.last = None; // a status says where the run is now, not its last progress
        match &self.bar {
            Some(bar) => {
                bar.set_length(0);
                bar.set_message(shown(text));
            }
            None if self.shown.0 != text => {
                self.shown = (text.to_owned(), -1);
                anstream::eprintln!("  {}", shown(text));
            }
            None => {}
        }
    }

    fn progress(&mut self, progress: &Progress) {
        let stepped = match &self.last {
            Some((last, at)) if (&last.stage, last.completed) == (&progress.stage, progress.completed) => *at,
            _ => Instant::now(),
        };
        self.last = Some((progress.clone(), stepped));
        self.show_progress();
    }

    /// The last progress again, as time passes: a step running long says so.
    fn tick(&mut self) {
        if self.bar.is_some() && self.last.is_some() {
            self.show_progress();
        }
    }

    fn show_progress(&mut self) {
        let Some((progress, stepped)) = self.last.clone() else { return };
        let fraction = fraction(&progress);
        let mut text = said(&progress);
        // At what rate bytes move, and so how long is left: a download that crawls says so.
        let (done, total) = (progress.bytes_done, progress.bytes_total);
        if (done > 0 || total > 0)
            && let Some(rate) = rate(&mut self.moving, &progress.stage, done, Instant::now())
        {
            let _ = write!(text, " at {}/s", output::bytes(rate as u64));
            if total > done && rate >= 1.0 {
                let left = ((total - done) as f64 / rate) as i64;
                let _ = write!(text, ", {} left", output::span(left));
            }
        }
        // A step running twice its stage's pace or more is named, with how long it has run.
        let pace = if progress.step_avg_ms > 0.0 { progress.step_avg_ms } else { progress.step_ms };
        let running = stepped.elapsed().as_secs_f64() * 1000.0;
        if pace > 0.0 && progress.completed < progress.total && running > 2.0 * pace {
            let _ = write!(text, " · step {} running {}", progress.completed + 1, took(running));
        }
        if let Some(waiting) = &self.waiting {
            text = format!("{waiting} · {text}");
        }
        match &self.bar {
            Some(bar) => {
                bar.set_message(shown(&text));
                if let Some(fraction) = fraction {
                    bar.set_length(1000);
                    bar.set_position((fraction.clamp(0.0, 1.0) * 1000.0) as u64);
                }
            }
            None => {
                let tenth = fraction.map_or(-1, |f| (f * 10.0) as i64);
                if self.shown.0 != progress.stage || tenth > self.shown.1 {
                    self.shown = (progress.stage.clone(), tenth);
                    anstream::eprintln!("  {}", shown(&text));
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

/// How far a run is: its own fraction, else its steps', else its bytes'.
pub(crate) fn fraction(progress: &Progress) -> Option<f64> {
    let of = |done: u64, total: u64| (total > 0).then(|| done as f64 / total as f64);
    (progress.fraction.filter(|f| *f >= 0.0))
        .or_else(|| of(progress.completed, progress.total))
        .or_else(|| of(progress.bytes_done, progress.bytes_total))
}

/// Where a run is: its stage, its steps of their total and their times, and its bytes of theirs.
/// A stage's first step carries its setup: its time is said as the first's, never as the pace.
pub(crate) fn said(progress: &Progress) -> String {
    let mut text = progress.stage.clone();
    if progress.total > 0 {
        let _ = write!(text, " {}/{}", progress.completed, progress.total);
    }
    let (done, total) = (progress.bytes_done, progress.bytes_total);
    if done > 0 || total > 0 {
        let _ = write!(text, " {}", output::bytes(done));
        if total > 0 {
            let _ = write!(text, "/{}", output::bytes(total));
        }
    }
    let (last, steady) = (progress.step_ms, progress.step_avg_ms);
    let _ = match (last > 0.0, progress.completed, steady > 0.0) {
        (false, ..) => Ok(()),
        (true, 1, _) => write!(text, " · first step {}", took(last)),
        (true, _, true) => write!(text, " · {} last, {} avg", took(last), took(steady)),
        (true, ..) => write!(text, " · {} last", took(last)),
    };
    text
}

/// A step's time: tenths of a second under a minute, else minutes and seconds.
fn took(ms: f64) -> String {
    match ms < 60_000.0 {
        true => format!("{:.1}s", ms / 1000.0),
        false => output::span((ms / 1000.0) as i64),
    }
}

/// A stage's bytes a second over its last 10 s (and at least its last two counts), once a second
/// of them is seen. A new stage, or a count that went back, starts again.
fn rate(moving: &mut (String, VecDeque<(Instant, u64)>), stage: &str, done: u64, now: Instant) -> Option<f64> {
    let (name, seen) = moving;
    if name != stage || seen.back().is_some_and(|(_, last)| done < *last) {
        (*name, *seen) = (stage.to_owned(), VecDeque::new());
    }
    seen.push_back((now, done));
    while seen.len() > 2 && seen.front().is_some_and(|(at, _)| now - *at > Duration::from_secs(10)) {
        seen.pop_front();
    }
    let (at, first) = *seen.front()?;
    let secs = (now - at).as_secs_f64();
    (secs >= 1.0).then(|| (done - first) as f64 / secs)
}
