//! `cozy run show`: one run whole, with what its machine measured of it and of each call it
//! made: where its execution time went (executor start, construction, each stage and its steps,
//! and what no measurement covers), the models it ran as resolved (release and lane), each
//! component's attention and GEMM kernels, every GPU's attention kernels (the one that served,
//! the others and why not), each model's memory per step, its calls, and its log.
use std::{collections::BTreeMap, process::ExitCode};

use protocol::{
    calcifer::{Call, Event, Run},
    error::Error,
};
use serde::Deserialize;

use super::follow;
use crate::{Context, output};

/// What an executor measured (`Outcome.measurements`, `Call.measurements`), read tolerantly: a
/// newer machine's fields are ignored and absent ones are empty.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Measurements {
    #[serde(deserialize_with = "lenient")]
    attribution: Attribution,
    #[serde(deserialize_with = "lenient")]
    execution: Execution,
    #[serde(deserialize_with = "lenient")]
    observations: Vec<Observation>,
    /// The models it ran, as its machine resolved them.
    #[serde(deserialize_with = "lenient")]
    models: Vec<Model>,
    #[serde(deserialize_with = "lenient")]
    memory: Memory,
}

/// A field of a shape this CLI does not read (`null`, a newer one) reads as empty: it never
/// takes the other measurements with it.
fn lenient<'de, D: serde::Deserializer<'de>, T: Default + serde::de::DeserializeOwned>(d: D) -> Result<T, D::Error> {
    Ok(T::deserialize(serde_json::Value::deserialize(d)?).unwrap_or_default())
}

/// Its executor's memory (TensorD `memory::call::CallMemory`); `stats` is the executor's own:
/// each model's passes, the evictions with their rung and reason, the activation peak per
/// shape, and what each GPU still held at the end.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Memory {
    #[serde(deserialize_with = "lenient")]
    stats: Stats,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Stats {
    rows: Vec<Step>,
    evictions: Vec<Eviction>,
    peaks: Vec<Peak>,
    leftover_by_gpu: Vec<(u32, u64)>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Step {
    model: String,
    gpu: u32,
    resident: u64,
    paged_in: u64,
    streamed: u64,
    stall_ns: u64,
    activations: u64,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Eviction {
    model: String,
    rung: u32,
    reason: String,
    nbytes: u64,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Peak {
    model: String,
    shape: String,
    peak: u64,
    cuts: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Attribution {
    stages: BTreeMap<String, Track>,
    steps: BTreeMap<String, Track>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Track {
    count: u64,
    total_ms: f64,
    started_unix_ms: i64,
    ended_unix_ms: Option<i64>,
    first_ms: Option<f64>,
    last_ms: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Execution {
    gpus: Vec<Gpu>,
    /// An older Runtime's GPUs: `ordinal` is the GPU's number.
    ranks: Vec<Gpu>,
    /// The executor's start: its fork, imports and CUDA initialization, one after another.
    executor: Option<Legs>,
    /// Its construction's preparation (qualifying, constructing, registering its models).
    construction: Option<Legs>,
    /// Each on-demand model's build, by slot, made inside the stage that first used it.
    on_demand: BTreeMap<String, Legs>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Legs {
    legs_ms: BTreeMap<String, f64>,
    started_unix_ms: Option<i64>,
    prepare_ms: Option<f64>,
    prepared_unix_ms: Option<i64>,
}

impl Legs {
    /// When it ran: an executor from its start for its legs' sum, a construction for its
    /// `prepare_ms` until it was prepared.
    fn span(&self) -> Option<(f64, f64)> {
        match (self.started_unix_ms, self.prepare_ms, self.prepared_unix_ms) {
            (Some(start), _, _) => Some((
                start as f64,
                start as f64 + self.legs_ms.values().sum::<f64>(),
            )),
            (None, Some(ms), Some(end)) if ms > 0.0 => Some((end as f64 - ms, end as f64)),
            _ => None,
        }
    }

    /// Its three longest legs, a construction's sub-legs aside: `package_import 10.0s, …`.
    fn longest(&self) -> String {
        let mut legs: Vec<(&String, &f64)> = self
            .legs_ms
            .iter()
            .filter(|(name, ms)| **ms >= 1.0 && !name.contains("construct."))
            .collect();
        legs.sort_by(|a, b| b.1.total_cmp(a.1));
        let said: Vec<String> = legs
            .iter()
            .take(3)
            .map(|(name, ms)| format!("{name} {}", duration(**ms)))
            .collect();
        said.join(", ")
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Gpu {
    gpu: Option<i64>,
    ordinal: Option<i64>,
    arch: String,
    attention: Attention,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Attention {
    /// The request's pin ("" = the machine's own selection).
    requested: String,
    kernels: Vec<Kernel>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Kernel {
    kernel: String,
    state: String,
    served: bool,
    progress: Option<f64>,
    compile_ms: f64,
    detail: String,
}

/// One kernel choice the executor confessed: `attention.selected` and `gemm.selected` per
/// component, `gemm.fallback` when a component's GEMM leaf could not serve (its log warns).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Observation {
    name: String,
    value: serde_json::Value,
    fields: Fields,
}

impl Observation {
    fn value(&self) -> String {
        match &self.value {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Fields {
    component: String,
    format: String,
    wanted: String,
    served: String,
    reason: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Model {
    slot: String,
    repository: String,
    release: String,
    lane: String,
    /// The header's precision lane (`int8`, …), shown beside the variant's path.
    precision: String,
    manifest: String,
}

pub async fn exec(ctx: &Context, id: &str, once: bool) -> Result<ExitCode, Error> {
    let calcifer = ctx.calcifer().await?;
    let run: Run = calcifer
        .get_query(&format!("/v1/runs/{id}"), &[("detail", "full")])
        .await?;
    // A run not yet settled is followed on a terminal, as `cozy run watch` follows it.
    if !once && !ctx.json && ctx.term.stdout && !follow::settled(&run) {
        return Ok(follow::exit_code(&follow::follow(ctx, run, false).await?));
    }
    if ctx.json {
        output::json(&run);
        return Ok(ExitCode::SUCCESS);
    }
    let execution = run
        .settled_at_ms
        .zip(run.execution_ms)
        .map(|(end, ms)| (end - ms as i64, end));
    let measured = measured(
        run.measurements.as_ref().map(|m| (m, execution)),
        &run.calls,
    );
    let mut fields = follow::fields(ctx, &run);
    if let Some(pin) = requested(&measured) {
        fields = fields.field("attention kernel", pin);
    }
    fields.print(ctx);
    let stages = grid(&["STAGE", "TIME", "DETAIL"], &measured, stage_rows);
    stages.print(ctx, "stages");
    grid(&["SLOT", "MODEL", "LANE"], &measured, model_rows).print(ctx, "models");
    let components = grid(
        &["COMPONENT", "ATTENTION", "GEMM"],
        &measured,
        component_rows,
    );
    components.print(ctx, "components");
    let gpus = grid(
        &["GPU", "ARCH", "KERNEL", "STATE", "COMPILE", "DETAIL"],
        &measured,
        gpu_rows,
    );
    gpus.print(ctx, "attention kernels");
    let memory = grid(
        &["MODEL", "GPU", "STEPS", "RESIDENT", "PAGED IN", "STREAMED", "STALL", "PEAK"],
        &measured,
        memory_rows,
    );
    memory.print(ctx, "memory");
    grid(&["MODEL", "RUNG", "REASON", "COUNT", "BYTES"], &measured, eviction_rows).print(ctx, "evictions");
    grid(&["MODEL", "SHAPE", "PEAK", "CUTS"], &measured, peak_rows).print(ctx, "activation peaks");
    if !run.calls.is_empty() {
        anstream::println!("\ncalls");
        let mut table =
            output::Table::new(ctx, ["#", "FUNCTION", "LABEL", "STATUS", "TIME", "REASON"]);
        for call in &run.calls {
            let ms = (call.finished_at_ms - call.called_at_ms).max(0) as f64;
            let status = if call.memoized {
                format!("{} (memoized)", call.status)
            } else {
                call.status.clone()
            };
            let reason = call
                .reason
                .as_ref()
                .map(|r| r.message.clone())
                .unwrap_or_default();
            table = table.row([
                call.index.to_string(),
                call.function.clone(),
                call.label.clone(),
                status,
                duration(ms),
                reason,
            ]);
        }
        table.print();
    }
    if follow::settled(&run) {
        // A settled run's stream replays its log and closes.
        let mut events = calcifer.events(run.id.as_str(), 0);
        let mut lines = Vec::new();
        // Each stage the run entered, at its time since the run was accepted: what a run whose
        // machine was lost has to say about where it got to.
        let start = run.accepted_at_ms.unwrap_or(run.created_at_ms);
        while let Some(event) = events.next().await {
            match event? {
                Event::Log { level, text, at_ms, .. } if level == "stage" => {
                    lines.push(format!("  stage: {text} (at {})", duration((at_ms - start).max(0) as f64)));
                }
                Event::Log { level, text, .. } => lines.push(format!("  {level}: {text}")),
                _ => {}
            }
        }
        if !lines.is_empty() {
            anstream::println!("\nlog");
            for line in lines {
                anstream::println!("{line}");
            }
        }
    }
    match follow::settled(&run) {
        true => follow::toolchain_offer(ctx, &run),
        false => follow::next(ctx, &run),
    }
    Ok(ExitCode::SUCCESS)
}

/// One executor's measurements, by who measured them (`run`, or the call's index), and the
/// span they account for: the run's execution, or the call from its call to its end.
struct Measured {
    who: String,
    span: Option<(i64, i64)>,
    m: Measurements,
}

type Span = Option<(i64, i64)>;

fn measured(own: Option<(&serde_json::Value, Span)>, calls: &[Call]) -> Vec<Measured> {
    let read = |who: String, m: &serde_json::Value, span: Span| {
        let m = Measurements::deserialize(m).ok()?;
        Some(Measured { who, span, m })
    };
    let own = own.and_then(|(m, span)| read("run".into(), m, span));
    let calls = calls.iter().filter_map(|c| {
        let span =
            (c.finished_at_ms >= c.called_at_ms).then_some((c.called_at_ms, c.finished_at_ms));
        read(c.index.to_string(), c.measurements.as_ref()?, span)
    });
    own.into_iter().chain(calls).collect()
}

/// A table's rows before they meet the terminal. With more than one measured, each row leads
/// with who measured it.
struct Grid {
    header: Vec<&'static str>,
    rows: Vec<Vec<String>>,
}

fn grid(
    header: &[&'static str],
    measured: &[Measured],
    rows_of: fn(&Measured) -> Vec<Vec<String>>,
) -> Grid {
    let lead = measured.len() > 1;
    let header = lead
        .then_some("CALL")
        .into_iter()
        .chain(header.iter().copied());
    let rows = measured.iter().flat_map(|m| {
        rows_of(m).into_iter().map(move |row| {
            let who = lead.then(|| m.who.clone());
            who.into_iter().chain(row).collect::<Vec<_>>()
        })
    });
    Grid {
        header: header.collect(),
        rows: rows.collect(),
    }
}

impl Grid {
    fn print(self, ctx: &Context, title: &str) {
        if self.rows.is_empty() {
            return;
        }
        anstream::println!("\n{title}");
        let table = output::Table::new(ctx, self.header);
        self.rows
            .into_iter()
            .fold(table, output::Table::row)
            .print();
    }
}

/// Where the span's time went, in the order it ran: the executor's start and its construction
/// when they ran inside it (a warm executor's were an earlier call's), each stage with its steps,
/// and last what no measurement covers.
fn stage_rows(measured: &Measured) -> Vec<Vec<String>> {
    let (attribution, execution) = (&measured.m.attribution, &measured.m.execution);
    let inside = |(start, end): (f64, f64)| {
        measured
            .span
            .is_none_or(|(from, to)| end > from as f64 && start < to as f64)
    };
    // Each row by its start, and the times it covers: a stage met twice covers its first and
    // its last occurrence; those between are counted, not placed.
    let (mut rows, mut covers, mut unplaced) = (vec![], vec![], 0.0);
    let built = execution
        .on_demand
        .iter()
        .map(|(slot, legs)| (format!("loading {slot}"), Some(legs)));
    let prepared = [
        ("executor start".to_string(), execution.executor.as_ref()),
        ("construction".to_string(), execution.construction.as_ref()),
    ];
    for (name, legs) in prepared.into_iter().chain(built) {
        if let Some((legs, span)) = legs
            .and_then(|l| Some((l, l.span()?)))
            .filter(|(_, s)| inside(*s))
        {
            covers.push(span);
            let row = vec![name, duration(span.1 - span.0), legs.longest()];
            rows.push((span.0, row));
        }
    }
    let mut stages: Vec<(&String, &Track)> = attribution.stages.iter().collect();
    for (name, track) in &attribution.steps {
        if !attribution.stages.contains_key(name) {
            stages.push((name, track));
        }
    }
    for (name, track) in stages {
        let steps = attribution.steps.get(name).map_or_else(String::new, |s| {
            let each = s.total_ms / s.count.max(1) as f64;
            format!("{} × {}", s.count, duration(each))
        });
        let start = track.started_unix_ms as f64;
        let end = track
            .ended_unix_ms
            .map_or(start + track.total_ms, |end| end as f64);
        match (track.count, track.first_ms, track.last_ms) {
            (2.., Some(first), Some(last)) => {
                covers.extend([(start, start + first), (end - last, end)]);
                unplaced += track.total_ms - first - last;
            }
            _ => covers.push((start, end)),
        }
        rows.push((start, vec![name.clone(), duration(track.total_ms), steps]));
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1[0].cmp(&b.1[0])));
    covers.sort_by(|a, b| a.0.total_cmp(&b.0));
    let unaccounted = measured.span.map(|(from, to)| {
        let (from, to) = (from as f64, to as f64);
        let (mut covered, mut reached) = (unplaced, from);
        for (start, end) in covers {
            let (start, end) = (start.max(reached), end.min(to));
            if end > start {
                covered += end - start;
                reached = end;
            }
        }
        to - from - covered
    });
    let mut rows: Vec<Vec<String>> = rows.into_iter().map(|(_, row)| row).collect();
    if let Some(ms) = unaccounted.filter(|ms| *ms >= 1.0) {
        rows.push(vec![
            "unaccounted".into(),
            duration(ms),
            "no measurement covers it".into(),
        ]);
    }
    rows
}

/// Each model slot's checkpoint: its repository and release, or its hash when it was passed
/// by value, and the variant it was resolved in with its header's precision lane.
fn model_rows(measured: &Measured) -> Vec<Vec<String>> {
    let measured = &measured.m;
    let row = |m: &Model| {
        let model = match m.repository.is_empty() {
            true => m.manifest.chars().take(19).collect(),
            false => format!("{}@{}", m.repository, m.release),
        };
        let lane = match m.precision.is_empty() {
            true => or_dash(&m.lane),
            false => format!("{} ({})", or_dash(&m.lane), m.precision),
        };
        vec![m.slot.clone(), model, lane]
    };
    measured.models.iter().map(row).collect()
}

/// Each component's attention and GEMM kernels (a 16-bit lane's components have no GEMM row),
/// a GEMM that served below its lane with what it wanted and why.
fn component_rows(measured: &Measured) -> Vec<Vec<String>> {
    let measured = &measured.m;
    let mut components: BTreeMap<&str, [String; 2]> = BTreeMap::new();
    for o in &measured.observations {
        let fields = &o.fields;
        let (column, said) = match o.name.as_str() {
            "attention.selected" => (0, o.value()),
            "gemm.selected" => (1, format!("{} ({})", o.value(), fields.format)),
            "gemm.fallback" => {
                let kernel = o.value();
                let served = [&fields.format, &fields.served]
                    .into_iter()
                    .find(|s| !s.is_empty() && **s != kernel);
                let served = served.map_or(String::new(), |s| format!("{s}; "));
                let (wanted, reason) = (&fields.wanted, &fields.reason);
                (1, format!("{kernel} ({served}wanted {wanted}: {reason})"))
            }
            _ => continue,
        };
        let cell = &mut components.entry(&fields.component).or_default()[column];
        if !cell.is_empty() {
            cell.push_str("; ");
        }
        cell.push_str(&said);
    }
    let row = |(component, [attention, gemm]): (&str, [String; 2])| {
        vec![component.to_owned(), or_dash(&attention), or_dash(&gemm)]
    };
    components.into_iter().map(row).collect()
}

/// The attention kernel the run pinned, as its executors recorded it.
fn requested(measured: &[Measured]) -> Option<String> {
    let mut pins: Vec<&str> = measured
        .iter()
        .flat_map(|m| m.m.execution.gpus.iter().chain(&m.m.execution.ranks))
        .map(|gpu| gpu.attention.requested.as_str())
        .filter(|pin| !pin.is_empty())
        .collect();
    pins.sort_unstable();
    pins.dedup();
    (!pins.is_empty()).then(|| pins.join(", "))
}

/// Every GPU's attention kernels: the one that served, and the others with why not.
fn gpu_rows(measured: &Measured) -> Vec<Vec<String>> {
    let measured = &measured.m;
    let execution = &measured.execution;
    let gpus = if execution.gpus.is_empty() {
        &execution.ranks
    } else {
        &execution.gpus
    };
    let mut rows = vec![];
    for gpu in gpus {
        let number = gpu
            .gpu
            .or(gpu.ordinal)
            .filter(|n| *n >= 0)
            .map_or("?".into(), |n| n.to_string());
        for kernel in &gpu.attention.kernels {
            let state = match (kernel.served, kernel.progress) {
                (true, _) => "served".to_owned(),
                (false, Some(p)) => format!("{} ({:.0}%)", kernel.state, p * 100.0),
                (false, None) => kernel.state.clone(),
            };
            let compile = match kernel.compile_ms > 0.0 {
                true => duration(kernel.compile_ms),
                false => "-".into(),
            };
            rows.push(vec![
                number.clone(),
                gpu.arch.clone(),
                kernel.kernel.clone(),
                state,
                compile,
                or_dash(&kernel.detail),
            ]);
        }
    }
    rows
}

/// Each model's passes in the order it first ran: the last pass's resident bytes, the bytes
/// copied home and streamed over all of them, the prefetch stall, and the largest activation
/// peak; then any torch bytes a GPU still held at the request's end.
fn memory_rows(measured: &Measured) -> Vec<Vec<String>> {
    let memory = &measured.m.memory.stats;
    let mut order: Vec<(&str, u32)> = vec![];
    for step in &memory.rows {
        if !order.contains(&(step.model.as_str(), step.gpu)) {
            order.push((&step.model, step.gpu));
        }
    }
    let mut rows: Vec<Vec<String>> = order
        .into_iter()
        .map(|(model, gpu)| {
            let steps: Vec<&Step> = memory.rows.iter().filter(|s| s.model == model && s.gpu == gpu).collect();
            let sum = |f: fn(&Step) -> u64| steps.iter().map(|s| f(s)).sum::<u64>();
            vec![
                model.into(),
                gpu.to_string(),
                steps.len().to_string(),
                output::bytes(steps.last().map_or(0, |s| s.resident)),
                output::bytes(sum(|s| s.paged_in)),
                output::bytes(sum(|s| s.streamed)),
                duration(sum(|s| s.stall_ns) as f64 / 1e6),
                output::bytes(steps.iter().map(|s| s.activations).max().unwrap_or(0)),
            ]
        })
        .collect();
    for (gpu, bytes) in memory.leftover_by_gpu.iter().filter(|(_, bytes)| *bytes > 0) {
        let dash = || "-".to_string();
        rows.push(vec!["leftover".into(), gpu.to_string(), dash(), output::bytes(*bytes), dash(), dash(), dash(), dash()]);
    }
    rows
}

/// Evictions by model, rung and reason.
fn eviction_rows(measured: &Measured) -> Vec<Vec<String>> {
    let mut counted: BTreeMap<(&str, u32, &str), (u64, u64)> = BTreeMap::new();
    for e in &measured.m.memory.stats.evictions {
        let entry = counted.entry((&e.model, e.rung, &e.reason)).or_default();
        *entry = (entry.0 + 1, entry.1 + e.nbytes);
    }
    let row = |((model, rung, reason), (count, bytes)): ((&str, u32, &str), (u64, u64))| {
        vec![model.into(), rung.to_string(), reason.into(), count.to_string(), output::bytes(bytes)]
    };
    counted.into_iter().map(row).collect()
}

/// The activation peak each model measured at each shape key, and the cuts in force there.
fn peak_rows(measured: &Measured) -> Vec<Vec<String>> {
    let row = |p: &Peak| vec![p.model.clone(), p.shape.clone(), output::bytes(p.peak), or_dash(&p.cuts.join(", "))];
    measured.m.memory.stats.peaks.iter().map(row).collect()
}

fn or_dash(text: &str) -> String {
    match text.is_empty() {
        true => "-".into(),
        false => text.into(),
    }
}

/// A time as people read it: `850ms`, `7.6s`, `3m48.7s`.
fn duration(ms: f64) -> String {
    match ms {
        ms if ms < 1000.0 => format!("{ms:.0}ms"),
        ms if ms < 60_000.0 => format!("{:.1}s", ms / 1000.0),
        ms => format!(
            "{}m{:.1}s",
            (ms / 60_000.0).floor(),
            (ms % 60_000.0) / 1000.0
        ),
    }
}
