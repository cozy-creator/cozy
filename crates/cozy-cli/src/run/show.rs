//! `cozy run show`: one run whole, with what its machine measured of it and of each call it
//! made: where its execution time went (executor start, construction, each stage and its steps,
//! and what no measurement covers), the models it ran as resolved (release and lane), each
//! component's attention and GEMM kernels, every GPU's attention kernels (the one that served,
//! the others and why not), its calls, and its log.
use std::{collections::BTreeMap, process::ExitCode};

use cozy_proto::{
    calcifer::{Call, Event, Run},
    error::Error,
};
use serde::Deserialize;

use super::follow;
use crate::{Context, output, tui};

/// What an executor measured (`Outcome.measurements`, `Call.measurements`), read tolerantly: a
/// newer machine's fields are ignored and absent ones are empty.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Measurements {
    attribution: Attribution,
    execution: Execution,
    observations: Vec<Observation>,
    /// The models it ran, as its machine resolved them.
    models: Vec<Model>,
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
    manifest: String,
}

pub async fn exec(ctx: &Context, id: &str, once: bool) -> Result<ExitCode, Error> {
    let calcifer = ctx.calcifer().await?;
    let run: Run = calcifer
        .get_query(&format!("/v1/runs/{id}"), &[("detail", "full")])
        .await?;
    if !once && !ctx.json && ctx.term.stdout && !follow::settled(&run) {
        let view = super::live::Live::open(ctx, run.id.as_str()).await?;
        tui::show(ctx, tui::Screen::Run(Box::new(view))).await?;
        return Ok(ExitCode::SUCCESS);
    }
    if ctx.json {
        output::json(&run);
        return Ok(ExitCode::SUCCESS);
    }
    follow::fields(ctx, &run).print(ctx);
    let execution = run
        .settled_at_ms
        .zip(run.execution_ms)
        .map(|(end, ms)| (end - ms as i64, end));
    let measured = measured(
        run.measurements.as_ref().map(|m| (m, execution)),
        &run.calls,
    );
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
        while let Some(event) = events.next().await {
            if let Event::Log { level, text, .. } = event? {
                lines.push(format!("  {level}: {text}"));
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
    for (name, legs) in [
        ("executor start", &execution.executor),
        ("construction", &execution.construction),
    ] {
        if let Some((legs, span)) = legs
            .as_ref()
            .and_then(|l| Some((l, l.span()?)))
            .filter(|(_, s)| inside(*s))
        {
            covers.push(span);
            let row = vec![name.into(), duration(span.1 - span.0), legs.longest()];
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
/// by value, and the lane it was resolved in.
fn model_rows(measured: &Measured) -> Vec<Vec<String>> {
    let measured = &measured.m;
    let row = |m: &Model| {
        let model = match m.repository.is_empty() {
            true => m.manifest.chars().take(19).collect(),
            false => format!("{}@{}", m.repository, m.release),
        };
        vec![m.slot.clone(), model, or_dash(&m.lane)]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> serde_json::Value {
        let path = format!(
            "{}/tests/fixtures/measurements/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
    }

    fn measured_alone(name: &str) -> Measured {
        let document = fixture(name);
        let m = Measurements::deserialize(document.get("measurements").unwrap_or(&document));
        Measured {
            who: "run".into(),
            span: None,
            m: m.unwrap(),
        }
    }

    /// Run 10236 (H3 long_form on an A100): its deviceless root measured nothing; its calls
    /// measured everything.
    fn h3_calls() -> Vec<Call> {
        serde_json::from_value(fixture("h3-run-10236-calls")).unwrap()
    }

    fn rows(grid: &Grid) -> Vec<Vec<&str>> {
        grid.rows
            .iter()
            .map(|r| r.iter().map(String::as_str).collect())
            .collect()
    }

    #[test]
    fn a_real_h3_runs_stages_read_in_the_order_they_ran() {
        let h3 = measured_alone("h3-run-5243-measurements");
        let stages = stage_rows(&h3);
        assert_eq!(stages[0][0], "executor start");
        assert_eq!(stages[2][0], "prepare");
        let denoise = stages.iter().find(|r| r[0] == "denoise").unwrap();
        assert_eq!(denoise[1], "3m48.7s");
        let video = stages.iter().find(|r| r[0] == "decode_video").unwrap();
        assert_eq!(video[2], "22 × 490ms");
    }

    #[test]
    fn each_gpus_kernels_say_which_served_and_why_the_others_did_not() {
        let rows = gpu_rows(&measured_alone("kernel-gpus"));
        let gpus: Vec<&str> = rows.iter().map(|r| r[0].as_str()).collect();
        assert!(gpus.iter().all(|g| ["2", "3"].contains(g)), "{gpus:?}");
        let served: Vec<&str> = rows
            .iter()
            .filter(|r| r[0] == "2" && r[3] == "served")
            .map(|r| r[2].as_str())
            .collect();
        assert_eq!(served, ["sageattention"]);
        assert_eq!(rows[1][4], "3m5.3s");
        let sol = rows.iter().find(|r| r[2] == "sol-attn").unwrap();
        assert_eq!(sol[3], "compiling (42%)");
        assert!(rows[0][5].contains("serves sm90 only"));
    }

    /// Each call's measurements are its own table rows, led by the call: what ran where, and
    /// the root (deviceless here) adds nothing.
    #[test]
    fn a_runs_calls_each_show_their_stages_components_and_gpus() {
        let measured = measured(None, &h3_calls());
        let components = grid(
            &["COMPONENT", "ATTENTION", "GEMM"],
            &measured,
            component_rows,
        );
        assert_eq!(
            components.header,
            ["CALL", "COMPONENT", "ATTENTION", "GEMM"]
        );
        let segment = |call| {
            vec![
                vec![call, "base_model/audio_vae", "sdpa", "-"],
                vec![
                    call,
                    "base_model/ref2va_dit",
                    "kitchen-sol-producer-fast-shared-qkv",
                    "-",
                ],
                vec![call, "base_model/video_vae", "sdpa", "-"],
            ]
        };
        assert_eq!(rows(&components), [segment("1"), segment("2")].concat());

        let gpus = grid(
            &["GPU", "ARCH", "KERNEL", "STATE", "COMPILE", "DETAIL"],
            &measured,
            gpu_rows,
        );
        assert_eq!(
            rows(&gpus)[..3],
            [
                ["1", "0", "sm_80", "kitchen-sol", "ready", "804ms", "-"],
                [
                    "1",
                    "0",
                    "sm_80",
                    "kitchen-sol-producer-fast-shared-qkv",
                    "served",
                    "-",
                    "-"
                ],
                ["1", "0", "sm_80", "sdpa", "served", "-", "-"],
            ]
        );

        let stages = grid(&["STAGE", "TIME", "DETAIL"], &measured, stage_rows);
        let first: Vec<&str> = rows(&stages).iter().map(|r| r[0]).collect();
        assert_eq!(first.first(), Some(&"0"));
        let denoise = rows(&stages)
            .into_iter()
            .find(|r| r[0] == "1" && r[1] == "denoise")
            .unwrap();
        assert!(denoise[3].starts_with("8 × "), "{denoise:?}");
    }

    /// The machine's resolution and the executor's GEMM leaves, as TensorD and Runtime (#1519,
    /// #1522) record them: each call's slot with its release and lane (run 10236's background
    /// removal resolved `paul/birefnet@0.1.0 fp16`), and each component's GEMM kernel beside its
    /// attention, one that served below its lane with what it wanted and why.
    #[test]
    fn a_calls_resolved_models_and_gemm_leaves_show_beside_its_attention() {
        let mut calls = h3_calls();
        let birefnet = serde_json::json!([{"slot": "model", "repository": "paul/birefnet",
            "release": "0.1.0", "lane": "fp16", "manifest": format!("sha256:{}", "ab".repeat(32))}]);
        calls[0].measurements.as_mut().unwrap()["models"] = birefnet;
        let measurements = calls[1].measurements.as_mut().unwrap();
        let observations = measurements["observations"].as_array_mut().unwrap();
        observations.extend([
            serde_json::json!({"kind": "confession", "name": "gemm.selected", "value": "cozy-int8",
                "fields": {"component": "base_model/ref2va_dit", "format": "int8",
                "leaf": "cozy.int8-convrot256.native-leaf/1", "sites": 416}}),
            serde_json::json!({"kind": "confession", "name": "gemm.fallback", "value": "torch.linear",
                "fields": {"component": "base_model/video_vae", "wanted": "fp8", "served": "bf16",
                "reason": "the leaf refused its shape", "sites": 4}}),
            serde_json::json!({"kind": "confession", "name": "gemm.fallback", "value": "torch._int_mm",
                "fields": {"component": "base_model/audio_vae", "wanted": "cozy-int8",
                "served": "torch._int_mm", "reason": "the op library did not load", "sites": 1}}),
        ]);
        let measured = measured(None, &calls);
        let models = grid(&["SLOT", "MODEL", "LANE"], &measured, model_rows);
        assert_eq!(
            rows(&models),
            [["0", "model", "paul/birefnet@0.1.0", "fp16"]]
        );
        let components = grid(
            &["COMPONENT", "ATTENTION", "GEMM"],
            &measured,
            component_rows,
        );
        assert_eq!(
            rows(&components)[..3],
            [
                [
                    "1",
                    "base_model/audio_vae",
                    "sdpa",
                    "torch._int_mm (wanted cozy-int8: the op library did not load)"
                ],
                [
                    "1",
                    "base_model/ref2va_dit",
                    "kitchen-sol-producer-fast-shared-qkv",
                    "cozy-int8 (int8)"
                ],
                [
                    "1",
                    "base_model/video_vae",
                    "sdpa",
                    "torch.linear (bf16; wanted fp8: the leaf refused its shape)"
                ],
            ]
        );
        // A model passed by value names its hash; one measured alone leads with no call.
        let held = Measurements {
            models: vec![Model {
                slot: "model".into(),
                manifest: format!("sha256:{}", "cd".repeat(32)),
                ..Default::default()
            }],
            ..Default::default()
        };
        let held = Measured {
            who: "run".into(),
            span: None,
            m: held,
        };
        let alone = grid(&["SLOT", "MODEL", "LANE"], &[held], model_rows);
        assert_eq!(rows(&alone), [["model", "sha256:cdcdcdcdcdcd", "-"]]);
        assert_eq!(alone.header, ["SLOT", "MODEL", "LANE"]);
    }

    /// Run 10264 (sdxl on a fresh A5000): 17.4 s of execution whose stages add to 5.8 s. The
    /// rest is its executor's start and its construction; what no measurement covers is said.
    #[test]
    fn a_runs_execution_is_accounted_for_whole() {
        let run = fixture("sdxl-run-10264");
        let end = run["settled_at_ms"].as_i64().unwrap();
        let span = Some((end - run["execution_ms"].as_i64().unwrap(), end));
        let sdxl = measured(Some((&run["measurements"], span)), &[]);
        let stages = grid(&["STAGE", "TIME", "DETAIL"], &sdxl, stage_rows);
        assert_eq!(stages.header, ["STAGE", "TIME", "DETAIL"]);
        assert_eq!(
            rows(&stages),
            [
                vec![
                    "executor start",
                    "10.9s",
                    "package_import 10.0s, cuda_init 658ms, fork 206ms"
                ],
                vec![
                    "construction",
                    "10.4s",
                    "qualify 7.1s, construct_and_register 2.4s, resolve 352ms"
                ],
                vec!["tokenize", "4ms", ""],
                vec!["loading weights", "465ms", ""],
                vec!["encode", "434ms", ""],
                vec!["denoise", "4.6s", "20 × 224ms"],
                vec!["decode", "311ms", ""],
                vec!["encode_webp", "1ms", ""],
                vec!["unaccounted", "696ms", "no measurement covers it"],
            ]
        );
        // A warm executor's start was an earlier call's: run 10236's second segment reused the
        // first's, and its own stages leave 2.2 s uncovered.
        let calls = h3_calls();
        let segment = measured(None, &calls)
            .into_iter()
            .find(|m| m.who == "2")
            .unwrap();
        let rows = stage_rows(&segment);
        assert_eq!(rows[0][0], "prepare");
        assert_eq!(rows.last().unwrap()[..2], ["unaccounted", "2.2s"]);
    }

    #[test]
    fn durations_read_as_people_say_them() {
        for (ms, said) in [(850.0, "850ms"), (7_620.0, "7.6s"), (228_735.0, "3m48.7s")] {
            assert_eq!(duration(ms), said);
        }
    }
}
