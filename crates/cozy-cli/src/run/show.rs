//! `cozy run show`: one run whole, with what its machine measured of it and of each call it
//! made: each stage's time and steps, the models it ran as resolved (release and lane), each
//! component's attention and GEMM kernels, every GPU's attention kernels (the one that served,
//! the others and why not), its calls, and its log.
use std::{collections::BTreeMap, process::ExitCode};

use cozy_proto::{
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
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Execution {
    gpus: Vec<Gpu>,
    /// An older Runtime's GPUs: `ordinal` is the GPU's number.
    ranks: Vec<Gpu>,
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
/// component, `gemm.fallback` when a component's GEMM leaf could not serve.
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

pub async fn exec(ctx: &Context, id: &str) -> Result<ExitCode, Error> {
    let calcifer = ctx.calcifer().await?;
    let run: Run = calcifer
        .get_query(&format!("/v1/runs/{id}"), &[("detail", "full")])
        .await?;
    if ctx.json {
        output::json(&run);
        return Ok(ExitCode::SUCCESS);
    }
    follow::fields(ctx, &run).print(ctx);
    let measured = measured(run.measurements.as_ref(), &run.calls);
    let stages = grid(&["STAGE", "TIME", "STEPS"], &measured, stage_rows);
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
    let fallbacks = grid(&["COMPONENT", "WARNING"], &measured, fallback_rows);
    fallbacks.print(ctx, "warnings");
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
    if !follow::settled(&run) {
        follow::next(ctx, &run);
    }
    Ok(ExitCode::SUCCESS)
}

/// The run's own measurements and each call's, by who measured them: `run`, or the call's
/// index.
fn measured(own: Option<&serde_json::Value>, calls: &[Call]) -> Vec<(String, Measurements)> {
    let read = |m: Option<&serde_json::Value>| m.and_then(|m| Measurements::deserialize(m).ok());
    let own = read(own).map(|m| ("run".to_owned(), m));
    let calls = calls
        .iter()
        .filter_map(|c| read(c.measurements.as_ref()).map(|m| (c.index.to_string(), m)));
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
    measured: &[(String, Measurements)],
    rows_of: fn(&Measurements) -> Vec<Vec<String>>,
) -> Grid {
    let lead = measured.len() > 1;
    let header = lead
        .then_some("CALL")
        .into_iter()
        .chain(header.iter().copied());
    let rows = measured.iter().flat_map(|(who, m)| {
        rows_of(m).into_iter().map(move |row| {
            let who = lead.then(|| who.clone());
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

/// Each stage in the order it started, with its time and, for a stepped stage, its steps.
fn stage_rows(measured: &Measurements) -> Vec<Vec<String>> {
    let attribution = &measured.attribution;
    let mut stages: Vec<(&String, &Track)> = attribution.stages.iter().collect();
    for (name, track) in &attribution.steps {
        if !attribution.stages.contains_key(name) {
            stages.push((name, track));
        }
    }
    stages.sort_by_key(|(name, track)| (track.started_unix_ms, (*name).clone()));
    let row = |(name, track): (&String, &Track)| {
        let steps = attribution.steps.get(name).map_or_else(String::new, |s| {
            let each = s.total_ms / s.count.max(1) as f64;
            format!("{} × {}", s.count, duration(each))
        });
        vec![name.clone(), duration(track.total_ms), steps]
    };
    stages.into_iter().map(row).collect()
}

/// Each model slot's checkpoint: its repository and release, or its hash when it was passed
/// by value, and the lane it was resolved in.
fn model_rows(measured: &Measurements) -> Vec<Vec<String>> {
    let row = |m: &Model| {
        let model = match m.repository.is_empty() {
            true => m.manifest.chars().take(19).collect(),
            false => format!("{}@{}", m.repository, m.release),
        };
        vec![m.slot.clone(), model, or_dash(&m.lane)]
    };
    measured.models.iter().map(row).collect()
}

/// Each component's attention and GEMM kernels (a 16-bit lane's components have no GEMM row).
fn component_rows(measured: &Measurements) -> Vec<Vec<String>> {
    let mut components: BTreeMap<&str, [String; 2]> = BTreeMap::new();
    for o in &measured.observations {
        let fields = &o.fields;
        let (column, said) = match o.name.as_str() {
            "attention.selected" => (0, o.value()),
            "gemm.selected" => (1, format!("{} ({})", o.value(), fields.format)),
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

/// Each component whose GEMM leaf could not serve, as one warning line: what it wanted, what
/// served instead and why.
fn fallback_rows(measured: &Measurements) -> Vec<Vec<String>> {
    let fallbacks = measured
        .observations
        .iter()
        .filter(|o| o.name == "gemm.fallback");
    let row = |o: &Observation| {
        let (f, kernel) = (&o.fields, o.value());
        let on = match kernel == f.served {
            true => String::new(),
            false => format!(" on {kernel}"),
        };
        let said = format!(
            "GEMM wanted {}, served {}{on}: {}",
            f.wanted, f.served, f.reason
        );
        vec![f.component.clone(), said]
    };
    fallbacks.map(row).collect()
}

/// Every GPU's attention kernels: the one that served, and the others with why not.
fn gpu_rows(measured: &Measurements) -> Vec<Vec<String>> {
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

    fn measured_alone(name: &str) -> Measurements {
        let document = fixture(name);
        Measurements::deserialize(document.get("measurements").unwrap_or(&document)).unwrap()
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
        assert_eq!(stages[0][0], "prepare");
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

        let stages = grid(&["STAGE", "TIME", "STEPS"], &measured, stage_rows);
        let first: Vec<&str> = rows(&stages).iter().map(|r| r[0]).collect();
        assert_eq!(first.first(), Some(&"0"));
        let denoise = rows(&stages)
            .into_iter()
            .find(|r| r[0] == "1" && r[1] == "denoise")
            .unwrap();
        assert!(denoise[3].starts_with("8 × "), "{denoise:?}");
    }

    /// The machine's resolution and the executor's GEMM leaves, as TensorD and Runtime (#1519)
    /// record them: each call's slot with its release and lane (run 10236's background removal
    /// resolved `paul/birefnet@0.1.0 fp16`), each component's GEMM kernel beside its attention,
    /// and a leaf that could not serve as one warning line.
    #[test]
    fn a_calls_resolved_models_and_gemm_leaves_show_beside_its_attention() {
        let mut calls = h3_calls();
        let birefnet = serde_json::json!([{"slot": "model", "repository": "paul/birefnet",
            "release": "0.1.0", "lane": "fp16", "manifest": format!("sha256:{}", "ab".repeat(32))}]);
        calls[0].measurements.as_mut().unwrap()["models"] = birefnet;
        let measurements = calls[1].measurements.as_mut().unwrap();
        let observations = measurements["observations"].as_array_mut().unwrap();
        observations.push(
            serde_json::json!({"kind": "confession", "name": "gemm.selected",
            "value": "cozy-int8", "fields": {"component": "base_model/ref2va_dit", "format": "int8",
            "leaf": "cozy.int8-convrot256.native-leaf/1", "sites": 416}}),
        );
        observations.push(serde_json::json!({"kind": "confession", "name": "gemm.fallback",
            "value": "torch.linear", "fields": {"component": "base_model/video_vae", "wanted": "fp8",
            "served": "bf16", "reason": "the leaf refused its shape", "sites": 4}}));
        let mut measured = measured(None, &calls);
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
                ["1", "base_model/audio_vae", "sdpa", "-"],
                [
                    "1",
                    "base_model/ref2va_dit",
                    "kitchen-sol-producer-fast-shared-qkv",
                    "cozy-int8 (int8)"
                ],
                ["1", "base_model/video_vae", "sdpa", "-"],
            ]
        );
        // A failed 8-bit library's fallback names its kernel once.
        let library = serde_json::json!({"kind": "confession", "name": "gemm.fallback",
            "value": "torch._int_mm", "fields": {"component": "base_model/audio_vae", "wanted": "cozy-int8",
            "served": "torch._int_mm", "reason": "the op library did not load", "sites": 1}});
        measured[1]
            .1
            .observations
            .push(serde_json::from_value(library).unwrap());
        let warnings = grid(&["COMPONENT", "WARNING"], &measured, fallback_rows);
        assert_eq!(
            rows(&warnings),
            [
                [
                    "1",
                    "base_model/video_vae",
                    "GEMM wanted fp8, served bf16 on torch.linear: the leaf refused its shape"
                ],
                [
                    "1",
                    "base_model/audio_vae",
                    "GEMM wanted cozy-int8, served torch._int_mm: the op library did not load"
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
        let alone = grid(
            &["SLOT", "MODEL", "LANE"],
            &[("run".into(), held)],
            model_rows,
        );
        assert_eq!(rows(&alone), [["model", "sha256:cdcdcdcdcdcd", "-"]]);
        assert_eq!(alone.header, ["SLOT", "MODEL", "LANE"]);
    }

    #[test]
    fn durations_read_as_people_say_them() {
        for (ms, said) in [(850.0, "850ms"), (7_620.0, "7.6s"), (228_735.0, "3m48.7s")] {
            assert_eq!(duration(ms), said);
        }
    }
}
