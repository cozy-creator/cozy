//! `cozy run show`: one run whole, with what its machine measured: each stage's time, the steps
//! it took, every GPU's attention kernels (the one that served, the others and why not), its
//! calls, and its log.
use std::{collections::BTreeMap, process::ExitCode};

use cozy_proto::{
    calcifer::{Event, Run},
    error::Error,
};
use serde::Deserialize;

use super::follow;
use crate::{Context, output};

/// What a run's executor measured (Runtime's `Outcome.measurements`), read tolerantly: a newer
/// Runtime's fields are ignored and absent ones are empty.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Measurements {
    attribution: Attribution,
    execution: Execution,
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
    let measured: Measurements = run
        .measurements
        .clone()
        .and_then(|m| serde_json::from_value(m).ok())
        .unwrap_or_default();
    if let Some(table) = stages(ctx, &measured) {
        anstream::println!("\nstages");
        table.print();
    }
    if let Some(table) = kernels(ctx, &measured) {
        anstream::println!("\nattention kernels");
        table.print();
    }
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

/// Each stage in the order it started, with its time and, for a stepped stage, its steps.
fn stages(ctx: &Context, measured: &Measurements) -> Option<output::Table> {
    let attribution = &measured.attribution;
    let mut stages: Vec<(&String, &Track)> = attribution.stages.iter().collect();
    for (name, track) in &attribution.steps {
        if !attribution.stages.contains_key(name) {
            stages.push((name, track));
        }
    }
    if stages.is_empty() {
        return None;
    }
    stages.sort_by_key(|(name, track)| (track.started_unix_ms, (*name).clone()));
    let mut table = output::Table::new(ctx, ["STAGE", "TIME", "STEPS"]);
    for (name, track) in stages {
        let steps = attribution.steps.get(name).map_or_else(String::new, |s| {
            let each = s.total_ms / s.count.max(1) as f64;
            format!("{} × {}", s.count, duration(each))
        });
        table = table.row([name.clone(), duration(track.total_ms), steps]);
    }
    Some(table)
}

/// Every GPU's attention kernels: the one that served, and the others with why not.
fn kernels(ctx: &Context, measured: &Measurements) -> Option<output::Table> {
    let execution = &measured.execution;
    let gpus = if execution.gpus.is_empty() {
        &execution.ranks
    } else {
        &execution.gpus
    };
    if gpus.iter().all(|g| g.attention.kernels.is_empty()) {
        return None;
    }
    let mut table =
        output::Table::new(ctx, ["GPU", "ARCH", "KERNEL", "STATE", "COMPILE", "DETAIL"]);
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
            let detail = if kernel.detail.is_empty() {
                "-".into()
            } else {
                kernel.detail.clone()
            };
            table = table.row([
                number.clone(),
                gpu.arch.clone(),
                kernel.kernel.clone(),
                state,
                compile,
                detail,
            ]);
        }
    }
    Some(table)
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

    fn measured(name: &str) -> Measurements {
        let path = format!(
            "{}/tests/fixtures/measurements/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let document: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        serde_json::from_value(document.get("measurements").cloned().unwrap_or(document)).unwrap()
    }

    #[test]
    fn a_real_h3_runs_stages_read_in_the_order_they_ran() {
        let h3 = measured("h3-run-5243-measurements");
        let mut stages: Vec<(&String, &Track)> = h3.attribution.stages.iter().collect();
        stages.sort_by_key(|(name, track)| (track.started_unix_ms, (*name).clone()));
        let names: Vec<&str> = stages.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names.first(), Some(&"prepare"));
        assert!(
            names.contains(&"denoise") && names.contains(&"decode_video"),
            "{names:?}"
        );
        assert_eq!(
            duration(h3.attribution.stages["denoise"].total_ms),
            "3m48.7s"
        );
        let video = &h3.attribution.steps["decode_video"];
        assert_eq!(
            (video.count, duration(video.total_ms / video.count as f64)),
            (22, "490ms".into())
        );
    }

    #[test]
    fn each_gpus_kernels_say_which_served_and_why_the_others_did_not() {
        let gpus = measured("kernel-gpus");
        let gpus = &gpus.execution.gpus;
        assert_eq!(
            gpus.iter().map(|g| g.gpu).collect::<Vec<_>>(),
            [Some(2), Some(3)]
        );
        let kernels = &gpus[0].attention.kernels;
        let served: Vec<&str> = kernels
            .iter()
            .filter(|k| k.served)
            .map(|k| k.kernel.as_str())
            .collect();
        assert_eq!(served, ["sageattention"]);
        assert_eq!(duration(kernels[1].compile_ms), "3m5.3s");
        let sol = kernels.iter().find(|k| k.kernel == "sol-attn").unwrap();
        assert_eq!(
            (sol.state.as_str(), sol.progress),
            ("compiling", Some(0.42))
        );
        assert!(kernels[0].detail.contains("serves sm90 only"));
    }

    #[test]
    fn durations_read_as_people_say_them() {
        for (ms, said) in [(850.0, "850ms"), (7_620.0, "7.6s"), (228_735.0, "3m48.7s")] {
            assert_eq!(duration(ms), said);
        }
    }
}
