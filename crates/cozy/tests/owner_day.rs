//! The owner's day with cozy, end to end: the real binary, a real Calcifer and real TensorDs
//! (COZY_TEST_TENSORD: the newest cozy-runtime wheel's, or this workspace's with
//! COZY_TENSORD=workspace) against the CI Tensorhub that Tensorhub's `live-hub-client.sh` starts
//! (TENSORHUB_ORIGIN, TENSORHUB_LOG; its docker provider rents CPU machines). `owner-day.yml`
//! runs it, with the other scenario targets, on every release candidate. `h3-standin` has
//! minimax-h3's request shape without torch. Each test has its own home, account and machines,
//! so they run in parallel. Elsewhere it skips.
#![cfg(unix)]
mod common;

use std::{
    io::BufRead,
    path::{Path, PathBuf},
    process::{Child, Stdio},
    time::{Duration, Instant},
};

use common::{Home, Out, tensord};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

/// A hang bound for one command; the product waits without one.
const PATIENCE: Duration = Duration::from_secs(600);

/// A home on the CI Hub, signed in as `email`, that published `h3-standin` 1.0.0: its account and
/// the package. `None` off the job.
fn owner(email: &str) -> Option<(Home, String, String)> {
    let home = Home::with_hub()?;
    if std::env::var_os("COZY_TEST_TENSORD").is_none() {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return None;
    }
    let account = home.sign_in(email);
    let package = common::package::publish(&home, "h3-standin", "1.0.0");
    Some((home, account, package))
}

fn cozy(home: &Home, dir: &Path, args: &[&str]) -> Out {
    common::out(home.command().current_dir(dir).args(args).output().unwrap())
}

/// `cozy args` in `dir`, which must exit within `limit`: a view that never ends fails here.
fn bounded(home: &Home, dir: &Path, args: &[&str], limit: Duration) -> Out {
    let mut child = home
        .command()
        .current_dir(dir)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let until = Instant::now() + limit;
    while child.try_wait().unwrap().is_none() {
        if Instant::now() > until {
            child.kill().unwrap();
            let out = common::out(child.wait_with_output().unwrap());
            panic!("cozy {args:?} did not exit within {limit:?}: {out:?}");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    common::out(child.wait_with_output().unwrap())
}

/// The events a followed `--json` run printed: one JSON document per line.
fn events(out: &Out) -> Vec<Value> {
    out.stdout
        .lines()
        .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}\n{out:?}")))
        .collect()
}

/// A followed `--json` run's record at its completion.
fn completed(out: &Out) -> Value {
    assert_eq!(out.code, 0, "{out:?}");
    let end = events(out)
        .into_iter()
        .rfind(|e| e["type"] == "run.completed");
    end.unwrap_or_else(|| panic!("no completion: {out:?}"))["run"].clone()
}

fn sha256(bytes: &[u8]) -> String {
    format!("sha256:{}", hex::encode(Sha256::digest(bytes)))
}

/// Every file under `dir`.
fn files(dir: &Path) -> Vec<PathBuf> {
    let mut out = vec![];
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        match entry.file_type().unwrap().is_dir() {
            true => out.extend(files(&entry.path())),
            false => out.push(entry.path()),
        }
    }
    out
}

/// `tests/fixtures/media/<name>`, copied to `to`.
fn media(name: &str, to: &Path) -> Vec<u8> {
    let from = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/media")
        .join(name);
    std::fs::create_dir_all(to.parent().unwrap()).unwrap();
    std::fs::copy(from, to).unwrap();
    std::fs::read(to).unwrap()
}

/// The owner's shoot: request files beside their media, as in `~/cozy_v2/r`. Answers the shoot's
/// directory and each reference's expected `(name, kind, sha256, description)`.
fn shoot(home: &Home) -> (PathBuf, Vec<(String, String, String, String)>) {
    let dir = home.path().join("work/shoot");
    let azul = media("azul.png", &dir.join("azul.png"));
    let sophia = media("sophia.jpg", &dir.join("images/sophia.jpg"));
    let clip = media("motion.mp4", &dir.join("videos/motion.mp4"));
    let voice = media("voice.wav", &home.path().join("voices/voice.wav"));
    let references = "references:\n\
        \x20 - name: Azul\n    kind: character\n    image: azul.png\n\
        \x20 - name: Sophia\n    kind: character\n    image: images/sophia.jpg\n\
        \x20 - name: Background\n    kind: background\n    description: A wooden dojo at dusk.\n\
        \x20 - name: Video 1\n    kind: video\n    video: videos/motion.mp4\n\
        \x20 - name: Voice\n    kind: audio\n    audio: ~/voices/voice.wav\n";
    let write = |name: &str, text: String| std::fs::write(dir.join(name), text).unwrap();
    write(
        "long.yaml",
        format!(
            "# long_form. File paths are relative to this file.\nstyle: Realistic action.\n{references}\
             segments:\n\
             \x20 - prompt: <Azul> lies face-first on the ground; <Sophia> stands over her.\n    duration_s: 5\n\
             \x20 - prompt: <Sophia> kneels and rolls <Azul> over, as in <Video 1>.\n    duration_s: 5\n"
        ),
    );
    write(
        "clip.yaml",
        format!(
            "# ref2va, as roll-over.yaml.\nduration_s: 5\nresolution: 768p\nstyle: Realistic action.\n\
             {references}prompt: |\n  summary:\n  <Azul> and <Sophia> follow the motion of <Video 1>.\n"
        ),
    );
    let seen = |name: &str, kind: &str, bytes: &[u8], description: &str| {
        let digest = match bytes.is_empty() {
            true => String::new(),
            false => hex::encode(Sha256::digest(bytes)),
        };
        (
            name.to_owned(),
            kind.to_owned(),
            digest,
            description.to_owned(),
        )
    };
    let expected = vec![
        seen("Azul", "character", &azul, ""),
        seen("Sophia", "character", &sophia, ""),
        seen("Background", "background", b"", "A wooden dojo at dusk."),
        seen("Video 1", "video", &clip, ""),
        seen("Voice", "audio", &voice, ""),
    ];
    (dir, expected)
}

/// The references a run's result says the machine read.
fn seen(run: &Value) -> Vec<(String, String, String, String)> {
    let text = |r: &Value, k: &str| r[k].as_str().unwrap_or_default().to_owned();
    run["result"]["references"]
        .as_array()
        .unwrap_or_else(|| panic!("no references in {run}"))
        .iter()
        .map(|r| {
            (
                text(r, "name"),
                text(r, "kind"),
                text(r, "sha256"),
                text(r, "description"),
            )
        })
        .collect()
}

/// The run's one landed video: whole, its digest the record's and the machine's.
fn landed_video(run: &Value) -> PathBuf {
    let item = &run["output"][0];
    assert_eq!(item["name"], "video", "{run}");
    let path = PathBuf::from(
        item["path"]
            .as_str()
            .unwrap_or_else(|| panic!("no path: {run}")),
    );
    let bytes = std::fs::read(&path).unwrap();
    let digest = sha256(&bytes);
    assert_eq!(item["sha256"].as_str(), Some(digest.as_str()), "{run}");
    let made = format!("sha256:{}", run["result"]["video_sha256"].as_str().unwrap());
    assert_eq!(
        digest, made,
        "the bytes differ from those the machine wrote: {run}"
    );
    assert!(!Path::new(&format!("{}.part", path.display())).exists());
    path
}

/// A `--json` run followed in the background; its stdout read line by line.
fn follow(
    home: &Home,
    dir: &Path,
    args: &[&str],
) -> (Child, std::io::BufReader<std::process::ChildStdout>) {
    let mut child = home
        .command()
        .current_dir(dir)
        .args(args)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = std::io::BufReader::new(child.stdout.take().unwrap());
    (child, stdout)
}

/// Reads NDJSON events until one satisfies `want`.
fn until(reader: &mut impl BufRead, want: impl Fn(&Value) -> bool) -> Value {
    loop {
        let mut line = String::new();
        assert!(reader.read_line(&mut line).unwrap() > 0, "stdout closed");
        let event: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line}"));
        if want(&event) {
            return event;
        }
    }
}

/// Polls `cozy run show N` until its status is `status`.
fn reaches(home: &Home, number: &str, status: &str) -> Value {
    let until = Instant::now() + PATIENCE;
    loop {
        let run = home.json(["run", "show", number, "--json"]);
        if run["status"] == status {
            return run;
        }
        assert!(
            Instant::now() < until,
            "run {number} never reached {status}: {run}"
        );
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// minimax-h3's long_form and ref2va as the owner sends them: a YAML of named references whose
/// files sit beside it, in subfolders and under `~`, run from another directory. Each file reaches
/// the machine as itself; the video lands in the package's folder (or `--out`), whole, before
/// `cozy run` returns; followed `--json` is NDJSON. A request in last week's shape is refused in
/// one line per problem, naming the release it was checked against, with no argument dump.
#[test]
fn an_h3_request_binds_its_references_from_a_yaml_and_its_video_lands_whole() {
    let Some((home, account, package)) = owner("h3@cozy.test") else {
        return;
    };
    let _machine = tensord::added(&home, "m1");
    let (shoot, expected) = shoot(&home);
    let elsewhere = home.path().join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();

    let long_form = format!("{package}/long_form");
    let input = "../work/shoot/long.yaml";
    let run = completed(&cozy(
        &home,
        &elsewhere,
        &["--json", "run", &long_form, "--input", input, "--on", "m1"],
    ));
    assert_eq!(seen(&run), expected, "{run}");
    let video = landed_video(&run);
    let folder = home
        .root()
        .join("outputs")
        .join(format!("{account}-h3-standin"));
    assert_eq!(
        video,
        folder.join(format!("{}-video.mp4", run["number"])),
        "{run}"
    );
    assert_eq!(std::fs::metadata(&video).unwrap().len(), 32 + (10 << 20));
    assert_eq!(
        files(&folder),
        [video.clone()],
        "each revision rewrote the one file"
    );

    let ref2va = format!("{package}/ref2va");
    let input = "../work/shoot/clip.yaml";
    let args = [
        "--json", "run", &ref2va, "--input", input, "--on", "m1", "--out", "saved",
    ];
    let run = completed(&cozy(&home, &elsewhere, &args));
    assert_eq!(seen(&run), expected, "{run}");
    let video = landed_video(&run);
    assert_eq!(
        video,
        elsewhere.join(format!("saved/{}-video.mp4", run["number"]))
    );

    // Last week's shape: a reference kind that is gone, and clips in a `videos` list.
    std::fs::write(
        shoot.join("old.yaml"),
        "duration_s: 5\nprompt: <Azul> trains in <Dojo>.\nreferences:\n  \
         - {name: Azul, kind: character, image: azul.png}\n  \
         - {name: Dojo, kind: scene, description: A wooden dojo at dusk.}\n\
         videos: [videos/motion.mp4]\n",
    )
    .unwrap();
    let old = [
        "run",
        &ref2va,
        "--input",
        "../work/shoot/old.yaml",
        "--on",
        "m1",
    ];
    let refused = cozy(&home, &elsewhere, &[&["--json"][..], &old].concat());
    assert_eq!(refused.code, 1, "{refused:?}");
    let refusal = refused.refusal();
    assert_eq!(refusal["code"], "request_payload_invalid", "{refused:?}");
    let lines = [
        format!("{ref2va} cannot take this request (checked against {package} 1.0.0):"),
        "  references[1].kind: \"scene\" is not one of character, background, audio, video".into(),
        format!("warning: ignored unknown field videos — not in {ref2va}'s interface"),
    ];
    assert_eq!(
        refusal["message"]
            .as_str()
            .unwrap()
            .lines()
            .collect::<Vec<_>>(),
        lines,
        "{refused:?}"
    );
    assert_eq!(
        refusal["next"],
        serde_json::json!([format!("cozy run {ref2va}@1.0.0 --describe")])
    );
    let person = cozy(&home, &elsewhere, &old);
    assert_eq!(person.code, 1, "{person:?}");
    assert!(person.stdout.contains(&lines[1]), "{person:?}");
    assert!(
        !person.stdout.contains("trains in") && person.stdout.lines().count() <= 5,
        "{person:?}"
    );
}

/// A long run outlives Calcifer: `cozy down` then `cozy up` while it renders, the follower
/// reattaches, the video lands once and the history stays. Left idle, Calcifer keeps serving:
/// the next `cozy run list` answers within a second from the same process.
#[test]
fn a_run_outlives_a_calcifer_restart_and_calcifer_answers_after_idle() {
    let Some((home, account, package)) = owner("restart@cozy.test") else {
        return;
    };
    let _machine = tensord::added(&home, "m1");
    let (_, expected) = shoot(&home);
    let elsewhere = home.path().join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    let ref2va = format!("{package}/ref2va");
    let before = completed(&cozy(
        &home,
        &elsewhere,
        &[
            "--json",
            "run",
            &ref2va,
            "--input",
            "../work/shoot/clip.yaml",
            "--on",
            "m1",
        ],
    ));

    let long_form = format!("{package}/long_form");
    let args = [
        "--json",
        "run",
        &long_form,
        "--input",
        "../work/shoot/long.yaml",
        "render_s:=6",
        "--on",
        "m1",
    ];
    let (mut follower, mut stdout) = follow(&home, &elsewhere, &args);
    let accepted = until(&mut stdout, |e| e["type"] == "run.in_progress");
    let number = accepted["run"]["number"].to_string();
    let down = home.json(["down", "--json"]);
    assert_eq!(down["changed"], true, "{down}");
    let up = home.json(["up", "--json"]);
    let pid = up["calcifer"]["pid"].clone();
    let end = until(&mut stdout, |e| {
        ["run.completed", "run.failed", "run.canceled"]
            .contains(&e["type"].as_str().unwrap_or_default())
    });
    assert_eq!(end["type"], "run.completed", "{end}");
    assert_eq!(follower.wait().unwrap().code(), Some(0));

    let run = home.json(["run", "show", &number, "--json"]);
    assert_eq!(seen(&run), expected, "{run}");
    let video = landed_video(&run);
    let folder = home
        .root()
        .join("outputs")
        .join(format!("{account}-h3-standin"));
    let mine: Vec<PathBuf> = files(&folder)
        .into_iter()
        .filter(|f| {
            f.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(&format!("{number}-"))
        })
        .collect();
    assert_eq!(mine, [video], "the video landed once");
    let listed = home.json(["run", "list", "--json"]);
    let numbers: Vec<u64> = listed["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["number"].as_u64().unwrap())
        .collect();
    for kept in [before["number"].as_u64().unwrap(), number.parse().unwrap()] {
        assert!(
            numbers.contains(&kept),
            "run {kept} left the history: {listed}"
        );
    }

    // Idle, then the owner's next command.
    std::thread::sleep(Duration::from_secs(120));
    let began = Instant::now();
    let list = bounded(&home, &elsewhere, &["run", "list"], PATIENCE);
    let took = began.elapsed();
    eprintln!("cozy run list after 120 s idle: {took:?}");
    assert_eq!(list.code, 0, "{list:?}");
    assert!(
        took < Duration::from_secs(1),
        "cozy run list took {took:?} after idle"
    );
    let status = home.json(["status", "--json"]);
    assert_eq!(
        status["calcifer"]["pid"], pid,
        "Calcifer started again: {status}"
    );
}

/// Off a terminal, every list and `run show` prints its plain view once and exits, even while a
/// run is in progress; `--json` prints one document. `cozy run cancel` settles a running run and
/// a queued one canceled, and neither leaves a file behind.
#[test]
fn views_print_once_off_a_terminal_and_cancel_settles_running_and_queued_runs() {
    let Some((home, _, package)) = owner("cancel@cozy.test") else {
        return;
    };
    let mut machine = tensord::added(&home, "m1");
    shoot(&home);
    let elsewhere = home.path().join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    let long_form = format!("{package}/long_form");
    let args = [
        "--json",
        "run",
        &long_form,
        "--input",
        "../work/shoot/long.yaml",
        "render_s:=600",
        "--on",
        "m1",
        "--detach",
    ];
    let sent = cozy(&home, &elsewhere, &args);
    assert_eq!(sent.code, 0, "{sent:?}");
    let number = serde_json::from_str::<Value>(&sent.stdout).unwrap()["number"].to_string();
    reaches(&home, &number, "in_progress");

    let views: [(Vec<&str>, [&str; 2]); 4] = [
        (vec!["run", "list"], ["TARGET", "STATUS"]),
        (
            vec!["run", "show", number.as_str()],
            ["target", "/long_form"],
        ),
        (vec!["machine", "list"], ["KIND", "STATE"]),
        (vec!["package", "list"], ["PACKAGE", "RELEASE"]),
    ];
    for (view, header) in views {
        let out = bounded(&home, &elsewhere, &view, Duration::from_secs(60));
        assert_eq!(out.code, 0, "{view:?}: {out:?}");
        assert!(
            !out.stdout.contains('\x1b'),
            "{view:?} drew a terminal view: {out:?}"
        );
        let headers = out
            .stdout
            .lines()
            .filter(|l| header.iter().all(|h| l.contains(h)))
            .count();
        assert_eq!(
            headers, 1,
            "{view:?} printed its view {headers} times: {out:?}"
        );
        let json = bounded(
            &home,
            &elsewhere,
            &[view.clone(), vec!["--json"]].concat(),
            Duration::from_secs(60),
        );
        assert_eq!(json.code, 0, "{view:?}: {json:?}");
        serde_json::from_str::<Value>(&json.stdout)
            .unwrap_or_else(|e| panic!("{view:?} --json: {e}: {json:?}"));
    }

    let canceled = |number: &str| {
        let out = home.cozy(["run", "cancel", number, "--json"]);
        assert_eq!(out.code, 0, "{out:?}");
        reaches(&home, number, "canceled")
    };
    let run = canceled(number.as_str());
    assert!(
        run["output"]
            .as_array()
            .is_none_or(|o| o.iter().all(|i| i["path"].is_null())),
        "{run}"
    );

    // A run its machine cannot take yet waits, unsent, and cancels as it waits.
    machine.stop();
    let ref2va = format!("{package}/ref2va");
    let args = [
        "--json",
        "run",
        &ref2va,
        "--input",
        "../work/shoot/clip.yaml",
        "--on",
        "m1",
        "--detach",
    ];
    let sent = cozy(&home, &elsewhere, &args);
    assert_eq!(sent.code, 0, "{sent:?}");
    let queued: Value = serde_json::from_str(&sent.stdout).unwrap();
    assert_eq!(queued["status"], "queued", "{queued}");
    canceled(&queued["number"].to_string());
    let left: Vec<PathBuf> = files(&home.root().join("outputs"));
    assert!(left.is_empty(), "canceled runs left files: {left:?}");
}

/// A CPU machine rented from the CI Hub's docker provider runs a long_form with a 15 MiB video;
/// `cozy machine end` the moment `cozy run` returns loses nothing: the file is whole, and the
/// rental is gone. The rental runs its image's TensorD, whichever COZY_TEST_TENSORD names.
#[test]
fn a_rental_ended_the_moment_its_run_returns_loses_nothing() {
    let Some((home, _, package)) = owner("rental-day@cozy.test") else {
        return;
    };
    let skus = home.json(["machine", "rent", "--json"]);
    assert!(
        skus.as_array().unwrap().iter().any(|s| s["name"] == "cpu"),
        "the CI Hub offers no cpu product (its docker provider is off): {skus}"
    );
    let rented = home.json(["machine", "rent", "cpu", "--detach", "--json"]);
    let name = rented["machine"]["name"].as_str().unwrap().to_owned();
    let id = rented["machine"]["rental_id"].as_str().unwrap().to_owned();
    let (shoot, expected) = shoot(&home);
    let yaml = std::fs::read_to_string(shoot.join("long.yaml")).unwrap();
    let third = "  - prompt: <Azul> stands.\n    duration_s: 5\n";
    std::fs::write(shoot.join("long.yaml"), format!("{yaml}{third}")).unwrap();

    let long_form = format!("{package}/long_form");
    let run = completed(&cozy(
        &home,
        &shoot,
        &[
            "--json",
            "run",
            &long_form,
            "--input",
            "long.yaml",
            "--on",
            &name,
        ],
    ));
    let ended = home.json(["machine", "end", &id, "--json"]);
    assert_eq!(
        (
            ended["changed"].as_bool(),
            ended["machine"]["state"].as_str()
        ),
        (Some(true), Some("ended")),
        "{ended}"
    );
    assert_eq!(seen(&run), expected, "{run}");
    let video = landed_video(&run);
    assert_eq!(std::fs::metadata(&video).unwrap().len(), 32 + (15 << 20));
    let list = home.json(["machine", "list", "--json"]);
    assert!(
        !list["machines"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["rental_id"] == id.as_str() && m["state"] != "ended"),
        "{list}"
    );
}

/// A clip uploads to a machine over a link shaped like the owner's uplink (`owner-day.yml` shapes
/// the loopback for COZY_TEST_SHAPED_PORT at COZY_TEST_SHAPED_BPS, 400 ms round trip): the run
/// says how far the upload is as it moves, and completes with the clip whole on the machine.
#[test]
fn a_clip_uploads_over_a_slow_link_with_progress_and_the_run_completes() {
    let (Ok(port), Ok(rate)) = (
        std::env::var("COZY_TEST_SHAPED_PORT"),
        std::env::var("COZY_TEST_SHAPED_BPS"),
    ) else {
        eprintln!("skipped: no shaped link (COZY_TEST_SHAPED_PORT, COZY_TEST_SHAPED_BPS)");
        return;
    };
    let (port, rate): (u16, f64) = (port.parse().unwrap(), rate.parse().unwrap());
    let Some((home, _, package)) = owner("slow@cozy.test") else {
        return;
    };
    let _machine = tensord::added_on(&home, "slow", Some(port), &[]);
    let (shoot, _) = shoot(&home);
    // A 4 MiB take: the small clip, then a `free` box filling the rest.
    let mut take = std::fs::read(shoot.join("videos/motion.mp4")).unwrap();
    let pad = (4 << 20) - take.len();
    take.extend_from_slice(&(pad as u32).to_be_bytes());
    take.extend_from_slice(b"free");
    take.extend((0..pad - 8).map(|i| (i as u32).wrapping_mul(2_654_435_761).to_be_bytes()[0]));
    std::fs::write(shoot.join("videos/take.mp4"), &take).unwrap();
    std::fs::write(
        shoot.join("take.yaml"),
        "duration_s: 5\nprompt: <Azul> follows <Take>.\nreferences:\n  \
         - {name: Azul, kind: character, image: azul.png}\n  \
         - {name: Take, kind: video, video: videos/take.mp4}\n",
    )
    .unwrap();

    let ref2va = format!("{package}/ref2va");
    let out = cozy(
        &home,
        &shoot,
        &[
            "--json",
            "run",
            &ref2va,
            "--input",
            "take.yaml",
            "--on",
            "slow",
        ],
    );
    let run = completed(&out);
    let held = &seen(&run)[1];
    assert_eq!(held.2, hex::encode(Sha256::digest(&take)), "{run}");
    // `uploading <field>: <done>/4.0 MiB[ at <rate>/s]`: rising, and ending at the whole take.
    let progress: Vec<f64> = events(&out)
        .iter()
        .filter(|e| e["type"] == "run.waiting")
        .filter_map(|e| e["reason"].as_str())
        .filter(|r| r.starts_with("uploading ") && r.contains("/4.0 MiB"))
        .map(|r| {
            let done = r.rsplit_once(": ").unwrap().1.split('/').next().unwrap();
            let (value, unit) = done.split_once(' ').unwrap();
            let scale = match unit {
                "B" => 1.0,
                "KiB" => 1024.0,
                _ => 1048576.0,
            };
            value.parse::<f64>().unwrap() * scale
        })
        .collect();
    eprintln!("upload progress: {progress:?}");
    assert!(
        progress.len() >= 3 && progress.is_sorted() && progress.last() == Some(&(4.0 * 1048576.0)),
        "{progress:?}\n{out:?}"
    );
    let took = (run["accepted_at_ms"].as_i64().unwrap() - run["created_at_ms"].as_i64().unwrap())
        as f64
        / 1e3;
    let floor = take.len() as f64 / rate;
    eprintln!("created to accepted: {took:.1} s; the link carries the take in {floor:.1} s");
    assert!(
        took >= 0.8 * floor,
        "the upload beat the shaped link: is the loopback shaped?"
    );
}
