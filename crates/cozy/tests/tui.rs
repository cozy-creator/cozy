//! The live views, drawn on ratatui's `TestBackend` and fed by a real Calcifer, a real TensorD and
//! real runs that move through their states. TensorD runs on Linux only.
#![cfg(unix)]
mod common;

use std::{
    path::PathBuf,
    sync::mpsc::{Receiver, channel},
    thread::JoinHandle,
    time::Duration,
};

use common::{Home, package, tensord};
use cozy_cli::tui::testing::{View, draw};
use cozy_proto::error::Error;

/// A hang bound for a view that stops drawing; the product waits without one.
const PATIENCE: Duration = Duration::from_secs(600);

/// `view` drawn on its own thread until `done` says a frame is the last; each frame is sent.
fn drawing(
    root: PathBuf,
    view: View,
    done: fn(&str) -> bool,
) -> (JoinHandle<Result<(), Error>>, Receiver<String>) {
    let (frames, drawn) = channel();
    let thread = std::thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(draw(&root, view, |frame| {
            let _ = frames.send(frame.to_owned());
            done(frame)
        }))
    });
    (thread, drawn)
}

/// The update run's row of a frame of `cozy run list`.
fn update_row(frame: &str) -> Option<String> {
    frame
        .lines()
        .find(|l| l.contains("update"))
        .map(str::to_owned)
}

fn settled(row: &str) -> bool {
    ["completed", "failed", "canceled"]
        .iter()
        .any(|s| row.contains(s))
}

/// `cozy run list`'s live view draws a run again each time Calcifer's stream says it changed: an
/// update run on a real machine is drawn before it completed, then completed.
#[test]
fn the_live_run_list_redraws_a_run_as_it_moves() {
    let Some(home) = Home::new() else { return };
    let _machine = tensord::added(&home, "live");
    let (view, drawn) = drawing(home.root(), View::Runs, |frame| {
        update_row(frame).is_some_and(|row| settled(&row))
    });
    let first = drawn.recv_timeout(PATIENCE).expect("the view drew");
    assert!(first.contains("TARGET"), "{first}");
    // Subscribed: the run starts now, and the view draws each change of it meanwhile.
    let out = home.cozy(["machine", "update", "live", "--json"]);
    assert_eq!(out.code, 0, "{out:?}");
    let mut rows = vec![];
    while !rows.last().is_some_and(|row: &String| settled(row)) {
        let frame = drawn
            .recv_timeout(PATIENCE)
            .expect("the view stopped drawing");
        rows.extend(update_row(&frame));
    }
    view.join().unwrap().expect("the view followed the run");
    assert!(rows.last().unwrap().contains("completed"), "{rows:?}");
    assert!(
        rows.iter().any(|row| !settled(row)),
        "drawn before it completed: {rows:?}"
    );
}

/// `cozy package list`'s live view draws a package when the machine's install run lands it, told
/// by Calcifer's machine stream.
#[test]
fn the_live_package_list_draws_an_install_when_it_lands() {
    let Some(home) = Home::new() else { return };
    let _machine = tensord::added(&home, "live");
    let path = package::project(&home, "tiny");
    let (view, drawn) = drawing(home.root(), View::Packages, |frame| {
        frame.contains("local/tiny")
    });
    let first = drawn.recv_timeout(PATIENCE).expect("the view drew");
    assert!(
        first.contains("PACKAGE") && !first.contains("local/tiny"),
        "{first}"
    );
    let out = home.cozy([
        "--json".as_ref(),
        "package".as_ref(),
        "install".as_ref(),
        path.as_os_str(),
        "--on".as_ref(),
        "live".as_ref(),
    ]);
    assert_eq!(out.code, 0, "{out:?}");
    let last = loop {
        let frame = drawn
            .recv_timeout(PATIENCE)
            .expect("the view stopped drawing");
        if frame.contains("local/tiny") {
            break frame;
        }
    };
    view.join().unwrap().expect("the view followed the machine");
    let row = last.lines().find(|l| l.contains("local/tiny")).unwrap();
    assert!(row.contains("live") && row.contains("0.1.0"), "{row}");
}
