//! Transfers between Calcifer and its machine over a link shaped like a slow, far uplink: an
//! input's upload and a run output's ranged download, each held to at least 80% of the shaped rate
//! end to end. Run with `--ignored` once `scripts/shaped-link.sh` has shaped the loopback for the
//! machine's port (`COZY_TEST_SHAPED_BPS` is its rate in bytes per second):
//!
//! ```sh
//! scripts/shaped-link.sh --rate 2600kbit --delay 200ms   # the owner's uplink; 400 ms round trip
//! COZY_TEST_SHAPED_BPS=325000 cargo test -p calcifer --test shaped -- --ignored --nocapture
//! ```
mod common;

use std::{
    path::Path,
    time::{Duration, Instant},
};

use calcifer::Calcifer;
use common::{SilentHub, Tensord};
use cozy_proto::calcifer::{Event, Kind, On, OutputItem, SubmitRun};

/// The machine's port the shaping names.
const SHAPED: &str = "127.0.0.1:47123";
/// How far progress may run ahead of what the link could have carried: what sits in buffers
/// (HTTP/2's 1 MiB per stream, the socket's bounded unsent bytes, the link's own queue).
const AHEAD: f64 = 2.0 * (1 << 20) as f64;

/// The run output the download test lands.
const BIG: u64 = 20 << 20;

/// The shaped rate, in bytes per second.
fn shaped_rate() -> f64 {
    std::env::var("COZY_TEST_SHAPED_BPS")
        .expect("COZY_TEST_SHAPED_BPS names the shaped rate in bytes per second")
        .parse()
        .unwrap()
}

/// `3.2 MiB` as bytes.
fn parse(text: &str) -> f64 {
    let (value, unit) = text.trim().split_once(' ').unwrap();
    let scale = match unit {
        "B" => 1.0,
        "KiB" => 1024.0,
        "MiB" => 1024.0 * 1024.0,
        _ => 1024.0 * 1024.0 * 1024.0,
    };
    value.parse::<f64>().unwrap() * scale
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs the loopback shaped for port 47123; see the module doc"]
async fn an_input_uploads_near_the_link_rate_over_a_long_round_trip() {
    let shaped = shaped_rate();
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let machine = Tensord::start_at(SHAPED, &[]);
    let hub = SilentHub::new();
    let calcifer =
        Calcifer::open(&home.path().join("calcifer"), &home.path().join("outputs")).unwrap();
    calcifer.put_machine(&machine.row("m", &key)).unwrap();
    let mut events = calcifer.subscribe();
    let total = 20u32 << 20;
    let bytes: Vec<u8> = (0..total)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    let (staged, input) = common::input_file(Path::new(home.path()), &bytes);
    let submit = SubmitRun {
        on: On::from("m".to_owned()),
        hub: "https://tensorhub.com".into(),
        kind: Kind::Warm,
        target: "cozy-test/held".into(),
        spec: common::warm_spec(&hub.origin(), vec![input]),
        inputs: vec![staged],
        out: None,
        warnings: vec![],
    };
    let id = calcifer.submit("k", submit).await.unwrap().run.id;
    let mut seen: Vec<(Instant, f64)> = vec![];
    // Written, then sent: the machine's acceptance follows the upload's end.
    // A hang bound for a broken link; the product waits without one.
    let accepted = tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            let (run, event) = events.recv().await.unwrap();
            if run != id {
                continue;
            }
            match event {
                Event::Waiting { reason } => {
                    if let Some(progress) = reason.strip_prefix("writing image: ") {
                        seen.push((Instant::now(), parse(progress.split('/').next().unwrap())));
                    }
                }
                Event::InProgress { .. } => return Instant::now(),
                _ => {}
            }
        }
    })
    .await
    .expect("the input was written and the run accepted");
    let (first, from) = seen[0];
    let rate = (total as f64 - from) / (accepted - first).as_secs_f64();
    // How far progress ran ahead of what the link could have carried by then (negative: behind).
    let ahead = (seen.iter())
        .map(|(at, done)| done - from - shaped * (*at - first).as_secs_f64())
        .fold(f64::MIN, f64::max);
    eprintln!(
        "{} MiB written and the run accepted in {:.1} s over {} progress events: {:.0} B/s, {:.0}% of the shaped {shaped:.0} B/s; progress at most {:.0} KiB ahead of the link",
        total >> 20,
        (accepted - first).as_secs_f64(),
        seen.len(),
        rate,
        100.0 * rate / shaped,
        ahead / 1024.0,
    );
    let link = calcifer::Link::new(&machine.endpoint, common::key()).unwrap();
    let digest = cozy_proto::ids::Digest::of(&bytes);
    assert_eq!(
        link.held(&digest, total as u64).await.unwrap(),
        total as u64
    );
    assert!(
        ahead <= AHEAD,
        "progress ran {ahead:.0} B ahead of what the link carried"
    );
    assert!(
        seen.len() >= 10,
        "progress moved as the bytes did: {} events",
        seen.len()
    );
    assert!(
        rate >= 0.8 * shaped,
        "{rate:.0} B/s is under 80% of {shaped:.0} B/s"
    );
    calcifer.cancel(&id).unwrap();
    calcifer.close();
}

/// The item an output event carries, for the output named `name`.
fn item<'a>(event: &'a Event, name: &str) -> Option<&'a OutputItem> {
    match event {
        Event::OutputAdded { item, .. }
        | Event::OutputDone { item, .. }
        | Event::OutputLanded { item, .. } => Some(item).filter(|i| i.name == name),
        _ => None,
    }
}

/// A run's 20 MiB output, made by a real CPU package on a real TensorD across the shaped link,
/// lands on this computer through Calcifer's ranged reads at ≥80% of the shaped rate, timed from
/// the output's first word to its landing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs the loopback shaped for port 47123; see the module doc"]
async fn a_runs_output_downloads_near_the_link_rate_over_a_long_round_trip() {
    let shaped = shaped_rate();
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let machine = common::installing_machine_at(SHAPED);
    let calcifer =
        Calcifer::open(&home.path().join("calcifer"), &home.path().join("outputs")).unwrap();
    calcifer.put_machine(&machine.row("m", &key)).unwrap();
    let mut events = calcifer.subscribe();
    let gate = home.path().join("gate");
    std::fs::write(&gate, b"").unwrap();
    let id = calcifer
        .submit("k", common::package_run(home.path(), &gate, BIG))
        .await
        .unwrap()
        .run
        .id;
    // A hang bound for a run that never ends (it installs its package from PyPI first); the
    // product waits without one.
    let (began, landed) = tokio::time::timeout(Duration::from_secs(900), async {
        let mut began = None;
        loop {
            let (run, event) = events.recv().await.unwrap();
            if run != id {
                continue;
            }
            let Some(big) = item(&event, "big") else {
                continue;
            };
            let began = *began.get_or_insert_with(Instant::now);
            if let Event::OutputLanded { item, .. } = &event
                && item.sha256.is_some()
            {
                return (began, (Instant::now(), big.clone()));
            }
        }
    })
    .await
    .expect("the run's big output landed");
    let (at, item) = landed;
    let seconds = (at - began).as_secs_f64();
    let rate = BIG as f64 / seconds;
    eprintln!(
        "{} MiB landed in {seconds:.1} s: {rate:.0} B/s, {:.0}% of the shaped {shaped:.0} B/s",
        BIG >> 20,
        100.0 * rate / shaped,
    );
    let path = item.path.expect("a landed item names its file");
    assert_eq!(std::fs::metadata(&path).unwrap().len(), BIG);
    assert!(
        rate >= 0.8 * shaped,
        "{rate:.0} B/s is under 80% of {shaped:.0} B/s"
    );
    calcifer.close();
}
