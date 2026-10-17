//! An input's upload to its machine over a link shaped like the owner's laptop uplink: 2.6 Mbit/s
//! with a 400 ms round trip. Run with `--ignored` once the loopback is shaped for the machine's
//! port, as CI does (`COZY_TEST_SHAPED_BPS` is the shaped rate in bytes per second):
//!
//! ```sh
//! sudo ip link set dev lo mtu 1500
//! sudo tc qdisc add dev lo root handle 1: prio bands 4 priomap 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//! sudo tc qdisc add dev lo parent 1:4 handle 40: netem delay 200ms rate 2600kbit limit 100000
//! sudo tc filter add dev lo parent 1: protocol ip prio 1 u32 match ip dport 47123 0xffff flowid 1:4
//! sudo tc filter add dev lo parent 1: protocol ip prio 1 u32 match ip sport 47123 0xffff flowid 1:4
//! ```
mod common;

use std::{
    path::Path,
    time::{Duration, Instant},
};

use calcifer::Calcifer;
use common::{SilentHub, Tensord};
use cozy_proto::calcifer::{Event, Kind, On, SubmitRun};

/// The machine's port the shaping names.
const SHAPED: &str = "127.0.0.1:47123";
/// How far progress may run ahead of what the link could have carried: what sits in buffers.
const AHEAD: f64 = 1.5 * (1 << 20) as f64;

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
    let shaped: f64 = std::env::var("COZY_TEST_SHAPED_BPS")
        .expect("COZY_TEST_SHAPED_BPS names the shaped rate in bytes per second")
        .parse()
        .unwrap();
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
    // A hang bound for a broken link; the product waits without one.
    tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            let (run, event) = events.recv().await.unwrap();
            if run != id {
                continue;
            }
            if let Event::Waiting { reason } = event
                && let Some(progress) = reason.strip_prefix("writing image: ")
            {
                let done = parse(progress.split('/').next().unwrap());
                seen.push((Instant::now(), done));
                if done >= total as f64 {
                    return;
                }
            }
        }
    })
    .await
    .expect("the input was written");
    let (first, from) = seen[0];
    let (last, _) = *seen.last().unwrap();
    let rate = (total as f64 - from) / (last - first).as_secs_f64();
    eprintln!(
        "{} MiB in {:.1} s over {} progress events: {:.0} B/s, {:.0}% of the shaped {shaped:.0} B/s",
        total >> 20,
        (last - first).as_secs_f64(),
        seen.len(),
        rate,
        100.0 * rate / shaped,
    );
    for (at, done) in &seen {
        let carried = shaped * (*at - first).as_secs_f64();
        assert!(
            done - from <= carried + AHEAD,
            "progress said {done:.0} B after {:.1} s, more than the link carried",
            (*at - first).as_secs_f64()
        );
    }
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
