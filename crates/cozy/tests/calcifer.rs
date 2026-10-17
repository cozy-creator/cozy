//! `cozy calcifer`, the role, through the real binary on its own temp home (hosted runners only).
mod common;

use std::{
    process::Stdio,
    time::{Duration, Instant},
};

use common::Home;
use cozy_platform::{durable, lock::LockFile, paths};
use cozy_proto::calcifer::{Down, Holder};

/// `cozy calcifer` is named `calcifer`, serves its home, and exits 0 on `down`.
#[test]
fn cozy_calcifer_is_a_process_named_calcifer_that_down_stops() {
    let Some(home) = Home::new() else { return };
    let cozy = paths::Home::at(home.root());
    durable::private_dir(&cozy.calcifer()).unwrap();
    let mut calcifer = home
        .command()
        .arg("calcifer")
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let pid = calcifer.id();
    let lock = LockFile::new(cozy.calcifer_lock());
    // A hang bound for a Calcifer that never publishes; the product waits without one.
    let until = Instant::now() + Duration::from_secs(120);
    let holder: Holder = loop {
        let record = lock.holder().unwrap();
        if let Some(holder) = record.and_then(|r| serde_json::from_slice::<Holder>(&r).ok())
            && holder.pid == pid
        {
            break holder;
        }
        assert!(calcifer.try_wait().unwrap().is_none(), "Calcifer exited");
        assert!(
            Instant::now() < until,
            "Calcifer never published its record"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    #[cfg(target_os = "linux")]
    assert_eq!(
        std::fs::read_to_string(format!("/proc/{pid}/comm")).unwrap(),
        "calcifer\n"
    );

    let down: Down = tokio::runtime::Runtime::new().unwrap().block_on(async {
        reqwest::Client::new()
            .post(format!("http://{}/v1/down", holder.address))
            .bearer_auth(&holder.token)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    });
    assert!(down.stopping && down.owed.is_empty());
    assert!(calcifer.wait().unwrap().success());
    assert_eq!(lock.holder().unwrap(), None);
}
