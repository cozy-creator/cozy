use super::*;

#[test]
fn every_rank_contributes_host_headroom_and_a_shared_process_counts_once() {
    let call = Call {
        ranks: vec![
            Rank { pid: 10, gpu: "first".into() },
            Rank { pid: 20, gpu: "second".into() },
            Rank { pid: 20, gpu: "third".into() },
        ],
        package: "package".into(), shape: "shape".into(),
    };
    let rows = call.host(|gpu| Some(match gpu { "first" => 100, "second" => 200, _ => 150 }), |pid| match pid { 10 => 30, _ => 60 });
    assert_eq!(rows, [
        HostCall { pid: 10, peak: Some(100), anon: 30, running: 1 },
        HostCall { pid: 20, peak: Some(200), anon: 60, running: 1 },
    ]);
    assert_eq!(Keep::default().look(rows), 210, "70 bytes for rank 0 and 140 for its follower");
}

#[test]
fn an_unmeasured_follower_still_enters_the_host_growth_rule() {
    let call = Call {
        ranks: vec![Rank { pid: 10, gpu: "known".into() }, Rank { pid: 20, gpu: "unknown".into() }],
        package: "package".into(), shape: "shape".into(),
    };
    let rows = call.host(|gpu| (gpu == "known").then_some(100), |_| 30);
    assert_eq!(rows[1], HostCall { pid: 20, peak: None, anon: 30, running: 1 });
}

#[test]
fn persisted_peaks_are_canonical_maxima_across_cards() {
    let input = vec![
        Peak::Request { key: "package|shape".into(), bytes: 100 },
        Peak::Request { key: "package|shape".into(), bytes: 200 },
        Peak::Compute { key: "package|shape".into(), bytes: 50 },
        Peak::Compute { key: "package|shape".into(), bytes: 20 },
        Peak::Host { key: "package|shape".into(), bytes: 70 },
    ];
    let merged = merge_peaks(input.clone().into_iter());
    assert_eq!(merged, merge_peaks(input.into_iter().rev()));
    assert_eq!(merged, vec![
        Peak::Request { key: "package|shape".into(), bytes: 200 },
        Peak::Compute { key: "package|shape".into(), bytes: 50 },
        Peak::Host { key: "package|shape".into(), bytes: 70 },
    ]);
}
