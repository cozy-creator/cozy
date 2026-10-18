//! Live marker names (`meta::now_nanos_unique`) never repeat across threads.

#[test]
fn concurrent_marker_timestamp_reservations_are_unique() {
    use std::sync::{Arc, Barrier};
    let barrier = Arc::new(Barrier::new(16));
    let values = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    let mut values = Vec::with_capacity(4096);
                    for _ in 0..4096 {
                        barrier.wait();
                        values.push(tensorfs_core::meta::now_nanos_unique());
                    }
                    values
                })
            })
            .collect();
        threads
            .into_iter()
            .flat_map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>()
    });
    let count = values.len();
    let unique = values.into_iter().collect::<std::collections::HashSet<_>>();
    assert_eq!(
        unique.len(),
        count,
        "independent live marker owners received the same name"
    );
}
