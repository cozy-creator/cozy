//! Physical observations and logical admission envelopes are separate facts. The memory
//! mechanism supplies unique committed backing; these tests exercise the cap arithmetic.
use varena::table::{Class, Page, Remote, Req, Set, Table};
const M: u64 = 1 << 20;

fn own() -> Table {
    Table {
        capacity: 24 * M, physical: 8 * M, vmm: 4 * M, context: M, reserved: 3 * M,
        sets: vec![Some(Set { name: "weights".into(), pages: vec![Page { bytes: 4 * M, mapped: true, ..Page::default() }], ..Set::default() })],
        requests: vec![Req { id: 1, ..Req::default() }], ..Table::default()
    }
}

#[test]
fn own_and_peer_promises_cannot_reduce_physical_foreign_usage() {
    let mut t = own();
    t.others.push(Remote { row: 2, context: M, torch: 3 * M, physical: 4 * M, ..Remote::default() });
    t.orphans = M;
    t.foreign = t.foreign_used(24 * M, 16 * M);
    assert_eq!(t.foreign, 3 * M);
    let free = t.free();
    t.promised = 2 * M;
    assert_eq!(t.foreign_used(24 * M, 16 * M), 3 * M);
    assert_eq!(t.free(), free - 2 * M);
    t.promised = 0;
    t.requests[0].claimed = 5 * M; // Two bytes beyond the three already physically reserved.
    assert_eq!(t.foreign_used(24 * M, 16 * M), 3 * M);
    assert_eq!(t.free(), free - 2 * M);
    t.requests[0].claimed = 0;
    t.others[0].torch += 2 * M;
    assert_eq!(t.foreign_used(24 * M, 16 * M), 3 * M);
    assert_eq!(t.free(), free - 2 * M);
}

#[test]
fn successful_credit_conversion_does_not_charge_the_same_bytes_twice() {
    let mut t = own();
    t.promised = 2 * M;
    t.foreign = t.foreign_used(24 * M, 10 * M);
    let admitted = t.free();
    // Consuming the promise provisionally changes admission fields, not actual backing.
    t.promised = 0;
    t.reserved += 2 * M;
    assert_eq!(t.foreign_used(24 * M, 10 * M), 2 * M);
    assert_eq!(t.free(), admitted);
    // Only the successful driver allocation adds physical bytes; actual driver used agrees.
    t.physical += 2 * M;
    assert_eq!(t.foreign_used(24 * M, 12 * M), 2 * M);
    assert_eq!(t.free(), admitted);
}

#[test]
fn a_placeholder_is_admitted_without_becoming_physical() {
    let mut t = own();
    t.foreign = t.foreign_used(24 * M, 10 * M);
    let before = t.free();
    t.sets[0].as_mut().unwrap().pages.push(Page { bytes: 2 * M, mapped: true, filling: true, ..Page::default() });
    t.mapping = 2 * M;
    assert_eq!(t.foreign_used(24 * M, 10 * M), 2 * M);
    assert_eq!(t.free(), before - 2 * M);
    t.physical += 2 * M; // Actual cuMemCreate completed, without a second admission charge.
    t.mapping = 0;
    t.vmm += 2 * M;
    assert_eq!(t.foreign_used(24 * M, 12 * M), 2 * M);
    assert_eq!(t.free(), before - 2 * M);
}

#[test]
fn shared_retirement_and_custody_handover_count_unique_backing_once() {
    let mut t = Table { capacity: 8 * M, physical: 4 * M, vmm: 4 * M, returning: 4 * M, pending: 4 * M, inflight: 4 * M, ..Table::default() };
    // One backing has both an unmap and a custody-withdrawal ticket. Neither counter is added
    // to the caller's unique physical observation again.
    t.foreign = t.foreign_used(8 * M, 6 * M);
    assert_eq!(t.foreign, 2 * M);
    assert_eq!(t.torch_limit(), 2 * M, "overlapping tickets cannot cause a false allocator refusal");
    assert_eq!(t.classes()[Class::P.index()], 4 * M);
    t.inflight = 0;
    assert_eq!(t.foreign_used(8 * M, 6 * M), 2 * M);
    t.pending = 0;
    t.physical = 0;
    t.vmm = 0;
    t.returning = 0;
    assert_eq!(t.foreign_used(8 * M, 2 * M), 2 * M);
    // The same physical backing is orphaned, then imported by a single active owner.
    t.orphans = 4 * M;
    assert_eq!(t.foreign_used(8 * M, 6 * M), 2 * M);
    t.orphans = 0;
    let mut peer = Remote { row: 2, physical: 4 * M, ..Remote::default() };
    peer.classes[Class::N.index()] = 4 * M;
    t.others.push(peer);
    assert_eq!(t.foreign_used(8 * M, 6 * M), 2 * M);
}

#[test]
fn foreign_usage_is_charged_only_beyond_the_cards_room_outside_the_cap() {
    let t = Table { capacity: 8 * M, physical: 4 * M, ..Table::default() };
    assert_eq!(t.foreign_used(16 * M, 10 * M), 0);
    assert_eq!(t.foreign_used(16 * M, 14 * M), 2 * M);
}


#[test]
fn quarantined_backing_stays_charged_but_is_not_available_returning_memory() {
    let t = Table { capacity: 8 * M, physical: 4 * M, vmm: 4 * M, ..Table::default() };
    assert_eq!(t.held(), 4 * M);
    assert_eq!(t.classes()[Class::P.index()], 4 * M);
    assert_eq!(t.returning(), 0);
    assert_eq!(t.torch_limit(), 4 * M);
}

#[test]
fn unstable_physical_counts_cannot_reduce_foreign_usage() {
    let t = Table { capacity: 8 * M, physical: 4 * M, vmm: 4 * M, ..Table::default() };
    assert_eq!(t.foreign_from_known(8 * M, 6 * M, 4 * M), 2 * M);
    assert_eq!(t.foreign_from_known(8 * M, 6 * M, 0), 6 * M);
    assert_eq!(t.held(), 4 * M, "invalid physical evidence does not release its admission charge");
}
