//! `make_room`'s order over tables, pure: other sets' idle pages, then the current step's own
//! pages, furthest next use first; never a pinned page. A small simulator runs real plans through
//! it at real capacities (each bind takes its page's room, each block its activations'), plus
//! prefetch, and the ledger's admission and placement while they last.

use varena::decide::{fits, place, prefetch, victims, Admitted, Victim};
use varena::table::{Key, Page, Req, Set, Table};

const MB: u64 = 1 << 20;
const GB: u64 = 1 << 30;

fn set(name: &str, pages: usize, bytes: u64) -> Option<Set> {
    Some(Set { name: name.into(), pages: vec![Page { bytes, ..Page::default() }; pages], declared: true })
}

/// A table with these sets and one request over `steps` (set indices, passes).
fn table(capacity: u64, sets: Vec<Option<Set>>, steps: Vec<(Vec<u32>, u32)>) -> Table {
    let req = Req { id: 1, name: "r".into(), steps, ..Req::default() };
    Table { limit: capacity, sets, requests: vec![req] }
}

fn map(t: &mut Table, k: Key, clock: &mut u64) {
    *clock += 1;
    let p = t.page_mut(k).unwrap();
    (p.mapped, p.filled, p.last_use) = (true, true, *clock);
}

fn pages(v: &[Victim]) -> Vec<Key> {
    v.iter().map(|v| v.key).collect()
}

fn free(t: &Table) -> u64 {
    t.limit.saturating_sub(t.mapped())
}

/// Takes `bytes` as `make_room` does: victims in order until free covers them. Returns the
/// pages taken; `None` when nothing is left to take.
fn make_room(t: &mut Table, bytes: u64, keep: &[Key]) -> Option<Vec<Key>> {
    let mut taken = vec![];
    while free(t) < bytes {
        let key = victims(t, keep).first()?.key;
        let p = t.page_mut(key).unwrap();
        (p.mapped, p.filled) = (false, false);
        taken.push(key);
    }
    Some(taken)
}

#[derive(Default, Debug)]
struct Run {
    /// Pages taken while they were in the running step (refilled later in the pass).
    own: u64,
    maps: u64,
}

/// Runs the plan: each pass binds its step's blocks in order, each block needing `acts` free as
/// its activations while bound. The bound page is the one pin.
fn play(t: &mut Table, acts: u64) -> Option<Run> {
    let (mut run, mut clock) = (Run::default(), 0);
    let steps = t.requests[0].steps.clone();
    for (i, (sets, passes)) in steps.iter().enumerate() {
        t.requests[0].cursor = varena::table::Cursor { step: i as u32, ..Default::default() };
        for _ in 0..*passes {
            for &s in sets {
                let begun = t.requests[0].cursor.passes.get(&s).copied().unwrap_or(0);
                t.requests[0].cursor.passes.insert(s, begun + 1);
                t.requests[0].cursor.next.insert(s, 0);
                let n = t.sets[s as usize].as_ref().unwrap().pages.len() as u32;
                for b in 0..n {
                    let k = (s, b);
                    if !t.page(k).unwrap().mapped {
                        let bytes = t.page(k).unwrap().bytes;
                        let taken = make_room(t, bytes, &[k])?;
                        run.own += taken.iter().filter(|k| sets.contains(&k.0)).count() as u64;
                        map(t, k, &mut clock);
                        run.maps += 1;
                    }
                    t.page_mut(k).unwrap().pins = vec![1];
                    t.requests[0].cursor.next.insert(s, b + 1);
                    let taken = make_room(t, acts, &[k])?;
                    run.own += taken.iter().filter(|k| sets.contains(&k.0)).count() as u64;
                    t.page_mut(k).unwrap().pins.clear();
                }
            }
        }
    }
    Some(run)
}

/// H3: [te, {avae, vae}, dit×8, avae, vae] (8-bit DiT in 40 blocks).
fn h3(capacity: u64) -> Table {
    let sets = vec![set("te", 24, 256 * MB), set("avae", 2, 256 * MB), set("vae", 2, 256 * MB), set("dit", 40, 352 * MB)];
    table(capacity, sets, vec![(vec![0], 1), (vec![1, 2], 1), (vec![3], 8), (vec![1], 1), (vec![2], 1)])
}

/// SDXL: [te, unet×30, vae].
fn sdxl(capacity: u64) -> Table {
    let sets = vec![set("te", 8, 200 * MB), set("unet", 26, 100 * MB), set("vae", 2, 80 * MB)];
    table(capacity, sets, vec![(vec![0], 1), (vec![1], 30), (vec![2], 1)])
}

/// The audit's bar: one loop serves every capacity; a model bigger than the card streams (its
/// own blocks behind the cursor go), one that fits loads once and stays.
#[test]
fn h3_and_sdxl_complete_at_24_12_and_8_gib() {
    for capacity in [24 * GB, 12 * GB, 8 * GB] {
        let run = play(&mut sdxl(capacity), GB).unwrap_or_else(|| panic!("SDXL at {capacity}"));
        assert_eq!((run.own, run.maps), (0, 36), "SDXL fits at {}: each page loads once", capacity / GB);
        let run = play(&mut h3(capacity), 2 * GB).unwrap_or_else(|| panic!("H3 at {capacity}"));
        let dit = 40 * 352 * MB;
        if capacity > dit + 3 * GB {
            assert_eq!(run.own, 0, "H3's DiT fits at {}", capacity / GB);
        } else {
            assert!(run.own > 0, "H3's DiT streams at {}", capacity / GB);
        }
    }
}

/// The order: pages of sets no plan uses, least recently used first; then a later step's pages,
/// furthest first; then the current step's pages, furthest next use first. Never a pinned or
/// filling page, nor one being bound; a page whose release is pending is marked.
#[test]
fn victims_go_other_sets_pages_then_own_pages_furthest_first() {
    let mut t = table(64 * MB, vec![set("now", 2, MB), set("later", 2, MB), set("idle", 2, MB)], vec![(vec![0], 1), (vec![1], 1)]);
    let mut clock = 0;
    for k in [(2, 1), (2, 0), (1, 0), (1, 1), (0, 0), (0, 1)] {
        map(&mut t, k, &mut clock);
    }
    t.requests[0].cursor.passes.insert(0, 1);
    t.requests[0].cursor.next.insert(0, 1);
    t.page_mut((0, 1)).unwrap().releasing = vec![3];
    let v = victims(&t, &[(1, 1)]);
    let page = |key, releasing| Victim { key, bytes: MB, releasing };
    assert_eq!(
        v,
        vec![page((2, 1), false), page((2, 0), false), page((1, 0), false), page((0, 0), false), page((0, 1), true)],
        "idle by age; later's; then now's block behind the cursor (next used a pass ahead), then the block ahead"
    );
    t.page_mut((0, 0)).unwrap().pins = vec![1];
    t.page_mut((2, 1)).unwrap().filling = true;
    assert!(!pages(&victims(&t, &[])).iter().any(|k| *k == (0, 0) || *k == (2, 1)), "pinned and filling pages never go");
}

/// Prefetch maps the next blocks past the cursor into free bytes only: never more than fits.
#[test]
fn prefetch_maps_the_next_blocks_into_free_bytes_only() {
    let mut t = table(3 * MB, vec![set("s", 4, MB)], vec![(vec![0], 1)]);
    t.requests[0].cursor.passes.insert(0, 1);
    t.requests[0].cursor.next.insert(0, 1);
    assert_eq!(prefetch(&t, 1, free(&t)), vec![(0, 1), (0, 2)], "the next two blocks");
    let mut clock = 0;
    map(&mut t, (0, 0), &mut clock);
    map(&mut t, (0, 3), &mut clock);
    assert_eq!(prefetch(&t, 1, free(&t)), vec![(0, 1)], "one block of room left");
}

#[test]
fn admission_counts_each_byte_once() {
    let row = |remaining, running, unmeasured| Admitted { context: GB, outside: 0, remaining, running, unmeasured, yielded: [(0, 0); 3] };
    assert!(fits(24 * GB, 0, &[row(10 * GB, 1, 0)], &[], Some(12 * GB), None));
    assert!(!fits(24 * GB, 0, &[row(10 * GB, 1, 0)], &[], Some(13 * GB + 1), None));
    assert!(!fits(24 * GB, 2 * GB, &[row(10 * GB, 1, 0)], &[], Some(12 * GB), None), "foreign bytes are taken off");
    assert!(!fits(24 * GB, 0, &[row(10 * GB, 1, 0)], &[Some(4 * GB)], Some(12 * GB), None), "a dispatched request counts");
    // Unmeasured: only on an idle GPU, which admits anything (a request larger than it streams).
    assert!(!fits(24 * GB, 0, &[row(GB, 1, 0)], &[], None, None));
    assert!(fits(24 * GB, 0, &[row(0, 0, 0)], &[], None, None));
    assert!(fits(8 * GB, 0, &[row(0, 0, 0)], &[], Some(20 * GB), None));
    assert!(!fits(24 * GB, 0, &[row(GB, 1, 1)], &[], Some(GB), None));
}

/// Review 3, R4: a row with requests yielded to two families publishes each family's peak
/// apart. A's child is admitted beside A's parent as if it were gone, never beside B's.
#[test]
fn a_rows_yielded_w_is_per_family() {
    let (a, b) = (11, 22);
    let rows = [Admitted { context: GB, outside: 0, remaining: 0, running: 1, unmeasured: 0, yielded: [(a, 8 * GB), (b, 6 * GB), (0, 0)] }];
    assert!(fits(24 * GB, 0, &rows, &[], Some(17 * GB), Some(a)) && !fits(24 * GB, 0, &rows, &[], Some(17 * GB + 1), Some(a)), "A's child: beside B's parent (6) only");
    assert!(fits(24 * GB, 0, &rows, &[], Some(15 * GB), Some(b)) && !fits(24 * GB, 0, &rows, &[], Some(15 * GB + 1), Some(b)), "B's child: beside A's parent (8) only");
    assert!(fits(24 * GB, 0, &rows, &[], Some(9 * GB), None) && !fits(24 * GB, 0, &rows, &[], Some(9 * GB + 1), None), "a stranger's child: beside both");
}

/// S5 r3: largest first; GPU 0 keeps what fits, else the GPU with the most room that holds it,
/// else GPU 0 (it pages there); an unknown set goes where the most room is left.
#[test]
fn sets_place_largest_first_and_stay_on_gpu_0_when_they_fit() {
    assert_eq!(place(&[10 * MB, 20 * MB], &[Some(4 * MB), Some(6 * MB)]), vec![0, 0], "both fit on GPU 0");
    assert_eq!(place(&[8 * MB, 20 * MB], &[Some(4 * MB), Some(6 * MB)]), vec![1, 0], "the larger first: GPU 0 keeps 6, the 4 moves");
    assert_eq!(place(&[2 * MB, 5 * MB, 7 * MB], &[Some(4 * MB)]), vec![2], "the most room that holds it");
    assert_eq!(place(&[2 * MB, 3 * MB], &[Some(4 * MB)]), vec![0], "nowhere holds it: GPU 0 pages it");
    assert_eq!(place(&[2 * MB, 9 * MB, 3 * MB], &[None, Some(4 * MB)]), vec![1, 1], "known first; the unknown where the most is left");
}
