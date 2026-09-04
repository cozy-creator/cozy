//! Choosing what to stop naming — the query half of reclamation. Nothing here deletes.
//!
//! **GC collects what nothing names. Reclamation chooses what to stop naming.** They are
//! two acts and they get two names: `gc.rs` is the sweep, this is the chooser, and the
//! chooser never runs the sweep. Deletion stays the consequence of an explicit act
//! (decisions.md 1135), so this module produces a document and an operator or an
//! orchestrator decides what to do with it.
//!
//! ## The question, and why it is not "how big is this model"
//!
//! Owner, verbatim: *"We can't just delete models outright, we need to know how much space
//! is reclaimable; deleting a model might reclaim almost no space because it was de-duped
//! with one we want to keep."*
//!
//! On a content-addressed store that is not a corner case. Two models can be backed by one
//! object; H3 carries two byte-identical index files resolving to a single blob. A 6.9 GB
//! SDXL closure sharing its text encoder and VAE with the model beside it may free a few
//! hundred megabytes. Evicting it looks like progress, frees almost nothing, leaves the
//! disk just as full, and buys a re-fetch later for the privilege.
//!
//! So every number here is a property of the candidate SET, computed by
//! [`Census::reclaim_plan`] — the same walk the collection performs, with the deletions not
//! yet applied. A separate, faster estimator is exactly where two expressions that agree on
//! every fixture anyone thought to write come to disagree on the store that matters.
//!
//! ## Why the ranking is not LRU
//!
//! LRU is the familiar name and on its own it is the wrong key here: **LRU ranks access,
//! the budget is about bytes, and on a content-addressed store those orders differ
//! arbitrarily.** The least-recently-used model can be the one that frees nothing.
//!
//! The ranking is **marginal bytes freed per drop, descending**, over roots whose marginal
//! is non-zero, with the oldest-recorded root breaking ties. Three reasons, and they are
//! judgement calls rather than derivations:
//!
//! 1. **Every candidate is already off the serving set.** The orchestrator passes the
//!    serving set as `keep`, so recency is a weak signal among what remains — they are all
//!    cold by construction — while byte yield is the strong one.
//! 2. **Each eviction has a fixed operational cost** (a model leaves the set, a placement
//!    is torn down, a later refetch is a round trip), so the thing to minimise is the
//!    NUMBER of drops needed to reach the need, which is what maximising marginal per drop
//!    does.
//! 3. **Per-byte refetch cost is uniform today, so it cannot rank anything.** Bytes
//!    recoverable from the `/workspace` cache and bytes recoverable from R2 are counted
//!    apart and reported apart, but they are not weighted apart: the ingest pool measures
//!    516 MB/s and the volume's read rate against this workload has never been measured.
//!    Inventing a multiplier would be decisions.md 698's shape exactly — a number that
//!    looks measured because it is precise.
//!
//! A root whose marginal is zero is **never proposed at all**, at any need. The owner's
//! worry is made structurally unrepresentable rather than merely unlikely.
//!
//! ## What "recency" is here, precisely
//!
//! It is the repository document's mtime, and it is named `last_recorded_unix` rather than
//! "last used" because that is what it measures: when this root was last WRITTEN — pulled,
//! or had a release lane moved — not when it was last read. TensorFS keeps no read-time
//! stamp, and inventing one would mean a hot-path write and a new authority for a signal
//! that only breaks ties. If a true last-read stamp is ever wanted it is a separate issue
//! with a named consumer.

use std::collections::BTreeSet;

use crate::canon::Value;
use crate::err::{refuse, Code, Result};
use crate::repository::RepositoryName;
use crate::storage::{Census, HeldKey};
use crate::store::Store;

/// One retention root, and what it would cost to lose it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootFacts {
    pub org: String,
    pub name: String,
    /// Everything this root reaches, shared bytes included.
    pub bytes_closure: u64,
    /// What dropping this root ALONE would free. The gap between this and `bytes_closure`
    /// IS the dedup, which is why both are reported and neither is reported alone.
    pub bytes_solo: u64,
    /// When the repository document was last written. See the module docs: not last read.
    pub last_recorded_unix: u64,
}

/// One chosen victim, in drop order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Victim {
    pub org: String,
    pub name: String,
    /// What dropping this root frees GIVEN the victims listed above it. Superadditive: an
    /// object shared only between two victims belongs to whichever is named second, and to
    /// neither of them alone.
    pub marginal_bytes: u64,
    /// The plan's running total through this victim.
    pub cumulative_bytes: u64,
    pub bytes_closure: u64,
    pub last_recorded_unix: u64,
}

/// What a reclamation would drop, and what it would free. Deletes nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub need_bytes: u64,
    /// Bytes no repository reaches at all. A plain `tfs gc` frees these WITHOUT dropping
    /// anything, so they are counted first and reported apart: telling an operator to evict
    /// a model for bytes a collection would have handed back for free is the same error as
    /// telling them to evict one that frees nothing.
    pub unreferenced_bytes: u64,
    /// `unreferenced_bytes` plus every victim's marginal.
    pub reclaimable_bytes: u64,
    pub victims: Vec<Victim>,
}

impl Plan {
    pub fn line(&self) -> String {
        format!(
            "{} B needed: {} B already unreferenced plus {} B from {} root{}",
            self.need_bytes,
            self.unreferenced_bytes,
            self.reclaimable_bytes - self.unreferenced_bytes,
            self.victims.len(),
            if self.victims.len() == 1 { "" } else { "s" },
        )
    }

    pub fn json(&self) -> Vec<u8> {
        crate::canon::write(&Value::obj(vec![
            ("need_bytes", Value::uint(self.need_bytes)),
            ("reclaimable_bytes", Value::uint(self.reclaimable_bytes)),
            ("unreferenced_bytes", Value::uint(self.unreferenced_bytes)),
            (
                "victims",
                Value::arr(
                    self.victims
                        .iter()
                        .map(|victim| {
                            Value::obj(vec![
                                ("bytes_closure", Value::uint(victim.bytes_closure)),
                                ("cumulative_bytes", Value::uint(victim.cumulative_bytes)),
                                (
                                    "last_recorded_unix",
                                    Value::uint(victim.last_recorded_unix),
                                ),
                                ("marginal_bytes", Value::uint(victim.marginal_bytes)),
                                ("name", Value::str(victim.name.clone())),
                                ("org", Value::str(victim.org.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
        ]))
    }
}

/// Every reclaim verb refuses on a Store that was never given a disk budget.
///
/// The budget IS the opt-in (tfs-064). The owner's requirement was that reclamation must
/// not be default-on — *"users would be confused if files start disappearing"* — so a
/// personal `$HOME/.tensorfs`, which has no budget, cannot be reclaimed even by asking.
fn permitted(store: &Store) -> Result<()> {
    if store.disk_budget().is_some() {
        return Ok(());
    }
    refuse(
        Code::RECLAIM_NOT_PERMITTED,
        format!(
            "{} was never given a disk budget, so nothing may be reclaimed from it; \
             `tfs store ensure <root> --disk-budget <bytes>` is the opt-in",
            store.root().display()
        ),
    )
}

fn recorded_unix(store: &Store, repo: &RepositoryName) -> u64 {
    std::fs::metadata(store.repository_path(repo))
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |since| since.as_secs())
}

/// Every root, with what it reaches and what dropping it alone would free.
pub fn usage(store: &Store, holds: &[HeldKey]) -> Result<Vec<RootFacts>> {
    permitted(store)?;
    let census = Census::open(store.root())?;
    let index = census.holder_index(holds)?;
    let mut rows = Vec::with_capacity(index.roots.len());
    for (position, repo) in index.roots.iter().enumerate() {
        rows.push(RootFacts {
            org: repo.org.clone(),
            name: repo.name.clone(),
            bytes_closure: index.closure_bytes(position),
            bytes_solo: index.reclaimable(&BTreeSet::from([position])),
            last_recorded_unix: recorded_unix(store, repo),
        });
    }
    Ok(rows)
}

/// Choose the roots to drop to free `need` bytes, keeping everything in `keep`.
///
/// Refuses `RECLAIM_INSUFFICIENT` when the whole droppable set frees less than `need`,
/// rather than returning a plan that empties the Store and still cannot admit the fetch.
pub fn plan(store: &Store, need: u64, keep: &[RepositoryName], holds: &[HeldKey]) -> Result<Plan> {
    permitted(store)?;
    let census = Census::open(store.root())?;
    let index = census.holder_index(holds)?;

    let kept: BTreeSet<usize> = keep.iter().filter_map(|repo| index.position(repo)).collect();
    let candidates: Vec<usize> = (0..index.roots.len())
        .filter(|position| !kept.contains(position))
        .collect();

    // `reclaimable(∅)` is exactly the unreferenced bytes, so the running total starts
    // there and every marginal below is measured against it.
    let unreferenced = index.unreferenced();
    let mut chosen: BTreeSet<usize> = BTreeSet::new();
    let mut order: Vec<usize> = Vec::new();
    let mut cumulative = unreferenced;

    while cumulative < need {
        let mut best: Option<(u64, u64, usize)> = None;
        let mut coldest: Option<(u64, usize)> = None;
        for &candidate in &candidates {
            if chosen.contains(&candidate) {
                continue;
            }
            let recorded = recorded_unix(store, &index.roots[candidate]);
            if coldest.is_none_or(|(top, _)| recorded < top) {
                coldest = Some((recorded, candidate));
            }
            let mut trial = chosen.clone();
            trial.insert(candidate);
            let marginal = index.reclaimable(&trial).saturating_sub(cumulative);
            if marginal == 0 {
                continue;
            }
            // Marginal benefit per drop, descending; oldest-recorded breaks ties, because
            // between two roots that free the same bytes the colder one is the one less
            // likely to be wanted back.
            let better = match best {
                None => true,
                Some((top, top_recorded, _)) => {
                    marginal > top || (marginal == top && recorded < top_recorded)
                }
            };
            if better {
                best = Some((marginal, recorded, candidate));
            }
        }
        let pick = match best {
            Some((_, _, candidate)) => candidate,
            // NOTHING FREES A BYTE ON ITS OWN, WHICH IS NOT THE SAME AS NOTHING CAN BE
            // FREED. An object reachable from exactly two candidates is unique to neither,
            // so the PAIR frees it and no member of the pair does. Refusing here — which an
            // earlier draft did, by skipping every zero-marginal root — would report
            // "insufficient" over a store where dropping two models frees plenty. So the
            // search advances on the coldest root and the prune below throws away anything
            // that turns out not to have earned its place.
            None => match coldest {
                Some((_, candidate)) => candidate,
                None => break,
            },
        };
        chosen.insert(pick);
        order.push(pick);
        cumulative = index.reclaimable(&chosen);
    }

    // NOTHING USELESS SURVIVES INTO THE PLAN. A victim whose absence still leaves the plan
    // at or above `need` was never needed, and proposing it would be a model lost for no
    // bytes — the owner's exact worry. Pruning runs from the last pick backwards, so the
    // high-yield roots the search chose first are the ones that stay.
    if cumulative >= need {
        let mut position = order.len();
        while position > 0 {
            position -= 1;
            let mut trial = chosen.clone();
            trial.remove(&order[position]);
            if index.reclaimable(&trial) >= need {
                chosen = trial;
                order.remove(position);
            }
        }
        cumulative = index.reclaimable(&chosen);
    }

    // Marginals are reported in the order the plan drops them, over the plan that survived
    // pruning — not the order the search happened to visit.
    let mut victims = Vec::with_capacity(order.len());
    let mut running = unreferenced;
    let mut prefix: BTreeSet<usize> = BTreeSet::new();
    for &candidate in &order {
        prefix.insert(candidate);
        let total = index.reclaimable(&prefix);
        victims.push(Victim {
            org: index.roots[candidate].org.clone(),
            name: index.roots[candidate].name.clone(),
            marginal_bytes: total - running,
            cumulative_bytes: total,
            bytes_closure: index.closure_bytes(candidate),
            last_recorded_unix: recorded_unix(store, &index.roots[candidate]),
        });
        running = total;
    }

    if cumulative < need {
        return refuse(
            Code::RECLAIM_INSUFFICIENT,
            format!(
                "dropping every droppable root frees {cumulative} B, {} B short of the \
                 {need} B asked for",
                need - cumulative
            ),
        );
    }

    // THE ACCELERATOR IS NOT THE ORACLE. The greedy search ran on the holder index because
    // asking `reclaim_plan` once per candidate per step re-parses every CozyTensors header
    // from disk. The number that gets REPORTED is re-derived through `reclaim_plan` — the
    // same walk the collection will perform — so a wrong index cannot become a wrong
    // promise. They are pinned together by `the_index_and_the_oracle_agree`.
    let drop: Vec<RepositoryName> = victims
        .iter()
        .map(|victim| RepositoryName::new(victim.org.clone(), victim.name.clone()))
        .collect::<Result<Vec<_>>>()?;
    let oracle: u64 = census
        .reclaim_plan(&drop, holds)?
        .iter()
        .map(|row| row.length)
        .sum();
    if oracle != cumulative {
        return refuse(
            Code::CENSUS_INCOMPLETE,
            format!(
                "the reclaim index promised {cumulative} B and the census plan frees \
                 {oracle} B; the two disagree and the plan is not trustworthy"
            ),
        );
    }

    Ok(Plan {
        need_bytes: need,
        unreferenced_bytes: unreferenced,
        reclaimable_bytes: cumulative,
        victims,
    })
}
