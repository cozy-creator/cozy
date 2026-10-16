//! Choosing what to stop naming. Nothing here deletes: `gc::collect_cached_for` asks this
//! planner which cache roots to drop and drops them.
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
//! ## The ranking: the fewest drops, the least recently used among them
//!
//! Every candidate is already off the serving set (the caller passes that as `keep`), and
//! each eviction costs a refetch, so the plan minimises the NUMBER of drops: while no single
//! root finishes the need, the root that frees the most goes (the older of two that free
//! the same). **Once a root can finish the need alone, the least recently used such root
//! goes**, never the largest: a dip of a few bytes takes the coldest model that covers it.
//! LRU alone would be the wrong key, because the least recently used root can be the one
//! that frees nothing.
//!
//! A root whose marginal is zero is **never proposed at all**, at any need, and a plan that
//! cannot cover the need is refused whole rather than returned short.
//!
//! ## What "recency" is here
//!
//! The later of the repository document's mtime (when the root was last written) and, for
//! a cache root, its last use (`cache_roots::touch`). It is reported as
//! `last_recorded_unix`.

use std::collections::BTreeSet;

use crate::err::{refuse, Code, Result};
use crate::repository::RepositoryName;
use crate::storage::{Census, HeldKey};
use crate::store::Store;

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

fn recorded_unix(store: &Store, repo: &RepositoryName) -> u64 {
    std::fs::metadata(store.repository_path(repo))
        .and_then(|metadata| metadata.modified())
        .ok()
        .max(crate::cache_roots::used(store, repo))
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |since| since.as_secs())
}

/// What dropping every root outside `keep` would free, unreferenced bytes included.
pub(crate) fn droppable_bytes(
    store: &Store,
    keep: &[RepositoryName],
    holds: &[HeldKey],
) -> Result<u64> {
    let index = Census::open(store.root())?.holder_index(holds)?;
    let all = (0..index.roots.len())
        .filter(|position| !keep.contains(&index.roots[*position]))
        .collect();
    Ok(index.reclaimable(&all))
}

/// Choose the roots to drop to free `need` bytes, keeping everything in `keep`.
///
/// Refuses `RECLAIM_INSUFFICIENT` when the whole droppable set frees less than `need`,
/// rather than returning a plan that empties the Store and still cannot admit the fetch.
pub fn plan(
    store: &Store,
    need: u64,
    keep: &[RepositoryName],
    holds: &[HeldKey],
) -> Result<Plan> {
    let census = Census::open(store.root())?;
    let index = census.holder_index(holds)?;

    let kept: BTreeSet<usize> = keep
        .iter()
        .filter_map(|repo| index.position(repo))
        .collect();
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
        // (marginal bytes, last use, root) of every root still standing.
        let rest: Vec<(u64, u64, usize)> = candidates
            .iter()
            .filter(|candidate| !chosen.contains(candidate))
            .map(|&candidate| {
                let mut trial = chosen.clone();
                trial.insert(candidate);
                (
                    index.reclaimable(&trial).saturating_sub(cumulative),
                    recorded_unix(store, &index.roots[candidate]),
                    candidate,
                )
            })
            .collect();
        let coldest = |rows: &mut dyn Iterator<Item = &(u64, u64, usize)>| {
            rows.min_by_key(|(_, recorded, _)| *recorded).copied()
        };
        let finishing = coldest(&mut rest.iter().filter(|row| cumulative + row.0 >= need));
        let largest = rest
            .iter()
            .rev() // of two that tie on both, the first
            .filter(|row| row.0 > 0)
            .max_by_key(|(marginal, recorded, _)| (*marginal, std::cmp::Reverse(*recorded)))
            .copied();
        // NOTHING FREES A BYTE ON ITS OWN, WHICH IS NOT THE SAME AS NOTHING CAN BE FREED.
        // An object reachable from exactly two candidates is unique to neither, so the PAIR
        // frees it and no member of the pair does. The search then advances on the coldest
        // root and the prune below throws away anything that did not earn its place.
        let Some((_, _, pick)) = finishing.or(largest).or_else(|| coldest(&mut rest.iter()))
        else {
            break;
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
