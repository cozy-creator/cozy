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
use crate::ids::ObjectRef;
use crate::repo_cache::{CacheKind, CacheWrite};
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
                                ("last_recorded_unix", Value::uint(victim.last_recorded_unix)),
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

/// The victims a plan document names, read back for `run`.
///
/// Only the NAMES are read. Every byte figure in the document is a record of what was true
/// when the plan was made; `run` re-derives the delete set from the census, so a stale plan
/// costs a re-plan and can never cost the wrong bytes.
pub fn victims_of(document: &[u8]) -> Result<Vec<RepositoryName>> {
    let value = crate::canon::parse_canonical(document, crate::limits::DOC_MAX_BYTES)?;
    let mut fields = crate::canon::Fields::new("ReclaimPlan", &value)?;
    let victims = crate::canon::as_arr("ReclaimPlan", "victims", fields.req("victims")?)?;
    let mut names = Vec::with_capacity(victims.len());
    for victim in victims {
        let mut row = crate::canon::Fields::new("ReclaimPlan.victim", victim)?;
        let name = row.req_str("name")?.to_string();
        let org = row.req_str("org")?.to_string();
        names.push(RepositoryName::new(org, name)?);
    }
    Ok(names)
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

/// What a whole store answers when asked what it could give back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    /// One row per candidate. With no candidate list, every root is a candidate.
    pub roots: Vec<RootFacts>,
    /// Bytes freed by dropping EVERY candidate at once.
    ///
    /// **This is not the sum of the per-root marginals, and that gap is the entire reason
    /// the field exists.** Two models sharing a 2 GB base each exclude it from their own
    /// figure — it is not unique to either — so dropping both frees 2 GB MORE than the sum.
    /// A caller that adds up marginals systematically under-estimates what eviction can
    /// achieve, evicts too little, and lands back on the full filesystem this work exists
    /// to prevent. This is the number a refuse-early check must use.
    pub bytes_total: u64,
    /// Bytes no repository reaches, which a plain `tfs gc` frees without dropping anything.
    pub bytes_unreferenced: u64,
    /// What the Store is holding, and the budget it is held to. **This is deliberately not
    /// a filesystem statistic**: TensorFS stats no filesystem for policy (decisions.md
    /// 241/941), because only the caller can see the image layers, install roots and caches
    /// sharing the disk. It is the one answer to "how full is this STORE", so a caller
    /// consuming it never has a second number that can disagree.
    pub occupancy_bytes: u64,
    pub budget_bytes: u64,
    /// `budget - occupancy`, floored at zero.
    pub headroom_bytes: u64,
}

impl Usage {
    /// `repo usage`-shaped JSONL: one `kind:"root"` row per candidate, then one
    /// `kind:"all"` row for the whole set.
    pub fn json_lines(&self) -> Vec<Vec<u8>> {
        let mut lines: Vec<Vec<u8>> = self
            .roots
            .iter()
            .map(|root| {
                crate::canon::write(&Value::obj(vec![
                    ("bytes_closure", Value::uint(root.bytes_closure)),
                    ("bytes_marginal", Value::uint(root.bytes_solo)),
                    ("kind", Value::str("root")),
                    ("last_recorded_unix", Value::uint(root.last_recorded_unix)),
                    ("name", Value::str(root.name.clone())),
                    ("org", Value::str(root.org.clone())),
                ]))
            })
            .collect();
        lines.push(crate::canon::write(&Value::obj(vec![
            ("budget_bytes", Value::uint(self.budget_bytes)),
            ("bytes_total", Value::uint(self.bytes_total)),
            ("bytes_unreferenced", Value::uint(self.bytes_unreferenced)),
            ("candidates", Value::uint(self.roots.len() as u64)),
            ("headroom_bytes", Value::uint(self.headroom_bytes)),
            ("kind", Value::str("all")),
            ("occupancy_bytes", Value::uint(self.occupancy_bytes)),
        ])));
        lines
    }
}

/// What each candidate would free alone, and what all of them would free together.
///
/// `bytes_marginal` on a root is what dropping THAT ONE frees while everything else stands
/// — the keep set and **every other candidate**. That is the greedy ranking key, and it is
/// deliberately not `Census::usage()`'s `bytes_unique`, which answers the same question one
/// repository at a time and therefore cannot be summed.
///
/// `keep` and `candidates` are CALLER INPUTS, exactly like free space. TensorFS does not
/// know and must never decide which models a pod is serving; it knows what they reach and
/// what they share.
pub fn usage(
    store: &Store,
    keep: &[RepositoryName],
    candidates: &[RepositoryName],
    holds: &[HeldKey],
) -> Result<Usage> {
    permitted(store)?;
    let census = Census::open(store.root())?;
    let index = census.holder_index(holds)?;

    for repo in keep {
        if candidates.contains(repo) {
            return refuse(
                Code::DISPOSITION_CONFLICT,
                format!(
                    "{}/{} is named as both kept and a candidate; one of the two lists is \
                     wrong and guessing which would be a deletion",
                    repo.org, repo.name
                ),
            );
        }
    }

    // No candidate list means every root is one, which is what an operator asking "what
    // could this store give back" means.
    let chosen: Vec<usize> = if candidates.is_empty() {
        (0..index.roots.len())
            .filter(|position| !keep.contains(&index.roots[*position]))
            .collect()
    } else {
        candidates
            .iter()
            .map(|repo| {
                index.position(repo).ok_or_else(|| crate::err::Refusal {
                    code: Code::REPOSITORY_ABSENT,
                    detail: format!("{}/{} is not a root of this store", repo.org, repo.name),
                })
            })
            .collect::<Result<Vec<_>>>()?
    };

    let roots = chosen
        .iter()
        .map(|position| {
            let repo = &index.roots[*position];
            RootFacts {
                org: repo.org.clone(),
                name: repo.name.clone(),
                bytes_closure: index.closure_bytes(*position),
                bytes_solo: index.reclaimable(&BTreeSet::from([*position])),
                last_recorded_unix: recorded_unix(store, repo),
            }
        })
        .collect();

    let all: BTreeSet<usize> = chosen.into_iter().collect();
    let occupancy = store.occupancy()?;
    let budget = store.disk_budget().unwrap_or(0);
    Ok(Usage {
        roots,
        bytes_total: index.reclaimable(&all),
        bytes_unreferenced: index.unreferenced(),
        occupancy_bytes: occupancy,
        budget_bytes: budget,
        headroom_bytes: budget.saturating_sub(occupancy),
    })
}

/// Choose the roots to drop to free `need` bytes, keeping everything in `keep`.
///
/// Refuses `RECLAIM_INSUFFICIENT` when the whole droppable set frees less than `need`,
/// rather than returning a plan that empties the Store and still cannot admit the fetch.
pub fn plan(store: &Store, need: u64, keep: &[RepositoryName], holds: &[HeldKey]) -> Result<Plan> {
    permitted(store)?;
    plan_managed(store, need, keep, holds)
}

// Managed pressure already has an explicit owner opt-in; it shares the same planner.
pub(crate) fn plan_managed(
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

// ------------------------------------------------------------------ the act (tfs-066)

/// What `run` was asked to do beyond the plan itself.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunOptions {
    /// Best-effort cache warming before reclaim. This never authorizes deletion;
    /// assume_upstream is still required because the cache is disposable.
    pub demote: bool,
    /// The caller declares these roots recoverable from the hub. Same class of caller-input
    /// as free space (decisions.md 241): TensorFS cannot see the hub, the caller can.
    pub assume_upstream: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunReport {
    pub roots_dropped: Vec<String>,
    /// Objects published to the cache by THIS run.
    pub demoted_objects: u64,
    pub demoted_bytes: u64,
    /// Objects that were already on the cache, so demotion had nothing to do.
    pub already_cached_objects: u64,
    pub reclaimed_bytes: u64,
    pub reclaimed_objects: u64,
}

impl RunReport {
    pub fn line(&self) -> String {
        format!(
            "dropped {} root{}, reclaimed {} B in {} objects ({} demoted to the cache here, \
             {} already there)",
            self.roots_dropped.len(),
            if self.roots_dropped.len() == 1 {
                ""
            } else {
                "s"
            },
            self.reclaimed_bytes,
            self.reclaimed_objects,
            self.demoted_objects,
            self.already_cached_objects,
        )
    }

    pub fn json(&self) -> Vec<u8> {
        crate::canon::write(&Value::obj(vec![
            (
                "already_cached_objects",
                Value::uint(self.already_cached_objects),
            ),
            ("demoted_bytes", Value::uint(self.demoted_bytes)),
            ("demoted_objects", Value::uint(self.demoted_objects)),
            ("reclaimed_bytes", Value::uint(self.reclaimed_bytes)),
            ("reclaimed_objects", Value::uint(self.reclaimed_objects)),
            (
                "roots_dropped",
                Value::arr(
                    self.roots_dropped
                        .iter()
                        .map(|root| Value::str(root.as_str()))
                        .collect(),
                ),
            ),
        ]))
    }
}

/// Reclaim named roots only after the caller confirms durable upstream custody.
/// Optional cache warming precedes repository tombstones and ordinary GC; cache
/// presence never authorizes either step. Retained roots and active holds still
/// protect their objects through the existing reachability census.
pub fn run(store: &Store, drop: &[RepositoryName], options: RunOptions) -> Result<RunReport> {
    permitted(store)?;
    let mut report = RunReport::default();
    if drop.is_empty() {
        return Ok(report);
    }
    if !options.assume_upstream {
        return refuse(
            Code::DURABILITY_UNPROVEN,
            "named local roots require --assume-upstream after confirming durable upstream \
             custody; an optional cache or --demote cannot preserve the only copy",
        );
    }

    // EXCLUSIVITY IS PROVED BEFORE ANYTHING IS DONE, not discovered at the end.
    //
    // Phase 3 is `gc::collect`, which takes the store-wide recovery lock — and a read lease
    // holds a SHARED lock on that same file for the life of a placement, while a repository
    // mutation does not. So without this probe the order of events on a busy pod was:
    // demote, tombstone, then refuse — leaving the model un-named, the space still
    // occupied, and a retry that finds no root to drop and reports success over an empty
    // plan. A half-evicted store is a worse outcome than a refusal.
    //
    // The lock is taken and released rather than held, because the tombstone in phase 2 is
    // an ordinary writer and cannot run under it. That leaves a window in which a reader
    // may arrive; if it does, phase 3 refuses as before and the roots are already gone, so
    // that refusal says how to finish. The probe removes the case that actually happens —
    // a placement that has been live all along.
    std::mem::drop(crate::catalog::WriterGuard::lock_rebuild(store.root())?);

    // The plan is re-derived here and never read out of the caller's document. A plan is a
    // record of what WAS true; the census is what is true now.
    let census = Census::open(store.root())?;
    let (holds, _) = crate::gc::session_holds(store)?;

    // WHAT THIS RUN IS EVICTING, which is not everything the sweep will take. Objects that
    // are ALREADY unreferenced are what a plain `tfs gc` frees without anybody dropping a
    // model, so they are neither demoted nor asked to prove themselves:
    //
    //  - demoting them would push bytes no model names onto a SHARED volume cache that has
    //    no garbage collector by design, where they would stay forever;
    //  - and asking them for a durable upstream would make one orphan blob refuse every
    //    reclaim run on the store, which is a false refusal — those bytes are going whether
    //    or not this run happens.
    //
    // They are still collected in phase 3, by the same sweep that would have taken them.
    let already: BTreeSet<String> = census
        .reclaim_plan(&[], &holds)?
        .into_iter()
        .map(|row| row.key)
        .collect();
    let condemned: Vec<_> = census
        .reclaim_plan(drop, &holds)?
        .into_iter()
        .filter(|row| !already.contains(&row.key))
        .collect();

    if options.demote {
        if let Some(cache) = store.repo_cache() {
            for row in &condemned {
                let kind = if row.kind == "manifest" {
                    CacheKind::Manifest
                } else {
                    CacheKind::Blob
                };
                let object = ObjectRef {
                    sha256: row.sha256.clone(),
                    length: row.length,
                };
                // Demotion drops the local copy: prove the cached one first.
                match cache.repair_store(store, kind, &object) {
                    Ok(CacheWrite::Stored) => {
                        report.demoted_objects += 1;
                        report.demoted_bytes = report.demoted_bytes.saturating_add(row.length);
                    }
                    Ok(CacheWrite::Present) => report.already_cached_objects += 1,
                    Ok(CacheWrite::Unavailable) | Err(_) => {}
                }
            }
        }
    }

    let fault = crate::store::Fault::default();
    for repo in drop {
        let path = store.repository_path(repo);
        let observed = std::fs::read(&path).ok();
        if observed.is_none() {
            continue;
        }
        store.apply_repository(
            observed.as_deref(),
            &crate::repository::Mutation::DeleteRepository { repo: repo.clone() },
            &fault,
        )?;
        report
            .roots_dropped
            .push(format!("{}/{}", repo.org, repo.name));
    }

    let collected = crate::gc::collect(store.root(), false).map_err(|error| {
        if error.code == Code::STORE_BUSY {
            return crate::err::Refusal {
                code: Code::STORE_BUSY,
                detail: format!(
                    "{error} — a reader arrived after the roots were dropped, so \
                     {} root(s) are gone and their bytes are not yet collected; \
                     `tfs gc` finishes this once the store is quiet",
                    report.roots_dropped.len()
                ),
            };
        }
        error
    })?;
    report.reclaimed_bytes = collected.reclaimed_bytes;
    report.reclaimed_objects = collected.reclaimed_blobs + collected.reclaimed_manifests;
    Ok(report)
}
