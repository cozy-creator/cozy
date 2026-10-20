//! Pure placement policy. CUDA, byte transfers, clocks and waits belong to the mechanism.
//!
//! A decision uses one immutable snapshot. The mechanism applies its actions in order and
//! takes a fresh snapshot after asynchronous work; it must not treat an ask as freed memory.

use std::collections::{BTreeMap, BTreeSet};

pub type PageId = u64;
pub type ModelId = u64;
pub type RequestId = u64;
pub type RowId = u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum Class {
    Free,
    Orphan,
    Speculative,
    Unused,
    Declared,
    Working,
    Pinned,
}

impl Class {
    pub const ALL: [Self; 7] = [
        Self::Free,
        Self::Orphan,
        Self::Speculative,
        Self::Unused,
        Self::Declared,
        Self::Working,
        Self::Pinned,
    ];
    pub const fn as_usize(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tier {
    Disk,
    Ram,
    Gpu,
    Kept,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Weight,
    Cache,
    Slack,
    Context,
    Scratch(RequestId),
}

/// The Runtime records this order; the policy never infers it from past binds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Position {
    pub step: u32,
    pub order: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Step {
    pub request: RequestId,
    pub index: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Use {
    pub request: RequestId,
    pub position: Position,
}

#[derive(Clone, Debug)]
pub struct Page {
    pub id: PageId,
    pub row: RowId,
    pub model: Option<ModelId>,
    pub bytes: u64,
    pub tier: Tier,
    pub kind: Kind,
    pub pins: u32,
    pub filling: bool,
    pub loaded: bool,
    pub last_used: u64,
    /// Increasing spread rank: a lower watermark removes the highest ranks first.
    pub spread_rank: u32,
    pub uses: Vec<Use>,
    pub speculative_for: Option<Step>,
    /// Kept until the corresponding request ends, even if a speculative fill is dropped.
    pub attempted: BTreeSet<Step>,
    pub ram_ready: bool,
    pub lowered: BTreeSet<RequestId>,
}

impl Page {
    pub fn mapped(&self) -> bool {
        matches!(self.tier, Tier::Gpu | Tier::Kept)
    }
}

#[derive(Clone, Debug)]
pub struct Row {
    pub id: RowId,
    pub alive: bool,
    pub reaping: bool,
}

#[derive(Clone, Debug)]
pub struct Request {
    pub id: RequestId,
    pub row: RowId,
    pub position: Position,
    /// Additional working-set room per step, including activations and streaming slots.
    /// Resident current-step pages, contexts and pins are accounted separately by the table.
    pub working_sets: Vec<Option<u64>>,
    pub starved: bool,
    /// Shared mode: footprint captured at step start; exclusive mode: None.
    pub sharing_limit: Option<u64>,
    /// Multi-pass steps whose unpinned pages should stay resident if they fit.
    pub resident_steps: BTreeSet<u32>,
    /// Members of these steps are needed together, with no order between models.
    pub joint_steps: BTreeSet<u32>,
    /// Consumed pages within each member of the current joint step.
    pub model_positions: BTreeMap<ModelId, u32>,
    pub can_cut: bool,
    pub can_lower: bool,
    /// Increment after a completed demand so a later demand can ask rows again.
    pub reclaim_round: u64,
}

impl Request {
    fn pending(&self, page: &Page, usage: &Use) -> bool {
        if usage.request != self.id || usage.position.step < self.position.step {
            return false;
        }
        let order = if self.joint_steps.contains(&self.position.step) {
            page.model
                .and_then(|id| self.model_positions.get(&id))
                .copied()
                .unwrap_or(0)
        } else {
            self.position.order
        };
        usage.position.step > self.position.step || usage.position.order >= order
    }
    pub fn step(&self) -> Step {
        Step {
            request: self.id,
            index: self.position.step,
        }
    }
    pub fn measured(&self) -> Option<u64> {
        usize::try_from(self.position.step)
            .ok()
            .and_then(|i| self.working_sets.get(i))
            .copied()
            .flatten()
    }
}

/// A completed cross-row attempt in this request/step. A new step discards these records.
#[derive(Clone, Debug)]
pub struct Reclaim {
    pub step: Step,
    pub round: u64,
    pub row: RowId,
    pub ceiling: Class,
    pub complete: bool,
}

/// Another process publishes aggregate facts; its page identities stay private.
#[derive(Clone, Debug)]
pub struct Remote {
    pub row: RowId,
    /// These classes exclude the separately reported context.
    pub classes: [u64; 7],
    pub context: u64,
    /// Largest remaining W per request, excluding pinned bytes and context.
    pub remaining: Vec<Option<u64>>,
    pub reaping: bool,
}

impl Remote {
    fn class(&self, class: Class) -> u64 {
        self.classes.get(class.as_usize()).copied().unwrap_or(0)
    }
    fn reclaimable(&self, ceiling: Class) -> u64 {
        Class::ALL
            .into_iter()
            .zip(self.classes)
            .filter(|(class, _)| *class <= ceiling)
            .fold(0_u64, |sum, (_, bytes)| sum.saturating_add(bytes))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Table {
    pub capacity: u64,
    pub pages: Vec<Page>,
    pub rows: Vec<Row>,
    pub requests: Vec<Request>,
    pub reclaims: Vec<Reclaim>,
    pub remote: Vec<Remote>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Trigger {
    Begin(RequestId),
    StepStart(RequestId),
    Bind {
        request: RequestId,
        page: PageId,
    },
    Oom {
        request: RequestId,
        need: u64,
    },
    Ask {
        row: RowId,
        need: u64,
        ceiling: Class,
    },
    /// Admission needs idle cache references released, not any weight bytes evicted.
    ReleaseUnusedCaches {
        row: RowId,
        after: PageId,
        through: PageId,
    },
    Load {
        row: RowId,
        model: ModelId,
        level: Tier,
    },
    FillLanded(RequestId),
    PinReleased(RequestId),
    /// Explicit permission to lower only this request's current-step watermark.
    Lower {
        request: RequestId,
        need: u64,
    },
    End(RequestId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Action {
    Trim(PageId),
    ReleaseKept(PageId),
    Shrink {
        page: PageId,
        bytes: u64,
    },
    Unmap(PageId),
    DropHome {
        request: RequestId,
        page: PageId,
    },
    Map(PageId),
    Fill {
        page: PageId,
        from: Tier,
        to: Tier,
        speculative: Option<Step>,
    },
    Starve(Step),
    Ask {
        row: RowId,
        need: u64,
        ceiling: Class,
    },
    Acknowledge {
        row: RowId,
        freed: u64,
    },
    Cut(RequestId),
    Lower {
        request: RequestId,
        need: u64,
    },
    Exhausted {
        request: RequestId,
        need: u64,
    },
    Forget(RequestId),
    Invalid(&'static str),
}

type Distance = (u32, u32);

impl Table {
    pub fn available(&self) -> u64 {
        self.occupied()
            .map_or(0, |bytes| self.capacity.saturating_sub(bytes))
    }

    pub fn working_set(&self, request: RequestId) -> Option<u64> {
        let request = self.request(request)?;
        let home = self
            .pages
            .iter()
            .filter(|page| {
                self.class(page) == Class::Working
                    && page.uses.iter().any(|usage| {
                        usage.request == request.id && usage.position.step == request.position.step
                    })
            })
            .try_fold(request.measured()?, |bytes, page| {
                bytes.checked_add(page.bytes)
            })?;
        home.checked_add(self.stream_footprint(request.id, request.position.step)?)
    }

    fn stream_footprint(&self, request: RequestId, step: u32) -> Option<u64> {
        let mut groups: BTreeMap<ModelId, (u64, u64, u64)> = BTreeMap::new();
        for page in self.pages.iter().filter(|page| {
            page.lowered.contains(&request)
                && page
                    .uses
                    .iter()
                    .any(|usage| usage.request == request && usage.position.step == step)
        }) {
            let Some(model) = page.model else {
                continue;
            };
            let group = groups.entry(model).or_default();
            group.0 = group.0.max(page.bytes);
            group.1 = group.1.saturating_add(1);
            if page.mapped() && self.class(page) == Class::Pinned {
                group.2 = group.2.checked_add(page.bytes)?;
            }
        }
        groups
            .values()
            .try_fold(0_u64, |sum, (largest, count, pinned)| {
                sum.checked_add(
                    largest
                        .checked_mul((*count).min(3))?
                        .saturating_sub(*pinned),
                )
            })
    }
    pub fn speculative_budget(&self, request: RequestId) -> Option<u64> {
        let request = self.request(request)?;
        if request.starved {
            return Some(0);
        }
        let reclaim = self
            .pages
            .iter()
            .filter(|page| {
                page.mapped()
                    && (self.class(page) == Class::Orphan
                        || (page.row == request.row
                            && matches!(self.class(page), Class::Free | Class::Unused)))
            })
            .try_fold(0_u64, |sum, page| sum.checked_add(page.bytes))?;
        let budget = self
            .capacity
            .saturating_sub(self.occupied()?.saturating_sub(reclaim))
            .saturating_sub(self.protected_room()?);
        let own = self
            .pages
            .iter()
            .filter(|page| {
                page.mapped()
                    && page.row == request.row
                    && !matches!(self.class(page), Class::Free | Class::Unused)
            })
            .try_fold(0_u64, |sum, page| sum.checked_add(page.bytes))?;
        Some(
            request
                .sharing_limit
                .map_or(budget, |limit| budget.min(limit.saturating_sub(own))),
        )
    }

    pub fn request(&self, id: RequestId) -> Option<&Request> {
        self.requests.iter().find(|request| request.id == id)
    }

    pub fn page(&self, id: PageId) -> Option<&Page> {
        self.pages.iter().find(|page| page.id == id)
    }

    pub fn bound(&mut self, request: RequestId, page: PageId) {
        let Some(current) = self.request(request) else {
            return;
        };
        let next = self.page(page).and_then(|page| {
            page.uses
                .iter()
                .filter(|usage| current.pending(page, usage))
                .map(|usage| (page.model, usage.position))
                .min()
        });
        if let (Some((model, next)), Some(row)) =
            (next, self.requests.iter_mut().find(|row| row.id == request))
        {
            if row.position.step != next.step {
                row.model_positions.clear();
            }
            row.position.step = next.step;
            if row.joint_steps.contains(&next.step) {
                if let Some(model) = model {
                    row.model_positions
                        .insert(model, next.order.saturating_add(1));
                }
                row.position.order = 0;
            } else {
                row.position.order = next.order.saturating_add(1);
            }
        }
    }

    fn next_use(&self, page: &Page) -> Option<Distance> {
        page.uses
            .iter()
            .filter_map(|usage| {
                let request = self.request(usage.request)?;
                request.pending(page, usage).then(|| {
                    (
                        usage.position.step.saturating_sub(request.position.step),
                        usage.position.order,
                    )
                })
            })
            .min()
    }

    pub fn class(&self, page: &Page) -> Class {
        if !self.rows.iter().any(|row| row.id == page.row && row.alive) {
            return Class::Orphan;
        }
        if page.pins != 0 || page.filling || matches!(page.kind, Kind::Context | Kind::Scratch(_)) {
            return Class::Pinned;
        }
        if page.kind == Kind::Slack {
            return Class::Free;
        }
        let next = self.next_use(page);
        if self
            .requests
            .iter()
            .any(|request| !page.lowered.contains(&request.id) && current(page, request))
        {
            return Class::Working;
        }
        if page.speculative_for.is_some() {
            return Class::Speculative;
        }
        if next.is_some() {
            Class::Declared
        } else {
            Class::Unused
        }
    }

    /// No authoritative byte counters are stored beside the page table.
    pub fn bytes_by_class(&self, row: RowId) -> Result<[u64; 7], &'static str> {
        self.counts(row, false)
    }

    pub fn published_classes(&self, row: RowId) -> Result<[u64; 7], &'static str> {
        self.counts(row, true)
    }

    fn counts(&self, row: RowId, separate_context: bool) -> Result<[u64; 7], &'static str> {
        let mut counts = [0_u64; 7];
        for page in self.pages.iter().filter(|page| {
            page.row == row && page.mapped() && (!separate_context || page.kind != Kind::Context)
        }) {
            let Some(count) = counts.get_mut(self.class(page).as_usize()) else {
                return Err("invalid page class");
            };
            *count = count
                .checked_add(page.bytes)
                .ok_or("page byte count overflow")?;
        }
        Ok(counts)
    }

    fn occupied(&self) -> Option<u64> {
        let local = self
            .pages
            .iter()
            .filter(|page| page.mapped())
            .try_fold(0_u64, |sum, page| sum.checked_add(page.bytes))?;
        self.remote.iter().try_fold(local, |sum, row| {
            row.classes
                .iter()
                .try_fold(sum.checked_add(row.context)?, |sum, bytes| {
                    sum.checked_add(*bytes)
                })
        })
    }

    fn stream_reserve(&self) -> Option<u64> {
        let mut models: BTreeMap<(RequestId, ModelId), (u64, u64, u64)> = BTreeMap::new();
        for request in &self.requests {
            for page in self
                .pages
                .iter()
                .filter(|page| page.lowered.contains(&request.id) && current(page, request))
            {
                let Some(model) = page.model else {
                    continue;
                };
                let row = models.entry((request.id, model)).or_default();
                row.0 = row.0.max(page.bytes);
                row.1 = row.1.saturating_add(1);
                if page.mapped() {
                    row.2 = row.2.saturating_add(page.bytes);
                }
            }
        }
        models
            .values()
            .try_fold(0_u64, |sum, (largest, count, mapped)| {
                sum.checked_add(
                    largest
                        .checked_mul((*count).min(3))?
                        .saturating_sub(*mapped),
                )
            })
    }

    fn activation_reserve(&self, measured: bool) -> Option<u64> {
        let need = self.requests.iter().try_fold(0_u64, |sum, request| {
            let peak = if measured {
                request.measured()?
            } else {
                request.measured().unwrap_or(0)
            };
            sum.checked_add(peak)
        })?;
        let scratch = self
            .pages
            .iter()
            .filter(|page| page.mapped() && matches!(page.kind, Kind::Scratch(_)))
            .try_fold(0_u64, |sum, page| sum.checked_add(page.bytes))?;
        Some(need.saturating_sub(scratch))
    }

    fn reserve_known(&self, measured: bool) -> Option<u64> {
        self.activation_reserve(measured)?
            .checked_add(self.stream_reserve()?)
    }

    fn reserve(&self) -> Option<u64> {
        self.reserve_known(true)
    }

    fn protected_room(&self) -> Option<u64> {
        let local = self
            .pages
            .iter()
            .filter(|page| !page.mapped() && self.class(page) == Class::Working)
            .try_fold(self.reserve()?, |bytes, page| bytes.checked_add(page.bytes))?;
        self.remote.iter().try_fold(local, |sum, row| {
            let reserve = row
                .remaining
                .iter()
                .try_fold(0_u64, |sum, bytes| sum.checked_add((*bytes)?))?;
            sum.checked_add(reserve.saturating_sub(row.class(Class::Working)))
        })
    }

    fn valid(&self) -> bool {
        let pages: BTreeSet<_> = self.pages.iter().map(|page| page.id).collect();
        let rows: BTreeSet<_> = self.rows.iter().map(|row| row.id).collect();
        let requests: BTreeSet<_> = self.requests.iter().map(|request| request.id).collect();
        let remote: BTreeSet<_> = self.remote.iter().map(|row| row.row).collect();
        pages.len() == self.pages.len()
            && rows.len() == self.rows.len()
            && requests.len() == self.requests.len()
            && self.occupied().is_some()
            && remote.len() == self.remote.len()
            && remote.is_subset(&rows)
            && self.pages.iter().all(|page| !remote.contains(&page.row))
            && self.pages.iter().all(|page| rows.contains(&page.row))
            && self
                .requests
                .iter()
                .all(|request| rows.contains(&request.row))
    }

    /// Largest remaining declared working set, excluding separately charged pins/context.
    pub fn remaining(&self, id: RequestId) -> Option<u64> {
        let request = self.request(id)?;
        let start = usize::try_from(request.position.step).ok()?;
        let remaining = request.working_sets.get(start..)?;
        if remaining.is_empty() {
            return None;
        }
        remaining
            .iter()
            .enumerate()
            .try_fold(0_u64, |largest, (offset, extra)| {
                let step = u32::try_from(start.checked_add(offset)?).ok()?;
                let weights = self
                    .pages
                    .iter()
                    .filter(|page| {
                        page.kind == Kind::Weight
                            && !page.lowered.contains(&id)
                            && self.class(page) != Class::Pinned
                            && page.uses.iter().any(|usage| {
                                usage.request == id
                                    && usage.position.step == step
                                    && request.pending(page, usage)
                            })
                    })
                    .try_fold(0_u64, |sum, page| sum.checked_add(page.bytes))?;
                Some(
                    largest.max(
                        weights
                            .checked_add((*extra)?)?
                            .checked_add(self.stream_footprint(id, step)?)?,
                    ),
                )
            })
    }

    /// Admission is conservative: an unmeasured request runs alone, never beside a reaping row.
    /// `peaks` describes the prospective request's total per-step footprint, excluding contexts.
    pub fn fits(&self, peaks: &[Option<u64>]) -> bool {
        if !self.valid()
            || peaks.is_empty()
            || self.rows.iter().any(|row| row.reaping)
            || self.remote.iter().any(|row| row.reaping)
        {
            return false;
        }
        let unknown = peaks.iter().any(Option::is_none);
        if unknown {
            return self.requests.is_empty()
                && self.remote.iter().all(|row| row.remaining.is_empty());
        }
        let mut needed = self
            .pages
            .iter()
            .filter(|page| {
                page.mapped() && (page.kind == Kind::Context || self.class(page) == Class::Pinned)
            })
            .try_fold(0_u64, |sum, page| sum.checked_add(page.bytes));
        for row in &self.remote {
            needed = needed
                .and_then(|sum| sum.checked_add(row.context))
                .and_then(|sum| sum.checked_add(row.class(Class::Pinned)));
            for remaining in &row.remaining {
                needed = needed.and_then(|sum| sum.checked_add((*remaining)?));
            }
        }
        for request in &self.requests {
            needed = needed.and_then(|sum| sum.checked_add(self.remaining(request.id)?));
        }
        needed
            .and_then(|sum| sum.checked_add(peaks.iter().flatten().copied().max().unwrap_or(0)))
            .is_some_and(|bytes| bytes <= self.capacity)
    }
}

/// Decide from facts only. Applying actions, including keeping them atomic with the snapshot,
/// is the mechanism's responsibility. Ask/Cut/Lower require a new snapshot before retrying.
pub fn decide(table: &Table, trigger: Trigger) -> Vec<Action> {
    if !table.valid() {
        return vec![Action::Invalid("invalid or overflowing page table")];
    }
    match trigger {
        Trigger::End(request) => vec![Action::Forget(request)],
        Trigger::Ask { row, need, ceiling } => reclaim(table, row, need, ceiling),
        Trigger::ReleaseUnusedCaches {
            row,
            after,
            through,
        } => {
            if table.requests.iter().any(|request| request.row == row) {
                return vec![];
            }
            table
                .pages
                .iter()
                .filter(|page| {
                    page.row == row
                        && page.id > after
                        && page.id <= through
                        && page.mapped()
                        && page.kind == Kind::Cache
                        && table.class(page) == Class::Unused
                })
                .min_by_key(|page| page.id)
                .map(|page| {
                    vec![Action::Shrink {
                        page: page.id,
                        bytes: page.bytes,
                    }]
                })
                .unwrap_or_default()
        }
        Trigger::Load { row, model, level } => load(table, row, model, level),
        Trigger::Lower { request, need } => lower(table, request, need),
        Trigger::Oom { request, need } => demand(table, request, need, None),
        Trigger::Bind { request, page } => {
            let Some(item) = table.page(page) else {
                return vec![Action::Invalid("unknown page")];
            };
            let Some(owner) = table.request(request) else {
                return vec![Action::Invalid("unknown request")];
            };
            if owner.row != item.row {
                return vec![Action::Invalid("page belongs to another row")];
            }
            if item.mapped() || item.filling {
                return Vec::new();
            }
            demand(table, request, item.bytes, Some(page))
        }
        Trigger::Begin(request) | Trigger::StepStart(request) => step_start(table, request),
        Trigger::FillLanded(request) | Trigger::PinReleased(request) => {
            let stream = stream(table, request);
            if stream.is_empty() {
                speculate(table, request)
            } else {
                stream
            }
        }
    }
}

#[derive(Clone)]
struct Planner<'a> {
    table: &'a Table,
    actions: Vec<Action>,
    removed: BTreeSet<PageId>,
    free: u64,
    allocated: u64,
}

impl<'a> Planner<'a> {
    fn new(table: &'a Table) -> Self {
        let allocated = table.occupied().unwrap_or(u64::MAX);
        Self {
            table,
            actions: Vec::new(),
            removed: BTreeSet::new(),
            allocated,
            free: table.capacity.saturating_sub(allocated),
        }
    }

    fn shortfall(&self, need: u64) -> u64 {
        self.allocated
            .saturating_sub(self.table.capacity)
            .saturating_add(need.saturating_sub(self.free))
    }

    fn evict(&mut self, page: &Page) {
        if !self.removed.insert(page.id) {
            return;
        }
        let action = match self.table.class(page) {
            Class::Free => Action::Trim(page.id),
            Class::Orphan => Action::ReleaseKept(page.id),
            _ if page.kind == Kind::Cache => Action::Shrink {
                page: page.id,
                bytes: page.bytes,
            },
            _ => Action::Unmap(page.id),
        };
        self.actions.push(action);
        if let Some(step) = page
            .speculative_for
            .and_then(|step| self.table.request(step.request))
            .map(Request::step)
        {
            if !self.actions.contains(&Action::Starve(step)) {
                self.actions.push(Action::Starve(step));
            }
        }
        self.allocated = self.allocated.saturating_sub(page.bytes);
        self.free = self.table.capacity.saturating_sub(self.allocated);
    }

    fn candidates(&self, accept: impl Fn(&Page, Class) -> bool) -> Vec<&'a Page> {
        let mut pages: Vec<_> = self
            .table
            .pages
            .iter()
            .filter(|page| {
                page.mapped()
                    && !self.removed.contains(&page.id)
                    && accept(page, self.table.class(page))
            })
            .collect();
        pages.sort_by_key(|page| {
            let class = self.table.class(page);
            let category = if page.kind == Kind::Cache {
                1
            } else if page.loaded {
                2
            } else {
                0
            };
            (
                class,
                if class == Class::Unused { category } else { 0 },
                if class == Class::Unused {
                    None
                } else {
                    Some(std::cmp::Reverse(self.table.next_use(page)))
                },
                if class == Class::Unused {
                    page.last_used
                } else {
                    0
                },
                page.id,
            )
        });
        pages
    }

    fn take(&mut self, need: u64, accept: impl Fn(&Page, Class) -> bool) {
        for page in self.candidates(accept) {
            if self.free >= need {
                break;
            }
            self.evict(page);
        }
    }

    fn ask(&mut self, request: &Request, need: u64, ceiling: Class) -> bool {
        let mut rows: BTreeSet<_> = self
            .table
            .pages
            .iter()
            .filter(|page| {
                page.row != request.row
                    && page.mapped()
                    && self.table.class(page) <= ceiling
                    && !self.removed.contains(&page.id)
            })
            .map(|page| page.row)
            .collect();
        rows.extend(
            self.table
                .remote
                .iter()
                .filter(|row| row.reclaimable(ceiling) != 0)
                .map(|row| row.row),
        );
        for row in rows {
            let attempt = self.table.reclaims.iter().find(|attempt| {
                attempt.step == request.step()
                    && attempt.round == request.reclaim_round
                    && attempt.row == row
                    && attempt.ceiling >= ceiling
            });
            match attempt {
                Some(attempt) if attempt.complete => continue,
                Some(_) => return true,
                None => {
                    self.actions.push(Action::Ask {
                        row,
                        need: self.shortfall(need),
                        ceiling,
                    });
                    return true;
                }
            }
        }
        false
    }

    fn room(&mut self, request: &Request, need: u64) -> bool {
        self.take(need, |page, class| {
            page.row == request.row && class == Class::Free
        });
        self.take(need, |_, class| class == Class::Orphan);
        if self.free < need && self.ask(request, need, Class::Orphan) {
            return false;
        }
        self.take(need, |page, class| {
            page.row == request.row && class <= Class::Unused
        });
        if self
            .removed
            .iter()
            .filter_map(|id| self.table.page(*id))
            .any(|page| self.table.class(page) == Class::Speculative)
            && !self.actions.contains(&Action::Starve(request.step()))
        {
            self.actions.push(Action::Starve(request.step()));
        }
        if self.free >= need {
            return true;
        }
        if self.ask(request, need, Class::Unused) {
            return false;
        }
        self.take(need, |page, class| {
            page.row == request.row && class == Class::Declared
        });
        if self.free >= need {
            return true;
        }
        if self.ask(request, need, Class::Declared) {
            return false;
        }
        self.actions.push(if request.can_cut {
            Action::Cut(request.id)
        } else if request.can_lower {
            Action::Lower {
                request: request.id,
                need: self.shortfall(need),
            }
        } else {
            Action::Exhausted {
                request: request.id,
                need: self.shortfall(need),
            }
        });
        false
    }

    fn fill(&mut self, page: &Page, speculative: Option<Step>) {
        self.actions.push(Action::Map(page.id));
        self.actions.push(Action::Fill {
            page: page.id,
            from: if page.ram_ready {
                Tier::Ram
            } else {
                Tier::Disk
            },
            to: Tier::Gpu,
            speculative,
        });
        self.allocated = self.allocated.saturating_add(page.bytes);
        self.free = self.table.capacity.saturating_sub(self.allocated);
    }
}

fn demand(table: &Table, id: RequestId, need: u64, page: Option<PageId>) -> Vec<Action> {
    let Some(request) = table.request(id) else {
        return vec![Action::Invalid("unknown request")];
    };
    let mut plan = Planner::new(table);
    if let Some(target) = page.and_then(|id| table.page(id)) {
        if !stream_slot(&mut plan, request, target, true) {
            return vec![Action::Exhausted { request: id, need }];
        }
    }
    let reserve = table.reserve_known(false).unwrap_or(0);
    let need = if page
        .and_then(|id| table.page(id))
        .is_some_and(|page| page.lowered.contains(&id))
    {
        need.max(reserve)
    } else {
        let Some(total) = need.checked_add(reserve) else {
            return vec![Action::Invalid("working set overflow")];
        };
        total
    };
    if plan.room(request, need) {
        if let Some(page) = page.and_then(|id| table.page(id)) {
            plan.fill(page, None);
        }
    }
    plan.actions
}

fn reclaim(table: &Table, row: RowId, need: u64, ceiling: Class) -> Vec<Action> {
    let mut plan = Planner::new(table);
    let before = plan.allocated;
    for page in
        plan.candidates(|page, class| page.row == row && class <= ceiling.min(Class::Declared))
    {
        if before.saturating_sub(plan.allocated) >= need {
            break;
        }
        plan.evict(page);
    }
    plan.actions.push(Action::Acknowledge {
        row,
        freed: before.saturating_sub(plan.allocated),
    });
    plan.actions
}

fn current(page: &Page, request: &Request) -> bool {
    page.uses.iter().any(|usage| {
        usage.request == request.id
            && usage.position.step == request.position.step
            && request.pending(page, usage)
    })
}

fn lower(table: &Table, id: RequestId, need: u64) -> Vec<Action> {
    let Some(request) = table.request(id) else {
        return vec![Action::Invalid("unknown request")];
    };
    let mut pages: Vec<_> = table
        .pages
        .iter()
        .filter(|page| {
            page.row == request.row
                && table.class(page) == Class::Working
                && current(page, request)
                && !table
                    .requests
                    .iter()
                    .any(|other| other.id != id && current(page, other))
        })
        .collect();
    pages.sort_by_key(|page| (std::cmp::Reverse(page.spread_rank), page.id));
    let mut home = pages
        .iter()
        .map(|page| page.bytes)
        .fold(0_u64, u64::saturating_add);
    let mapped = pages
        .iter()
        .filter(|page| page.mapped())
        .map(|page| page.bytes)
        .fold(0_u64, u64::saturating_add);
    let fixed = table.occupied().unwrap_or(u64::MAX).saturating_sub(mapped);
    let mut largest: BTreeMap<ModelId, (u64, u64, u64)> = BTreeMap::new();
    for page in table
        .pages
        .iter()
        .filter(|page| page.row == request.row && current(page, request))
    {
        if let Some(model) = page.model {
            let row = largest.entry(model).or_default();
            row.0 = row.0.max(page.bytes);
            row.1 = row.1.saturating_add(1);
            if page.mapped() && page.lowered.contains(&id) {
                row.2 = row.2.saturating_add(page.bytes);
            }
        }
    }
    let ring = largest
        .values()
        .map(|(max, count, mapped)| max.saturating_mul((*count).min(3)).saturating_sub(*mapped))
        .fold(0_u64, u64::saturating_add);
    let budget = table
        .capacity
        .saturating_sub(fixed)
        .saturating_sub(table.activation_reserve(false).unwrap_or(0))
        .saturating_sub(ring)
        .saturating_sub(need);
    let mut plan = Planner::new(table);
    let before = plan.allocated;
    for page in pages {
        if home <= budget && before.saturating_sub(plan.allocated) >= need {
            break;
        }
        plan.actions.push(Action::DropHome {
            request: id,
            page: page.id,
        });
        home = home.saturating_sub(page.bytes);
        if page.mapped() {
            plan.evict(page);
        }
    }
    if before.saturating_sub(plan.allocated) < need {
        plan.actions.push(Action::Exhausted {
            request: id,
            need: need.saturating_sub(before.saturating_sub(plan.allocated)),
        });
    }
    plan.actions
}

fn stream(table: &Table, id: RequestId) -> Vec<Action> {
    let Some(request) = table.request(id) else {
        return Vec::new();
    };
    if request.measured().is_none() || request.joint_steps.contains(&request.position.step) {
        return Vec::new();
    }
    let mut next: Vec<_> = table
        .pages
        .iter()
        .filter(|page| {
            page.lowered.contains(&id) && current(page, request) && !page.mapped() && !page.filling
        })
        .collect();
    next.sort_by_key(|page| (table.next_use(page), page.id));
    let Some(page) = next.first().copied() else {
        return Vec::new();
    };
    let mut plan = Planner::new(table);
    if !stream_slot(&mut plan, request, page, false) {
        return Vec::new();
    }
    let reserve = table.protected_room().unwrap_or(u64::MAX).max(page.bytes);
    plan.take(reserve, |candidate, class| {
        class <= Class::Orphan || (candidate.row == request.row && class <= Class::Unused)
    });
    if plan.free >= reserve {
        plan.fill(page, None);
        plan.actions
    } else {
        Vec::new()
    }
}

fn stream_slot(plan: &mut Planner<'_>, request: &Request, target: &Page, demand: bool) -> bool {
    if !target.lowered.contains(&request.id) {
        return true;
    }
    let mut mapped = plan
        .table
        .pages
        .iter()
        .filter(|page| {
            page.mapped() && page.model == target.model && page.lowered.contains(&request.id)
        })
        .count();
    for page in plan.candidates(|page, class| {
        page.model == target.model
            && page.lowered.contains(&request.id)
            && class < Class::Working
            && (demand || class <= Class::Unused)
            && page.id != target.id
    }) {
        if mapped < 3 {
            break;
        }
        plan.evict(page);
        mapped = mapped.saturating_sub(1);
    }
    mapped < 3
}

fn step_start(table: &Table, id: RequestId) -> Vec<Action> {
    let Some(request) = table.request(id) else {
        return vec![Action::Invalid("unknown request")];
    };
    if request.measured().is_none() {
        return Vec::new();
    }
    let mut plan = Planner::new(table);
    let reserve = table.reserve().unwrap_or(0);
    if !plan.room(request, reserve) {
        return plan.actions;
    }
    if request.resident_steps.contains(&request.position.step) {
        let mut pages: Vec<_> = table
            .pages
            .iter()
            .filter(|page| {
                page.row == request.row
                    && !page.mapped()
                    && !page.filling
                    && table.class(page) == Class::Working
                    && page.uses.iter().any(|usage| {
                        usage.request == id
                            && usage.position.step == request.position.step
                            && request.pending(page, usage)
                    })
            })
            .collect();
        pages.sort_by_key(|page| (table.next_use(page), page.id));
        for page in pages {
            let Some(need) = reserve.checked_add(page.bytes) else {
                return vec![Action::Invalid("working set overflow")];
            };
            if !plan.room(request, need) {
                return plan.actions;
            }
            plan.fill(page, None);
        }
    }
    // A separate snapshot after these fills establishes measured, actually free surplus.
    if plan.actions.is_empty() {
        speculate(table, id)
    } else {
        plan.actions
    }
}

fn load(table: &Table, row: RowId, model: ModelId, level: Tier) -> Vec<Action> {
    if !table.rows.iter().any(|item| item.id == row && item.alive) {
        return vec![Action::Invalid("unknown live row")];
    }
    let mut plan = Planner::new(table);
    let reserve = if level == Tier::Gpu {
        let Some(reserve) = table.protected_room() else {
            return Vec::new();
        };
        reserve
    } else {
        0
    };
    let mut pages: Vec<_> = table
        .pages
        .iter()
        .filter(|page| page.row == row && page.model == Some(model))
        .collect();
    pages.sort_by_key(|page| page.id);
    for page in pages {
        if level == Tier::Ram {
            if !page.ram_ready && !page.filling {
                plan.actions.push(Action::Fill {
                    page: page.id,
                    from: Tier::Disk,
                    to: Tier::Ram,
                    speculative: None,
                });
            }
        } else if level == Tier::Gpu && !page.mapped() && !page.filling {
            let Some(need) = page.bytes.checked_add(reserve) else {
                break;
            };
            let mut trial = plan.clone();
            trial.take(need, |victim, class| {
                class == Class::Orphan
                    || (victim.row == row && class == Class::Free)
                    || (victim.row == row
                        && class == Class::Unused
                        && !victim.loaded
                        && victim.model != Some(model))
            });
            if trial.free < need {
                break;
            }
            plan = trial;
            plan.fill(page, None);
        }
    }
    plan.actions
}

fn speculate(table: &Table, id: RequestId) -> Vec<Action> {
    let Some(request) = table.request(id) else {
        return vec![Action::Invalid("unknown request")];
    };
    if request.starved {
        return Vec::new();
    }
    let Some(reserve) = table.protected_room() else {
        return Vec::new();
    };
    let mut candidates: Vec<_> = table
        .pages
        .iter()
        .filter_map(|page| {
            if page.row != request.row
                || page.mapped()
                || page.filling
                || page.kind != Kind::Weight
                || page.attempted.contains(&request.step())
                || page.lowered.contains(&id)
                || table.class(page) == Class::Working
            {
                return None;
            }
            let usage = page
                .uses
                .iter()
                .filter(|usage| usage.request == id && usage.position.step > request.position.step)
                .map(|usage| usage.position)
                .min()?;
            Some((usage, page))
        })
        .collect();
    candidates.sort_by_key(|(usage, page)| (*usage, page.id));
    let mut plan = Planner::new(table);
    let mut own_bytes = table
        .pages
        .iter()
        .filter(|page| page.row == request.row && page.mapped())
        .fold(0_u64, |sum, page| sum.saturating_add(page.bytes));
    for (_, page) in candidates {
        let Some(need) = reserve.checked_add(page.bytes) else {
            break;
        };
        let mut trial = plan.clone();
        trial.take(need, |victim, class| {
            class == Class::Orphan
                || (victim.row == request.row && matches!(class, Class::Free | Class::Unused))
        });
        let removed = table
            .pages
            .iter()
            .filter(|item| item.row == request.row && trial.removed.contains(&item.id))
            .fold(0_u64, |sum, item| sum.saturating_add(item.bytes));
        if trial.free < need
            || request.sharing_limit.is_some_and(|limit| {
                own_bytes.saturating_sub(removed).saturating_add(page.bytes) > limit
            })
        {
            break;
        }
        plan = trial;
        plan.fill(page, Some(request.step()));
        own_bytes = own_bytes.saturating_add(page.bytes);
    }
    plan.actions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(id: RequestId) -> Request {
        Request {
            id,
            row: 1,
            position: Position { step: 0, order: 0 },
            working_sets: vec![Some(0), Some(0)],
            starved: false,
            sharing_limit: None,
            resident_steps: BTreeSet::new(),
            joint_steps: BTreeSet::new(),
            model_positions: BTreeMap::new(),
            can_cut: false,
            can_lower: false,
            reclaim_round: 0,
        }
    }

    fn page(id: PageId, bytes: u64) -> Page {
        Page {
            id,
            row: 1,
            model: Some(1),
            bytes,
            tier: Tier::Gpu,
            kind: Kind::Weight,
            pins: 0,
            filling: false,
            loaded: false,
            last_used: id,
            spread_rank: id as u32,
            uses: Vec::new(),
            speculative_for: None,
            attempted: BTreeSet::new(),
            ram_ready: false,
            lowered: BTreeSet::new(),
        }
    }

    fn usage(request: RequestId, step: u32) -> Use {
        Use {
            request,
            position: Position { step, order: 0 },
        }
    }

    #[test]
    fn admission_releases_only_idle_unused_caches_even_when_the_cache_has_no_bytes() {
        let mut pinned = page(1, 600);
        pinned.pins = 1;
        let mut cache = page(2, 0);
        cache.kind = Kind::Cache;
        cache.model = None;
        let ordinary = page(3, 40);
        let mut active_cache = cache.clone();
        active_cache.id = 4;
        active_cache.pins = 1;
        let mut table = Table {
            capacity: 1024,
            pages: vec![pinned, cache, ordinary, active_cache],
            rows: vec![Row {
                id: 1,
                alive: true,
                reaping: false,
            }],
            ..Default::default()
        };
        assert_eq!(
            decide(
                &table,
                Trigger::ReleaseUnusedCaches {
                    row: 1,
                    after: 0,
                    through: 4
                }
            ),
            vec![Action::Shrink { page: 2, bytes: 0 }]
        );
        // Even a callback that leaves its zero-byte cache present is attempted
        // once. A cache created by that callback lies beyond the initial bound.
        let mut new_cache = page(5, 0);
        new_cache.kind = Kind::Cache;
        new_cache.model = None;
        table.pages.push(new_cache);
        assert!(
            decide(
                &table,
                Trigger::ReleaseUnusedCaches {
                    row: 1,
                    after: 2,
                    through: 4
                }
            )
            .is_empty(),
            "processed and newly created caches must not extend the release"
        );
        table.pages.retain(|page| page.id != 2);
        assert!(
            decide(
                &table,
                Trigger::ReleaseUnusedCaches {
                    row: 1,
                    after: 2,
                    through: 4
                }
            )
            .is_empty(),
            "ordinary warm weights must remain mapped"
        );
        table.requests.push(request(7));
        assert!(
            decide(
                &table,
                Trigger::ReleaseUnusedCaches {
                    row: 1,
                    after: 0,
                    through: 4
                }
            )
            .is_empty(),
            "a newly admitted request prevents the idle-only release"
        );
    }

    /// A small table oracle, not a simulated GPU or workload benchmark.
    #[test]
    fn placement_invariants() {
        let mut orphan = page(1, 20);
        orphan.row = 9;
        orphan.tier = Tier::Kept;
        let mut speculative = page(2, 20);
        speculative.speculative_for = Some(Step {
            request: 1,
            index: 0,
        });
        speculative.uses.push(usage(1, 1));
        let unused = page(3, 20);
        let mut working = page(4, 40);
        working.uses.push(usage(1, 0));
        let mut table = Table {
            capacity: 100,
            pages: vec![orphan, speculative, unused, working],
            rows: vec![
                Row {
                    id: 1,
                    alive: true,
                    reaping: false,
                },
                Row {
                    id: 9,
                    alive: false,
                    reaping: false,
                },
            ],
            requests: vec![request(1)],
            reclaims: Vec::new(),
            remote: Vec::new(),
        };
        assert_eq!(
            decide(
                &table,
                Trigger::Oom {
                    request: 1,
                    need: 25
                }
            ),
            vec![
                Action::ReleaseKept(1),
                Action::Unmap(2),
                Action::Starve(Step {
                    request: 1,
                    index: 0
                })
            ]
        );
        assert_eq!(table.bytes_by_class(1), Ok([0, 0, 20, 20, 0, 40, 0]));

        // An explicit lower cannot take a shared current-step page or a live pin.
        table.requests.push(request(2));
        if let Some(page) = table.pages.iter_mut().find(|page| page.id == 4) {
            page.uses.push(usage(2, 0));
        }
        assert_eq!(
            decide(
                &table,
                Trigger::Lower {
                    request: 1,
                    need: 10
                }
            ),
            vec![Action::Exhausted {
                request: 1,
                need: 10
            }]
        );

        // A cap can be below the existing allocation: reclaim pays that debt first.
        table.requests.retain(|request| request.id == 1);
        table.capacity = 70;
        assert_eq!(
            decide(
                &table,
                Trigger::Oom {
                    request: 1,
                    need: 5
                }
            ),
            vec![
                Action::ReleaseKept(1),
                Action::Unmap(2),
                Action::Starve(Step {
                    request: 1,
                    index: 0
                })
            ]
        );

        // Speculation cannot use declared pages to create room, nor run unmeasured.
        table.capacity = 100;
        table
            .pages
            .retain(|page| page.id != 1 && page.id != 2 && page.id != 3);
        let mut declared = page(5, 40);
        declared.uses.push(usage(1, 1));
        let mut future = page(6, 30);
        future.tier = Tier::Disk;
        future.uses.push(usage(1, 1));
        table.pages.extend([declared, future]);
        assert_eq!(decide(&table, Trigger::PinReleased(1)), Vec::new());
        if let Some(request) = table.requests.first_mut() {
            request.working_sets = vec![None];
        }
        assert_eq!(decide(&table, Trigger::StepStart(1)), Vec::new());
        assert_eq!(decide(&table, Trigger::PinReleased(1)), Vec::new());
        if let Some(request) = table.requests.first_mut() {
            request.working_sets = vec![Some(0), Some(0)];
        }
        table.capacity = 110;
        if let Some(page) = table.pages.iter_mut().find(|page| page.id == 6) {
            page.attempted.insert(Step {
                request: 1,
                index: 0,
            });
        }
        assert_eq!(decide(&table, Trigger::PinReleased(1)), Vec::new());

        // A lowered home target persists; step start must not restore the page.
        let mut lowered = Table {
            capacity: 100,
            pages: vec![page(7, 40)],
            rows: vec![Row {
                id: 1,
                alive: true,
                reaping: false,
            }],
            requests: vec![request(1)],
            ..Default::default()
        };
        if let Some(page) = lowered.pages.first_mut() {
            page.uses.push(usage(1, 0));
        }
        assert_eq!(
            decide(
                &lowered,
                Trigger::Lower {
                    request: 1,
                    need: 10
                }
            ),
            vec![
                Action::DropHome {
                    request: 1,
                    page: 7
                },
                Action::Unmap(7)
            ]
        );
        if let Some(page) = lowered.pages.first_mut() {
            page.lowered.insert(1);
            page.tier = Tier::Disk;
        }
        if let Some(request) = lowered.requests.first_mut() {
            request.resident_steps.insert(0);
        }
        assert_eq!(decide(&lowered, Trigger::StepStart(1)), Vec::new());
        assert_eq!(
            lowered.pages.first().map(|page| lowered.class(page)),
            Some(Class::Declared)
        );

        // An ask must complete before the caller evicts its own declared future.
        if let Some(page) = table.pages.iter_mut().find(|page| page.id == 6) {
            page.tier = Tier::Gpu;
            page.row = 2;
            page.uses.clear();
        }
        table.rows.push(Row {
            id: 2,
            alive: true,
            reaping: false,
        });
        assert_eq!(
            decide(
                &table,
                Trigger::Oom {
                    request: 1,
                    need: 10
                }
            ),
            vec![Action::Ask {
                row: 2,
                need: 10,
                ceiling: Class::Unused
            }]
        );
        table.pages.retain(|page| page.row != 2);
        table.remote.push(Remote {
            row: 2,
            classes: [0, 0, 0, 30, 0, 0, 0],
            context: 0,
            remaining: vec![],
            reaping: false,
        });
        assert_eq!(
            decide(
                &table,
                Trigger::Oom {
                    request: 1,
                    need: 10
                }
            ),
            vec![Action::Ask {
                row: 2,
                need: 10,
                ceiling: Class::Unused
            }]
        );
        table.reclaims.push(Reclaim {
            step: Step {
                request: 1,
                index: 0,
            },
            round: 0,
            row: 2,
            ceiling: Class::Unused,
            complete: false,
        });
        assert_eq!(
            decide(
                &table,
                Trigger::Oom {
                    request: 1,
                    need: 10
                }
            ),
            Vec::new()
        );
        let mut joint = request(1);
        joint.joint_steps.insert(0);
        let mut a = page(1, 10);
        a.uses.push(usage(1, 0));
        let mut b = page(2, 10);
        b.model = Some(2);
        b.uses.push(Use {
            request: 1,
            position: Position { step: 0, order: 0 },
        });
        let mut group = Table {
            capacity: 100,
            pages: vec![a, b],
            requests: vec![joint],
            rows: vec![Row {
                id: 1,
                alive: true,
                reaping: false,
            }],
            ..Default::default()
        };
        // Joint members may bind B before A; neither model is an earlier ordered step.
        group.bound(1, 2);
        assert_eq!(group.class(group.page(1).expect("A")), Class::Working);
        assert_eq!(group.class(group.page(2).expect("B")), Class::Unused);
        group.bound(1, 1);
        assert!(group
            .pages
            .iter()
            .all(|page| group.class(page) == Class::Unused));

        let mut ring = Table {
            capacity: 100,
            requests: vec![request(1)],
            rows: vec![Row {
                id: 1,
                alive: true,
                reaping: false,
            }],
            ..Default::default()
        };
        for id in 1..=4 {
            let mut item = page(id, 10);
            item.lowered.insert(1);
            item.tier = if id < 4 { Tier::Gpu } else { Tier::Disk };
            item.uses.push(Use {
                request: 1,
                position: Position {
                    step: 0,
                    order: id as u32 - 1,
                },
            });
            ring.pages.push(item);
        }
        // A full ring may not exchange one unconsumed future page for another.
        assert!(decide(&ring, Trigger::PinReleased(1)).is_empty());
        ring.bound(1, 1);
        let refill = decide(&ring, Trigger::PinReleased(1));
        assert!(refill.contains(&Action::Unmap(1)));
        assert!(refill
            .iter()
            .any(|action| matches!(action, Action::Fill { page: 4, .. })));
        assert!(!refill.contains(&Action::Unmap(2)) && !refill.contains(&Action::Unmap(3)));
        ring.requests[0].joint_steps.insert(0);
        assert!(
            decide(&ring, Trigger::PinReleased(1)).is_empty(),
            "joint groups refill only on actual demand"
        );
    }
}
