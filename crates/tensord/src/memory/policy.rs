//! Per-GPU ledger and room decisions. Pure: callers bring NVML samples and executor facts, carry
//! out the step returned, sample again and ask again. Only idle tenants give room; a running
//! call's cap is lowered only to keep the driver's floor (`below_floor`), and the executor reads
//! the driver for everything else.
use std::collections::{BTreeMap, BTreeSet};

pub const MIB: u64 = 1 << 20;
/// Allocator fragmentation beside a want.
pub const MARGIN: u64 = 64 * MIB;
/// Free bytes kept on a GPU that drives a display.
pub const DISPLAY_FLOOR: u64 = 512 * MIB;
/// Free bytes kept on any other GPU.
pub const HEADLESS_FLOOR: u64 = 256 * MIB;
/// A new executor's context before any executor on this GPU measured one.
pub const UNMEASURED_CONTEXT: u64 = 1 << 30;

/// One NVML reading of a GPU.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sample {
    pub total: u64,
    pub free: u64,
    pub display: bool,
    /// Device bytes NVML charges each process; None where it cannot attribute them.
    pub processes: Option<BTreeMap<u32, u64>>,
}

/// What an executor reported of itself in this machine run. None is unknown, never zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Facts {
    pub context: Option<u64>,
    /// Context + torch reserved + the plane's own maps.
    pub process: Option<u64>,
    /// Weights as stages count them (decoded copies included), and their lowest rung.
    pub weights: Option<u64>,
    pub weights_floor: Option<u64>,
    /// The weight bytes a call kept mapped with nothing evicted: what its entrypoint's stages
    /// use (H3 registers two denoisers, 101.8 GB, and a call maps 79.9 GB).
    pub mapped: Option<u64>,
    /// The largest activation growth a call measured.
    pub activation: Option<u64>,
    /// Its private host bytes (PSS less shared memory) after a call or an import.
    pub host: Option<u64>,
    /// The budget that bounds its pinned host tier.
    pub pinned_budget: Option<u64>,
}
impl Facts {
    /// Newer facts replace older ones field by field; an absent field keeps what was known.
    /// Peaks only grow.
    pub fn merge(&mut self, newer: Facts) {
        self.context = newer.context.or(self.context);
        self.process = newer.process.or(self.process);
        self.weights = newer.weights.or(self.weights);
        self.weights_floor = newer.weights_floor.or(self.weights_floor);
        self.pinned_budget = newer.pinned_budget.or(self.pinned_budget);
        self.mapped = self.mapped.max(newer.mapped);
        self.activation = self.activation.max(newer.activation);
        self.host = self.host.max(newer.host);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Spawned or loading.
    Starting,
    Idle,
    /// In a call: never asked to give room.
    Active,
}

#[derive(Clone, Debug)]
pub struct Tenant {
    pub pid: u32,
    pub phase: Phase,
    pub last_used: u64,
    /// The cap last written to its call's cell (`u64::MAX`: none this call).
    pub cap: u64,
    /// Whether its plane holds device weights a `Budget{vram 0}` would free.
    pub mapped: bool,
}

/// One Degree 2 holding as custody reports it (pull): counted once on this GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Holding {
    /// Custody's name for it: layout and generation.
    pub id: String,
    pub bytes: u64,
    /// Executor pids that map it.
    pub readers: Vec<u32>,
    pub idle_ms: u64,
    pub revoking: bool,
}

/// A holding's name without its generation (`<name>#<generation>`): one weight set on one GPU.
pub fn holding_name(id: &str) -> &str {
    id.rsplit_once('#').map_or(id, |(name, _)| name)
}

/// What the caller does next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// `Budget{vram 0}`: the idle executor's weights leave the GPU, its pinned tier keeps them.
    Unmap(String),
    /// End the idle executor for the device context it holds.
    End(String),
    /// Revoke a Degree 2 holding no running call reads.
    Revoke(String),
    /// Revoke only enough of a holding's regions for these bytes; the rest stays attachable.
    Trim(String, u64),
}

/// Who asks for room, and so whose room it may take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ask {
    /// A running call (`DeviceRoom`): the holdings it let go first, then any idle tenant.
    Call,
    /// A spawn: any idle tenant, never the holdings it attaches.
    Spawn,
    /// A warm set member: idle tenants outside every warm set only.
    Member,
}

/// One room-making's memory of what it already asked, so no step repeats.
#[derive(Default)]
pub struct Round {
    tried: BTreeSet<String>,
}

#[derive(Default)]
pub struct Gpu {
    /// Configured minimum free bytes; the larger of it and the default floor applies.
    pub floor_bytes: u64,
    tenants: BTreeMap<String, Tenant>,
    /// Facts per plan, kept past its executor so a respawn starts from them.
    facts: BTreeMap<String, Facts>,
    /// Degree 2 holdings custody charges on this GPU; the caller refreshes them before each step.
    pub holdings: Vec<Holding>,
    /// Plans in a caller's warm set: among idle tenants, the last to give anything up.
    pub members: BTreeSet<String>,
    /// Custody's names (without the generation) of the weight sets each plan's executors asked
    /// for or offered: what its next executor attaches when they are held.
    named: BTreeMap<String, BTreeSet<String>>,
}

/// One clock for every GPU's ledger, so recency compares across GPUs.
fn tick() -> u64 {
    static CLOCK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    CLOCK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
}

impl Gpu {
    pub fn with_floor(floor_bytes: u64) -> Self {
        Self {
            floor_bytes,
            ..Self::default()
        }
    }

    pub fn floor(&self, sample: &Sample) -> u64 {
        let default = if sample.display {
            DISPLAY_FLOOR.max(sample.total / 16)
        } else {
            HEADLESS_FLOOR
        };
        default.max(self.floor_bytes)
    }

    pub fn tenant(&self, plan: &str) -> Option<&Tenant> {
        self.tenants.get(plan)
    }

    pub fn facts(&self, plan: &str) -> Facts {
        self.facts.get(plan).copied().unwrap_or_default()
    }

    /// What another GPU measured of `plan`, for one that never ran it (not its process there).
    pub fn adopt(&mut self, plan: &str, facts: Facts) {
        let known = Facts { context: None, process: None, pinned_budget: None, ..facts };
        self.facts.entry(plan.into()).or_insert(known);
    }

    pub fn starting(&mut self, plan: &str) {
        let (phase, last_used, cap) = (Phase::Starting, tick(), u64::MAX);
        let tenant = Tenant { pid: 0, phase, last_used, cap, mapped: false };
        self.tenants.insert(plan.into(), tenant);
    }

    pub fn spawned(&mut self, plan: &str, pid: u32) {
        if let Some(tenant) = self.tenants.get_mut(plan) {
            tenant.pid = pid;
        }
    }

    /// Facts from a reply. A starting executor that reports what it holds is loaded.
    pub fn observe(&mut self, plan: &str, facts: Facts, mapped: Option<bool>) {
        self.facts.entry(plan.into()).or_default().merge(facts);
        if let Some(tenant) = self.tenants.get_mut(plan) {
            if let Some(mapped) = mapped {
                tenant.mapped = mapped;
            }
            if tenant.phase == Phase::Starting && facts.process.is_some() {
                tenant.phase = Phase::Idle;
            }
        }
    }

    /// A call starts uncapped.
    pub fn active(&mut self, plan: &str) {
        if let Some(tenant) = self.tenants.get_mut(plan) {
            tenant.phase = Phase::Active;
            tenant.cap = u64::MAX;
            tenant.last_used = tick();
            tenant.mapped = true;
        }
    }

    pub fn idle(&mut self, plan: &str) {
        if let Some(tenant) = self.tenants.get_mut(plan) {
            tenant.phase = Phase::Idle;
        }
    }

    pub fn set_cap(&mut self, plan: &str, cap: u64) {
        if let Some(tenant) = self.tenants.get_mut(plan) {
            tenant.cap = cap;
        }
    }

    pub fn unmapped(&mut self, plan: &str) {
        if let Some(tenant) = self.tenants.get_mut(plan) {
            tenant.mapped = false;
        }
    }

    pub fn ended(&mut self, plan: &str) {
        self.tenants.remove(plan);
    }

    /// `plan`'s executor named holding `id` to custody: its next executor attaches it.
    pub fn named(&mut self, plan: &str, id: &str) {
        self.named.entry(plan.into()).or_default().insert(holding_name(id).into());
    }

    /// Bytes `tenant` holds now: NVML's charge where it can say, else its report. With Degree 2
    /// holdings on the GPU its report comes first: NVML charges a shared allocation to whichever
    /// process the driver picks, and custody counts it once.
    fn actual(&self, plan: &str, tenant: &Tenant, sample: &Sample) -> u64 {
        let measured = sample
            .processes
            .as_ref()
            .and_then(|table| table.get(&tenant.pid).copied())
            .filter(|_| tenant.pid != 0);
        let reported = self.facts(plan).process;
        let held = if self.holdings.is_empty() {
            measured.or(reported)
        } else {
            reported.or(measured)
        };
        held.unwrap_or(0)
    }

    /// What `plan`'s process may hold now: what it holds and everything free above the floor.
    /// Below the floor it is less than the process holds (run 5424: a grant 110 MB above it
    /// drove an 8 GB laptop card to 10 MB free).
    pub fn room(&self, plan: &str, sample: &Sample) -> u64 {
        let held = self.tenant(plan).map_or(0, |tenant| self.actual(plan, tenant, sample));
        (held + sample.free).saturating_sub(self.floor(sample))
    }

    /// A new executor's context: the largest one measured on this GPU, else `UNMEASURED_CONTEXT`.
    pub fn context_estimate(&self) -> u64 {
        match self.facts.values().filter_map(|f| f.context).max() {
            Some(measured) if measured > 0 => measured + MARGIN,
            _ => UNMEASURED_CONTEXT,
        }
    }

    /// Whether `holding` is `plan`'s own: mapped by its process, or, while its executor has
    /// not loaded yet, named by an earlier executor of it and held now (it attaches them at
    /// its load). Its bytes are on the GPU already: counted once, never again in what `plan`
    /// asks for, and never revoked for `plan` (phil, 12 GB: each Anima grant revoked 1.79 GB of
    /// Anima's own idle holdings, then loaded them again). A loaded executor that let a holding
    /// go never maps it again: then it is not its own (huaisang).
    fn own(&self, plan: &str, holding: &Holding) -> bool {
        let tenant = self.tenant(plan);
        let pid = tenant.map_or(0, |t| t.pid);
        (pid != 0 && holding.readers.contains(&pid))
            || (!holding.revoking
                && tenant.is_none_or(|t| t.phase == Phase::Starting)
                && self.names(plan, holding))
    }

    fn names(&self, plan: &str, holding: &Holding) -> bool {
        self.named.get(plan).is_some_and(|names| names.contains(holding_name(&holding.id)))
    }

    /// Whether `holding` is weights a warm set member's executor named.
    fn kept(&self, holding: &Holding) -> bool {
        self.members.iter().any(|member| self.names(member, holding))
    }

    /// Whether room for `plan` may revoke or trim `holding`: another plan's, read by no call in
    /// progress.
    fn reclaimable(&self, plan: &str, holding: &Holding) -> bool {
        !holding.revoking
            && !self.own(plan, holding)
            && !holding.readers.iter().any(|pid| {
                self.tenants
                    .iter()
                    .any(|(other, t)| other != plan && t.pid == *pid && t.phase != Phase::Idle)
            })
    }

    fn own_bytes(&self, plan: &str) -> u64 {
        self.holdings.iter().filter(|h| self.own(plan, h)).map(|h| h.bytes).sum()
    }

    /// Whether `plan`'s whole construction and its activations fit beside the other tenants
    /// (Degree 2 keeps every component resident). What room for it may revoke or trim is room
    /// (huaisang: counting only holdings nobody read, two models that do not both fit on a 12 GB
    /// card each detached in turn and reloaded whole at every switch). False until a call of it
    /// measured its activations.
    pub fn fits_resident(&self, plan: &str, sample: &Sample) -> bool {
        let reclaimable: u64 = self
            .holdings
            .iter()
            .filter(|h| self.reclaimable(plan, h))
            .map(|h| h.bytes)
            .sum();
        self.want(plan)
            .is_some_and(|want| self.room(plan, sample) + reclaimable >= want)
    }

    /// Whether a call of `plan` fits in the room there is now with no idle tenant giving any:
    /// its measured want, or, never measured, a GPU that holds nothing at all.
    pub fn fits(&self, plan: &str, sample: &Sample) -> bool {
        match self.want(plan) {
            Some(want) => self.room(plan, sample) >= want,
            None => self.tenants.is_empty() && self.holdings.is_empty(),
        }
    }

    /// Everything resident beside what it already maps: context, the weights its calls map,
    /// activations. None: not measured yet.
    pub fn want(&self, plan: &str) -> Option<u64> {
        let facts = self.facts(plan);
        let weights = facts.mapped.or(facts.weights)?;
        Some(
            facts.context.unwrap_or(self.context_estimate())
                + weights.saturating_sub(self.own_bytes(plan))
                + facts.activation?
                + MARGIN,
        )
    }

    /// What a spawn of `plan` needs to start: a context and, as measured, its weights' lowest
    /// rung and activations. The weights it attaches are resident already (H3 on an RTX PRO
    /// 6000: a replacement's floor would have revoked a holding it was about to map).
    pub fn spawn_need(&self, plan: &str) -> u64 {
        let facts = self.facts(plan);
        self.context_estimate()
            + facts.weights_floor.unwrap_or(0).saturating_sub(self.own_bytes(plan))
            + facts.activation.unwrap_or(0)
    }

    /// What `plan`'s process holds with its weights mapped and nothing running: its context
    /// and the weights its calls map, less what it maps already. None: never measured.
    pub fn mapped_need(&self, plan: &str) -> Option<u64> {
        let facts = self.facts(plan);
        let weights = facts.mapped.or(facts.weights)?.saturating_sub(self.own_bytes(plan));
        Some(facts.context.unwrap_or(self.context_estimate()) + weights + MARGIN)
    }

    /// The next idle tenant to give room toward `short` bytes for `plan`, least recently used
    /// first: idle weights (an executor's mapped ones, then custody's), then idle processes for
    /// their contexts; a warm set member's last. A call first takes back the holdings it named
    /// and nobody reads. None: nothing idle is left. A running call is never touched.
    pub fn step(&self, plan: &str, ask: Ask, short: u64, round: &mut Round) -> Option<Step> {
        if ask == Ask::Call {
            let unread = self.holdings.iter().filter(|h| {
                !h.revoking && h.readers.is_empty() && !round.tried.contains(&h.id) && self.names(plan, h)
            });
            if let Some(holding) = unread.max_by_key(|h| h.idle_ms) {
                round.tried.insert(holding.id.clone());
                return Some(Step::Revoke(holding.id.clone()));
            }
        }
        let mut idle: Vec<(&String, &Tenant)> = self
            .tenants
            .iter()
            .filter(|(other, t)| other.as_str() != plan && t.phase == Phase::Idle)
            .filter(|(other, _)| ask != Ask::Member || !self.members.contains(*other))
            .collect();
        idle.sort_by_key(|(other, tenant)| (self.members.contains(*other), tenant.last_used));
        let sets: &[bool] = if ask == Ask::Member { &[false] } else { &[false, true] };
        for &kept in sets {
            let unmap = idle.iter().find(|(other, t)| {
                t.mapped && self.members.contains(*other) == kept && !round.tried.contains(*other)
            });
            if let Some((other, _)) = unmap {
                round.tried.insert(other.to_string());
                return Some(Step::Unmap(other.to_string()));
            }
            let holding = self
                .holdings
                .iter()
                .filter(|h| !round.tried.contains(&h.id) && self.reclaimable(plan, h) && self.kept(h) == kept)
                .max_by_key(|h| h.idle_ms);
            if let Some(holding) = holding {
                round.tried.insert(holding.id.clone());
                // Only the shortfall goes: a 12 GB card switching SDXL and Anima revoked the
                // UNet's 5.3 GB for a 1.6 GB one, and loaded it all again at the next switch.
                return Some(match short < holding.bytes {
                    true => Step::Trim(holding.id.clone(), short),
                    false => Step::Revoke(holding.id.clone()),
                });
            }
        }
        let end = idle.into_iter().find(|(other, _)| round.tried.insert(format!("end:{other}")));
        end.map(|(other, _)| Step::End(other.clone()))
    }

    /// The idle tenant other than `plan` the host gives up first: one outside every warm
    /// set before a member, least recently used first. `spare`: never a member.
    pub fn lru_idle(&self, plan: &str, spare: bool) -> Option<String> {
        self.tenants
            .iter()
            .filter(|(other, t)| other.as_str() != plan && t.phase == Phase::Idle)
            .filter(|(other, _)| !(spare && self.members.contains(*other)))
            .min_by_key(|(other, t)| (self.members.contains(*other), t.last_used))
            .map(|(other, _)| other.clone())
    }

    /// Every live tenant as `(plan, weights, last used)`: what host pinned budgets are shared by.
    pub fn pinned_rows(&self) -> Vec<(String, u64, u64)> {
        let row = |(plan, tenant): (&String, &Tenant)| {
            (plan.clone(), self.facts(plan).weights.unwrap_or(0), tenant.last_used)
        };
        self.tenants.iter().map(row).collect()
    }

    /// A sample below the floor during a call: a running tenant's cap, lowered from what it
    /// holds by the deficit (never below its context), so the driver keeps its own floor (runs
    /// 5422/5423: libcuda faulted at 10 MB free). Pressure its ladder absorbs. None while the
    /// floor holds, nobody runs, or the cap is already that low.
    pub fn below_floor(&self, sample: &Sample) -> Option<(String, u64)> {
        let floor = self.floor(sample);
        if sample.free >= floor {
            return None;
        }
        let deficit = floor - sample.free + MARGIN;
        self.tenants
            .iter()
            .filter(|(_, t)| t.phase == Phase::Active)
            .map(|(plan, t)| {
                let lowered = self.actual(plan, t, sample).saturating_sub(deficit);
                (plan, lowered.max(self.facts(plan).context.unwrap_or(0)), t.cap)
            })
            .find(|(_, lowered, cap)| lowered < cap)
            .map(|(plan, lowered, _)| (plan.clone(), lowered))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1 << 30;

    fn sample(total: u64, free: u64, processes: &[(u32, u64)]) -> Sample {
        Sample {
            total,
            free,
            display: false,
            processes: Some(processes.iter().copied().collect()),
        }
    }

    fn loaded(gpu: &mut Gpu, plan: &str, pid: u32, weights: u64, activation: u64) {
        gpu.starting(plan);
        gpu.spawned(plan, pid);
        let facts = Facts {
            context: Some(GIB / 2),
            process: Some(GIB / 2),
            weights: Some(weights),
            weights_floor: Some(weights / 8),
            activation: Some(activation),
            ..Facts::default()
        };
        gpu.observe(plan, facts, Some(false));
    }

    fn held(id: &str, bytes: u64, readers: &[u32], idle_ms: u64) -> Holding {
        Holding { id: id.into(), bytes, readers: readers.to_vec(), idle_ms, revoking: false }
    }

    #[test]
    fn an_unmeasured_plan_fits_only_a_gpu_with_nothing_to_evict() {
        // A fresh pod's cold burst: its first call starts on GPU 0, the next fits empty GPU 1.
        let (mut first, empty) = (Gpu::default(), Gpu::default());
        let card = sample(96 * GIB, 95 * GIB, &[]);
        assert!(first.fits("image", &card));
        first.starting("image");
        assert!(empty.fits("image", &card));
        // Beside another tenant only a measured want fits, as another GPU measured it too.
        let mut shared = Gpu::default();
        loaded(&mut shared, "film", 7, 20 * GIB, 4 * GIB);
        let card = sample(96 * GIB, 70 * GIB, &[(7, 26 * GIB)]);
        assert!(!shared.fits("image", &card));
        let measured = Facts { context: Some(GIB), weights: Some(30 * GIB), activation: Some(4 * GIB), ..Facts::default() };
        shared.adopt("image", Facts { weights: Some(70 * GIB), ..measured });
        assert!(!shared.fits("image", &card));
        shared.observe("image", measured, None);
        assert!(shared.fits("image", &card));
    }

    #[test]
    fn a_display_gpu_keeps_its_floor() {
        let gpu = Gpu::default();
        let mut s = sample(8 * GIB, 8 * GIB - 166 * MIB, &[]);
        s.display = true;
        assert_eq!(gpu.floor(&s), DISPLAY_FLOOR);
        assert_eq!(gpu.room("sdxl", &s), 8 * GIB - DISPLAY_FLOOR - 166 * MIB);
    }

    #[test]
    fn room_for_a_call_unmaps_idle_weights_then_ends_idle_processes() {
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, 3 * GIB);
        gpu.active("sdxl");
        gpu.idle("sdxl");
        loaded(&mut gpu, "anima", 11, 6 * GIB, 3 * GIB);
        gpu.active("anima");
        let mut round = Round::default();
        assert_eq!(gpu.step("anima", Ask::Call, GIB, &mut round), Some(Step::Unmap("sdxl".into())));
        gpu.unmapped("sdxl");
        assert_eq!(gpu.step("anima", Ask::Call, GIB, &mut round), Some(Step::End("sdxl".into())));
        gpu.ended("sdxl");
        assert_eq!(gpu.step("anima", Ask::Call, GIB, &mut round), None);
    }

    #[test]
    fn a_running_call_never_gives_room() {
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, 3 * GIB);
        gpu.active("sdxl");
        gpu.holdings = vec![held("GPU-1/sha256:unet#1", 5 * GIB, &[10], 9_000)];
        for ask in [Ask::Call, Ask::Spawn, Ask::Member] {
            assert_eq!(gpu.step("anima", ask, 2 * GIB, &mut Round::default()), None);
        }
        assert_eq!(gpu.tenant("sdxl").unwrap().cap, u64::MAX);
    }

    #[test]
    fn below_the_floor_a_running_cap_is_lowered_by_the_deficit() {
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, 3 * GIB);
        gpu.active("sdxl");
        let s = sample(16 * GIB, 100 * MIB, &[(10, 12 * GIB)]);
        let (plan, cap) = gpu.below_floor(&s).expect("floor breached");
        assert_eq!((plan.as_str(), cap), ("sdxl", 12 * GIB - (HEADLESS_FLOOR - 100 * MIB + MARGIN)));
        gpu.set_cap("sdxl", cap);
        assert!(gpu.below_floor(&s).is_none(), "already that low");
        assert!(gpu.below_floor(&sample(16 * GIB, GIB, &[(10, 12 * GIB)])).is_none());
        // Never below its context.
        let tight = sample(16 * GIB, 0, &[(10, GIB / 2 + 100 * MIB)]);
        gpu.set_cap("sdxl", u64::MAX);
        assert_eq!(gpu.below_floor(&tight), Some(("sdxl".into(), GIB / 2)));
    }

    /// Two idle tenants with their weights mapped, "member" the less recently used.
    fn two_idle(members: &[&str]) -> Gpu {
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "member", 10, 2 * GIB, GIB);
        loaded(&mut gpu, "other", 11, 2 * GIB, GIB);
        gpu.observe("member", Facts::default(), Some(true));
        gpu.observe("other", Facts::default(), Some(true));
        gpu.members = members.iter().map(|plan| plan.to_string()).collect();
        gpu
    }

    #[test]
    fn room_comes_from_tenants_outside_the_warm_set_before_a_member() {
        let mut gpu = two_idle(&["member"]);
        let mut round = Round::default();
        // By the clock alone the member, the older of the two, would go first.
        assert_eq!(gpu.step("c", Ask::Call, GIB, &mut round), Some(Step::Unmap("other".into())));
        // Work is never refused: with nothing else idle, the member gives its room too.
        assert_eq!(gpu.step("c", Ask::Call, GIB, &mut round), Some(Step::Unmap("member".into())));
        assert_eq!(gpu.step("c", Ask::Call, GIB, &mut round), Some(Step::End("other".into())));
        gpu.ended("other");
        assert_eq!(gpu.lru_idle("c", false), Some("member".into()));
        assert_eq!(gpu.lru_idle("c", true), None);
        // A member joining takes only from outside the set: weights, then the process.
        let gpu = two_idle(&["member", "c"]);
        let mut round = Round::default();
        assert_eq!(gpu.step("c", Ask::Member, GIB, &mut round), Some(Step::Unmap("other".into())));
        assert_eq!(gpu.step("c", Ask::Member, GIB, &mut round), Some(Step::End("other".into())));
        assert_eq!(gpu.step("c", Ask::Member, GIB, &mut round), None);
    }

    #[test]
    fn idle_weights_outside_the_warm_set_are_reclaimed_before_a_members() {
        // sounosuke, 20 GB: SDXL as a member and another SDXL checkpoint, both idle with their
        // weights in custody, then Anima: the member's holding, the older one, went first.
        let mut gpu = two_idle(&["member"]);
        gpu.unmapped("member");
        gpu.unmapped("other");
        gpu.named("member", "GPU-1/sha256:unet-a");
        gpu.named("other", "GPU-1/sha256:unet-b");
        gpu.holdings = vec![
            held("GPU-1/sha256:unet-a#1", 2 * GIB, &[], 9_000),
            held("GPU-1/sha256:unet-b#1", 2 * GIB, &[], 1_000),
        ];
        let mut round = Round::default();
        let mut step = |ask| gpu.step("c", ask, 4 * GIB, &mut round);
        assert_eq!(step(Ask::Spawn), Some(Step::Revoke("GPU-1/sha256:unet-b#1".into())));
        assert_eq!(step(Ask::Spawn), Some(Step::Revoke("GPU-1/sha256:unet-a#1".into())));
        // A member joining never takes the member's.
        gpu.members.insert("c".into());
        let mut round = Round::default();
        let mut step = || gpu.step("c", Ask::Member, 4 * GIB, &mut round);
        assert_eq!(step(), Some(Step::Revoke("GPU-1/sha256:unet-b#1".into())));
        assert_eq!(step(), Some(Step::End("other".into())));
        assert_eq!(step(), None);
    }

    #[test]
    fn room_never_revokes_what_the_asking_plans_executor_maps() {
        // phil, 12 GB, SDXL and Anima alternating: room for Anima revoked its own holdings and
        // it loaded them again. What its executor maps stays; one it let go (huaisang:
        // detached when Degree 2 did not fit) comes back first, before anyone else gives.
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, GIB);
        loaded(&mut gpu, "anima", 11, 6 * GIB, GIB);
        gpu.active("anima");
        gpu.observe("sdxl", Facts::default(), Some(true));
        gpu.named("anima", "GPU-1/sha256:qwen");
        gpu.named("anima", "GPU-1/sha256:dit");
        gpu.holdings = vec![
            held("GPU-1/sha256:dit#7", 4 * GIB, &[11], 95_000),
            held("GPU-1/sha256:qwen#6", GIB, &[], 90_000),
            held("GPU-1/sha256:unet#5", 5 * GIB, &[10], 30_000),
        ];
        let mut round = Round::default();
        let mut step = || gpu.step("anima", Ask::Call, 2 * GIB, &mut round);
        assert_eq!(step(), Some(Step::Revoke("GPU-1/sha256:qwen#6".into())));
        assert_eq!(step(), Some(Step::Unmap("sdxl".into())));
        assert_eq!(step(), Some(Step::Trim("GPU-1/sha256:unet#5".into(), 2 * GIB)), "its own transformer stays");
        assert_eq!(step(), Some(Step::End("sdxl".into())));
        assert_eq!(step(), None);
    }

    #[test]
    fn a_holding_the_shortfall_exceeds_goes_whole() {
        let gpu = Gpu { holdings: vec![held("GPU-1/sha256:te#4", GIB, &[], 5_000)], ..Gpu::default() };
        let step = gpu.step("anima", Ask::Spawn, 2 * GIB, &mut Round::default());
        assert_eq!(step, Some(Step::Revoke("GPU-1/sha256:te#4".into())));
    }

    #[test]
    fn a_model_that_fits_once_an_idle_tenants_holdings_are_trimmed_shares() {
        // huaisang, 12 GB: SDXL (7 GiB) and Anima (5.5 GiB) do not both fit; Anima's idle
        // holdings count as room at SDXL's fit, until Anima's call reads them.
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, GIB);
        loaded(&mut gpu, "anima", 11, 11 * GIB / 2, GIB / 2);
        gpu.holdings = vec![held("GPU-1/sha256:dit#3", 11 * GIB / 2, &[11], 1_000)];
        let s = sample(12 * GIB, 6 * GIB - GIB / 2, &[(10, GIB / 2), (11, GIB / 2)]);
        assert!(gpu.room("sdxl", &s) < gpu.want("sdxl").unwrap());
        assert!(gpu.fits_resident("sdxl", &s));
        gpu.active("anima");
        assert!(!gpu.fits_resident("sdxl", &s));
    }

    #[test]
    fn weights_a_plan_already_maps_are_counted_once() {
        // 24 GiB card, Degree 2: SDXL and Anima both kept by custody, each read by its own idle
        // executor. Anima's want is its context and activations: its weights are resident.
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, GIB / 2);
        loaded(&mut gpu, "anima", 11, 6 * GIB, 3 * GIB);
        gpu.holdings = vec![held("sdxl#1", 7 * GIB, &[10], 1_000), held("anima#1", 6 * GIB, &[11], 1_000)];
        assert_eq!(gpu.want("anima"), Some(GIB / 2 + 3 * GIB + MARGIN));
    }

    #[test]
    fn a_weight_set_another_tenant_reads_is_not_counted_again_at_the_fit() {
        // 16 GiB card: an SDXL checkpoint (7 GiB) held and read by one package's running call.
        // A second package binds the same checkpoint; its earlier executor named that set.
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "txt2img", 10, 7 * GIB, 3 * GIB);
        gpu.active("txt2img");
        gpu.holdings = vec![held("GPU-1/sha256:sdxl#4", 7 * GIB, &[10], 1_000)];
        let measured = Facts { context: Some(GIB / 2), weights: Some(7 * GIB), activation: Some(3 * GIB), ..Facts::default() };
        gpu.observe("img2img", measured, None);
        let s = sample(16 * GIB, 16 * GIB - 7 * GIB - GIB / 2, &[(10, GIB / 2)]);
        assert!(!gpu.fits_resident("img2img", &s));
        // It attaches the held set: only its context and activations are new bytes.
        gpu.named("img2img", "GPU-1/sha256:sdxl#3");
        assert!(gpu.fits_resident("img2img", &s));
        // A set being revoked, or one of another checkpoint, is no help.
        gpu.holdings[0].revoking = true;
        assert!(!gpu.fits_resident("img2img", &s));
        gpu.holdings[0] = held("GPU-1/sha256:other#1", 7 * GIB, &[10], 1_000);
        assert!(!gpu.fits_resident("img2img", &s));
    }

    #[test]
    fn a_replacement_spawns_beside_the_holdings_it_attaches() {
        // An executor of SDXL (7 GiB of weights, 1 GiB floor, 7 GiB of activations) was killed:
        // custody keeps its weights and nobody reads them.
        let mut gpu = Gpu::default();
        let measured = Facts { weights: Some(7 * GIB), weights_floor: Some(GIB), activation: Some(7 * GIB), ..Facts::default() };
        gpu.observe("sdxl", measured, None);
        gpu.named("sdxl", "GPU-1/sha256:sdxl");
        gpu.holdings = vec![held("GPU-1/sha256:sdxl#1", 7 * GIB, &[], 1_000)];
        assert_eq!(gpu.spawn_need("sdxl"), gpu.context_estimate() + 7 * GIB);
        assert_eq!(gpu.step("sdxl", Ask::Spawn, GIB, &mut Round::default()), None, "the holding it maps next stays");
        // Another checkpoint's holding is no part of its floor and gives room.
        gpu.holdings = vec![held("GPU-1/sha256:other#1", 7 * GIB, &[], 1_000)];
        assert_eq!(gpu.spawn_need("sdxl"), gpu.context_estimate() + GIB + 7 * GIB);
        let step = gpu.step("sdxl", Ask::Spawn, GIB, &mut Round::default());
        assert_eq!(step, Some(Step::Trim("GPU-1/sha256:other#1".into(), GIB)));
    }

    #[test]
    fn a_plan_is_priced_by_the_weights_its_calls_map() {
        // H3 on one RTX PRO 6000 (run 4285): the construction registers two denoisers and a
        // call maps one. Priced whole, Degree 2 never fits; priced as mapped, it does.
        let mut gpu = Gpu::default();
        let measured = Facts { context: Some(700 * MIB), weights: Some(102 * GIB), activation: Some(20 * GIB), ..Facts::default() };
        gpu.observe("h3", measured, None);
        let free = sample(102 * GIB, 102 * GIB, &[]);
        assert!(!gpu.fits_resident("h3", &free));
        gpu.observe("h3", Facts { mapped: Some(80 * GIB), ..Facts::default() }, None);
        assert_eq!(gpu.want("h3"), Some(700 * MIB + 80 * GIB + 20 * GIB + MARGIN));
        assert!(gpu.fits_resident("h3", &free));
    }
}
