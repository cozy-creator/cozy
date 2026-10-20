//! Executor lifecycle and reported measurements. Placement decisions live in varena.
use super::learned::Learned;
use std::collections::{BTreeMap, BTreeSet};

/// What an executor last reported of itself. None is unknown, never zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Facts {
    pub context: Option<u64>,
    /// Context + torch reserved + the plane's own maps.
    pub process: Option<u64>,
    /// Weights as stages count them (decoded copies included), and their lowest rung.
    pub weights: Option<u64>,
    pub weights_floor: Option<u64>,
    pub activation: Option<u64>,
    /// The budget that bounds its pinned host tier.
    pub pinned_budget: Option<u64>,
}
impl Facts {
    /// Newer facts replace older ones field by field; an absent field keeps what was known.
    pub fn merge(&mut self, newer: Facts) {
        self.context = newer.context.or(self.context);
        self.process = newer.process.or(self.process);
        self.weights = newer.weights.or(self.weights);
        self.weights_floor = newer.weights_floor.or(self.weights_floor);
        self.pinned_budget = newer.pinned_budget.or(self.pinned_budget);
        self.activation = match (self.activation, newer.activation) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => b.or(a),
        };
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Starting,
    Idle,
    Active,
}
#[derive(Clone, Debug)]
pub struct Tenant {
    pub pid: u32,
    pub phase: Phase,
    pub last_used: u64,
    pub mapped: bool,
}
#[derive(Default)]
pub struct Gpu {
    tenants: BTreeMap<String, Tenant>,
    facts: BTreeMap<String, Facts>,
    shapes: BTreeMap<String, String>,
    pub learned: Learned,
    pub members: BTreeSet<String>,
    pub device: String,
    clock: u64,
}
impl Gpu {
    pub fn tenant(&self, key: &str) -> Option<&Tenant> {
        self.tenants.get(key)
    }
    pub fn facts(&self, key: &str) -> Facts {
        self.facts.get(key).copied().unwrap_or_default()
    }
    pub fn starting(&mut self, key: &str) {
        self.clock = self.clock.saturating_add(1);
        self.tenants.insert(
            key.into(),
            Tenant {
                pid: 0,
                phase: Phase::Starting,
                last_used: self.clock,
                mapped: false,
            },
        );
    }
    pub fn spawned(&mut self, key: &str, pid: u32) {
        if let Some(t) = self.tenants.get_mut(key) {
            t.pid = pid;
        }
    }
    pub fn ended(&mut self, key: &str) {
        self.tenants.remove(key);
    }
    pub fn idle(&mut self, key: &str) {
        if let Some(t) = self.tenants.get_mut(key) {
            t.phase = Phase::Idle;
        }
    }
    pub fn active(&mut self, key: &str) {
        self.clock = self.clock.saturating_add(1);
        if let Some(t) = self.tenants.get_mut(key) {
            t.phase = Phase::Active;
            t.last_used = self.clock;
        }
    }
    pub fn unmapped(&mut self, key: &str) {
        if let Some(t) = self.tenants.get_mut(key) {
            t.mapped = false;
        }
    }
    pub fn observe(&mut self, key: &str, facts: Facts, mapped: Option<bool>) {
        self.facts.entry(key.into()).or_default().merge(facts);
        if let Some(t) = self.tenants.get_mut(key) {
            if let Some(mapped) = mapped {
                t.mapped = mapped;
            }
            if t.phase == Phase::Starting {
                t.phase = Phase::Idle;
            }
        }
    }
    /// Host-pressure cleanup only: GPU pages are never selected here.
    pub fn lru_idle(&self, own: &str, spare: bool) -> Option<String> {
        self.tenants
            .iter()
            .filter(|(key, t)| {
                key.as_str() != own
                    && t.phase == Phase::Idle
                    && (!spare || !self.members.contains(*key))
            })
            .min_by_key(|(key, t)| (self.members.contains(*key), t.last_used))
            .map(|(key, _)| key.clone())
    }
    pub fn set_shape(&mut self, key: &str, shape: &str) {
        self.shapes.insert(key.into(), shape.into());
    }
    pub fn shape_key(&self, key: &str) -> &str {
        self.shapes.get(key).map(String::as_str).unwrap_or("")
    }
    pub fn shape(&self, key: &str) -> Option<super::learned::Shape> {
        self.learned
            .plans
            .get(key)?
            .shapes
            .get(self.shape_key(key))
            .cloned()
    }
    pub fn seeds(&self, key: &str) -> BTreeMap<String, u64> {
        self.shape(key).map(|s| s.methods).unwrap_or_default()
    }
    pub fn squeezed(&self, key: &str) -> BTreeMap<String, u64> {
        self.shape(key).map(|s| s.squeezed).unwrap_or_default()
    }
}
