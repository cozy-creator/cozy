//! The order this machine offers its ready runs in (xs-047): model affinity with elevator
//! fairness. A turn serves one model's runs queued when its first run started (the machine switched
//! to it), oldest first, as placement lets them start; then the oldest run of another model takes
//! the next turn, and the first model's later arrivals wait for its next one. Calls of started runs come first, in their roots' order;
//! the rest follow by acceptance. Whether a run starts is placement's; nothing here waits.
use crate::journal::Execution;
use std::collections::HashMap;

#[derive(Default)]
pub(crate) struct Queue {
    turn: Option<Turn>,
    /// Each ready root's model: its executors' construction.
    models: HashMap<String, String>,
}

/// One model's turn: its runs up to `through`, the newest root queued when it started.
struct Turn {
    model: String,
    through: u64,
    started: bool,
}

impl Queue {
    /// `ready` (by root acceptance) in offer order; `model` names a root run's model.
    pub(crate) fn order(&mut self, mut ready: Vec<Execution>, mut model: impl FnMut(&Execution) -> String) -> Vec<Execution> {
        let mut models = HashMap::new();
        for root in ready.iter().filter(|record| record.invocation.parent.is_empty()) {
            let known = self.models.remove(&root.id).unwrap_or_else(|| model(root));
            models.insert(root.id.clone(), known);
        }
        let roots: Vec<_> = ready.iter().filter_map(|r| Some((r.root_seq(), models.get(&r.id)?))).collect();
        let newest = roots.iter().map(|(seq, _)| *seq).max().unwrap_or_default();
        // Until its first run starts, a turn takes in what arrives: the switch is when it starts.
        if let Some(turn) = self.turn.as_mut().filter(|turn| !turn.started) {
            turn.through = turn.through.max(newest);
        }
        let rank = |turn: &Option<Turn>, record: &Execution| match models.get(&record.id) {
            None => 0,
            Some(m) if turn.as_ref().is_some_and(|t| &t.model == m && record.root_seq() <= t.through) => 1,
            Some(_) => 2,
        };
        if !ready.iter().any(|record| rank(&self.turn, record) == 1) {
            let previous = self.turn.take().map(|turn| turn.model);
            let next = roots.iter().find(|(_, m)| Some(*m) != previous.as_ref()).or(roots.first());
            self.turn = next.map(|(_, model)| Turn { model: (*model).clone(), through: newest, started: false });
        }
        ready.sort_by_key(|record| rank(&self.turn, record));
        self.models = models;
        ready
    }

    /// Root `run` started: a turn it belongs to takes in no more arrivals.
    pub(crate) fn started(&mut self, run: &Execution) {
        let model = self.models.get(&run.id);
        if let Some(turn) = self.turn.as_mut().filter(|turn| model == Some(&turn.model) && run.root_seq() <= turn.through) {
            turn.started = true;
        }
    }
}
