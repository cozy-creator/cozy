//! The order this machine offers its ready runs in (xs-047): model affinity with elevator
//! fairness. A turn serves one model's runs queued when it began, oldest first, as placement lets
//! them start; then the oldest run of another model takes the next turn, and the first model's
//! later arrivals wait for its next one. Calls of started runs come first, in their roots' order;
//! the rest follow by acceptance. Whether a run starts is placement's; nothing here waits.
use crate::journal::Execution;
use std::collections::HashMap;

#[derive(Default)]
pub(crate) struct Queue {
    /// The model whose turn it is, and the newest root queued when its turn began.
    turn: Option<(String, u64)>,
    /// Each ready root's model: its executors' construction.
    models: HashMap<String, String>,
}

impl Queue {
    /// `ready` (by root acceptance) in offer order; `model` names a root run's model.
    pub(crate) fn order(&mut self, mut ready: Vec<Execution>, mut model: impl FnMut(&Execution) -> String) -> Vec<Execution> {
        let mut models = HashMap::new();
        for root in ready.iter().filter(|record| record.invocation.parent.is_empty()) {
            let known = self.models.remove(&root.id).unwrap_or_else(|| model(root));
            models.insert(root.id.clone(), known);
        }
        let rank = |turn: &Option<(String, u64)>, record: &Execution| match models.get(&record.id) {
            None => 0,
            Some(m) if turn.as_ref().is_some_and(|(t, through)| t == m && record.root_seq() <= *through) => 1,
            Some(_) => 2,
        };
        if !ready.iter().any(|record| rank(&self.turn, record) == 1) {
            let previous = self.turn.take().map(|(model, _)| model);
            let roots: Vec<_> = ready.iter().filter_map(|r| Some((r.root_seq(), models.get(&r.id)?))).collect();
            let next = roots.iter().find(|(_, m)| Some(*m) != previous.as_ref()).or(roots.first());
            let through = roots.iter().map(|(seq, _)| *seq).max().unwrap_or_default();
            self.turn = next.map(|(_, model)| ((*model).clone(), through));
        }
        ready.sort_by_key(|record| rank(&self.turn, record));
        self.models = models;
        ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{execution::Engine, journal::Invocation};

    #[test]
    fn a_turn_serves_its_models_queued_runs_then_the_oldest_other_model_and_no_one_starves() {
        let root = std::env::temp_dir().join(format!("queue-{}", uuid::Uuid::new_v4()));
        let engine = Engine::open(&root).unwrap();
        let mut queue = Queue::default();
        // Runs of models `a` and `b` (their modules here), accepted in turn: ids 1, 2, …
        let accept = |models: &str| for model in models.chars() {
            let invocation = Invocation { package: "p".into(), generation: "0".repeat(32),
                module: model.into(), entrypoint: "run".into(), ..Default::default() };
            let key = uuid::Uuid::new_v4().to_string();
            let (run, _) = engine.accept_run("actor", &key, &key, invocation.clone()).unwrap();
            engine.bind_prepared(&run.id, invocation, "", Default::default(), None).unwrap();
        };
        let mut offered = || {
            let order = queue.order(engine.ready().unwrap(), |run| run.invocation.module.clone());
            order.iter().map(|run| run.root_seq()).collect::<Vec<_>>()
        };
        let start = |seq: u64| assert!(engine.dispatch_managed(&seq.to_string(), |_, _| Ok(())).unwrap());
        // Interleaved arrivals: one switch per model, not one per run.
        accept("ababa");
        assert_eq!(offered(), [1, 3, 5, 2, 4]);
        // 1 and 3 start; a's later arrival 6 waits for a's next turn, behind b's 2 and 4.
        start(1);
        start(3);
        accept("a");
        assert_eq!(offered(), [5, 2, 4, 6]);
        // a's turn ends with 5: b's turn, though more of a keeps arriving.
        start(5);
        accept("a");
        assert_eq!(offered(), [2, 4, 6, 7]);
        start(2);
        start(4);
        accept("b");
        assert_eq!(offered(), [6, 7, 8]);
        // b's 8 was queued when a's turn began: it has the next turn, before a's 9.
        start(6);
        start(7);
        accept("a");
        assert_eq!(offered(), [8, 9]);
        crate::scratch::remove(&root);
    }
}
