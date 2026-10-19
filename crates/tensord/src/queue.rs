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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{execution::Engine, journal::Invocation};
    use std::sync::Arc;

    /// A real journal's ready runs, offered in the queue's order; runs of models `a`, `b`, `h`
    /// (their modules) are accepted with ids 1, 2, …
    struct Bench(Arc<Engine>, Queue, std::path::PathBuf);
    impl Bench {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!("queue-{}", uuid::Uuid::new_v4()));
            Self(Engine::open(&root).unwrap(), Queue::default(), root)
        }
        fn accept(&self, models: &str) {
            for model in models.chars() {
                let invocation = Invocation { package: "p".into(), generation: "0".repeat(32),
                    module: model.into(), entrypoint: "run".into(), ..Default::default() };
                let key = uuid::Uuid::new_v4().to_string();
                let (run, _) = self.0.accept_run("actor", &key, &key, invocation.clone()).unwrap();
                self.0.bind_prepared(&run.id, invocation, "", Default::default(), None).unwrap();
            }
        }
        /// One dispatch pass's offer order.
        fn offered(&mut self) -> Vec<u64> {
            let order = self.1.order(self.0.ready().unwrap(), |run| run.invocation.module.clone());
            order.iter().map(|run| run.root_seq()).collect()
        }
        fn start(&mut self, seqs: &[u64]) {
            for seq in seqs {
                let run = self.0.get(&seq.to_string()).unwrap();
                assert!(self.0.dispatch_managed(&run.id, |_, _| Ok(())).unwrap());
                self.1.started(&run);
            }
        }
    }
    impl Drop for Bench {
        fn drop(&mut self) {
            crate::scratch::remove(&self.2);
        }
    }

    #[test]
    fn a_turn_serves_its_models_queued_runs_then_the_oldest_other_model_and_no_one_starves() {
        let mut bench = Bench::new();
        // Interleaved arrivals: one switch per model, not one per run.
        bench.accept("ababa");
        assert_eq!(bench.offered(), [1, 3, 5, 2, 4]);
        // 1 and 3 start; a's later arrival 6 waits for a's next turn, behind b's 2 and 4.
        bench.start(&[1, 3]);
        bench.accept("a");
        assert_eq!(bench.offered(), [5, 2, 4, 6]);
        // a's turn ends with 5: b's turn, though more of a keeps arriving.
        bench.start(&[5]);
        bench.accept("a");
        assert_eq!(bench.offered(), [2, 4, 6, 7]);
        bench.start(&[2, 4]);
        bench.accept("b");
        assert_eq!(bench.offered(), [6, 7, 8]);
        // b's 8 was queued when a's turn began: it has the next turn, before a's 9.
        bench.start(&[6, 7]);
        bench.accept("a");
        assert_eq!(bench.offered(), [8, 9]);
    }

    /// The rental proof's shape (runs 11922–11931): while h holds the machine, 5 of a, 1 of b and
    /// 3 of a arrive one pass at a time. a's turn takes them all in until its first run starts,
    /// so its 8 run back to back and b after; an arrival once it started waits for b.
    #[test]
    fn a_turn_takes_in_its_models_arrivals_until_it_starts_then_later_ones_wait() {
        let mut bench = Bench::new();
        bench.accept("h");
        assert_eq!(bench.offered(), [1]);
        bench.start(&[1]);
        for model in ["a", "a", "a", "a", "a", "b", "a", "a", "a"] {
            bench.accept(model);
            bench.offered();
        }
        assert_eq!(bench.offered(), [2, 3, 4, 5, 6, 8, 9, 10, 7]);
        bench.start(&[2]);
        bench.accept("a");
        assert_eq!(bench.offered(), [3, 4, 5, 6, 8, 9, 10, 7, 11]);
        bench.start(&[3, 4, 5, 6, 8, 9, 10]);
        assert_eq!(bench.offered(), [7, 11]);
    }
}
