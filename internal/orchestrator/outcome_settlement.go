package orchestrator

import pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"

// A device lane can hold two admitted requests and one released outcome; the job
// seat is independent. Bound owner work by that cohort, rather than starting a
// goroutine per terminal. An outcome still waits for verified local custody and
// its durable commit before ack; only control reception proceeds alongside it.
func (c *Orchestrator) startOutcomeSettlement(s *session, w *worker, devices int) func() {
	// Never allocate beyond this session's existing control-frame window, even if
	// a peer reports an impossible device count. A full window applies backpressure.
	depth := min(3*max(devices, 1)+1, cap(s.out))
	s.outcomes = make(chan *pb.AttemptOutcome, depth)
	s.completions = make(chan snapshotContinuation, depth)
	outcomesDone, completionsDone := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(completionsDone)
		for item := range s.completions {
			c.afterAck(item.request, item.attempt, w)
		}
	}()
	go func() {
		defer close(outcomesDone)
		for outcome := range s.outcomes {
			c.onOutcome(s, outcome)
		}
	}()
	return func() {
		close(s.outcomes)
		<-outcomesDone
		s.completionMu.Lock()
		s.completionClosed = true
		close(s.completions)
		s.completionMu.Unlock()
		<-completionsDone
	}
}

func (s *session) complete(item snapshotContinuation) bool {
	s.completionMu.RLock()
	defer s.completionMu.RUnlock()
	if s.completionClosed {
		return false
	}
	// A durable closed attempt still owes its continuation if transport cancels.
	// Teardown drains this lane, rather than dropping work on ctx.Done().
	s.completions <- item
	return true
}
