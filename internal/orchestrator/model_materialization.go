package orchestrator

import (
	"bytes"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// modelMaterializationMiss is only native absence reported against the exact
// accepted placement. Corruption, arbitrary I/O failures and historical faults
// keep their normal terminal handling.
func (w *worker) modelMaterializationMiss(r *pb.ObservedWorkerState) bool {
	if w.spec.Connection == nil || w.spec.IsJob() || r.AcceptedDesiredStateRevision == 0 ||
		r.ConvergedRevision >= r.AcceptedDesiredStateRevision || !bytes.Equal(r.AcceptedPlacementSetDigest, w.setDigest) {
		return false
	}
	desired, err := canonical.Read(w.setBytes, &pb.PlacementSet{})
	if err != nil {
		return false
	}
	current := map[string]bool{}
	for _, entry := range desired.List("placements") {
		current[entry.Str("placement_id")] = true
	}
	for _, p := range r.Placements {
		if p == nil || !current[p.PlacementId] || p.Serving != pb.ServingState_SERVING_STATE_OFFLINE || p.Materialization != pb.MaterializationState_MATERIALIZATION_STATE_ABSENT || !bytes.Equal(p.PlacementSetDigest, w.setDigest) {
			continue
		}
		for _, f := range append(append([]*pb.Fault(nil), r.Faults...), p.Faults...) {
			if f != nil && f.Kind == pb.FaultKind_FAULT_KIND_ARTIFACT_FETCH_FAILED && f.Reason == "model_materialization_required" && f.Subject == p.PlacementId {
				return true
			}
		}
	}
	return false
}

// reensureModels uses the existing logical desired input, not the detached
// PREPARED answer. Runtime has now accepted that intent and protects its models
// during acquisition, so a successful ensure cannot immediately be evicted by
// the same worker. A second miss after this ensure is an invariant failure.
func (c *Orchestrator) reensureModels(s *session, w *worker, revision uint64) {
	w.desiredMu.Lock()
	defer w.desiredMu.Unlock()
	c.mu.Lock()
	if c.workers[w.instanceID] != w || w.bootID != s.bootID || w.revision != revision || c.sessions[w.bootID] != s {
		if w.modelEnsureFromRevision == revision && w.modelEnsureRevision == 0 {
			w.modelEnsureFromRevision = 0
		}
		c.mu.Unlock()
		return
	}
	unpublished := cloneUnpublishedPlacementSet(w.desiredUnpublishedPlacement)
	packages, models := clonePackageRefs(w.desiredPackages), cloneModelRefs(w.desiredModels)
	c.mu.Unlock()
	var problem *exit.Error
	if unpublished != nil {
		problem = c.issueUnpublishedPlacementSet(s, w, unpublished)
	} else if len(packages) > 0 {
		problem = c.issuePackageSet(s, w, packages, models)
	} else {
		problem = exit.Named(exit.Structural, "worker.model_materialization_input_unavailable", "worker reported absent model bytes without a downloadable model selection")
	}
	c.mu.Lock()
	if w.modelEnsureFromRevision == revision {
		w.modelEnsureRevision = w.revision
		if problem != nil {
			w.modelEnsureFromRevision = 0
			w.desiredRefusal = problem
		}
	}
	c.mu.Unlock()
	if problem != nil {
		c.logf("worker %s: model re-ensure failed: %s", w.instanceID, problem.Message)
	}
}
