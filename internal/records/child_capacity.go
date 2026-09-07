package records

import (
	"bytes"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// CaptureOrchestrationDirective preserves the exact CPU parent contract before
// dispatch. Reconnect can reconstruct full replacement without guessing caps or
// accepting a worker-supplied replacement for the original owner declaration.
func (s *Store) CaptureOrchestrationDirective(id string, raw []byte) *exit.Error {
	var directive pb.JobDirective
	if canonical.Unmarshal(raw, &directive) != nil || !directive.Orchestration || directive.OrchestrationParent != nil || directive.ResourceCaps.GetDeviceRequired() || directive.DeviceCount != 0 {
		return exit.New(exit.Validation, "orchestration capture must be one exact CPU parent directive")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin orchestration capture: %s", err)
	}
	defer tx.Rollback()
	var old []byte
	var state string
	var retained bool
	if err := tx.QueryRow(`SELECT orchestration_directive,state,retain_work FROM requests WHERE id=?`, id).Scan(&old, &state, &retained); err != nil {
		return exit.Internalf("cannot read orchestration owner: %s", err)
	}
	if len(old) > 0 {
		if !bytes.Equal(old, raw) {
			return exit.Named(exit.Conflict, "child.parent_directive_changed", "the parent already captured another exact orchestration contract")
		}
		return nil
	}
	if !retained || (state != "submitted" && state != "queued" && state != "dispatching") {
		return exit.Named(exit.Conflict, "child.parent_stopped", "a stopped parent cannot acquire orchestration capacity")
	}
	if _, err := tx.Exec(`UPDATE requests SET orchestration_directive=? WHERE id=?`, raw, id); err != nil {
		return exit.Internalf("cannot capture orchestration contract: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit orchestration contract: %s", err)
	}
	return nil
}
