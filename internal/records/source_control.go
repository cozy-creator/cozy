package records

import "github.com/cozy-creator/cozy/internal/exit"

// Called only after the exact revision's Host pause acknowledgement proves that
// source work has drained. The checkpoint/provenance rows remain intact.
func (s *Store) PauseModelSourceMaterialization(id string, revision uint64) *exit.Error {
	_, err := s.db.Exec(`UPDATE request_model_transfers SET state='pending',updated_at=?
		WHERE request_id=? AND state='materializing' AND EXISTS(SELECT 1 FROM requests r
		WHERE r.id=request_id AND r.state='pausing' AND r.control_revision=?)`, now(), id, revision)
	if err != nil {
		return exit.Internalf("cannot settle paused source materialization: %s", err)
	}
	return nil
}
