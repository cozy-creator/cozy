package records

import (
	"database/sql"
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
)

// observationEvent records an accepted run's observation changing: lost when this computer
// stopped hearing its machine, restored when a stream from it opened again.
const observationEvent = "machine.observation"

// ObservationLoss is an accepted run its machine holds that this computer cannot see now.
type ObservationLoss struct {
	Since  string `json:"since"`
	Reason string `json:"reason"`
}

// LoseObservation records that the run's machine stopped answering, once per loss.
func (s *Store) LoseObservation(requestID, reason string) *exit.Error {
	if loss, problem := s.ObservationLost(requestID); problem != nil || loss != nil {
		return problem
	}
	return s.AppendEvent(requestID, observationEvent, 0, map[string]any{"lost": reason})
}

// RestoreObservation records that a stream from the run's machine opened after a loss, and
// answers whether one was open.
func (s *Store) RestoreObservation(requestID string) (bool, *exit.Error) {
	if loss, problem := s.ObservationLost(requestID); problem != nil || loss == nil {
		return false, problem
	}
	return true, s.AppendEvent(requestID, observationEvent, 0, map[string]any{"restored": true})
}

// ObservationLost is the run's open loss of observation; nil while its machine is heard.
func (s *Store) ObservationLost(requestID string) (*ObservationLoss, *exit.Error) {
	var body, at string
	err := s.db.QueryRow(`SELECT payload,at FROM request_events WHERE request_id=? AND type=?
		ORDER BY seq DESC LIMIT 1`, requestID, observationEvent).Scan(&body, &at)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read the observation of %s: %s", requestID, err)
	}
	var payload struct {
		Lost string `json:"lost"`
	}
	if json.Unmarshal([]byte(body), &payload) != nil || payload.Lost == "" {
		return nil, nil
	}
	return &ObservationLoss{Since: at, Reason: payload.Lost}, nil
}
