package api

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// LocalAbandonment reports only this client's disposition. It asserts no remote
// cancellation, workspace loss, resource destruction or collection acknowledgement.
type LocalAbandonment struct {
	ID                  string `json:"id"`
	Number              int64  `json:"number"`
	Status              string `json:"status"`
	Changed             bool   `json:"changed"`
	AbandonedLocally    bool   `json:"abandoned_locally"`
	RemoteStopConfirmed bool   `json:"remote_stop_confirmed"`
	RentalReleased      bool   `json:"rental_released"`
}

func (s *Server) abandonRequest(w http.ResponseWriter, r *http.Request) {
	var intent struct {
		Actor string `json:"actor"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096)).Decode(&intent); err != nil || strings.TrimSpace(intent.Actor) == "" {
		s.refuseTyped(w, r, exit.New(exit.Validation, "local abandonment requires an explicit actor"))
		return
	}
	row, problem := s.store.RequestByReference(r.PathValue("id"))
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if row == nil {
		s.refuse(w, r, http.StatusNotFound, "not_found", "no such recorded run on this host", "")
		return
	}
	changed, problem := s.store.AbandonMachineExecution(row.ID, intent.Actor)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if s.machineExecutions != nil {
		s.machineExecutions.Withdraw(row.ID)
	}
	latest, problem := s.store.RequestRow(row.ID)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if latest == nil {
		s.refuseTyped(w, r, exit.Internalf("abandoned run disappeared"))
		return
	}
	s.ok(w, r, http.StatusOK, LocalAbandonment{ID: row.ID, Number: row.Number, Status: s.publicStatusOf(*latest), Changed: changed, AbandonedLocally: true})
}
