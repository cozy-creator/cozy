package records

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/protobuf/proto"
)

// RunV1Accepted marks a run its machine accepted over cozy.machine.v1: its receipt column holds
// the run's first RunState and its outcome column the v1 Outcome. The run's log is recorded in
// the same request events a worker.v1 run's import writes, so every reader renders it alike.
const RunV1Accepted = "machine.api_v1"

// RunV1 is whether the run's machine accepted it over cozy.machine.v1.
func (s *Store) RunV1(id string) (bool, *exit.Error) {
	var found bool
	if err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM request_events WHERE request_id=? AND type=?)`, id, RunV1Accepted).Scan(&found); err != nil {
		return false, exit.Internalf("cannot read the run's machine API: %s", err)
	}
	return found, nil
}

// RunV1State is the run's accepted RunState (its machine number), nil before acceptance.
func RunV1State(link *MachineExecution) *v1.RunState {
	var state v1.RunState
	if link == nil || len(link.Receipt) == 0 || proto.Unmarshal(link.Receipt, &state) != nil {
		return nil
	}
	return &state
}

// RunV1Outcome is the run's recorded v1 Outcome, nil before it ended.
func RunV1Outcome(link *MachineExecution) *v1.Outcome {
	var outcome v1.Outcome
	if link == nil || len(link.Outcome) == 0 || proto.Unmarshal(link.Outcome, &outcome) != nil {
		return nil
	}
	return &outcome
}

// AcceptRunV1 records the machine's acceptance once: its first RunState, the marker event,
// and the request projected from that state.
func (s *Store) AcceptRunV1(id string, state *v1.RunState) *exit.Error {
	raw, err := proto.MarshalOptions{Deterministic: true}.Marshal(state)
	if err != nil {
		return exit.Internalf("cannot retain the run's acceptance: %s", err)
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin the run's acceptance: %s", err)
	}
	defer tx.Rollback()
	result, err := tx.Exec(`UPDATE machine_executions SET receipt=? WHERE request_id=? AND length(receipt)=0`, raw, id)
	if err != nil {
		return exit.Internalf("cannot record the run's acceptance: %s", err)
	}
	if n, _ := result.RowsAffected(); n == 0 {
		return nil // accepted before
	}
	if err := appendEventTx(tx, id, RunV1Accepted, int64(state.Attempt), map[string]any{"number": state.Number, "state": state.State}); err != nil {
		return exit.Internalf("cannot record the run's acceptance: %s", err)
	}
	if err := projectRunV1(tx, id, state); err != nil {
		return exit.Internalf("cannot project the run's acceptance: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit the run's acceptance: %s", err)
	}
	return nil
}

// projectRunV1 moves the request to the machine's state. A terminal word waits for the
// outcome, which carries the result; a canceled request stays canceled; a requested cancel
// shows canceling until the machine ends the run.
func projectRunV1(tx *sql.Tx, id string, state *v1.RunState) error {
	var current string
	var cancel bool
	if err := tx.QueryRow(`SELECT r.state, COALESCE(e.cancel_requested,0) FROM requests r LEFT JOIN machine_executions e ON e.request_id=r.id WHERE r.id=?`, id).Scan(&current, &cancel); err != nil {
		return err
	}
	next := ""
	switch state.State {
	case "queued", "preparing":
		next = "queued"
	case "running":
		next = "dispatching"
	case "paused":
		next = "paused"
	}
	if next == "" || Settled(current) {
		return nil
	}
	if cancel {
		next = "canceling"
	}
	if next == "dispatching" && current != "dispatching" {
		if err := appendEventTx(tx, id, "run.in_progress", int64(state.Attempt), map[string]any{"machine_execution": true}); err != nil {
			return err
		}
	}
	if next == "paused" && current != "paused" {
		if err := appendEventTx(tx, id, "request.paused", int64(state.Attempt), map[string]any{"machine_execution": true}); err != nil {
			return err
		}
	}
	_, err := tx.Exec(`UPDATE requests SET state=?, ordinal=? WHERE id=?`, next, max(state.Attempt, 1), id)
	return err
}

// ObserveRunV1 records one entry of the run's log, past the recorded cursor, in the shapes a
// worker.v1 import writes: progress as `machine.progress` samples (preparation's as
// `request.preparing`), a product as its output item events (`product` holds the file this
// client wrote, nil for none), a log line as `request.log`.
func (s *Store) ObserveRunV1(id string, event *v1.RunEvent, product *Product) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin the run's observation: %s", err)
	}
	defer tx.Rollback()
	var cursor int64
	var ordinal int64
	var current string
	if err := tx.QueryRow(`SELECT e.remote_cursor, r.ordinal, r.state FROM machine_executions e JOIN requests r ON r.id=e.request_id WHERE e.request_id=?`, id).Scan(&cursor, &ordinal, &current); err != nil {
		return exit.Internalf("cannot read the run's observation: %s", err)
	}
	if event.Sequence != 0 && int64(event.Sequence) <= cursor {
		return nil // recorded before
	}
	at := time.UnixMilli(event.AtMs).UTC().Format(time.RFC3339Nano)
	if event.AtMs == 0 {
		at = now()
	}
	insert := func(kind string, payload any) error {
		raw, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		_, err = tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,?,?,?,?)`, id, kind, max(ordinal, 1), string(raw), at)
		return err
	}
	switch value := event.Event.(type) {
	case *v1.RunEvent_State:
		err = projectRunV1(tx, id, value.State)
	case *v1.RunEvent_Progress:
		p := value.Progress
		if current == "queued" {
			detail := p.Stage
			if p.BytesTotal > 0 {
				detail = strings.TrimSpace(detail + " " + humanBytes(p.BytesDone) + " of " + humanBytes(p.BytesTotal))
			}
			err = insert("request.preparing", map[string]any{"stage": "machine", "detail": detail})
			break
		}
		sample := map[string]any{"stage": p.Stage}
		if p.Fraction >= 0 {
			sample["overall_fraction"] = p.Fraction
		}
		if p.Total > 0 {
			sample["position"], sample["total"] = p.Completed, p.Total
		}
		if p.BytesTotal > 0 {
			sample["bytes_done"], sample["bytes_total"] = p.BytesDone, p.BytesTotal
		}
		err = insert("machine.progress", map[string]any{"type": "progress", "payload": sample})
	case *v1.RunEvent_Product:
		if product != nil {
			for _, item := range outputItemEvents(*product) {
				if err = insert(item.kind, item.payload); err != nil {
					break
				}
			}
		}
	case *v1.RunEvent_Log:
		err = insert("request.log", map[string]any{"level": value.Log.Level, "message": value.Log.Text})
	}
	if err != nil {
		return exit.Internalf("cannot record the run's log: %s", err)
	}
	if event.Sequence != 0 {
		if _, err := tx.Exec(`UPDATE machine_executions SET remote_cursor=? WHERE request_id=?`, event.Sequence, id); err != nil {
			return exit.Internalf("cannot advance the run's cursor: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit the run's observation: %s", err)
	}
	return nil
}

func humanBytes(n uint64) string {
	if n >= 1<<30 {
		return fmt.Sprintf("%.1f GiB", float64(n)/(1<<30))
	}
	return fmt.Sprintf("%.1f MiB", float64(n)/(1<<20))
}

// RecordRunOutcomeV1 ends the run with its machine's outcome: its outputs finished, the
// terminal event with the result's status and reason, the request settled and its result
// collected (every output file this client writes is written before this is called).
func (s *Store) RecordRunOutcomeV1(id string, outcome *v1.Outcome) *exit.Error {
	raw, err := proto.MarshalOptions{Deterministic: true}.Marshal(outcome)
	if err != nil || len(raw) > 8<<20 {
		return exit.New(exit.Conflict, "the run's outcome is not bounded")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin the run's outcome: %s", err)
	}
	defer tx.Rollback()
	var ordinal int64
	var prior []byte
	if err := tx.QueryRow(`SELECT r.ordinal, e.outcome FROM requests r JOIN machine_executions e ON e.request_id=r.id WHERE r.id=?`, id).Scan(&ordinal, &prior); err != nil {
		return exit.Internalf("cannot read the run's outcome: %s", err)
	}
	if len(prior) > 0 {
		return nil // recorded before
	}
	attempt := uint64(max(ordinal, 1))
	state, kind, status, itemStatus := "failed", "run.failed", "FAILED", "incomplete"
	switch outcome.Status {
	case "succeeded":
		state, kind, status, itemStatus = "succeeded", "run.completed", "SUCCEEDED", "completed"
	case "canceled":
		state, kind, status = "canceled", "run.canceled", "CANCELED"
	}
	output, problem := finishOutputs(tx, id, attempt, itemStatus)
	if problem != nil {
		return problem
	}
	if output == nil {
		output = []OutputItem{}
	}
	facts := map[string]any{"machine_execution": true, "status": status, "outputs": []any{}, "output": output}
	if reason := outcome.Reason; reason != nil && outcome.Status != "succeeded" {
		facts["error"], facts["error_type"], facts["error_code"] = reason.Message, reason.Code, reason.Code
	}
	payload, _ := json.Marshal(facts)
	if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) SELECT ?,?,?,?,?
 WHERE NOT EXISTS(SELECT 1 FROM request_events WHERE request_id=? AND type='run.canceled')`, id, kind, attempt, string(payload), now(), id); err != nil {
		return exit.Internalf("cannot record the run's end: %s", err)
	}
	// Nothing of the run is owed after its outcome: the machine's cache manages what it keeps.
	if _, err := tx.Exec(`UPDATE machine_executions SET outcome=?, collected=1, pending_control=x'', cancel_requested=0 WHERE request_id=?`, raw, id); err != nil {
		return exit.Internalf("cannot retain the run's outcome: %s", err)
	}
	if _, err := tx.Exec(`UPDATE requests SET state=CASE WHEN state='canceled' THEN state ELSE ? END, retain_work=0 WHERE id=?`, state, id); err != nil {
		return exit.Internalf("cannot settle the run: %s", err)
	}
	for _, event := range []string{"client.machine_work_finished", MachineResultCollected, "machine.retention_released"} {
		if err := appendEventTx(tx, id, event, int64(attempt), map[string]any{"machine_execution": true, "state": outcome.Status}); err != nil {
			return exit.Internalf("cannot record the run's collection: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit the run's outcome: %s", err)
	}
	return nil
}
