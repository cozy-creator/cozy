package records

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/protobuf/proto"
)

// RunV1Accepted marks a run its machine accepted over cozy.machine.v1: its receipt column holds
// the run's first RunState and its outcome column the v1 Outcome. The run's log is recorded in
// the same request events a worker.v1 run's import writes, so every reader renders it alike.
const RunV1Accepted = "machine.api_v1"

// RunV1Sent marks permission to send a run's spec: the machine may hold it before
// its acceptance is recorded here. Cancellation after this mark needs a machine outcome.
const RunV1Sent = "machine.api_v1_sent"

// MarkRunV1Sent orders dispatch against cancellation in the same database transaction.
// If cancellation wins, no spec may be sent. If dispatch wins, cancellation must be
// delivered even when the connection fails before the machine records acceptance.
func (s *Store) MarkRunV1Sent(id string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin machine dispatch: %s", err)
	}
	defer tx.Rollback()
	var machine, state string
	var canceled, abandoned, sent bool
	err = tx.QueryRow(`SELECT e.machine_id, r.state, e.cancel_requested,
 EXISTS(SELECT 1 FROM request_events WHERE request_id=r.id AND type='client.machine_abandoned'),
 EXISTS(SELECT 1 FROM request_events WHERE request_id=r.id AND type=?)
 FROM requests r JOIN machine_executions e ON e.request_id=r.id WHERE r.id=?`, RunV1Sent, id).
		Scan(&machine, &state, &canceled, &abandoned, &sent)
	if err != nil {
		return false, exit.Internalf("cannot read machine dispatch: %s", err)
	}
	if machine == "" || canceled || abandoned || Settled(state) || state == "pausing" || state == "paused" || state == "blocked" {
		return false, nil
	}
	if !sent {
		if err := appendEventTx(tx, id, RunV1Sent, 0, map[string]any{"machine": machine}); err != nil {
			return false, exit.Internalf("cannot record machine dispatch: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit machine dispatch: %s", err)
	}
	return true, nil
}

// RunV1 is whether the run's machine accepted it over cozy.machine.v1.
func (s *Store) RunV1(id string) (bool, *exit.Error) {
	return s.RunV1Marked(id, RunV1Accepted)
}

// RunV1Marked is whether the run carries one of the marks above.
func (s *Store) RunV1Marked(id, mark string) (bool, *exit.Error) {
	var found bool
	if err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM request_events WHERE request_id=? AND type=?)`, id, mark).Scan(&found); err != nil {
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
	if err := projectRunV1(tx, id, state, 0); err != nil {
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
// startedMS is when the machine started the run running (its clock), 0 when not known.
func projectRunV1(tx *sql.Tx, id string, state *v1.RunState, startedMS int64) error {
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
	case "pausing", "paused":
		next = state.State
	}
	// A pausing job's root still reads running until it has stopped.
	if next == "" || Settled(current) || current == "pausing" && next == "dispatching" {
		return nil
	}
	if cancel {
		next = "canceling"
	}
	if next == "dispatching" && current != "dispatching" {
		running := map[string]any{"machine_execution": true}
		if startedMS > 0 {
			running["started_unix_ms"] = startedMS
		}
		if err := appendEventTx(tx, id, "run.in_progress", int64(state.Attempt), running); err != nil {
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
// worker.v1 import writes. Nothing here waits on fsync (see telemetryV1): progress and log
// lines join the run's batch; a state or a product (`product` holds the file this client
// wrote, nil for none) is written at once, with the run's pending rows ahead of it.
func (s *Store) ObserveRunV1(id string, event *v1.RunEvent, product *Product) *exit.Error {
	if telemetryEventV1(event) {
		s.queueTelemetryV1(id, event)
		return nil
	}
	s.telemetry.write.Lock()
	defer s.telemetry.write.Unlock()
	var sampled int
	err := s.unsynced(func(tx *sql.Tx) (err error) {
		if sampled, err = s.telemetryTx(tx, id); err != nil {
			return err
		}
		var cursor, ordinal int64
		if err := tx.QueryRow(`SELECT e.remote_cursor, r.ordinal FROM machine_executions e JOIN requests r ON r.id=e.request_id WHERE e.request_id=?`, id).Scan(&cursor, &ordinal); err != nil {
			return err
		}
		if event.Sequence != 0 && int64(event.Sequence) <= cursor {
			return nil // recorded before
		}
		at := time.UnixMilli(event.AtMs).UTC().Format(time.RFC3339Nano)
		if event.AtMs == 0 {
			at = now()
		}
		switch value := event.Event.(type) {
		case *v1.RunEvent_State:
			if err := projectRunV1(tx, id, value.State, event.AtMs); err != nil {
				return err
			}
		case *v1.RunEvent_Product:
			if product == nil {
				break
			}
			for _, item := range outputItemEvents(*product) {
				raw, err := json.Marshal(item.payload)
				if err != nil {
					return err
				}
				if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,?,?,?,?)`, id, item.kind, max(ordinal, 1), string(raw), at); err != nil {
					return err
				}
			}
		case *v1.RunEvent_Memo:
			// A memoized call's result: the package's later jobs carry it as known (KnownResults).
			if m := value.Memo; json.Valid(m.Result) && len(m.Result) <= maxKnownResult {
				if err := appendEventTx(tx, id, MemoRecorded, max(ordinal, 1), map[string]any{"operation": m.Operation,
					"computation_digest": m.ComputationDigest, "result": json.RawMessage(m.Result)}); err != nil {
					return err
				}
			}
		case *v1.RunEvent_Call:
			if err := insertCallV1(tx, id, max(ordinal, 1), value.Call); err != nil {
				return err
			}
		}
		if event.Sequence != 0 {
			_, err = tx.Exec(`UPDATE machine_executions SET remote_cursor=? WHERE request_id=?`, event.Sequence, id)
		}
		return err
	})
	if err != nil {
		return exit.Internalf("cannot record the run's observation: %s", err)
	}
	s.dropTelemetry(id, sampled)
	return nil
}

// insertCallV1 records one settled call of the run as `machine.call`, the record `run show`
// lists: the callee, the author's label, how it ended, when, and its stage and step tracks.
func insertCallV1(tx *sql.Tx, id string, attempt int64, call *v1.Call) error {
	var measured struct {
		Attribution struct {
			Stages json.RawMessage `json:"stages"`
			Steps  json.RawMessage `json:"steps"`
		} `json:"attribution"`
	}
	_ = json.Unmarshal(call.Measurements, &measured)
	record := map[string]any{"request": call.Run, "parent": id, "index": call.Index, "attempt": attempt,
		"export": call.Function, "label": call.Label, "status": call.Status, "error": call.GetReason().GetMessage(),
		"called_unix_ms": call.CalledAtMs, "stages": measured.Attribution.Stages, "steps": measured.Attribution.Steps}
	raw, err := json.Marshal(record)
	if err != nil {
		return err
	}
	at := time.UnixMilli(max(call.FinishedAtMs, call.CalledAtMs)).UTC().Format(time.RFC3339Nano)
	_, err = tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,'machine.call',?,?,?)`, id, attempt, string(raw), at)
	return err
}

func humanBytes(n uint64) string {
	if n >= 1<<30 {
		return fmt.Sprintf("%.1f GiB", float64(n)/(1<<30))
	}
	return fmt.Sprintf("%.1f MiB", float64(n)/(1<<20))
}

// RunEndV1 is what a run's collection settles in one transaction: the outcome, the export of
// its output files and the evidence kept beside it.
type RunEndV1 struct {
	Outcome  *v1.Outcome
	Refused  *exit.Error // why an output file was not written here; nil when every one was
	Export   bool        // the run has an output export, settled with Paths and Refused
	Paths    []string
	Evidence []byte // the triage bundle with the run's measurements; empty for none
	// FinishedMS is the machine's time of the outcome and ObservedMS when it reached this
	// client: their difference is the machine's delivery, the rest of a run's tail is ours.
	FinishedMS, ObservedMS int64
}

// RecordRunOutcomeV1 ends the run with its machine's outcome, once: its outputs finished, the
// terminal event with the result's status and reason, and the request settled. Its result is
// collected when every output file this client writes was written before the call; `Refused`
// is why one was not, and the result then stays with its machine until a later call collects it.
func (s *Store) RecordRunOutcomeV1(id string, end RunEndV1) *exit.Error {
	outcome, refused := end.Outcome, end.Refused
	raw, err := proto.MarshalOptions{Deterministic: true}.Marshal(outcome)
	if err != nil || len(raw) > 8<<20 {
		return exit.New(exit.Conflict, "the run's outcome is not bounded")
	}
	s.telemetry.write.Lock()
	defer s.telemetry.write.Unlock()
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
	attempt := uint64(max(ordinal, 1))
	sampled, err := s.telemetryTx(tx, id)
	if err != nil {
		return exit.Internalf("cannot record the run's telemetry: %s", err)
	}
	if end.Export {
		if problem := settleOutputExport(tx, id, end.Paths, refused); problem != nil {
			return problem
		}
	}
	if len(end.Evidence) > 0 {
		if problem := recordMachineTriage(tx, id, 1, end.Evidence); problem != nil {
			return problem
		}
	}
	if len(prior) == 0 {
		if problem := recordRunEndV1(tx, id, attempt, raw, end); problem != nil {
			return problem
		}
	}
	// The collection: every output in its folder, or why not. A refused one leaves the result
	// with its machine, and a reader asking about the run tries again.
	var collected bool
	if err := tx.QueryRow(`SELECT collected FROM machine_executions WHERE request_id=?`, id).Scan(&collected); err != nil {
		return exit.Internalf("cannot read the run's collection: %s", err)
	}
	switch {
	case collected:
	case refused != nil:
		code, message, problem := machineCollectionRefusal(tx, id)
		if problem != nil {
			return problem
		}
		if code != refused.ErrName() || message != refused.Message {
			if err := appendEventTx(tx, id, MachineCollectionRefused, int64(attempt),
				map[string]any{"machine_execution": true, "error_code": refused.ErrName(), "error": refused.Message}); err != nil {
				return exit.Internalf("cannot record collection refusal: %s", err)
			}
		}
	default:
		// Nothing of the run is owed after its collection: the machine's cache manages what it keeps.
		if _, err := tx.Exec(`UPDATE machine_executions SET collected=1 WHERE request_id=?`, id); err != nil {
			return exit.Internalf("cannot record the run's collection: %s", err)
		}
		for _, event := range []string{MachineResultCollected, "machine.retention_released"} {
			if err := appendEventTx(tx, id, event, int64(attempt), map[string]any{"machine_execution": true, "state": outcome.Status}); err != nil {
				return exit.Internalf("cannot record the run's collection: %s", err)
			}
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit the run's outcome: %s", err)
	}
	s.dropTelemetry(id, sampled)
	return nil
}

// recordRunEndV1 settles the run from its outcome, once.
func recordRunEndV1(tx *sql.Tx, id string, attempt uint64, raw []byte, end RunEndV1) *exit.Error {
	outcome := end.Outcome
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
	if _, err := tx.Exec(`UPDATE machine_executions SET outcome=?, pending_control=x'', cancel_requested=0 WHERE request_id=?`, raw, id); err != nil {
		return exit.Internalf("cannot retain the run's outcome: %s", err)
	}
	if _, err := tx.Exec(`UPDATE requests SET state=CASE WHEN state='canceled' THEN state ELSE ? END, retain_work=0 WHERE id=?`, state, id); err != nil {
		return exit.Internalf("cannot settle the run: %s", err)
	}
	finished := map[string]any{"machine_execution": true, "state": outcome.Status}
	if end.ObservedMS > 0 {
		finished["finished_unix_ms"], finished["observed_unix_ms"] = end.FinishedMS, end.ObservedMS
	}
	if err := appendEventTx(tx, id, "client.machine_work_finished", int64(attempt), finished); err != nil {
		return exit.Internalf("cannot record the run's end: %s", err)
	}
	if outcome.Status == "succeeded" {
		return recordPublishedWeightsV1(tx, id, outcome.Outputs)
	}
	return nil
}

// recordPublishedWeightsV1 records the checkpoints the machine published to the run's weights
// destination: each declared weights output's product is its checkpoint's manifest.
func recordPublishedWeightsV1(tx *sql.Tx, id string, products []*v1.Product) *exit.Error {
	var raw string
	err := tx.QueryRow(`SELECT intent FROM request_model_transfers WHERE request_id=? AND state NOT IN ('completed','canceled','failed')`, id).Scan(&raw)
	if err == sql.ErrNoRows {
		return nil
	}
	if err != nil {
		return exit.Internalf("cannot read the run's weights destination: %s", err)
	}
	var intent ModelTransferIntent
	if json.Unmarshal([]byte(raw), &intent) != nil || intent.Destination == "" {
		return nil
	}
	checkpoints := map[string]string{}
	for _, product := range products {
		for _, output := range intent.Outputs {
			if product.Output == output.Name && product.Index == 0 && product.Digest != "" {
				checkpoints[output.Name] = product.Digest
			}
		}
	}
	if len(checkpoints) == 0 {
		return nil
	}
	encoded, _ := json.Marshal(checkpoints)
	if _, err := tx.Exec(`UPDATE request_model_transfers SET state='completed',checkpoints=?,updated_at=? WHERE request_id=?`,
		string(encoded), now(), id); err != nil {
		return exit.Internalf("cannot record the run's published checkpoints: %s", err)
	}
	return nil
}
