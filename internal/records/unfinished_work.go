package records

import (
	"encoding/json"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Work ends completed or failed with a reason; nothing waits for a retry. EndUnfinishedWork
// ends, at daemon start, what an older daemon left waiting: each `blocked` request (a
// failure kept open for a --retry) fails with the reason it stopped for, and each weights
// destination of a run that already ended without publishing to it ends with that run. It
// answers the requests it failed.
func (s *Store) EndUnfinishedWork() ([]string, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return nil, exit.Internalf("cannot begin ending unfinished work: %s", err)
	}
	defer tx.Rollback()
	rows, err := tx.Query(`SELECT r.id,COALESCE((SELECT e.payload FROM request_events e WHERE e.request_id=r.id
 AND e.type='request.blocked' ORDER BY e.seq DESC LIMIT 1),'{}') FROM requests r WHERE r.state='blocked' ORDER BY r.created_at,r.id`)
	if err != nil {
		return nil, exit.Internalf("cannot read blocked work: %s", err)
	}
	type blocked struct{ id, payload string }
	var found []blocked
	var ids []string
	for rows.Next() {
		var b blocked
		if err := rows.Scan(&b.id, &b.payload); err != nil {
			rows.Close()
			return nil, exit.Internalf("cannot read blocked work: %s", err)
		}
		found = append(found, b)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return nil, exit.Internalf("cannot finish reading blocked work: %s", err)
	}
	for _, b := range found {
		payload := map[string]any{}
		_ = json.Unmarshal([]byte(b.payload), &payload)
		code, _ := payload["error_type"].(string)
		detail, _ := payload["error"].(string)
		if code == "" || detail == "" {
			code, detail = "request.stopped", "this run stopped before it started and recorded no reason"
		}
		payload["status"], payload["error_type"], payload["error"], payload["requeuing"] = "FAILED", code, detail, false
		if _, err := tx.Exec(`UPDATE requests SET state='failed' WHERE id=?`, b.id); err != nil {
			return nil, exit.Internalf("cannot fail blocked work %s: %s", b.id, err)
		}
		if problem := endModelTransferTx(tx, b.id, "failed", detail); problem != nil {
			return nil, problem
		}
		if problem := skipOutputExport(tx, b.id, "the run failed before it started: "+detail); problem != nil {
			return nil, problem
		}
		if err := appendEventTx(tx, b.id, "run.failed", 0, payload); err != nil {
			return nil, exit.Internalf("cannot journal blocked work %s: %s", b.id, err)
		}
		line, _, _ := strings.Cut(detail, "\n")
		ids = append(ids, b.id+": "+line)
	}
	// A machine run publishes its weights in its outcome, and the daemon's own pass-through
	// settles its request only after its transfer: an ended run's open destination is never
	// published to.
	if _, err := tx.Exec(`UPDATE request_model_transfers SET
 state=CASE WHEN r.state='canceled' THEN 'canceled' ELSE 'failed' END,
 error_code=CASE WHEN r.state='succeeded' THEN 'model_transfer.outputs_missing' ELSE 'model_transfer.run_'||r.state END,
 safe_error=CASE WHEN r.state='succeeded' THEN 'the run succeeded but published no checkpoint to its destination'
  ELSE 'the run ended '||r.state||' before publishing its weights' END,
 updated_at=?
 FROM requests r WHERE r.id=request_model_transfers.request_id
 AND request_model_transfers.state NOT IN ('completed','failed','canceled')
 AND r.state IN (`+settledRequestStates+`)
 AND NOT (r.package='cozy/platform' AND r.entrypoint='model-pass-through')
 AND (r.state<>'succeeded' OR COALESCE(json_extract(request_model_transfers.intent,'$.destination'),'')<>'')`, now()); err != nil {
		return nil, exit.Internalf("cannot end the weights destinations of ended runs: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return nil, exit.Internalf("cannot commit ending unfinished work: %s", err)
	}
	return ids, nil
}

// endModelTransferTx ends a run's weights destination with the run, when the run ended
// without success: nothing publishes to it afterwards.
func endModelTransferTx(q execer, id, state, reason string) *exit.Error {
	next := "failed"
	if state == "canceled" {
		next = "canceled"
	}
	message := "the run ended " + state + " before publishing its weights"
	if reason != "" {
		message += ": " + reason
	}
	if _, err := q.Exec(`UPDATE request_model_transfers SET state=?,error_code=?,safe_error=?,updated_at=?
 WHERE request_id=? AND state NOT IN ('completed','failed','canceled')`, next, "model_transfer.run_"+state, message, now(), id); err != nil {
		return exit.Internalf("cannot end the weights destination of %s: %s", id, err)
	}
	return nil
}
