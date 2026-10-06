package producttest

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"github.com/cozy-creator/cozy/internal/archive"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
)

// Retained producer fixtures supply an actual matching job invocation,
// successful outcome and native receipt. Only the historical bookkeeping failure
// is constructed here; recovery must never construct a second producer attempt.
func publicationRetryFixture(t *testing.T) (*records.Store, *sql.DB, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	st, problem := records.Open(path)
	fatal(t, problem)
	t.Cleanup(func() { st.Close() })
	db, err := sql.Open("sqlite", path)
	must(t, err)
	t.Cleanup(func() { db.Close() })
	read := func(name string) []byte {
		raw, err := os.ReadFile(filepath.Join("testdata", "record-archive", name+".json"))
		must(t, err)
		return raw
	}
	invocation := read("invocation_spec_job")
	outcome := read("attempt_outcome_body_job_succeeded")
	receipt := read("weights_receipt")
	rec, err := archive.Read(receipt, archive.WeightsReceipt)
	must(t, err)
	terminal, err := archive.Read(outcome, archive.TerminalBody)
	must(t, err)
	id := rec.Str("request_id")
	digest := func(raw []byte) string {
		result, err := canonical.Spell(canonical.Digest(raw))
		must(t, err)
		return result
	}
	run := func(query string, args ...any) { _, err := db.Exec(query, args...); must(t, err) }
	run(`INSERT INTO requests(id,idem_key,body_digest,package,entrypoint,plan_id,payload,state,ordinal,created_at,kind) VALUES(?,?,?,?,?,?,?,?,?,?,?)`, id, id, "fixture", "owner/tools", "quantize", "fixture", []byte("{}"), "failed", terminal.Int("attempt_ordinal"), "fixture", "job")
	run(`INSERT INTO worker_processes(instance_id,package,worker_id,devices,pid,birth,state,opened_at) VALUES('worker','owner/tools','worker','[]',1,'fixture','running','fixture')`)
	run(`INSERT INTO attempts(request_id,attempt,attempt_key,instance_id,session_id,invocation_digest,invocation,state,terminal_digest,terminal_status,terminal_body,dispatched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)`, id, terminal.Int("attempt_ordinal"), "attempt-fixture", "worker", "session", digest(invocation), invocation, "terminal", digest(outcome), "SUCCEEDED", outcome, "fixture")
	intent, err := json.Marshal(records.ModelTransferIntent{Kind: "model-upload", Destination: "owner/model", Outputs: []records.ModelTransferOutput{{Name: "model"}}})
	must(t, err)
	run(`INSERT INTO request_model_transfers(request_id,intent,state,error_code,safe_error,updated_at) VALUES(?,?,'failed','stale_preparation','old preparation failed','fixture')`, id, string(intent))
	run(`INSERT INTO request_model_transfer_outputs(request_id,output_slot,manifest_id,manifest_length,attempt,invocation_digest,transaction_id,receipt_digest,receipt,final_id) VALUES(?,?,?,?,?,?,?,?,?,?)`, id, "model", "sha256:bde1922331c714cc7a2cef772423b099169c5f41d11c7c4547d61c19aee65f2d", 16384, terminal.Int("attempt_ordinal"), digest(invocation), rec.Str("weights_transaction_id"), digest(receipt), receipt, records.ModelTransferOutputOperation(id, "model"))
	run(`INSERT INTO request_model_transfer_objects(request_id,attempt,output_slot,object_id,length,source_ref) SELECT request_id,attempt,output_slot,manifest_id,manifest_length,'fixture' FROM request_model_transfer_outputs`)
	return st, db, id
}

func TestVerifiedPublicationRetryPreservesSuccessfulAttempt(t *testing.T) {
	st, db, id := publicationRetryFixture(t)
	before, problem := st.Attempts(id)
	fatal(t, problem)
	changed, problem := st.RetryModelTransferPublication(id, "operator-proof")
	fatal(t, problem)
	if !changed {
		t.Fatal("verified failed publication did not resume")
	}
	request, problem := st.RequestRow(id)
	fatal(t, problem)
	transfer, problem := st.ModelTransferOf(id)
	fatal(t, problem)
	after, problem := st.Attempts(id)
	fatal(t, problem)
	if request.State != "finalizing" || transfer.State != "finalizing" || transfer.ErrorCode != "" || len(after) != 1 || len(before) != 1 || after[0].Attempt != before[0].Attempt || after[0].TerminalStatus != "SUCCEEDED" || after[0].State != "terminal" || !bytes.Equal(after[0].InvocationCanonical, before[0].InvocationCanonical) || !bytes.Equal(after[0].TerminalBody, before[0].TerminalBody) {
		t.Fatal("retry changed successful producer identity or failed to restore both lifecycle rows")
	}
	var payload string
	must(t, db.QueryRow(`SELECT payload FROM request_events WHERE request_id=? AND type='request.publication_retry_requested'`, id).Scan(&payload))
	var event map[string]any
	must(t, json.Unmarshal([]byte(payload), &event))
	if event["actor"] != "operator-proof" || event["recovered_request_failure"] != true {
		t.Fatal("explicit actor recovery event is absent")
	}
	changed, problem = st.RetryModelTransferPublication(id, "operator-proof")
	fatal(t, problem)
	if changed {
		t.Fatal("same recovery was applied twice")
	}
	var events int
	must(t, db.QueryRow(`SELECT COUNT(*) FROM request_events WHERE request_id=?`, id).Scan(&events))
	if events != 1 {
		t.Fatal("retry fabricated a completion or duplicate event")
	}
	st.Close()
	reopened, problem := records.Open(filepath.Join(filepath.Dir(dbPathForRetry(t, db)), "creator.sqlite"))
	fatal(t, problem)
	defer reopened.Close()
	restored, problem := reopened.RequestRow(id)
	fatal(t, problem)
	if restored.State != "finalizing" {
		t.Fatal("recovery did not survive owner restart")
	}
}

func dbPathForRetry(t *testing.T, db *sql.DB) string {
	t.Helper()
	var seq int
	var name, path string
	must(t, db.QueryRow(`PRAGMA database_list`).Scan(&seq, &name, &path))
	return path
}

func TestVerifiedPublicationRetryRefusesUnprovenOrCanceledWork(t *testing.T) {
	for name, query := range map[string]string{
		"request canceled":            `UPDATE requests SET state='canceled'`,
		"transfer canceled":           `UPDATE request_model_transfers SET state='canceled'`,
		"producer failed":             `UPDATE attempts SET terminal_status='FAILED'`,
		"producer still open":         `UPDATE attempts SET state='accepted'`,
		"no exact attempt":            `UPDATE requests SET ordinal=99`,
		"one checkpoint absent":       `UPDATE request_model_transfer_outputs SET final_id=''`,
		"manifest substituted":        `UPDATE request_model_transfer_outputs SET manifest_id='sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'`,
		"manifest length substituted": `UPDATE request_model_transfer_outputs SET manifest_length=42`,
		"publication substituted":     `UPDATE request_model_transfer_outputs SET final_id='another-operation'`,
		"root object absent":          `DELETE FROM request_model_transfer_objects`,
		"receipt absent":              `DELETE FROM request_model_transfer_outputs`,
		"receipt digest changed":      `UPDATE request_model_transfer_outputs SET receipt_digest='sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'`,
		"receipt bytes changed":       `UPDATE request_model_transfer_outputs SET receipt=x'7b7d'`,
		"transaction changed":         `UPDATE request_model_transfer_outputs SET transaction_id='different'`,
		"invocation bytes changed":    `UPDATE attempts SET invocation=x'7b7d'`,
		"outcome bytes changed":       `UPDATE attempts SET terminal_body=x'7b7d'`,
		"declaration changed":         `UPDATE request_model_transfers SET intent='{"kind":"model-upload","destination":"owner/model","outputs":[{"name":"other"}]}'`,
	} {
		t.Run(name, func(t *testing.T) {
			st, db, id := publicationRetryFixture(t)
			_, err := db.Exec(query)
			must(t, err)
			var beforeRequest, beforeTransfer string
			must(t, db.QueryRow(`SELECT state FROM requests WHERE id=?`, id).Scan(&beforeRequest))
			must(t, db.QueryRow(`SELECT state FROM request_model_transfers WHERE request_id=?`, id).Scan(&beforeTransfer))
			changed, problem := st.RetryModelTransferPublication(id, "operator-proof")
			if problem == nil || changed {
				t.Fatal("unverified or canceled work was restored")
			}
			var request, transfer string
			var events, attempts int
			must(t, db.QueryRow(`SELECT state FROM requests WHERE id=?`, id).Scan(&request))
			must(t, db.QueryRow(`SELECT state FROM request_model_transfers WHERE request_id=?`, id).Scan(&transfer))
			must(t, db.QueryRow(`SELECT COUNT(*) FROM request_events`).Scan(&events))
			must(t, db.QueryRow(`SELECT COUNT(*) FROM attempts`).Scan(&attempts))
			if request != beforeRequest || transfer != beforeTransfer || events != 0 || attempts != 1 {
				t.Fatal("refusal partially changed retained work")
			}
		})
	}
}
