package producttest

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func sealedExecution(t *testing.T, closed bool) (*records.Store, records.Terminal, string) {
	t.Helper()
	store, call, path := effectCancelFixture(t)
	fatal(t, store.StopNativeCall(call.ID, "canceled", "fixture"))
	body := assessmentDocument(t, &pb.AttemptOutcomeBody{RequestId: call.ParentRequestID, AttemptOrdinal: 1,
		InvocationSpecDigest: childDigest("1"), Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, ExecutionStarted: true,
		Observation: &pb.ExecutionObservation{Environment: &pb.ExecutionEnvironment{RuntimeVersion: "fixture", Accelerator: "CPU", WorkerBootId: "private-boot"}}})
	terminal := records.Terminal{RequestID: call.ParentRequestID, Attempt: 1, SessionID: "private-boot", InvocationDigest: childDigest("1"),
		TerminalID: "execution-identity", TerminalDigest: assessmentDigest(body), Status: "SUCCEEDED", RequestState: "succeeded", Body: body}
	_, problem := store.AcceptTerminal(terminal)
	fatal(t, problem)
	if closed {
		fatal(t, store.Closed(terminal.RequestID, 1))
	}
	return store, terminal, path
}

func TestKnownTerminalReplayKeepsItsExecutionIdentity(t *testing.T) {
	for _, closed := range []bool{false, true} {
		t.Run(map[bool]string{false: "terminal", true: "closed"}[closed], func(t *testing.T) {
			store, terminal, _ := sealedExecution(t, closed)
			before, problem := store.AttemptRow(terminal.RequestID, 1)
			fatal(t, problem)
			fatal(t, store.Recover(terminal.RequestID, 1, "different-recovery-peer"))
			replay := terminal
			replay.SessionID = "different-recovery-peer"
			applied, problem := store.AcceptTerminal(replay)
			fatal(t, problem)
			if applied {
				t.Fatal("a recovery peer applied the same terminal twice")
			}
			after, problem := store.AttemptRow(terminal.RequestID, 1)
			fatal(t, problem)
			if after.SessionID != before.SessionID || after.InstanceID != before.InstanceID || after.State != before.State || !bytes.Equal(after.TerminalBody, before.TerminalBody) {
				t.Fatal("ACK replay changed the sealed execution record")
			}
			replay.Body = []byte(`{"changed":true}`)
			if _, problem := store.AcceptTerminal(replay); problem == nil {
				t.Fatal("a supplied digest bypassed byte-exact replay validation")
			}
		})
	}
}

func TestRecoveryRepairsOnlyVerifiedTerminalExecutionIdentity(t *testing.T) {
	for _, mode := range []string{"verified", "missing", "digest", "request", "attempt", "spec"} {
		t.Run(mode, func(t *testing.T) {
			store, terminal, path := sealedExecution(t, true)
			db, err := sql.Open("sqlite", path)
			must(t, err)
			defer db.Close()
			// This is the exact old Recover corruption in an isolated fixture DB.
			_, err = db.Exec(`UPDATE attempts SET session_id='wrong-recovery-peer' WHERE request_id=? AND attempt=1`, terminal.RequestID)
			must(t, err)
			body, digest := terminal.Body, terminal.TerminalDigest
			if mode != "verified" {
				var document map[string]any
				must(t, json.Unmarshal(body, &document))
				switch mode {
				case "missing":
					delete(document, "observation")
				case "request":
					document["request_id"] = "another-request"
				case "attempt":
					document["attempt_ordinal"] = 2
				case "spec":
					document["invocation_spec_digest"] = childDigest("2")
				}
				body = assessmentJSON(t, document)
				digest = assessmentDigest(body)
				if mode == "digest" {
					digest = childDigest("f")
				}
				_, err = db.Exec(`UPDATE attempts SET terminal_body=?,terminal_digest=? WHERE request_id=? AND attempt=1`, body, digest, terminal.RequestID)
				must(t, err)
			}
			problem := store.Recover(terminal.RequestID, 1, "latest-recovery-peer")
			if mode == "verified" || mode == "missing" {
				fatal(t, problem)
				fatal(t, store.Recover(terminal.RequestID, 1, "another-recovery-peer"))
			} else if problem == nil || problem.ErrName() != "attempt.execution_identity_unverified" {
				t.Fatal("unbound terminal observation repaired execution identity", problem)
			}
			row, problem := store.AttemptRow(terminal.RequestID, 1)
			fatal(t, problem)
			want, events := "wrong-recovery-peer", 0
			if mode == "verified" {
				want, events = "private-boot", 1
			}
			if row.SessionID != want || row.State != "closed" || !bytes.Equal(row.TerminalBody, body) {
				t.Fatal("recovery changed unverified identity, closure, or terminal bytes")
			}
			var count int
			must(t, db.QueryRow(`SELECT count(*) FROM request_events WHERE request_id=? AND type='attempt.execution_identity_restored'`, terminal.RequestID).Scan(&count))
			if count != events {
				t.Fatal("execution identity repair lacks its exact audit event")
			}
		})
	}
}

func TestOpenRecoveryStillMovesItsCurrentPeer(t *testing.T) {
	store, call, _ := effectCancelFixture(t)
	fatal(t, store.Recover(call.ParentRequestID, 1, "recovery-peer"))
	row, problem := store.AttemptRow(call.ParentRequestID, 1)
	fatal(t, problem)
	if row.SessionID != "recovery-peer" || row.State != "recovered_open" {
		t.Fatal("open recovery lost its current peer")
	}
	if _, problem := store.AcceptTerminal(records.Terminal{RequestID: call.ParentRequestID,
		Attempt: 1, SessionID: "private-boot", InvocationDigest: childDigest("1")}); problem == nil {
		t.Fatal("the former peer sealed an open attempt after recovery moved ownership")
	}
}
