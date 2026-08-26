package records

import (
	"database/sql"
	"path/filepath"
	"testing"
)

func TestOpenNormalizesLegacyClosedRequeueThroughBudget(t *testing.T) {
	path := filepath.Join(t.TempDir(), "records.db")
	legacy, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	// This is the request/attempt shape from before assets and media cleanup were added.
	// More importantly, it carries the crash state produced after Closed but before the
	// old in-memory requeue continuation: queued request, current attempt already closed.
	for _, stmt := range []string{
		`CREATE TABLE requests (
		  id TEXT PRIMARY KEY, idem_key TEXT NOT NULL UNIQUE, body_digest TEXT NOT NULL,
		  endpoint TEXT NOT NULL, entrypoint TEXT NOT NULL, plan_id TEXT NOT NULL,
		  payload BLOB NOT NULL, outputs TEXT NOT NULL DEFAULT '', state TEXT NOT NULL,
		  ordinal INTEGER NOT NULL DEFAULT 0, requeues INTEGER NOT NULL DEFAULT 0,
		  created_at TEXT NOT NULL
		)`,
		`CREATE TABLE attempts (
		  request_id TEXT NOT NULL REFERENCES requests(id), attempt INTEGER NOT NULL,
		  attempt_key TEXT NOT NULL UNIQUE, instance_id TEXT NOT NULL, session_id TEXT NOT NULL,
		  invocation_digest TEXT NOT NULL, invocation BLOB NOT NULL, state TEXT NOT NULL,
		  plan_digest TEXT NOT NULL DEFAULT '', construction TEXT NOT NULL DEFAULT '',
		  plan_summary TEXT NOT NULL DEFAULT '', terminal_id TEXT NOT NULL DEFAULT '',
		  terminal_digest TEXT NOT NULL DEFAULT '', terminal_status TEXT NOT NULL DEFAULT '',
		  terminal_cause TEXT NOT NULL DEFAULT '', safe_message TEXT NOT NULL DEFAULT '',
		  triage_subject TEXT NOT NULL DEFAULT '', triage_digest TEXT NOT NULL DEFAULT '',
		  triage_length INTEGER NOT NULL DEFAULT 0, triage_path TEXT NOT NULL DEFAULT '',
		  terminal_body BLOB, dispatched_at TEXT NOT NULL, accepted_at TEXT NOT NULL DEFAULT '',
		  closed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (request_id, attempt)
		)`,
		`INSERT INTO requests(id,idem_key,body_digest,endpoint,entrypoint,plan_id,payload,
		  outputs,state,ordinal,requeues,created_at)
		  VALUES('req-legacy','idem-legacy','sha256:body','org/model','generate','plan',
		         '{}','image','queued',1,2,'2026-01-01T00:00:00Z')`,
		`INSERT INTO attempts(request_id,attempt,attempt_key,instance_id,session_id,
		  invocation_digest,invocation,state,terminal_digest,terminal_status,
		  dispatched_at,closed_at)
		  VALUES('req-legacy',1,'att-legacy','worker-legacy','boot-legacy',
		         'sha256:spec','spec','closed','sha256:terminal','ABANDONED',
		         '2026-01-01T00:00:00Z','2026-01-01T00:01:00Z')`,
	} {
		if _, err := legacy.Exec(stmt); err != nil {
			legacy.Close()
			t.Fatal(err)
		}
	}
	if err := legacy.Close(); err != nil {
		t.Fatal(err)
	}

	store, e := Open(path)
	if e != nil {
		t.Fatal(e)
	}
	defer store.Close()
	row, e := store.RequestRow("req-legacy")
	if e != nil || row == nil || row.State != "requeue_pending" || row.Requeues != 2 {
		t.Fatalf("normalized request = %#v, %v", row, e)
	}
	if owed, e := store.Owed(); e != nil || len(owed) != 0 {
		t.Fatalf("legacy request bypassed BeginRequeue before budget charge: %#v, %v", owed, e)
	}
	count, started, e := store.BeginRequeue("req-legacy", 3)
	if e != nil || !started || count != 3 {
		t.Fatalf("BeginRequeue = count %d, started %v, %v; want charged third requeue", count, started, e)
	}
	if owed, e := store.Owed(); e != nil || len(owed) != 1 || owed[0].ID != "req-legacy" {
		t.Fatalf("charged legacy request is not owed exactly once: %#v, %v", owed, e)
	}
	count, started, e = store.BeginRequeue("req-legacy", 3)
	if e != nil || started || count != 3 {
		t.Fatalf("second BeginRequeue = count %d, started %v, %v; want idempotent no-op", count, started, e)
	}
}
