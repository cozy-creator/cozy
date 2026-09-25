package producttest

import (
	"database/sql"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type machineChildProof struct {
	Request, State, Intent, Revision, Computation string
	Executions                                    int
	Result                                        []byte
}

// Read the actual Runtime journal only as evidence. These product tests submit,
// resume, observe and collect through the ordinary CLI, with no Creator children.
func machineChildren(t *testing.T, root string, store *records.Store, parentRef string) []machineChildProof {
	t.Helper()
	parent, problem := store.RequestByReference(parentRef)
	fatal(t, problem)
	if parent == nil {
		t.Fatalf("no parent %s", parentRef)
	}
	children, problem := store.Children(parent.ID)
	fatal(t, problem)
	attempts, problem := store.Attempts(parent.ID)
	fatal(t, problem)
	if len(children) != 0 || len(attempts) != 0 {
		t.Fatal("Creator owns Runtime attempts or children")
	}
	db, err := sql.Open("sqlite", "file:"+filepath.Join(root, "tensorfs", ".cozy-workspace", "journal.sqlite3")+"?mode=ro")
	must(t, err)
	defer db.Close()
	db.SetMaxOpenConns(1)
	_, err = db.Exec("PRAGMA busy_timeout=5000")
	must(t, err)
	rows, err := db.Query(`SELECT c.child_request,COALESCE(e.state,'succeeded'),hex(c.intent_digest),c.prepared,c.result,
 (SELECT count(*) FROM attempts a WHERE a.owner=c.owner AND a.request=c.child_request),
 COALESCE((SELECT a.outcome FROM attempts a WHERE a.owner=c.owner AND a.request=c.child_request ORDER BY ordinal DESC LIMIT 1),x'')
 FROM execution_calls c LEFT JOIN executions e ON e.owner=c.owner AND e.request=c.child_request
 WHERE c.parent_request=? ORDER BY c.call_index`, parent.ID)
	must(t, err)
	defer rows.Close()
	var found []machineChildProof
	for rows.Next() {
		var child machineChildProof
		var prepared, outcome []byte
		must(t, rows.Scan(&child.Request, &child.State, &child.Intent, &prepared, &child.Result, &child.Executions, &outcome))
		var plan struct {
			Revision    string `json:"revision"`
			Computation string `json:"computation"`
		}
		must(t, json.Unmarshal(prepared, &plan))
		child.Revision, child.Computation = plan.Revision, plan.Computation
		if len(child.Result) == 0 && len(outcome) > 0 {
			var body pb.AttemptOutcomeBody
			must(t, canonical.Unmarshal(outcome, &body))
			child.Result = body.GetResult().GetInlineResult()
		}
		found = append(found, child)
	}
	must(t, rows.Err())
	return found
}

func assertMachineChildScalar(t *testing.T, child machineChildProof, value int) {
	t.Helper()
	var result struct {
		Value int `json:"value"`
	}
	must(t, json.Unmarshal(child.Result, &result))
	if result.Value != value {
		t.Fatalf("Runtime child returned %d, want %d", result.Value, value)
	}
}
