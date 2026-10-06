package producttest

import (
	"database/sql"
	"encoding/json"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// Intent is the child's inputs; Computation is its operation identity, which an edit to
// its implementation also changes.
type machineChildProof struct {
	Request, State, Intent, Computation string
	Executions                          int
	Result                              []byte
}

// machineChildren reads a run's child calls from this computer's machine journal, only as
// evidence: the product tests submit, resume, observe and collect through the ordinary CLI.
// A child is its own run, `<parent>/<call index>`, in the caller's environment; Executions is
// how many attempts ran it (0 when a known result answered it), Intent its spec digest and
// Computation the memo computation of a memoized call, from the parent's call record.
func machineChildren(t *testing.T, root string, store *records.Store, parentRef string) []machineChildProof {
	t.Helper()
	parent, problem := store.RequestByReference(parentRef)
	fatal(t, problem)
	if parent == nil {
		t.Fatalf("no parent %s", parentRef)
	}
	children, problem := store.Children(parent.ID)
	fatal(t, problem)
	if len(children) != 0 {
		t.Fatal("Creator owns Runtime children")
	}
	db, err := sql.Open("sqlite", "file:"+rustJournal(root)+"?mode=ro&_pragma=busy_timeout(5000)")
	must(t, err)
	defer db.Close()
	computations := map[string]string{}
	calls, err := db.Query(`SELECT c.record FROM run_calls c JOIN executions p ON p.id=c.execution WHERE p.request_id=?`, parent.ID)
	must(t, err)
	for calls.Next() {
		var raw []byte
		must(t, calls.Scan(&raw))
		var call struct {
			Request     string `json:"request"`
			Computation string `json:"computation_digest"`
		}
		must(t, json.Unmarshal(raw, &call))
		computations[call.Request] = call.Computation
	}
	must(t, calls.Err())
	rows, err := db.Query(`SELECT request_id,record FROM executions WHERE request_id LIKE ? ORDER BY id`, parent.ID+"/%")
	must(t, err)
	defer rows.Close()
	type indexed struct {
		index int
		child machineChildProof
	}
	var found []indexed
	for rows.Next() {
		var request string
		var raw []byte
		must(t, rows.Scan(&request, &raw))
		index, err := strconv.Atoi(strings.TrimPrefix(request, parent.ID+"/"))
		if err != nil {
			continue // a grandchild, or another kind of run
		}
		var record struct {
			State      string `json:"state"`
			Attempt    int    `json:"attempt"`
			Submission struct {
				Invocation string `json:"invocation_digest"`
			} `json:"submission"`
			Result *struct {
				Value json.RawMessage `json:"value"`
			} `json:"result"`
		}
		must(t, json.Unmarshal(raw, &record))
		child := machineChildProof{Request: request, State: record.State, Intent: record.Submission.Invocation,
			Computation: computations[request], Executions: record.Attempt}
		if child.State == "completed" {
			child.State = "succeeded"
		}
		if record.Result != nil {
			child.Result = record.Result.Value
		}
		found = append(found, indexed{index, child})
	}
	must(t, rows.Err())
	sort.Slice(found, func(i, j int) bool { return found[i].index < found[j].index })
	out := make([]machineChildProof, len(found))
	for i := range found {
		out[i] = found[i].child
	}
	return out
}

// rustJournal is this computer's machine's execution journal.
func rustJournal(root string) string {
	return filepath.Join(root, "machine", "root", "var", "lib", "cozy", "rust-machine", "execution", "executions.sqlite3")
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
