package producttest

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// A memoizable run says in `cozy run show` how its machine's memo answered it: from a result
// the machine held (nothing ran), computed, or not memoized and why. A job's nested
// call says the same. A machine that predates the memo says nothing, and neither does show.
// The real store, Calcifer API and CLI, from the machine's outcome and call records.
func TestRunShowSaysHowTheMachineMemoAnsweredARun(t *testing.T) {
	o := hostOwner(t, "run-show-memo")
	computed, held := childDigest("c"), childDigest("d")
	record := func(id, kind, state string, outcome *v1.Outcome, calls ...*v1.Call) {
		t.Helper()
		_, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: id, Package: "local/memo", Entrypoint: "measure",
			Kind: kind, Payload: []byte(`{}`), BodyDigest: childDigest("b"), MachineExecutionObserver: true})
		fatal(t, problem)
		fatal(t, o.store.LinkMachineExecution(id, "pr-memo"))
		fatal(t, o.store.AcceptRunV1(id, "pr-memo", &v1.RunState{Id: id, Number: 1, State: state, Attempt: 1}))
		stamp := time.Now().UnixMilli()
		for i, call := range calls {
			call.Run, call.Index, call.Function, call.Status = fmt.Sprintf("%s/%d", id, i), uint32(i), "measure", "succeeded"
			call.CalledAtMs, call.FinishedAtMs = stamp-int64(10-i), stamp
			fatal(t, o.store.ObserveRunV1(id, &v1.RunEvent{Sequence: uint64(i + 1), AtMs: stamp,
				Event: &v1.RunEvent_Call{Call: call}}, nil))
		}
		outcome.Status, outcome.Result = "succeeded", []byte(`{"square":9}`)
		fatal(t, o.store.RecordRunOutcomeV1(id, records.RunEndV1{Outcome: outcome}))
	}
	// A memo hit runs nothing; the others ran.
	record("run-memo-served", "serving", "queued", &v1.Outcome{Memoized: true, ComputationDigest: held})
	record("run-memo-computed", "serving", "running", &v1.Outcome{ComputationDigest: computed, ExecutionMs: 40})
	record("run-memo-unmet", "serving", "running", &v1.Outcome{Unmemoized: "seed not given", ExecutionMs: 40})
	record("run-memo-older", "serving", "running", &v1.Outcome{ExecutionMs: 40})
	record("run-memo-calls", "serving", "running", &v1.Outcome{ExecutionMs: 40},
		&v1.Call{Unmemoized: "seed not given"}, &v1.Call{Memoized: true, ComputationDigest: held})
	defer publicationControlAPI(t, o)()

	type shown struct {
		Memoized       bool            `json:"memoized"`
		Computation    string          `json:"computation"`
		Unmemoized     string          `json:"unmemoized"`
		ExecutionMS    *int64          `json:"execution_ms"`
		ExecutionKnown bool            `json:"execution_known"`
		Result         json.RawMessage `json:"result"`
		Calls          []struct {
			Number      int    `json:"number"`
			Request     string `json:"request"`
			Status      string `json:"status"`
			Memoized    bool   `json:"memoized"`
			Computation string `json:"computation"`
			Unmemoized  string `json:"unmemoized"`
		} `json:"calls"`
	}
	show := func(id string) (shown, string) {
		t.Helper()
		code, raw := runCozy(t, o.root, "run", "show", id, "--json")
		var report shown
		if code != 0 || json.Unmarshal([]byte(raw), &report) != nil || string(report.Result) != `{"square":9}` {
			t.Fatalf("run show %s --json [%d]: %s", id, code, raw)
		}
		code, text := runCozy(t, o.root, "run", "show", id)
		if code != 0 {
			t.Fatalf("run show %s [%d]: %s", id, code, text)
		}
		return report, text
	}
	memoLine := regexp.MustCompile(`(?m)^memo: .*$`)
	for _, want := range []struct {
		id, memo string
		report   shown
	}{
		{"run-memo-served", "memo: served from this machine's memo (nothing ran) · " + held, shown{Memoized: true, Computation: held}},
		{"run-memo-computed", "memo: computed · " + computed, shown{Computation: computed}},
		{"run-memo-unmet", "memo: not memoized: seed not given", shown{Unmemoized: "seed not given"}},
		{"run-memo-older", "", shown{}},
	} {
		report, text := show(want.id)
		if report.Memoized != want.report.Memoized || report.Computation != want.report.Computation ||
			report.Unmemoized != want.report.Unmemoized || memoLine.FindString(text) != want.memo {
			t.Fatalf("%s: want %q and %+v, show says:\n%+v\n%s", want.id, want.memo, want.report, report, text)
		}
		// Nothing ran: a memo hit's execution is known, and none.
		if report.Memoized && (!report.ExecutionKnown || report.ExecutionMS == nil || *report.ExecutionMS != 0) {
			t.Fatalf("a memo hit's execution is not known as none: %+v", report)
		}
	}

	report, text := show("run-memo-calls")
	byRequest := map[string]int{}
	for i, call := range report.Calls {
		byRequest[call.Request] = i
	}
	unmet, reused := report.Calls[byRequest["run-memo-calls/0"]], report.Calls[byRequest["run-memo-calls/1"]]
	if report.Memoized || report.Computation != "" || report.Unmemoized != "" || memoLine.MatchString(text) ||
		unmet.Unmemoized != "seed not given" || unmet.Memoized || reused.Unmemoized != "" || !reused.Memoized || reused.Computation != held ||
		!strings.Contains(text, fmt.Sprintf("  call %d not memoized: seed not given\n", unmet.Number)) || !strings.Contains(text, "reused (memo)") {
		t.Fatalf("show does not say how each call was answered:\n%+v\n%s", report, text)
	}
	for number, memo := range map[int]string{unmet.Number: "memo: not memoized: seed not given",
		reused.Number: "memo: served from this machine's memo (nothing ran) · " + held} {
		code, call := runCozy(t, o.root, "run", "show", "run-memo-calls", "--call", fmt.Sprint(number))
		if code != 0 || memoLine.FindString(call) != memo {
			t.Fatalf("run show --call %d [%d] does not say %q:\n%s", number, code, memo, call)
		}
	}
}
