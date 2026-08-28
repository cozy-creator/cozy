package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/records"
)

// THE ARTIFACT CRASH MATRIX (cl-023).
//
// `kill -9` the record owner at each edge of the artifact transaction and ask the durable
// root what survived. The owner runs in a CHILD of this driver — the same trick apicrash
// plays with `cozy up`, minus the service, the store, and the card — so the observer
// outlives the crash and reads the SQLite root the dead process left behind.
//
// The four edges, and the one question each answers:
//
//	before-intent        the outcome never committed        — is an intent ever invented?
//	after-intent         the intent committed, unsent       — does the SAME decision survive?
//	during-disposition   the request was sent, unanswered   — does the replay stay identical?
//	after-result         one exact result journaled         — is a completion ever redone?
//
// After each kill the root is REOPENED BY A SECOND RECORD OWNER on the same directory,
// because "it converged" is a claim about the restart, not about the corpse.
//
// LOCAL LIMIT, stated rather than hidden: a local worker cannot outlive its orchestrator
// (Reconcile reaps every orphan at boot, and the peercred slot fence refuses to adopt a
// process this owner did not start). So the restarted owner cannot deliver the replayed
// ArtifactFinalizeRequest to the worker that owns the transaction — the release itself is
// the runtime's act, and proving it end to end needs cr-030's real rev-5 runtime peer.
// What IS proved here is Creator's whole half: one durable first-wins decision per slot,
// one scratch root, byte-identical replay, and nothing public at any edge.

type crashEdge struct {
	stage  string
	arm    string
	slots  []string
	marker string // the orchestrator log line that says this edge has been reached
	why    string
}

var artifactEdges = []crashEdge{
	{"before-intent", "idle", []string{"model"}, "AttemptAccepted",
		"the attempt is accepted and no outcome has arrived"},
	{"after-intent", "artifacthold", []string{"model"}, "durably recorded 1 artifact receipt(s)",
		"the receipt and the intent are committed, the request is not yet answered"},
	{"during-disposition", "artifacthold", []string{"model"}, "ArtifactFinalizeRequest",
		"the exact decision is on the wire and the runtime is mid-disposition"},
	{"after-result", "artifacthalf", []string{"model", "tokenizer"},
		"persisted before outcome ack", "one exact result is journaled, the outcome is unacked"},
}

func sectionArtifactCrash() {
	if stage := flag("stage", ""); stage != "" {
		artifactCrashChild(stage)
		return
	}
	for _, edge := range artifactEdges {
		artifactCrashEdge(edge)
	}
}

func artifactCrashEdge(edge crashEdge) {
	head("kill -9 the record owner " + edge.stage + " — " + edge.why)
	root := filepath.Join(os.TempDir(), "cozy-live", "cl023-crash-"+edge.stage)
	must("clearing the crash root", os.RemoveAll(root))
	must("creating the crash root", os.MkdirAll(root, 0o755))

	child, logPath := startCrashChild(edge.stage, root, true)
	line, reached := waitLine(logPath, "EDGE "+edge.stage, 90*time.Second)
	if !check("the owner reached the edge", reached, strings.TrimSpace(line)) {
		_ = killGroup(child.Process.Pid, syscall.SIGKILL)
		fmt.Println(indent(tail(logPath, 20)))
		return
	}
	requestID := strings.TrimSpace(field(readAll(logPath), "request"))
	_ = killGroup(child.Process.Pid, syscall.SIGKILL)
	_ = child.Wait()
	check("the record owner is gone, mid-transaction", child.ProcessState != nil &&
		!child.ProcessState.Success(), fmt.Sprint(child.ProcessState))

	before := readArtifactRoot(root, requestID)
	artifactCrashClaims(edge, before, "after the kill")

	// THE RESTART. A second record owner on the same directory, reconciling the orphaned
	// worker exactly as `cozy up` does after a crash.
	restart, restartLog := startCrashChild("recover", root, false)
	_, up := waitLine(restartLog, "EDGE recover", 60*time.Second)
	check("a SECOND record owner comes up on the same root", up, "")
	_ = restart.Wait()

	after := readArtifactRoot(root, requestID)
	artifactCrashClaims(edge, after, "after the restart")
	check("the restart minted no second decision and changed no byte of the first",
		sameDecisions(before, after), decisionBrief(after))
	check("and one scratch root per slot survived the crash, Creator's own derivation",
		sameRoots(requestID, after), rootBrief(after.finals))
}

// artifactState is everything the durable root says about one job's artifact transaction.
type artifactState struct {
	requestID string
	attempt   *records.Attempt
	receipts  []records.ArtifactReceipt
	finals    []records.ArtifactFinalization
	outputs   int
	published bool
	entries   []string
}

func readArtifactRoot(root, requestID string) artifactState {
	layout, e := home.Open(root)
	must("reopening the crashed layout", errOf(e))
	store, e := records.Open(layout.DB)
	must("reopening the crashed records root", errOf(e))
	defer store.Close()

	state := artifactState{requestID: requestID}
	attempts, _ := store.Attempts(requestID)
	if len(attempts) > 0 {
		row := attempts[len(attempts)-1]
		state.attempt = &row
		state.receipts, _ = store.ArtifactReceiptsOf(requestID, row.Attempt)
	}
	state.finals, _ = store.ArtifactFinalizationsOf(requestID)
	outs, _ := store.VisibleOutputs(requestID)
	state.outputs = len(outs)
	if pub, _ := store.PublicationOf(requestID); pub != nil {
		state.published = true
	}
	rows, err := os.ReadDir(layout.PublicationRoot("local", requestID))
	if err == nil {
		for _, row := range rows {
			if !strings.HasPrefix(row.Name(), ".") {
				state.entries = append(state.entries, row.Name())
			}
		}
	}
	return state
}

// artifactCrashClaims is the SAME set of questions at every edge and on both sides of the
// restart: what the journal holds, and that nothing became public at any of them.
func artifactCrashClaims(edge crashEdge, s artifactState, when string) {
	slots := len(edge.slots)
	switch edge.stage {
	case "before-intent":
		check("no receipt and no intent exist "+when,
			len(s.receipts) == 0 && len(s.finals) == 0,
			fmt.Sprintf("%d receipt(s), %s", len(s.receipts), finalizationBrief(s.finals)))
		check("and the attempt is still an OPEN obligation, not a terminal "+when,
			s.attempt != nil && s.attempt.TerminalID == "",
			fmt.Sprintf("state %q", stateOf(s.attempt)))
	case "after-intent", "during-disposition":
		check("the exact receipt and ONE ADOPT intent are durable "+when,
			len(s.receipts) == slots && len(s.finals) == slots && allAdopt(s.finals),
			fmt.Sprintf("%d receipt(s), %s", len(s.receipts), finalizationBrief(s.finals)))
		check("no result was journaled — the disposition is still owed "+when,
			pendingCount(s.finals) == slots, finalizationBrief(s.finals))
		check("and the attempt is TERMINAL but not CLOSED: the ack is owed too "+when,
			s.attempt != nil && s.attempt.State == "terminal",
			fmt.Sprintf("state %q", stateOf(s.attempt)))
	case "after-result":
		done := slots - pendingCount(s.finals)
		check("one slot's exact result is journaled and the other is still owed "+when,
			len(s.finals) == slots && done == 1, finalizationBrief(s.finals))
		check("the completed slot names its exact receipt as evidence "+when,
			completedEvidence(s.finals), finalizationBrief(s.finals))
		check("and the attempt is TERMINAL, never acknowledged behind an open slot "+when,
			s.attempt != nil && s.attempt.State == "terminal",
			fmt.Sprintf("state %q", stateOf(s.attempt)))
	}
	check("NOTHING is public "+when+": no output row, no publication, no addressable entry",
		s.outputs == 0 && !s.published && len(s.entries) == 0,
		fmt.Sprintf("%d output(s), publication %v, entries %v", s.outputs, s.published, s.entries))
}

func stateOf(a *records.Attempt) string {
	if a == nil {
		return "no attempt row"
	}
	return a.State
}

func allAdopt(rows []records.ArtifactFinalization) bool {
	for _, f := range rows {
		if f.Disposition != "ADOPT" || f.ScratchRootID == "" || f.ReceiptDigest == "" {
			return false
		}
	}
	return len(rows) > 0
}

func pendingCount(rows []records.ArtifactFinalization) int {
	n := 0
	for _, f := range rows {
		if f.ResultDigest == "" {
			n++
		}
	}
	return n
}

func completedEvidence(rows []records.ArtifactFinalization) bool {
	for _, f := range rows {
		if f.ResultDigest != "" {
			return f.ResultOutcome == "ADOPTED" && f.ResultReceiptDigest == f.ReceiptDigest &&
				f.CompletedAt != ""
		}
	}
	return false
}

// sameDecisions is the replay claim: the restarted owner would send the IDENTICAL request,
// because the decision digest and its exact bytes did not move.
func sameDecisions(before, after artifactState) bool {
	if len(before.finals) != len(after.finals) {
		return false
	}
	for i := range before.finals {
		a, b := before.finals[i], after.finals[i]
		if a.OutputSlot != b.OutputSlot || a.DecisionDigest != b.DecisionDigest ||
			string(a.DecisionBytes) != string(b.DecisionBytes) || a.Disposition != b.Disposition ||
			a.ScratchRootID != b.ScratchRootID || a.ResultDigest != b.ResultDigest {
			return false
		}
	}
	return true
}

func sameRoots(requestID string, s artifactState) bool {
	for _, f := range s.finals {
		if f.ScratchRootID != derivedScratchRoot(requestID, f.OutputSlot) {
			return false
		}
	}
	return true
}

func decisionBrief(s artifactState) string {
	out := make([]string, 0, len(s.finals))
	for _, f := range s.finals {
		out = append(out, fmt.Sprintf("%s=%s", f.OutputSlot, shortSHA(f.DecisionDigest)))
	}
	if len(out) == 0 {
		return "no decision rows"
	}
	return strings.Join(out, " ")
}

// ---------------------------------------------------------------- the child record owner

// startCrashChild runs THIS binary as a second process hosting the real orchestrator on
// `root`. It is its own process group, so the kill lands on the owner and leaves the
// worker for restart reconciliation to reap — which is what a real crash does.
func startCrashChild(stage, root string, fresh bool) (*exec.Cmd, string) {
	self, err := os.Executable()
	must("locating this binary", err)
	// OUTSIDE the root: a child that clears its own root would unlink the log the parent
	// is watching.
	logPath := root + "-owner-" + stage + ".log"
	file, err := os.Create(logPath)
	must("opening the child owner log", err)
	cmd := exec.Command("/usr/bin/nice", "-n", "19", self, "artifactcrash",
		"--stage", stage, "--home", root, "--fresh", fmt.Sprint(fresh))
	cmd.Stdout, cmd.Stderr = file, file
	setProcessGroup(cmd)
	must("starting the child record owner", cmd.Start())
	return cmd, logPath
}

// artifactCrashChild is the record owner that is about to die. It drives one artifact job
// to the named edge, prints the marker the parent is watching for, and then does nothing —
// there is no graceful path out of this role.
func artifactCrashChild(stage string) {
	lv := hostCoordinator("cl023-crash", flag("fresh", "") == "true")
	if stage == "recover" {
		// The restart: `Reconcile` already ran inside hostCoordinator and reaped the
		// orphaned worker. Nothing else is asked of it — the claim is about the journal.
		fmt.Println("EDGE recover")
		os.Stdout.Sync()
		time.Sleep(time.Second)
		lv.close()
		os.Exit(0)
	}
	edge := crashEdge{}
	for _, candidate := range artifactEdges {
		if candidate.stage == stage {
			edge = candidate
		}
	}
	if edge.stage == "" {
		fmt.Fprintf(os.Stderr, "cozy-live: unknown artifact crash stage %q\n", stage)
		os.Exit(2)
	}
	planID, ok := artifactWorker(lv, "crash"+stage, "0", edge.arm)
	if !ok {
		os.Exit(1)
	}
	sub := artifactSubmission(planID, "fake/crash"+stage, "cl023-crash-"+stage, edge.slots...)
	requestID, _, e := lv.c.Submit(sub)
	if e != nil {
		fmt.Fprintf(os.Stderr, "cozy-live: the crash job was not dispatched: %s\n", e.Message)
		os.Exit(1)
	}
	fmt.Printf("request: %s\n", requestID)
	os.Stdout.Sync()
	if _, seen := waitEvent(lv, edge.marker, 60*time.Second); !seen {
		fmt.Fprintf(os.Stderr, "cozy-live: the %s edge was never reached\n", stage)
		os.Exit(1)
	}
	// The journal is at the edge. Say so and wait to be killed.
	fmt.Printf("EDGE %s\n", stage)
	os.Stdout.Sync()
	select {}
}

// ------------------------------------------------------------------------- log watching

func readAll(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(data)
}

func waitLine(path, substr string, timeout time.Duration) (string, bool) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		for _, line := range strings.Split(readAll(path), "\n") {
			if strings.Contains(line, substr) {
				return line, true
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	return "", false
}
