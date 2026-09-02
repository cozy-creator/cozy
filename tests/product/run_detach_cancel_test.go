package producttest

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

// TestClientDeathNeverCancels is cl-108's ruling as behaviour: an ACCEPTED run belongs to
// the daemon, not to the client that submitted it. A watcher that dies — SIGKILL, or the
// SIGTERM an agent harness sends — merely detaches; the run completes, its export
// publishes, and it stays watchable. Only an explicit `cozy run cancel` cancels, every
// cancellation records its actor durably, and a canceled run renders LOUDLY with that
// actor in watch and list — never as a quiet no-output ending. The incident's own shape
// (submit, client dies, resubmit with different args) yields two completed runs.
func TestClientDeathNeverCancels(t *testing.T) {
	root, err := os.MkdirTemp(os.TempDir(), "cozy-product-")
	must(t, err)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		_ = os.RemoveAll(root)
	})

	project := weightlessProject(t)
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("editable install failed [exit %d]\n%s", code, out)
	}
	// Warm the worker once so delay_ms dominates every later arm's runtime.
	if code, out := runCozy(t, root, "--json", "run", localWeightlessRef+"/tile",
		"size=8", "seed=20", "--await"); code != 0 || !strings.Contains(out, `"status":"completed"`) {
		t.Fatalf("warmup run failed [exit %d]\n%s\n%s", code, out, productWorkerLogs(root))
	}

	// ------------------------------------------------ the incident's shape, survived
	// Submit --await, SIGKILL the client the moment the daemon owns the run, resubmit
	// with different args: BOTH runs complete. This is req-d0144ed0's exact sequence,
	// which used to end CLIENT_CANCELED with a skipped export.
	before := invocationIDSet(t, root)
	killed, kout, kerr := startCozyDetachArm(t, root, "--json", "run", localWeightlessRef+"/tile",
		"size=32", "seed=21", "delay_ms=4500", "--await")
	victim := awaitNewInvocation(t, root, before, kout, kerr)
	must(t, killed.Process.Kill())
	_ = killed.Wait()
	if code, out := runCozy(t, root, "--json", "run", localWeightlessRef+"/tile",
		"size=32", "seed=22", "delay_ms=200", "--await"); code != 0 ||
		!strings.Contains(out, `"status":"completed"`) {
		t.Fatalf("the resubmission after a client death did not complete [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "watch", victim.Number, "--json"); code != 0 ||
		!strings.Contains(out, `"status":"completed"`) || !strings.Contains(out, `"saved":[{`) {
		t.Fatalf("the killed client's run must complete and publish its export [exit %d]\n%s\n%s",
			code, out, productWorkerLogs(root))
	}

	// ------------------------------------------------ SIGTERM detaches, never cancels
	// This is the disconnect that used to be silently translated into a cancel. The CLI
	// now says it detached, exits cleanly, and the run runs on to completion.
	before = invocationIDSet(t, root)
	graceful, gout, gerr := startCozyDetachArm(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "seed=23", "delay_ms=4500", "--await")
	surviving := awaitNewInvocation(t, root, before, gout, gerr)
	awaitStderrProgress(t, gerr) // the watcher is attached: its signal handler stands
	must(t, graceful.Process.Signal(syscall.SIGTERM))
	_ = graceful.Wait()
	if code := graceful.ProcessState.ExitCode(); code != 0 {
		t.Fatalf("a SIGTERMed --await client must detach cleanly [exit %d]\nstdout:\n%s\nstderr:\n%s",
			code, gout.String(), gerr.String())
	}
	if !strings.Contains(gerr.String(), "detached — the run keeps running") ||
		strings.Contains(gerr.String(), "cancel requested") {
		t.Fatalf("SIGTERM must detach loudly and must not request cancellation\nstderr:\n%s", gerr.String())
	}
	awaitInvocationStatus(t, root, surviving.ID, "completed")

	// ------------------------------------------------ explicit cancel: live and queued
	// The LIVE path: a real running attempt, canceled explicitly; the attempt's own
	// journaled terminal settles it, attributed to its actor.
	code, dout, _ := runCozyStreams(t, root, "--json", "run", localWeightlessRef+"/tile",
		"size=32", "seed=24", "delay_ms=4500")
	dref := submittedRunReference(t, code, dout)
	awaitRunReferenceStatus(t, root, dref, "in_progress")
	if code, out := runCozy(t, root, "run", "cancel", dref, "--json"); code != 0 ||
		!strings.Contains(out, `"status":"canceled"`) ||
		!strings.Contains(out, `"canceled_by":"cozy run cancel"`) {
		t.Fatalf("live cancel was not attributed [exit %d]\n%s", code, out)
	}

	// The QUEUED path: the incident's exact pre-attempt state — a request parked with no
	// dispatchable worker — canceled explicitly and settled with its actor recorded.
	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	defer store.Close()
	const parkedID = "req-cl108-parked"
	if _, _, problem := store.Submit(records.Request{
		ID: parkedID, IdemKey: "idem-cl108-parked",
		BodyDigest: "sha256:" + strings.Repeat("ab", 32),
		Package:    "fake/parked", Entrypoint: "generate", Payload: []byte("{}"),
	}); problem != nil {
		t.Fatal(problem.Message)
	}
	if code, out := runCozy(t, root, "run", "cancel", parkedID, "--json"); code != 0 ||
		!strings.Contains(out, `"status":"canceled"`) ||
		!strings.Contains(out, `"canceled_by":"cozy run cancel"`) {
		t.Fatalf("queued cancel was not attributed [exit %d]\n%s", code, out)
	}

	// A canceled run renders LOUDLY with its actor: in watch (the operational error exit)…
	for _, ref := range []string{dref, parkedID} {
		code, out := runCozy(t, root, "run", "watch", ref)
		if code != 1 || !strings.Contains(out, "was canceled by cozy run cancel") {
			t.Fatalf("watch of canceled run %s is not loud about its actor [exit %d]\n%s", ref, code, out)
		}
	}
	// …and the queued settlement still says it never ran.
	if code, out := runCozy(t, root, "run", "watch", parkedID); code != 1 ||
		!strings.Contains(out, "before any attempt was dispatched") {
		t.Fatalf("the queued cancellation lost its pre-attempt cause [exit %d]\n%s", code, out)
	}
	// …and in list, where the incident read as a quiet no-output end.
	if code, out := runCozy(t, root, "run", "list"); code != 0 ||
		!strings.Contains(out, "canceled by cozy run cancel") {
		t.Fatalf("run list does not carry the cancellation cause [exit %d]\n%s", code, out)
	}

	// The attribution is DURABLE: the daemon's own records answer who canceled, for the
	// queued settlement and for the live attempt's cancel request alike.
	for _, ref := range []string{dref, parkedID} {
		row, problem := store.RequestByReference(ref)
		fatal(t, problem)
		if row == nil {
			t.Fatalf("canceled run %s has no request row", ref)
		}
		actor, _, _, problem := store.CancelAttribution(row.ID)
		fatal(t, problem)
		if actor != "cozy run cancel" {
			t.Fatalf("request %s records cancel actor %q, want %q", row.ID, actor, "cozy run cancel")
		}
	}
}

type listedInvocation struct {
	Number string `json:"number"`
	ID     string `json:"id"`
	Status string `json:"status"`
}

func listInvocations(t *testing.T, root string) []listedInvocation {
	t.Helper()
	code, out := runCozy(t, root, "run", "list", "--json", "--full")
	if code != 0 {
		t.Fatalf("run list failed [exit %d]\n%s", code, out)
	}
	var document struct {
		Invocations []listedInvocation `json:"invocations"`
	}
	if err := json.Unmarshal([]byte(out), &document); err != nil {
		t.Fatalf("run list returned invalid JSON: %v\n%s", err, out)
	}
	return document.Invocations
}

func invocationIDSet(t *testing.T, root string) map[string]bool {
	t.Helper()
	seen := map[string]bool{}
	for _, row := range listInvocations(t, root) {
		seen[row.ID] = true
	}
	return seen
}

// awaitNewInvocation waits until the daemon owns a run that was not there before —
// the moment the submission's fate no longer depends on its client staying alive.
func awaitNewInvocation(t *testing.T, root string, before map[string]bool,
	stdout, stderr *renderBuffer) listedInvocation {
	t.Helper()
	deadline := time.Now().Add(120 * time.Second)
	for time.Now().Before(deadline) {
		for _, row := range listInvocations(t, root) {
			if !before[row.ID] {
				return row
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("no new invocation appeared\nclient stdout:\n%s\nclient stderr:\n%s\n%s",
		stdout.String(), stderr.String(), productWorkerLogs(root))
	return listedInvocation{}
}

func awaitInvocationStatus(t *testing.T, root, id, wanted string) {
	t.Helper()
	deadline := time.Now().Add(180 * time.Second)
	status := ""
	for time.Now().Before(deadline) {
		for _, row := range listInvocations(t, root) {
			if row.ID != id {
				continue
			}
			status = row.Status
			if status == wanted {
				return
			}
			if strings.HasPrefix(status, "failed") || strings.HasPrefix(status, "canceled") {
				t.Fatalf("run %s settled %q, wanted %q\n%s", id, status, wanted, productWorkerLogs(root))
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("run %s stayed %q, wanted %q\n%s", id, status, wanted, productWorkerLogs(root))
}

func awaitRunReferenceStatus(t *testing.T, root, reference, wanted string) {
	t.Helper()
	deadline := time.Now().Add(120 * time.Second)
	status := ""
	for time.Now().Before(deadline) {
		for _, row := range listInvocations(t, root) {
			if row.Number != reference && row.ID != reference {
				continue
			}
			status = row.Status
			if status == wanted {
				return
			}
			if strings.HasPrefix(status, "completed") || strings.HasPrefix(status, "failed") ||
				strings.HasPrefix(status, "canceled") {
				t.Fatalf("run %s settled %q, wanted %q\n%s", reference, status, wanted,
					productWorkerLogs(root))
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("run %s stayed %q, wanted %q\n%s", reference, status, wanted, productWorkerLogs(root))
}

func submittedRunReference(t *testing.T, code int, stdout string) string {
	t.Helper()
	if code != 0 {
		t.Fatalf("detached submission failed [exit %d]\n%s", code, stdout)
	}
	match := regexp.MustCompile(`"run":"([^"]+)"`).FindStringSubmatch(stdout)
	if match == nil {
		t.Fatalf("detached submission printed no run reference\n%s", stdout)
	}
	return match[1]
}

// startCozyDetachArm launches the real CLI as its own process, with race-safe capture
// buffers, so a test can kill it exactly the way a terminal close or agent harness does.
func startCozyDetachArm(t *testing.T, root string, args ...string) (*exec.Cmd, *renderBuffer, *renderBuffer) {
	t.Helper()
	cmd := exec.Command(cozyBin, args...)
	cmd.Env = childEnv(t, root)
	stdout, stderr := &renderBuffer{}, &renderBuffer{}
	cmd.Stdout, cmd.Stderr = stdout, stderr
	must(t, cmd.Start())
	return cmd, stdout, stderr
}

// awaitStderrProgress waits for the watcher's own progress lane to speak — proof the
// event stream is attached and the detach signal handler is standing.
func awaitStderrProgress(t *testing.T, stderr *renderBuffer) {
	t.Helper()
	deadline := time.Now().Add(120 * time.Second)
	for time.Now().Before(deadline) {
		if strings.Contains(stderr.String(), "%") {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("the awaiting client never rendered progress\nstderr:\n%s", stderr.String())
}
