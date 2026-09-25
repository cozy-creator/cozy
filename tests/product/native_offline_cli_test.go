package producttest

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/records"
)

var nativeRecoveryHome = flag.String("native-recovery-home", "", "new isolated native recovery proof home to retain, including journals and captured scripts")

// An observed barrier separates submission from production: B and C must finish
// while Creator is demonstrably absent. All execution and collection use cozy.
func TestNativeCompositionCompletesWithClientOffline(t *testing.T) {
	root := *nativeRecoveryHome
	var err error
	if root == "" {
		root, err = os.MkdirTemp("", "cozy-offline-")
	} else {
		root, err = filepath.Abs(root)
		must(t, err)
		err = os.Mkdir(root, 0700) // Existing homes are never reused or removed.
	}
	must(t, err)
	control := filepath.Join(root, "control")
	uv := func(args ...string) {
		t.Helper()
		cmd := exec.Command("uv", args...)
		cmd.Env = childEnv(t, root)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("uv: %v\n%s", err, out)
		}
	}
	uv("venv", control, "--python", "3.12")
	runtime := "cozy-runtime>=" + hostruntime.ToolFloor
	if *privateChildRuntimeWheel != "" {
		runtime = *privateChildRuntimeWheel
	}
	install := []string{"pip", "install", "--python", filepath.Join(control, "bin", "python"), runtime}
	if *privateChildTensorFSWheel != "" {
		install = append(install, *privateChildTensorFSWheel)
	}
	uv(install...)
	path := filepath.Join(control, "bin")
	for _, env := range childEnv(t, root) {
		if strings.HasPrefix(env, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(env, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if t.Failed() || *nativeRecoveryHome != "" {
			t.Log("native offline evidence retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	run := func(args ...string) string {
		t.Helper()
		code, out := runCozyPath(t, root, path, append(args, "--json")...)
		if code != 0 {
			t.Fatalf("cozy %v [%d]: %s", args, code, out)
		}
		return out
	}
	entered, release := make(chan struct{}), make(chan struct{})
	var arrived, released sync.Once
	var calls atomic.Int32
	unblock := func() { released.Do(func() { close(release) }) }
	barrier := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		arrived.Do(func() { close(entered) })
		select {
		case <-release:
			_, _ = w.Write([]byte("continue"))
		case <-r.Context().Done():
		}
	}))
	defer func() {
		unblock()
		barrier.Close()
	}()
	// The barrier belongs to the trusted test Runtime, not package Python. Package
	// networking remains forbidden. Pause the real native source read for B;
	// after Creator is offline, allow that read and C's subsequent read through.
	wrapper := fmt.Sprintf(`#!%s
import sys
if "serve" in sys.argv:
    import urllib.request
    import tensorfs.derived as derived
    native = derived.serve_derived
    class ObservedWriter:
        def __init__(self, writer): self.writer = writer
        def __getattr__(self, name): return getattr(self.writer, name)
        def source_read_into(self, *args):
            with urllib.request.urlopen(%q, timeout=120) as response:
                response.read()
            return self.writer.source_read_into(*args)
    def observed(writer, *args, **kwargs):
        return native(ObservedWriter(writer), *args, **kwargs)
    derived.serve_derived = observed
from cozy_runtime.cli.main import main
sys.exit(main())
`, filepath.Join(control, "bin", "python"), barrier.URL)
	must(t, os.WriteFile(filepath.Join(control, "bin", "cozy-runtime"), []byte(wrapper), 0700))
	project := copyPrivateTensorProject(t, root, "")
	script := filepath.Join(project, "recipe.py")
	source, err := os.ReadFile(script)
	must(t, err)
	source = []byte(strings.Replace(string(source), "async def main():", "from cozy_runtime.author import ModelArtifact\n\nasync def main() -> ModelArtifact:", 1))
	source = []byte(strings.Replace(string(source), "    await candidate(source=original, factor=0)", "    prepared = await candidate(source=original, factor=2)\n    return await candidate(source=prepared, factor=3)", 1))
	must(t, os.WriteFile(script, source, 0600))
	accepted := run("run", script, "--idempotency-key", "native-offline")
	if !strings.Contains(accepted, `"machine_accepted":true`) {
		t.Fatalf("no durable machine acceptance: %s", accepted)
	}
	select {
	case <-entered:
	case <-time.After(2 * time.Minute):
		t.Fatal("native candidate never reached the controlled boundary")
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	parent, problem := store.RequestByIdempotencyKey("native-offline")
	fatal(t, problem)
	if parent == nil {
		t.Fatal("accepted request absent")
	}
	before, problem := store.MachineExecution(parent.ID)
	fatal(t, problem)
	if before == nil || len(before.Receipt) == 0 || before.Collected {
		t.Fatal("lost live accepted receipt")
	}
	first := machineChildren(t, root, store, "1")
	if len(first) != 2 || first[0].State != "succeeded" || first[0].Executions != 1 {
		t.Fatalf("unexpected before-disconnect work: %+v", first)
	}
	identity, err := os.ReadFile(filepath.Join(root, "runtime", "process.json"))
	must(t, err)
	run("down")
	if daemonOnRoot(root) != 0 {
		t.Fatal("Creator remains online")
	}
	unblock()
	journal, err := sql.Open("sqlite", "file:"+filepath.Join(root, "tensorfs", ".cozy-workspace", "journal.sqlite3")+"?mode=ro")
	must(t, err)
	defer journal.Close()
	journal.SetMaxOpenConns(1)
	_, err = journal.Exec("PRAGMA busy_timeout=5000")
	must(t, err)
	var state string
	var collected int
	for deadline := time.Now().Add(2 * time.Minute); ; {
		must(t, journal.QueryRow("SELECT state,collected FROM executions WHERE request=?", parent.ID).Scan(&state, &collected))
		if daemonOnRoot(root) != 0 {
			t.Fatal("Creator returned before independent completion")
		}
		if state == "succeeded" {
			break
		}
		if state == "failed" || state == "canceled" || time.Now().After(deadline) {
			t.Fatalf("offline root did not succeed: state=%s collected=%d", state, collected)
		}
		time.Sleep(50 * time.Millisecond)
	}
	if collected != 0 {
		t.Fatal("result acknowledged without a client")
	}
	offline := machineChildren(t, root, store, "1")
	if len(offline) != 3 {
		t.Fatalf("offline children: %+v", offline)
	}
	for _, child := range offline {
		if child.State != "succeeded" || child.Executions != 1 || len(child.Result) == 0 {
			t.Fatalf("offline production: %+v", child)
		}
	}
	artifact, problem := records.DecodeModelArtifact(offline[2].Result)
	fatal(t, problem)
	if artifact == nil {
		t.Fatal("final native artifact absent")
	}
	check := exec.Command(filepath.Join(control, "bin", "python"), filepath.Join("testdata", "private_child_read.py"), artifact.Manifest.Digest, strconv.Itoa(42), filepath.Join(root, "tensorfs"))
	check.Env = childEnv(t, root, "PATH="+path)
	verified, err := check.CombinedOutput()
	if err != nil {
		t.Fatalf("native result bytes: %v %s", err, verified)
	}
	run("run", "watch", parent.ID)
	after, problem := store.MachineExecution(parent.ID)
	fatal(t, problem)
	nowIdentity, err := os.ReadFile(filepath.Join(root, "runtime", "process.json"))
	must(t, err)
	if !after.Collected || after.CancelRequested || !bytes.Equal(before.Receipt, after.Receipt) || !bytes.Equal(before.Submission, after.Submission) || !bytes.Equal(identity, nowIdentity) {
		t.Fatal("reattachment replaced/canceled/resubmitted accepted execution")
	}
	edited := filepath.Join(project, "edited.py")
	must(t, os.WriteFile(edited, append(source, []byte("\n# The caller changed; deterministic operations did not.\n")...), 0600))
	run("run", edited, "--await")
	reused := machineChildren(t, root, store, "2")
	if len(reused) != 3 {
		t.Fatalf("edited caller children: %+v", reused)
	}
	for i, child := range reused {
		if child.Executions != 0 || child.Computation != offline[i].Computation || child.Revision != offline[i].Revision || !bytes.Equal(child.Result, offline[i].Result) {
			t.Fatalf("edited caller recomputed/changed child %d: before=%+v after=%+v", i, offline[i], child)
		}
	}
	if calls.Load() != 2 {
		t.Fatalf("native source reads occurred %d times; B and C each need exactly one", calls.Load())
	}
	evidence, err := json.MarshalIndent(map[string]any{"request": parent.ID, "offline_state": state, "offline_collected": collected, "children": offline, "edited_children": reused, "barrier_calls": calls.Load(), "native_read": string(verified)}, "", "  ")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, "proof.json"), evidence, 0600))
	t.Logf("native offline completion and edited caller reuse: %s", evidence)
}
