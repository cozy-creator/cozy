package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var cpuLongform = flag.String("cpu-longform", "", "cozy-machine tests/fixtures/cpu_longform: H3 long-form's shape on CPU")

var olderCozy = flag.String("older-cozy", "", "a released cozy whose daemon predates cozy.machine.v1 (0.1.26)")

// `cozy run <pkg>/long_form` on a machine that serves cozy.machine.v1, as a user types it: a
// job whose file input is a reference image, each segment a child run of the package's own
// invocable, the film published after every segment and saved to --out at its end.
func TestMachineV1JobRendersSegmentsThroughChildRuns(t *testing.T) {
	if *machineHostBinary == "" || *cpuLongform == "" {
		t.Skip("requires -machine-host=<cozy-machine> and -cpu-longform=<cozy-machine>/tests/fixtures/cpu_longform")
	}
	root, err := os.MkdirTemp(os.TempDir(), "czj")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("evidence retained at %s\nmachine log tail:\n%s", root, tail(filepath.Join(root, "machine", "host.log")))
		} else {
			_ = removeAllForce(root)
		}
	})
	project := filepath.Join(t.TempDir(), "cpu_longform")
	must(t, os.CopyFS(project, os.DirFS(*cpuLongform)))
	if out, err := exec.Command("uv", "lock", "--directory", project).CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v\n%s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}
	picture := image.NewRGBA(image.Rect(0, 0, 2, 2))
	for i := range 4 {
		picture.Set(i%2, i/2, color.RGBA{R: 7, G: 7, B: 7, A: 255})
	}
	var encoded bytes.Buffer
	must(t, png.Encode(&encoded, picture))
	reference := filepath.Join(root, "ref.png")
	must(t, os.WriteFile(reference, encoded.Bytes(), 0o600))
	segments := []string{"a dawn", "a storm", "a calm"}
	request, _ := json.Marshal(map[string]any{"segments": segments})
	in := filepath.Join(root, "req.json")
	must(t, os.WriteFile(in, request, 0o600))
	out := filepath.Join(root, "film")

	code, document := runCozy(t, root, "run", "local/cozy-machine-cpu-longform/long_form", "--input", in,
		"--asset", "reference="+reference, "--await", "--json", "--out", out)
	skipWithoutMachine(t, code, document)
	if code != 0 {
		t.Fatalf("the job did not succeed [exit %d]\n%s", code, document)
	}
	var run struct {
		Status string          `json:"status"`
		Result json.RawMessage `json:"result"`
	}
	if json.Unmarshal([]byte(lastJSONLine(document)), &run) != nil || run.Status != "completed" && run.Status != "succeeded" {
		t.Fatalf("the run's document is not a completed run\n%s", document)
	}
	// Each segment saw the job's reference (as the machine prepared it) and continued the
	// previous segment's context: the saved film is every segment's line, in order.
	files, err := filepath.Glob(filepath.Join(out, "*video*"))
	if err != nil || len(files) != 1 {
		entries, _ := os.ReadDir(out)
		t.Fatalf("want one saved video in %s, got %v (%v)\n%s", out, files, entries, document)
	}
	film, err := os.ReadFile(files[0])
	must(t, err)
	lines := strings.Split(strings.TrimSuffix(string(film), "\n"), "\n")
	if len(lines) != len(segments) {
		t.Fatalf("want %d segments in the film, got:\n%s", len(segments), film)
	}
	context := ""
	for index, line := range lines {
		fields := strings.Split(line, "|")
		if len(fields) != 4 || fields[0] != fmt.Sprint(index) || fields[1] != segments[index] || fields[3] != context {
			t.Fatalf("segment %d is not its own continuation: %q\n%s", index, line, film)
		}
		sum := sha256.Sum256([]byte(line + "\n"))
		context = hex.EncodeToString(sum[:])
	}
	if code, shown := runCozy(t, root, "run", "show", "1", "--json"); code != 0 || !strings.Contains(shown, "Video (segments 1-3)") {
		t.Fatalf("run show does not name the film's last revision [exit %d]\n%s", code, shown)
	}
	// A job's model choice addresses one of its callables and reaches that child: this
	// segment function declares no model slot, so its first child refuses it and the job
	// fails naming that segment.
	code, document = runCozy(t, root, "run", "local/cozy-machine-cpu-longform/long_form", "--input", in,
		"--asset", "reference="+reference, "model.render_segment.models.base=alice/model", "--await", "--json")
	if code == 0 || !strings.Contains(document, "segment 1 of 3") || !strings.Contains(document, "model choices name no declared model slot") {
		t.Fatalf("the child did not receive the job's choice for it [exit %d]\n%s", code, document)
	}
}

// A job on an explicit endpoint runs in the foreground (no daemon owns it): submit, pause and
// resume are each their own command, signed with this install's key. A machine admits that key
// as sshd does, by a line in its authorized_keys: without the line the job is refused before
// anything is sent, and with it again the job is accepted.
func TestEndpointJobSignsWithTheInstallKeyAndPauses(t *testing.T) {
	if *machineHostBinary == "" || *cpuLongform == "" {
		t.Skip("requires -machine-host=<cozy-machine> and -cpu-longform=<cozy-machine>/tests/fixtures/cpu_longform")
	}
	root, err := os.MkdirTemp(os.TempDir(), "cze")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("evidence retained at %s\nmachine log tail:\n%s", root, tail(filepath.Join(root, "machine", "host.log")))
		} else {
			_ = removeAllForce(root)
		}
	})
	project := filepath.Join(t.TempDir(), "cpu_longform")
	must(t, os.CopyFS(project, os.DirFS(*cpuLongform)))
	if out, err := exec.Command("uv", "lock", "--directory", project).CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v\n%s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "machine", "start"); code != 0 {
		t.Fatalf("machine start [exit %d]\n%s", code, out)
	}
	var agent struct {
		WorkerID   string `json:"worker_id"`
		WorkerPort int    `json:"worker_port"`
	}
	raw, err := os.ReadFile(filepath.Join(root, "machine", "agent.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &agent))
	leaf, err := os.ReadFile(filepath.Join(root, "machine", "leaf.pem"))
	must(t, err)
	address := fmt.Sprintf("127.0.0.1:%d", agent.WorkerPort)
	endpoint := filepath.Join(root, "endpoint.json")
	document, _ := json.Marshal(map[string]string{"format": "cozy.machine.endpoint/1", "address": address,
		"worker_id": agent.WorkerID, "worker_boot_id": "boot", "tls_certificate_pem": string(leaf), "execution_workspace_id": "workspace"})
	must(t, os.WriteFile(endpoint, document, 0o600))
	authorizedKeys := filepath.Join(root, "machine", "root", "authorized_keys")
	authorized, err := os.ReadFile(authorizedKeys)
	must(t, err)
	installed, err := os.ReadFile(filepath.Join(root, "id_ed25519.pub"))
	must(t, err)
	if !strings.Contains(string(authorized), strings.Join(strings.Fields(string(installed))[:2], " ")) {
		t.Fatalf("the machine's authorized_keys does not name this install's key\n%s", authorized)
	}
	// The daemon is not this run's controller: every command below is its own foreground one.
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}

	picture := image.NewRGBA(image.Rect(0, 0, 2, 2))
	var encoded bytes.Buffer
	must(t, png.Encode(&encoded, picture))
	reference := filepath.Join(root, "ref.png")
	must(t, os.WriteFile(reference, encoded.Bytes(), 0o600))
	in := filepath.Join(root, "req.json")
	request, _ := json.Marshal(map[string]any{"segments": []string{"one", "two", "three"}, "hold": 2.0})
	must(t, os.WriteFile(in, request, 0o600))
	out := filepath.Join(root, "film")
	run := func(extra ...string) (int, string) {
		args := append([]string{"run", "local/cozy-machine-cpu-longform/long_form", "--machine-endpoint-file", endpoint,
			"--input", in, "--asset", "reference=" + reference, "--json", "--out", out}, extra...)
		return runCozy(t, root, args...)
	}
	// The line deleted by hand: the machine drops the key at its next look and refuses the job.
	must(t, os.WriteFile(authorizedKeys, []byte("# no keys\n"), 0o600))
	landed(t, "the machine to drop the deleted key", func() bool {
		code, refused := run()
		return code != 0 && strings.Contains(refused, "machine.endpoint_unauthorized")
	})
	// Restored: a refused attempt sent nothing, so the first accepted one is the job.
	must(t, os.WriteFile(authorizedKeys, authorized, 0o600))
	var code int
	var submitted string
	landed(t, "the machine to admit the key again", func() bool {
		code, submitted = run()
		return code == 0 || !strings.Contains(submitted, "machine.endpoint_unauthorized")
	})
	var job struct {
		ID       string `json:"run"`
		Accepted bool   `json:"machine_accepted"`
	}
	if code != 0 || json.Unmarshal([]byte(lastJSONLine(submitted)), &job) != nil || job.ID == "" || !job.Accepted {
		t.Fatalf("the job signed with the install key was not accepted [exit %d]\n%s", code, submitted)
	}
	status := func() string {
		_, shown := runCozy(t, root, "run", "show", job.ID, "--json")
		var state struct {
			Status string `json:"status"`
		}
		_ = json.Unmarshal([]byte(lastJSONLine(shown)), &state)
		return state.Status
	}
	// Pause, then resume, each from its own foreground command on the recorded endpoint.
	if code, paused := runCozy(t, root, "run", "pause", job.ID, "--json"); code != 0 {
		t.Fatalf("run pause [exit %d]\n%s", code, paused)
	}
	landed(t, "the job to rest paused", func() bool { return status() == "paused" })
	if code, resumed := runCozy(t, root, "run", "resume", job.ID, "--json"); code != 0 {
		t.Fatalf("run resume [exit %d]\n%s", code, resumed)
	}
	if code, watched := runCozy(t, root, "run", "watch", job.ID, "--json"); code != 0 {
		t.Fatalf("the resumed job did not complete [exit %d]\n%s", code, watched)
	}
	files, _ := filepath.Glob(filepath.Join(out, "*video*"))
	if len(files) != 1 {
		t.Fatalf("want the film saved in %s, got %v", out, files)
	}
	film, err := os.ReadFile(files[0])
	must(t, err)
	if lines := strings.Count(string(film), "\n"); lines != 3 {
		t.Fatalf("the resumed job's film has %d segments:\n%s", lines, film)
	}
	if *olderCozy == "" {
		return
	}
	// Version skew: under a running daemon that predates cozy.machine.v1, a run naming the
	// rental goes from this command itself to the rental's machine; the daemon is not asked.
	if code, stopped := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("down [exit %d]\n%s", code, stopped)
	}
	up := exec.Command(*olderCozy, "up")
	up.Env = childEnv(t, root)
	if raw, err := up.CombinedOutput(); err != nil {
		t.Fatalf("the older cozy did not start its daemon: %v\n%s", err, raw)
	}
	daemonBinary := func() string {
		lock, _ := os.ReadFile(filepath.Join(root, "daemon.lock"))
		for line := range strings.Lines(string(lock)) {
			if pid, ok := strings.CutPrefix(strings.TrimSpace(line), "pid="); ok {
				exe, _ := os.Readlink("/proc/" + pid + "/exe")
				return exe
			}
		}
		return ""
	}
	older, err := filepath.EvalSymlinks(*olderCozy)
	must(t, err)
	if running := daemonBinary(); running != older {
		t.Fatalf("the daemon runs %q, not the older cozy %q", running, older)
	}
	must(t, os.WriteFile(in, []byte(`{"segments":["solo"]}`), 0o600))
	skewed := filepath.Join(root, "skewed")
	code, awaited := runCozy(t, root, "run", "local/cozy-machine-cpu-longform/long_form", "--rental=standin",
		"--input", in, "--asset", "reference="+reference, "--await", "--json", "--out", skewed)
	if files, _ := filepath.Glob(filepath.Join(skewed, "*video*")); code != 0 || len(files) != 1 {
		t.Fatalf("the run under the older daemon did not save its film [exit %d]: %v\n%s", code, files, awaited)
	}
	if running := daemonBinary(); running != older {
		t.Fatalf("the run replaced the older daemon with %q", running)
	}
}

// A run its machine already accepted never starts a stopped machine: observing attaches while
// the machine runs and waits while it is stopped, on cozy.machine.v1 as on the path before it.
func TestObservingAV1RunNeverStartsAStoppedMachine(t *testing.T) {
	if *machineHostBinary == "" || *cpuLongform == "" {
		t.Skip("requires -machine-host=<cozy-machine> and -cpu-longform=<cozy-machine>/tests/fixtures/cpu_longform")
	}
	root, err := os.MkdirTemp(os.TempDir(), "czs")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("evidence retained at %s\nmachine log tail:\n%s", root, tail(filepath.Join(root, "machine", "host.log")))
		} else {
			_ = removeAllForce(root)
		}
	})
	project := filepath.Join(t.TempDir(), "cpu_longform")
	must(t, os.CopyFS(project, os.DirFS(*cpuLongform)))
	if out, err := exec.Command("uv", "lock", "--directory", project).CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v\n%s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}
	picture := image.NewRGBA(image.Rect(0, 0, 2, 2))
	var encoded bytes.Buffer
	must(t, png.Encode(&encoded, picture))
	reference := filepath.Join(root, "ref.png")
	must(t, os.WriteFile(reference, encoded.Bytes(), 0o600))
	in := filepath.Join(root, "req.json")
	must(t, os.WriteFile(in, []byte(`{"segments":["one","two"],"hold":600}`), 0o600))
	code, submitted := runCozy(t, root, "run", "local/cozy-machine-cpu-longform/long_form", "--input", in,
		"--asset", "reference="+reference, "--json")
	if code != 0 {
		t.Fatalf("submit [exit %d]\n%s", code, submitted)
	}
	running := func() bool {
		_, shown := runCozy(t, root, "machine", "show", "--json")
		var machine struct {
			Running bool `json:"running"`
		}
		_ = json.Unmarshal([]byte(lastJSONLine(shown)), &machine)
		return machine.Running
	}
	status := func() string {
		_, shown := runCozy(t, root, "run", "show", "1", "--json")
		var state struct {
			Status string `json:"status"`
		}
		_ = json.Unmarshal([]byte(lastJSONLine(shown)), &state)
		return state.Status
	}
	landed(t, "the machine to accept the run", func() bool { return status() == "in_progress" })
	if code, out := runCozy(t, root, "machine", "stop"); code != 0 || running() {
		t.Fatalf("machine stop [exit %d]\n%s", code, out)
	}
	// The daemon's observer retries within seconds; a reader asks at once. Neither starts it.
	for deadline := time.Now().Add(8 * time.Second); time.Now().Before(deadline); time.Sleep(time.Second) {
		if status(); running() {
			t.Fatal("observing an accepted run started the stopped machine")
		}
	}
}

// `--timeout` is the run's deadline on cozy.machine.v1 as on worker.v1: when it passes the run
// is canceled on its machine, attributed to the deadline, whether or not a client waits.
func TestRunTimeoutCancelsTheRunOnItsMachine(t *testing.T) {
	if *machineHostBinary == "" || *cpuLongform == "" {
		t.Skip("requires -machine-host=<cozy-machine> and -cpu-longform=<cozy-machine>/tests/fixtures/cpu_longform")
	}
	root, err := os.MkdirTemp(os.TempDir(), "czt")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("evidence retained at %s\nmachine log tail:\n%s", root, tail(filepath.Join(root, "machine", "host.log")))
		} else {
			_ = removeAllForce(root)
		}
	})
	project := filepath.Join(t.TempDir(), "cpu_longform")
	must(t, os.CopyFS(project, os.DirFS(*cpuLongform)))
	if out, err := exec.Command("uv", "lock", "--directory", project).CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v\n%s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}
	var encoded bytes.Buffer
	must(t, png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 2, 2))))
	reference := filepath.Join(root, "ref.png")
	must(t, os.WriteFile(reference, encoded.Bytes(), 0o600))
	in := filepath.Join(root, "req.json")
	must(t, os.WriteFile(in, []byte(`{"segments":["one"],"hold":600}`), 0o600))
	run := []string{"run", "local/cozy-machine-cpu-longform/long_form", "--input", in, "--asset", "reference=" + reference, "--json", "--timeout", "20s"}
	status := func(number string) string {
		_, shown := runCozy(t, root, "run", "show", number, "--json")
		var state struct {
			Status string `json:"status"`
		}
		_ = json.Unmarshal([]byte(lastJSONLine(shown)), &state)
		return state.Status
	}

	// Waiting: the command ends with the deadline code once the machine stopped the run.
	code, out := runCozy(t, root, append(run, "--await")...)
	if code == 0 || !strings.Contains(out, `"code":"deadline"`) || !strings.Contains(out, "--timeout") {
		t.Fatalf("the awaited run did not end at its deadline [exit %d]\n%s", code, out)
	}
	if got := status("1"); got != "canceled" {
		t.Fatalf("the run its deadline ended is %q, not canceled", got)
	}
	// Detached: nobody waits, and the daemon still cancels it at the deadline.
	if code, out := runCozy(t, root, run...); code != 0 {
		t.Fatalf("detached submit [exit %d]\n%s", code, out)
	}
	landed(t, "the detached run to be canceled at its deadline", func() bool { return status("2") == "canceled" })
}
