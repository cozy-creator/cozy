package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
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

	"github.com/cozy-creator/cozy/internal/records"
)

var cpuLongform = flag.String("cpu-longform", "", "cozy-machine tests/fixtures/cpu_longform: H3 long-form's shape on CPU")

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

// A job on an explicit endpoint runs in the foreground (no daemon owns it), and an endpoint
// that is one of this host's rentals is signed with that rental's own key: a rental's machine
// authorizes it, not this computer's machine owner key. Here this computer's machine stands
// in for the rental: its key moves to a rental's credential and the owner key is replaced.
func TestEndpointJobSignsWithItsRentalKey(t *testing.T) {
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
	endpoint := filepath.Join(root, "endpoint.json")
	document, _ := json.Marshal(map[string]string{"format": "cozy.machine.endpoint/1", "address": fmt.Sprintf("127.0.0.1:%d", agent.WorkerPort),
		"worker_id": agent.WorkerID, "worker_boot_id": "boot", "tls_certificate_pem": string(leaf), "execution_workspace_id": "workspace"})
	must(t, os.WriteFile(endpoint, document, 0o600))
	// The machine authorizes the key it started with; that key becomes a rental's, and this
	// computer's owner key is now another one the machine does not know.
	owner := filepath.Join(root, "machine", "owner.pem")
	authorized, err := os.ReadFile(owner)
	must(t, err)
	must(t, os.MkdirAll(filepath.Join(root, "rentals"), 0o700))
	must(t, os.WriteFile(filepath.Join(root, "rentals", "pr-standin.creator.pem"), authorized, 0o600))
	_, private, _ := ed25519.GenerateKey(nil)
	der, err := x509.MarshalPKCS8PrivateKey(private)
	must(t, err)
	must(t, os.WriteFile(owner, pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der}), 0o600))

	picture := image.NewRGBA(image.Rect(0, 0, 2, 2))
	var encoded bytes.Buffer
	must(t, png.Encode(&encoded, picture))
	reference := filepath.Join(root, "ref.png")
	must(t, os.WriteFile(reference, encoded.Bytes(), 0o600))
	request, _ := json.Marshal(map[string]any{"segments": []string{"one", "two"}})
	in := filepath.Join(root, "req.json")
	must(t, os.WriteFile(in, request, 0o600))
	run := func(extra ...string) (int, string) {
		args := append([]string{"run", "local/cozy-machine-cpu-longform/long_form", "--machine-endpoint-file", endpoint,
			"--input", in, "--asset", "reference=" + reference, "--json"}, extra...)
		return runCozy(t, root, args...)
	}
	// Signed with the owner key, which the machine does not authorize: refused.
	if code, out := run("--await"); code == 0 {
		t.Fatalf("the machine accepted a key it does not authorize\n%s", out)
	}
	// Once the endpoint is a recorded rental, its rental key signs and the job runs.
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := store.RecordRental(records.Rental{ID: "pr-standin", MachineName: "standin", State: "ready",
		Address: fmt.Sprintf("127.0.0.1:%d", agent.WorkerPort), CertPath: filepath.Join(root, "machine", "leaf.pem"),
		ExpectedWorkerID: agent.WorkerID, ExpectedWorkerBootID: "boot", RentedAt: time.Now().UTC().Format(time.RFC3339)}); problem != nil {
		t.Fatal(problem)
	}
	store.Close()
	out := filepath.Join(root, "film")
	if code, document := run("--await", "--out", out); code != 0 {
		t.Fatalf("the rental-signed job did not succeed [exit %d]\n%s", code, document)
	}
	if files, _ := filepath.Glob(filepath.Join(out, "*video*")); len(files) != 1 {
		t.Fatalf("want the film saved in %s, got %v", out, files)
	}
}
