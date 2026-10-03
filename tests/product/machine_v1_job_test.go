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
