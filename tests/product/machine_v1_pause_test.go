package producttest

import (
	"bytes"
	"encoding/json"
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

// `cozy run pause` and `cozy run resume` on a machine that serves cozy.machine.v1, as a user
// types them: a job paused while a segment renders rests paused; resumed, it replays only its
// root, whose calls find the finished segments, and the saved film is every segment once.
func TestMachineV1JobPausesAndResumes(t *testing.T) {
	if *machineHostBinary == "" || *cpuLongform == "" {
		t.Skip("requires -machine-host=<cozy-machine> and -cpu-longform=<cozy-machine>/tests/fixtures/cpu_longform")
	}
	root, err := os.MkdirTemp(os.TempDir(), "czp")
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
		picture.Set(i%2, i/2, color.RGBA{R: 3, G: 3, B: 3, A: 255})
	}
	var encoded bytes.Buffer
	must(t, png.Encode(&encoded, picture))
	reference := filepath.Join(root, "ref.png")
	must(t, os.WriteFile(reference, encoded.Bytes(), 0o600))
	segments := []string{"a dawn", "a storm", "a calm"}
	request, _ := json.Marshal(map[string]any{"segments": segments, "hold": 6.0, "hold_at": 1})
	in := filepath.Join(root, "req.json")
	must(t, os.WriteFile(in, request, 0o600))
	out := filepath.Join(root, "film")

	if code, document := runCozy(t, root, "run", "local/cozy-machine-cpu-longform/long_form", "--input", in,
		"--asset", "reference="+reference, "--json", "--out", out); code != 0 {
		skipWithoutMachine(t, code, document)
		t.Fatalf("the job was not submitted [exit %d]\n%s", code, document)
	}
	show := func() (string, string) {
		code, shown := runCozy(t, root, "run", "show", "1", "--json")
		if code != 0 {
			t.Fatalf("run show [exit %d]\n%s", code, shown)
		}
		var run struct {
			Status string `json:"status"`
		}
		_ = json.Unmarshal([]byte(lastJSONLine(shown)), &run)
		return run.Status, shown
	}
	until := func(what string, done func(status, shown string) bool) {
		deadline := time.Now().Add(3 * time.Minute)
		for {
			status, shown := show()
			if done(status, shown) {
				return
			}
			if time.Now().After(deadline) {
				t.Fatalf("the run never showed %s (status %q)\n%s", what, status, shown)
			}
			time.Sleep(200 * time.Millisecond)
		}
	}
	// The first segment is in the film: the second, which holds, renders now.
	until("its first segment", func(_, shown string) bool { return strings.Contains(shown, "Video (segments 1-1)") })
	if code, paused := runCozy(t, root, "run", "pause", "1", "--json"); code != 0 {
		t.Fatalf("run pause [exit %d]\n%s", code, paused)
	}
	until("paused", func(status, _ string) bool { return status == "paused" })
	t.Logf("paused at %s", time.Now().Format(time.RFC3339Nano))
	if status, shown := show(); strings.Contains(shown, "Video (segments 1-2)") {
		t.Fatalf("a paused job went on publishing (status %q)\n%s", status, shown)
	}
	if code, resumed := runCozy(t, root, "run", "resume", "1", "--json"); code != 0 {
		t.Fatalf("run resume [exit %d]\n%s", code, resumed)
	}
	code, document := runCozy(t, root, "run", "watch", "1", "--json")
	if code != 0 {
		t.Fatalf("the resumed job did not succeed [exit %d]\n%s", code, document)
	}
	until("its result", func(status, _ string) bool { return status == "completed" || status == "succeeded" })
	_, shown := show()
	var run struct {
		Result struct {
			Segments int `json:"segments"`
			Attempts int `json:"attempts"`
			Replayed int `json:"replayed"`
		} `json:"result"`
	}
	if json.Unmarshal([]byte(lastJSONLine(shown)), &run) != nil || run.Result.Segments != 3 || run.Result.Attempts != 2 || run.Result.Replayed < 1 {
		t.Fatalf("the job's result does not show one resumed run of three segments\n%s", shown)
	}
	t.Logf("resumed job's result: %+v", run.Result)
	// The film is every segment once, in order: no segment ran twice or was lost.
	files, err := filepath.Glob(filepath.Join(out, "*video*"))
	if err != nil || len(files) != 1 {
		entries, _ := os.ReadDir(out)
		t.Fatalf("want one saved video in %s, got %v (%v)\n%s", out, files, entries, shown)
	}
	film, err := os.ReadFile(files[0])
	must(t, err)
	lines := strings.Split(strings.TrimSuffix(string(film), "\n"), "\n")
	if len(lines) != len(segments) {
		t.Fatalf("want %d segments in the film, got:\n%s", len(segments), film)
	}
	for index, line := range lines {
		if fields := strings.Split(line, "|"); len(fields) != 4 || fields[1] != segments[index] {
			t.Fatalf("segment %d is not in its place: %q\n%s", index, line, film)
		}
	}
}
