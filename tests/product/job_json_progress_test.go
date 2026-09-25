package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestAwaitedJobJSONSeparatesEventsAndResult(t *testing.T) {
	integration(t)
	root, err := os.MkdirTemp("", "cozy-job-json-")
	must(t, err)
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all"); _ = os.RemoveAll(root) })
	script := filepath.Join(t.TempDir(), "simple.py")
	metadata := "# /// script\n# requires-python = \">=3.12,<3.13\"\n# dependencies = [\"cozy-runtime>=" + runtimeFixtureVersion(t, *privateScriptRuntimeWheel) + "\"]\n"
	if wheel := *privateScriptRuntimeWheel; wheel != "" {
		metadata += fmt.Sprintf("# [tool.uv.sources]\n# cozy-runtime = {path = %q}\n", wheel)
	}
	must(t, os.WriteFile(script, []byte(metadata+`# ///
def main(ctx):
    ctx.raise_if_cancelled()
`), 0600))
	code, stdout, stderr := runCozyStreams(t, root, "--json", "run", script, "--await", "--full")
	var result map[string]any
	if code != 0 || json.Unmarshal([]byte(stdout), &result) != nil || result["status"] != "completed" {
		t.Fatalf("job did not return one final JSON document: code=%d stdout=%s stderr=%s", code, stdout, stderr)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	id, ok := result["id"].(string)
	if !ok {
		t.Fatalf("missing accepted request identity: %s", stdout)
	}
	row, problem := store.RequestRow(id)
	fatal(t, problem)
	terminalAt, problem := store.TerminalEventAt(id)
	fatal(t, problem)
	began, err := time.Parse(time.RFC3339Nano, row.CreatedAt)
	must(t, err)
	ended, err := time.Parse(time.RFC3339Nano, terminalAt)
	must(t, err)
	want := float64(ended.Sub(began).Milliseconds())
	if result["wall_ms"] != want {
		t.Fatalf("initial --await does not use durable request duration: got %v want %v", result["wall_ms"], want)
	}
	for repeat := 0; repeat < 2; repeat++ {
		code, replay, _ := runCozyStreams(t, root, "run", "watch", id, "--json", "--full")
		var watched map[string]any
		if code != 0 || json.Unmarshal([]byte(replay), &watched) != nil || watched["wall_ms"] != want {
			t.Fatalf("initial/replayed duration differs [%d]: %s", code, replay)
		}
	}
	if strings.TrimSpace(stderr) == "" || strings.ContainsAny(stderr, "\r\033") {
		t.Fatalf("missing JSONL events or terminal control bytes: %q", stderr)
	}
	for _, line := range strings.Split(strings.TrimSpace(stderr), "\n") {
		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil || event["type"] == nil {
			t.Fatalf("job stderr contains a non-event: %q (%v)", line, err)
		}
	}
}
