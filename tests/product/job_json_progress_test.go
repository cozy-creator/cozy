package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
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
	code, stdout, stderr := runCozyStreams(t, root, "--json", "run", script, "--await")
	var result map[string]any
	if code != 0 || json.Unmarshal([]byte(stdout), &result) != nil || result["status"] != "completed" {
		t.Fatalf("job did not return one final JSON document: code=%d stdout=%s stderr_bytes=%d", code, stdout, len(stderr))
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
