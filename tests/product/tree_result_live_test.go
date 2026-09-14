package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
)

// The ordinary CLI exports the complete native Tree, including nested and empty
// files, before collection, and keeps independent custody after the user edits it.
func TestOrdinaryScriptTreeResultExportsCompleteClosure(t *testing.T) {
	if *privateScriptRuntimeWheel == "" {
		t.Skip("requires an exact Runtime wheel")
	}
	wheel, err := filepath.Abs(*privateScriptRuntimeWheel)
	must(t, err)
	version := runtimeFixtureVersion(t, wheel)
	control := filepath.Join(t.TempDir(), "control")
	for _, args := range [][]string{{"venv", control, "--python", "3.12"}, {"pip", "install", "--python", filepath.Join(control, "bin/python"), wheel}} {
		if out, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
			t.Fatalf("uv: %v\n%s", err, out)
		}
	}
	root, err := os.MkdirTemp("", "cozy-tree-result-")
	must(t, err)
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("tree result evidence retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	script := filepath.Join(t.TempDir(), "reports.py")
	body := fmt.Sprintf(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime==%s"]
# [tool.uv.sources]
# cozy-runtime={path=%q}
# ///
from cozy_runtime.author import Outputs, Tree

def main(*, out: Outputs) -> Tree:
    directory = out.temporary_file()
    directory.mkdir()
    (directory / "nested").mkdir()
    data = b"verified report\\n" * 10000
    (directory / "nested" / "report.txt").write_bytes(data)
    (directory / "duplicate.txt").write_bytes(data)
    (directory / "empty.txt").write_bytes(b"")
    return out.save_tree(directory)
`, version, wheel)
	must(t, os.WriteFile(script, []byte(body), 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for _, explicit := range []bool{false, true} {
		args := []string{"run", script, "--await", "--json"}
		if explicit {
			args = append(args, "--out", filepath.Join(root, "custom-report"))
		}
		code, out := runCozyPath(t, root, path, args...)
		if code != 0 {
			t.Fatalf("tree CLI failed [%d]: %s\n%s", code, out, productWorkerLogs(root))
		}
		var result map[string]any
		for _, line := range strings.Split(out, "\n") {
			var value map[string]any
			if json.Unmarshal([]byte(line), &value) == nil && value["job"] != nil {
				result = value
			}
		}
		saved, ok := result["saved"].([]any)
		if !ok || len(saved) != 1 {
			t.Fatalf("tree result omitted export: %+v", result)
		}
		target := saved[0].(map[string]any)["path"].(string)
		if explicit && filepath.Dir(target) != filepath.Join(root, "custom-report") {
			t.Fatal("Tree ignored --out")
		}
		first, err := os.ReadFile(filepath.Join(target, "nested/report.txt"))
		must(t, err)
		second, err := os.ReadFile(filepath.Join(target, "duplicate.txt"))
		must(t, err)
		empty, err := os.ReadFile(filepath.Join(target, "empty.txt"))
		must(t, err)
		if len(first) < 48<<10 || string(first) != string(second) || len(empty) != 0 {
			t.Fatal("Tree closure lost nested, duplicate or empty bytes")
		}
		request, problem := store.RequestByReference(fmt.Sprint(result["job"]))
		fatal(t, problem)
		if request == nil || request.State != "succeeded" {
			t.Fatal("Tree root did not collect")
		}
		if children := machineChildren(t, root, store, request.ID); len(children) != 0 {
			t.Fatal("direct Tree output invented child execution")
		}
		files, problem := store.MachineFileResults(request.ID)
		fatal(t, problem)
		if len(files) != 1 || files[0].State != "released" || !files[0].Copied || files[0].Output.MimeType != resultfiles.TreeMediaType {
			t.Fatalf("Tree lacks copied receipt and released hold: %+v", files)
		}
		members, problem := resultfiles.ReadTreeManifest(files[0].Output.Path, files[0].Output.Digest, files[0].Output.Length, files[0].Source.ContentBytes)
		fatal(t, problem)
		if len(members) != 3 || files[0].Source.ContentBytes != int64(2*len(first)) {
			t.Fatal("Tree manifest changed its complete content size")
		}
		must(t, os.WriteFile(filepath.Join(target, "nested/report.txt"), []byte("user edit"), 0600))
		for _, member := range members {
			retained, err := os.ReadFile(filepath.Join(files[0].Output.Path+".files", strings.TrimPrefix(member.Digest, "sha256:")))
			must(t, err)
			if int64(len(retained)) != member.Length {
				t.Fatal("user edit changed Tree custody")
			}
		}
		t.Logf("collected Tree request=%s digest=%s members=%d bytes=%d exported=%s", request.ID, files[0].Output.Digest, len(members), files[0].Source.ContentBytes, target)
	}
}
