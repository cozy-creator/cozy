package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// Both script-owned and forwarded child files must reach the requested directory.
// The actual CLI/Runtime/native receipts are used; no file or outcome is seeded.
func TestClientScriptExportsDeclaredFiles(t *testing.T) {
	if *privateScriptRuntimeWheel == "" {
		t.Skip("requires an exact Runtime wheel")
	}
	wheel, err := filepath.Abs(*privateScriptRuntimeWheel)
	must(t, err)
	root, err := os.MkdirTemp("", "cozy-file-out-")
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
	uv("pip", "install", "--python", filepath.Join(control, "bin", "python"), wheel)
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("file-export evidence retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	run := func(args ...string) map[string]any {
		t.Helper()
		code, out := runCozyPath(t, root, path, append(args, "--json")...)
		if code != 0 {
			t.Fatalf("cozy %v [%d]: %s", args, code, out)
		}
		var result map[string]any
		must(t, json.Unmarshal([]byte(out), &result))
		return result
	}
	project := filepath.Join(root, "project")
	producer := filepath.Join(project, "producer")
	must(t, os.MkdirAll(producer, 0700))
	version := runtimeFixtureVersion(t, wheel)
	metadata := fmt.Sprintf(`[project]
name="file-producer"
version="0.1.0"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=%s"]
[tool.uv.sources]
cozy-runtime={path=%q}
[project.entry-points."cozy.application"]
default="file_producer:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["file_producer.py"]
`, version, wheel)
	must(t, os.WriteFile(filepath.Join(producer, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(producer, "package.toml"), []byte("[application]\nobject=\"file_producer:app\"\n"), 0600))
	must(t, os.WriteFile(filepath.Join(producer, "file_producer.py"), []byte(`from typing import Annotated
import msgspec
from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs, invocable
class Report(msgspec.Struct, frozen=True):
    facts: Annotated[FileAsset, AssetBound(max_bytes=1024, media_types=("application/json",))]
@invocable(memoize=True)
async def produce(ctx:Context, *, out:Outputs)->Report:
    return Report(out.save_bytes(b'{"ok":true}\n',media_type="application/json"))
app=App()
app.job(produce)
`), 0600))
	header := fmt.Sprintf(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime>=%s","file-producer>=0.1.0"]
# [tool.uv.sources]
# cozy-runtime={path=%q}
# file-producer={path="./producer"}
# ///
`, version, wheel)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for _, tc := range []struct {
		name, body, slot, media string
		data                    []byte
	}{
		{"bare", `from typing import Annotated
from cozy_runtime.author import AssetBound, FileAsset, Outputs
def main(*,out:Outputs)->Annotated[FileAsset,AssetBound(max_bytes=1024,media_types=("text/plain",))]:
    return out.save_bytes(b"bounded client-script output\n",media_type="text/plain")
`, "value", "text/plain", []byte("bounded client-script output\n")},
		{"forwarded", `from file_producer import Report, produce
async def main()->Report:
    return await produce()
`, "value.facts", "application/json", []byte("{\"ok\":true}\n")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			script := filepath.Join(project, tc.name+".py")
			must(t, os.WriteFile(script, []byte(header+tc.body), 0600))
			directory := filepath.Join(root, "export-"+tc.name)
			result := run("run", script, "--await", "--out", directory)
			saved, ok := result["saved"].([]any)
			if !ok || len(saved) != 1 {
				t.Fatalf("script omitted its exported file: %+v", result)
			}
			file := saved[0].(map[string]any)
			target, ok := file["path"].(string)
			if !ok || filepath.Dir(target) != directory {
				t.Fatalf("wrong requested directory: %+v", file)
			}
			data, err := os.ReadFile(target)
			must(t, err)
			if !bytes.Equal(data, tc.data) {
				t.Fatalf("wrong file bytes: %q", data)
			}
			digest := sha256.Sum256(data)
			if file["digest"] != "sha256:"+hex.EncodeToString(digest[:]) {
				t.Fatal("export changed the accepted digest")
			}
			request, problem := store.RequestByReference(fmt.Sprint(result["job"]))
			fatal(t, problem)
			export, problem := store.OutputExportOf(request.ID)
			fatal(t, problem)
			if export == nil || export.State != "published" || len(export.Outputs) != 1 || export.Outputs[0].OutputID != tc.slot || export.Outputs[0].MediaType != tc.media {
				t.Fatalf("file export was not recorded/settled: %+v", export)
			}
			files, problem := store.MachineFileResults(request.ID)
			fatal(t, problem)
			if len(files) != 1 || !files[0].Copied || files[0].State != "released" || files[0].Output.Path == target {
				t.Fatalf("native custody was not independently copied: %+v", files)
			}
			attempts, problem := store.Attempts(request.ID)
			fatal(t, problem)
			if len(attempts) != 0 {
				t.Fatal("Creator invented an execution attempt")
			}
			must(t, os.WriteFile(target, []byte("user edit"), 0600))
			internal, err := os.ReadFile(files[0].Output.Path)
			must(t, err)
			if !bytes.Equal(internal, data) {
				t.Fatal("editing the user copy changed retained native result")
			}
		})
	}
}
