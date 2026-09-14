package producttest

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestOrdinaryScriptNativeRootBytesSurviveOriginalEditAndClientExit(t *testing.T) {
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires exact Runtime wire55 input intake")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	root, err := os.MkdirTemp("", "cozy-root-byte-input-")
	must(t, err)
	control := filepath.Join(root, "control")
	for _, args := range [][]string{{"venv", control, "--python", "3.12"}, {"pip", "install", "--python", filepath.Join(control, "bin/python"), wheel}} {
		if out, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
			t.Fatalf("uv %v: %v\n%s", args, err, out)
		}
	}
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		if t.Failed() {
			_, _ = runCozyPath(t, root, path, "down", "--json")
			t.Log("root input proof and Runtime retained", root)
			return
		}
		compositionDown(t, root, path)
		if !t.Failed() {
			must(t, removeAllForce(root))
		}
	})
	script, source, report, data := rootByteInputProject(t, wheel)
	code, out := runCozyPath(t, root, path, "run", script, "--asset", "tree="+source, "--asset", "report="+report, "--json")
	if code != 0 {
		t.Fatalf("native root intake failed [%d]: %s\n%s", code, out, productWorkerLogs(root))
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByReference("1")
	fatal(t, problem)
	if request == nil {
		t.Fatal("CLI omitted root request")
	}
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if link == nil || len(link.Receipt) == 0 || link.Collected {
		t.Fatalf("root input has no live durable receipt: %s", out)
	}
	inputs, problem := store.MachineInputs(request.ID)
	fatal(t, problem)
	if len(inputs) != 2 {
		t.Fatalf("root byte intake omitted a native receipt: %+v", inputs)
	}
	must(t, os.WriteFile(report, []byte("changed original file"), 0600))
	must(t, os.WriteFile(filepath.Join(source, "report.json"), []byte("changed original tree"), 0600))
	must(t, os.Remove(filepath.Join(source, "extra.txt")))
	if code, out := runCozyPath(t, root, path, "down", "--json"); code != 0 {
		t.Fatalf("detach client [%d]: %s", code, out)
	}
	journal, err := sql.Open("sqlite", "file:"+filepath.Join(root, "tensorfs", ".cozy-workspace", "journal.sqlite3")+"?mode=ro&_pragma=busy_timeout(5000)")
	must(t, err)
	defer journal.Close()
	deadline := time.Now().Add(3 * time.Minute)
	for {
		var state string
		must(t, journal.QueryRow("SELECT state FROM executions WHERE request=?", request.ID).Scan(&state))
		if state == "succeeded" {
			break
		}
		if state == "failed" || state == "canceled" || time.Now().After(deadline) {
			t.Fatalf("Runtime did not finish with client absent: %s\n%s", state, productWorkerLogs(root))
		}
		time.Sleep(100 * time.Millisecond)
	}
	var collected int
	must(t, journal.QueryRow("SELECT collected FROM executions WHERE request=?", request.ID).Scan(&collected))
	if collected != 0 {
		t.Fatal("root was collected while its client was absent")
	}
	code, out = runCozyPath(t, root, path, "run", "watch", "1", "--json")
	if code != 0 {
		t.Fatalf("collect after original edit/client exit [%d]: %s\n%s", code, out, productWorkerLogs(root))
	}
	files, problem := store.MachineFileResults(request.ID)
	fatal(t, problem)
	if len(files) != 1 || !files[0].Copied {
		t.Fatal("root did not collect its returned input Tree")
	}
	exported, problem := store.OutputExportOf(request.ID)
	fatal(t, problem)
	if exported == nil || exported.State != "published" || len(exported.PublishedPaths) != 1 {
		t.Fatal("returned input Tree was not exported")
	}
	actual, err := os.ReadFile(filepath.Join(exported.PublishedPaths[0], "report.json"))
	must(t, err)
	if string(actual) != string(data) {
		t.Fatal("original edit changed Runtime input bytes")
	}
	children := machineChildren(t, root, store, "1")
	if len(children) != 1 || children[0].Executions != 1 || children[0].State != "succeeded" {
		t.Fatalf("native root inputs did not reach the managed reader: %+v", children)
	}
	var intakes, attempts int
	must(t, journal.QueryRow("SELECT count(*) FROM input_tree_intakes WHERE request=? AND state='released'", request.ID).Scan(&intakes))
	must(t, journal.QueryRow("SELECT count(*) FROM attempts").Scan(&attempts))
	if intakes != 2 || attempts != 2 {
		t.Fatalf("root intake fabricated an execution or lost exact cleanup: intakes=%d attempts=%d", intakes, attempts)
	}
	t.Logf("root=%s inputs=%d bytes=%d reader=%s; original mutated, client absent until Runtime succeeded", request.ID, intakes, len(data), children[0].Request)
}

func rootByteInputProject(t *testing.T, wheel string) (string, string, string, []byte) {
	t.Helper()
	version := runtimeFixtureVersion(t, wheel)
	project := t.TempDir()
	library := filepath.Join(project, "byte_tools")
	must(t, os.Mkdir(library, 0700))
	module := []byte(`from typing import Annotated
import hashlib
import msgspec
from cozy_runtime.author import App, AssetBound, Context, FileAsset, Tree, invocable
ReportFile = Annotated[FileAsset, AssetBound(max_bytes=200000, media_types=("text/plain",))]
class Verified(msgspec.Struct):
    digest: str
    length: int
@invocable()
async def verify(ctx: Context, *, report: ReportFile, tree: Tree) -> Verified:
    raw = report.read_bytes()
    assert raw == (tree.path / "report.json").read_bytes()
    assert (tree.path / "extra.txt").read_text() == msgspec.json.decode(raw)["variant"]
    return Verified(hashlib.sha256(raw).hexdigest(), len(raw))
app = App()
app.job(verify)
`)
	must(t, os.WriteFile(filepath.Join(library, "byte_tools.py"), module, 0600))
	metadata := fmt.Sprintf(`[project]
name="byte-tools"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime==%s"]
[project.entry-points."cozy.application"]
default="byte_tools:app"
[tool.uv.sources]
cozy-runtime={path=%q}
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["byte_tools.py"]
`, version, wheel)
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(metadata), 0600))
	script := filepath.Join(project, "consume.py")
	body := fmt.Sprintf(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime==%s", "byte-tools==0.0.1"]
# [tool.uv.sources]
# cozy-runtime={path=%q}
# byte-tools={path="./byte_tools",editable=true}
# ///
import asyncio
from typing import Annotated
from cozy_runtime.author import AssetBound, FileAsset, Outputs, Tree
from byte_tools import verify

async def main(ctx, *, report: Annotated[FileAsset, AssetBound(max_bytes=200000, media_types=("text/plain",))], tree: Tree, out: Outputs) -> Tree:
    await asyncio.sleep(10)
    checked = await verify(report=report, tree=tree)
    assert checked.length > 48000
    ctx.log("read the original native inputs")
    result = out.temporary_file()
    result.mkdir()
    (result / "report.json").write_bytes(report.read_bytes())
    (result / "extra.txt").write_bytes((tree.path / "extra.txt").read_bytes())
    return out.save_tree(result)
`, version, wheel)
	must(t, os.WriteFile(script, []byte(body), 0600))
	source := filepath.Join(project, "input")
	must(t, os.Mkdir(source, 0700))
	data, err := json.Marshal(map[string]string{"variant": "reviewed", "observations": strings.Repeat("verified", 12000)})
	must(t, err)
	report := filepath.Join(project, "report.json")
	must(t, os.WriteFile(report, data, 0600))
	must(t, os.WriteFile(filepath.Join(source, "report.json"), data, 0600))
	must(t, os.WriteFile(filepath.Join(source, "extra.txt"), []byte("reviewed"), 0600))
	return script, source, report, data
}
