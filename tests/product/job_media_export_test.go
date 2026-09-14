package producttest

import (
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

// Exercise the real CLI, daemon, Runtime, accepted publication and exporter.
func TestTopLevelJobMediaExportsAndTextDoesNotCreateDirectory(t *testing.T) {
	if *privateScriptRuntimeWheel == "" {
		t.Skip("requires an exact published Runtime wheel")
	}
	wheel, err := filepath.Abs(*privateScriptRuntimeWheel)
	must(t, err)
	version := runtimeFixtureVersion(t, wheel)
	root, err := os.MkdirTemp("", "cozy-job-export-")
	must(t, err)
	control := filepath.Join(t.TempDir(), "control")
	uv := func(args ...string) {
		t.Helper()
		if out, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
			t.Fatalf("uv: %v %s", err, out)
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
		_, _ = runCozyPath(t, root, path, "down", "--all")
		if !t.Failed() {
			_ = os.RemoveAll(root)
		}
	})
	project := t.TempDir()
	metadata := fmt.Sprintf(`[project]
name="job-media-export-proof"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime==%s"]
[tool.uv.sources]
cozy-runtime={path=%q}
[project.entry-points."cozy.application"]
default="proof:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["proof.py"]
`, version, wheel)
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"proof:app\"\n"), 0600))
	must(t, os.WriteFile(filepath.Join(project, "proof.py"), []byte(`from typing import Annotated
import msgspec
from cozy_runtime.author import App, AssetBound, ImageAsset, ImageFrame, Outputs
app=App()
class Request(msgspec.Struct):
    pass
class Picture(msgspec.Struct):
    image: Annotated[ImageAsset, AssetBound(media_types=("image/png",))]
class Text(msgspec.Struct):
    text: str
@app.job
def text(payload:Request)->Text:
    return Text("done")
@app.job(emits_media=True)
def picture(payload:Request,out:Outputs)->Picture:
    return Picture(out.save_image(ImageFrame(2,2,bytes([255,0,0])*4),format="png"))
`), 0600))
	uv("lock", "--project", project)
	run := func(args ...string) map[string]any {
		t.Helper()
		command := exec.Command(cozyBin, args...)
		command.Env = childEnv(t, root, "PATH="+path)
		out, err := command.Output()
		var result map[string]any
		if err != nil || json.Unmarshal(out, &result) != nil {
			t.Fatalf("cozy %v [%v]: %s", args, err, out)
		}
		return result
	}
	run("package", "install", project, "--editable", "--json")
	run("run", "local/job-media-export-proof/text", "--await", "--json")
	defaultDir := filepath.Join(root, "outputs", "local-job-media-export-proof")
	if _, err := os.Stat(defaultDir); !os.IsNotExist(err) {
		t.Fatalf("text-only job created outputs directory: %v", err)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for _, explicit := range []bool{false, true} {
		directory := defaultDir
		args := []string{"run", "local/job-media-export-proof/picture", "--await", "--json"}
		if explicit {
			directory = filepath.Join(root, "custom-output")
			args = append(args, "--out", directory)
		}
		result := run(args...)
		if result["status"] != "completed" {
			t.Fatalf("media job failed: %+v", result)
		}
		saved, ok := result["saved"].([]any)
		if !ok || len(saved) != 1 {
			t.Fatalf("job omitted saved media: %+v", result)
		}
		entry := saved[0].(map[string]any)
		target := entry["path"].(string)
		if filepath.Dir(target) != directory || !strings.HasSuffix(target, ".png") {
			t.Fatalf("wrong output path: %s", target)
		}
		data, err := os.ReadFile(target)
		must(t, err)
		digest := sha256.Sum256(data)
		if entry["digest"] != "sha256:"+hex.EncodeToString(digest[:]) {
			t.Fatal("saved bytes differ from accepted digest")
		}
		request, problem := store.RequestByReference(fmt.Sprint(result["job"]))
		fatal(t, problem)
		export, problem := store.OutputExportOf(request.ID)
		fatal(t, problem)
		if export == nil || export.State != "published" || len(export.PublishedPaths) != 1 {
			t.Fatalf("export obligation was not settled: %+v", export)
		}
		outputs, problem := store.VisibleOutputs(request.ID)
		fatal(t, problem)
		if len(outputs) != 1 || outputs[0].Path == target {
			t.Fatal("user copy replaced internal publication custody")
		}
		files, problem := store.MachineFileResults(request.ID)
		fatal(t, problem)
		if len(files) != 1 || !files[0].Copied || files[0].State != "released" || files[0].Source.Digest != outputs[0].Digest {
			t.Fatal("collected image has no exact copied receipt and released recipient hold")
		}
		attempts, problem := store.Attempts(request.ID)
		fatal(t, problem)
		if len(attempts) != 0 {
			t.Fatal("file collection created a Creator execution attempt")
		}
		must(t, os.WriteFile(target, []byte("user edit"), 0600))
		internal, err := os.ReadFile(outputs[0].Path)
		must(t, err)
		if string(internal) != string(data) {
			t.Fatal("editing exported media changed internal custody")
		}
	}
}
