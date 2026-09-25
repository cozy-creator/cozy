package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image/png"
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
	run := func(t *testing.T, args ...string) map[string]any {
		t.Helper()
		command := exec.Command(cozyBin, append(args, "--json")...)
		command.Env = childEnv(t, root, "PATH="+path)
		var stderr bytes.Buffer
		command.Stderr = &stderr
		out, err := command.Output()
		if err != nil {
			t.Fatalf("cozy %v: %v\n%s\n%s", args, err, out, stderr.String())
		}
		var result map[string]any
		must(t, json.Unmarshal(out, &result))
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
from cozy_runtime.author import App, AssetBound, Context, FileAsset, ImageAsset, Outputs, invocable
from PIL import Image
class Report(msgspec.Struct, frozen=True):
    facts: Annotated[FileAsset, AssetBound(max_bytes=1024, media_types=("application/json",))]
@invocable(memoize=True)
async def produce(ctx:Context, *, out:Outputs)->Report:
    return Report(out.save_bytes(b'{"ok":true}\n',media_type="application/json"))
class ImageReport(msgspec.Struct, frozen=True):
    image: Annotated[ImageAsset, AssetBound(max_bytes=4096, max_decoded_bytes=768, media_types=("image/png",))]
@invocable(memoize=True)
async def produce_image(ctx:Context, *, out:Outputs)->ImageReport:
    return ImageReport(out.save_image(Image.new("RGB", (16,16), (10,20,30)), format="png"))
app=App()
app.job(produce)
app.job(produce_image)
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
		{"forwarded-image", `from typing import Annotated
from cozy_runtime.author import AssetBound, ImageAsset
from file_producer import produce_image
async def main()->Annotated[ImageAsset,AssetBound(max_bytes=4096,max_decoded_bytes=768,media_types=("image/png",))]:
    return (await produce_image()).image
`, "value", "image/png", nil},
		{"forwarded", `from typing import Annotated
from cozy_runtime.author import AssetBound, FileAsset
from file_producer import produce
async def main()->Annotated[FileAsset,AssetBound(max_bytes=1024,media_types=("application/json",))]:
    return (await produce()).facts
`, "value", "application/json", []byte("{\"ok\":true}\n")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			script := filepath.Join(project, tc.name+".py")
			must(t, os.WriteFile(script, []byte(header+tc.body), 0600))
			directory := filepath.Join(root, "export-"+tc.name)
			result := run(t, "run", script, "--await", "--out", directory)
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
			if tc.media == "image/png" {
				image, err := png.Decode(bytes.NewReader(data))
				must(t, err)
				if image.Bounds().Dx() != 16 || image.Bounds().Dy() != 16 {
					t.Fatal("wrong forwarded image dimensions")
				}
				red, green, blue, _ := image.At(0, 0).RGBA()
				if red != 10*257 || green != 20*257 || blue != 30*257 {
					t.Fatal("wrong forwarded image pixels")
				}
			} else if !bytes.Equal(data, tc.data) {
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

	t.Run("oversized-forwarded-image", func(t *testing.T) {
		script := filepath.Join(project, "oversized-image.py")
		body := `from typing import Annotated
from cozy_runtime.author import AssetBound, ImageAsset
from file_producer import produce_image
async def main()->Annotated[ImageAsset,AssetBound(max_bytes=4096,max_decoded_bytes=767,media_types=("image/png",))]:
    return (await produce_image()).image
`
		must(t, os.WriteFile(script, []byte(header+body), 0600))
		directory := filepath.Join(root, "export-oversized-image")
		command := exec.Command(cozyBin, "run", script, "--await", "--out", directory, "--json")
		command.Env = childEnv(t, root, "PATH="+path)
		output, err := command.CombinedOutput()
		if err == nil || !strings.Contains(string(output), "captured decoded bound") {
			t.Fatalf("oversized image was not refused at collection: %v %s", err, output)
		}
		files, err := filepath.Glob(filepath.Join(directory, "*.png"))
		must(t, err)
		if len(files) != 0 {
			t.Fatal("oversized image was exported before bound validation")
		}
	})
}
