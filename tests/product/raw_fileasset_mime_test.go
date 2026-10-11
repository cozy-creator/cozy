package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// A real typed FileAsset callable on the Rust machine through ordinary CLI commands.
// The exact authored octet-stream/131072-byte policy remains in force; no GPU work occurs.
func TestRawFileAssetRetainsOctetStreamAndAuthoredBoundsThroughOrdinaryCLI(t *testing.T) {
	if *machineHostBinary == "" || *machineRuntimeWheel == "" {
		t.Skip("requires -machine-host and exact -machine-runtime-wheel/-machine-tensorfs-wheel CPU cohort")
	}
	t.Setenv("CUDA_VISIBLE_DEVICES", "")
	root, err := os.MkdirTemp(os.TempDir(), "czrawasset")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("evidence retained at %s\nmachine log:\n%s", root, tail(filepath.Join(root, "machine", "host.log")))
		} else {
			_ = removeAllForce(root)
		}
	})
	project := t.TempDir()
	wheel, err := filepath.Abs(*machineRuntimeWheel)
	must(t, err)
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(fmt.Sprintf(`[project]
name="raw-fileasset-proof"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=%s","msgspec>=0.19,<1"]
[tool.uv.sources]
cozy-runtime={path=%q}
[project.entry-points."cozy.application"]
default="raw_asset:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["raw_asset.py"]
`, runtimeFixtureVersion(t, wheel), wheel)), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"raw_asset:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "raw_asset.py"), []byte(`import hashlib
from typing import Annotated
import msgspec
from cozy_runtime.author import App, FileAsset, AssetBound
app = App()
class Request(msgspec.Struct):
    latent: Annotated[FileAsset, AssetBound(max_bytes=131072, media_types=("application/octet-stream",))]
class Result(msgspec.Struct):
    length: int
    digest: str
    media_type: str
@app.entrypoint
def measure(payload: Request) -> Result:
    data = payload.latent.read_bytes()
    return Result(len(data), hashlib.sha256(data).hexdigest(), payload.latent.media_type)
`), 0o600))
	if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v\n%s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", project); code != 0 {
		t.Fatalf("package install [%d]\n%s", code, out)
	}
	data := bytes.Repeat([]byte{0, 255, 127, 128}, 131072/4)
	path := filepath.Join(root, "opaque.bin")
	must(t, os.WriteFile(path, data, 0o600))
	digest := sha256.Sum256(data)
	code, out := runCozy(t, root, "run", "local/raw-fileasset-proof/measure", "--asset", "latent="+path, "--await", "--json")
	if code != 0 {
		t.Fatalf("explicit octet-stream input rejected [%d]\n%s", code, out)
	}
	check := func(document string) {
		var value struct {
			Result struct {
				Length    int64  `json:"length"`
				Digest    string `json:"digest"`
				MediaType string `json:"media_type"`
			} `json:"result"`
		}
		if err := json.Unmarshal([]byte(lastJSONLine(document)), &value); err != nil {
			t.Fatalf("result: %v\n%s", err, document)
		}
		if value.Result.Length != int64(len(data)) || value.Result.Digest != hex.EncodeToString(digest[:]) || value.Result.MediaType != "application/octet-stream" {
			t.Fatalf("machine changed byte identity/type: %s", document)
		}
	}
	check(out)
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("owned controller down [%d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "run", "show", "1", "--json")
	if code != 0 {
		t.Fatalf("reopened result [%d]\n%s", code, out)
	}
	check(out)
	tooLarge := filepath.Join(root, "too-large.bin")
	must(t, os.WriteFile(tooLarge, append(append([]byte(nil), data...), 0), 0o600))
	code, out = runCozy(t, root, "run", "local/raw-fileasset-proof/measure", "--asset", "latent="+tooLarge, "--await", "--json")
	if code == 0 || !strings.Contains(out, "131072") {
		t.Fatalf("authored byte bound was relaxed [%d]\n%s", code, out)
	}
	var picture bytes.Buffer
	must(t, png.Encode(&picture, image.NewRGBA(image.Rect(0, 0, 2, 2))))
	known := filepath.Join(root, "actual-png.bin")
	must(t, os.WriteFile(known, picture.Bytes(), 0o600))
	code, out = runCozy(t, root, "run", "local/raw-fileasset-proof/measure", "--asset", "latent="+known, "--await", "--json")
	if code == 0 || !strings.Contains(out, "image/png") {
		t.Fatalf("known MIME was relabeled or authored policy removed [%d]\n%s", code, out)
	}
}
