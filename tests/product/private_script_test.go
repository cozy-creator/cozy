package producttest

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

var privateScriptRuntimeWheel = flag.String("script-runtime-wheel", "", "Exact Runtime wheel used for local script product proofs")

func TestPrivateScriptRequiresMainWithoutExecutingSource(t *testing.T) {
	root, err := os.MkdirTemp("", "cozy-script-count-")
	must(t, err)
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all"); _ = os.RemoveAll(root) })
	p := filepath.Join(t.TempDir(), "two.py")
	metadata := "# /// script\n# requires-python = \">=3.12,<3.13\"\n# dependencies = [\"cozy-runtime\"]\n"
	if wheel := *privateScriptRuntimeWheel; wheel != "" {
		metadata += fmt.Sprintf("# [tool.uv.sources]\n# cozy-runtime = {path = %q}\n", wheel)
	}
	metadata += "# ///\n"
	must(t, os.WriteFile(p, []byte(metadata+`raise RuntimeError("client must never execute this module")
def first():
    pass
def second():
    pass
`), 0o600))
	status, out := runCozy(t, root, "run", p, "--json")
	if status == 0 || !strings.Contains(out, "script_main_missing") {
		t.Fatalf("script without main was accepted [%d]: %s", status, out)
	}
	if rows := listInvocations(t, root); len(rows) != 0 {
		t.Fatalf("script without main created work: %+v", rows)
	}
}

// Real private installation, generated descriptor, Worker attempt and output spool.
// No App, request/result DTO or service shim is supplied by the client script.
func TestPrivateScriptTypedOutputsUseNormalAttempt(t *testing.T) {
	integration(t)
	if *privateScriptRuntimeWheel == "" {
		t.Skip("requires the exact candidate Runtime wheel")
	}
	root, err := os.MkdirTemp("", "cozy-typed-")
	must(t, err)
	t.Cleanup(func() {
		path := ""
		for _, value := range childEnv(t, root) {
			if strings.HasPrefix(value, "PATH=") {
				path = strings.TrimPrefix(value, "PATH=")
			}
		}
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("typed script evidence retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	script := filepath.Join(t.TempDir(), "image.py")
	code := fmt.Sprintf(`# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime[media]"]
# [tool.uv.sources]
# cozy-runtime = {path = %q}
# ///
from typing import Annotated
from cozy_runtime.author import AssetBound, ImageFrame, ImageAsset, Outputs, Telemetry

def main(*, out: Outputs, tel: Telemetry) -> Annotated[ImageAsset, AssetBound(media_types=("image/png",))]:
    tel.log("typed main saved a real image")
    return out.save_image(ImageFrame(2, 2, b"\xff\x00\x00" * 4), format="png")
`, *privateScriptRuntimeWheel)
	must(t, os.WriteFile(script, []byte(code), 0o600))
	status, stdout, stderr := runCozyStreams(t, root, "run", script, "--await", "--json")
	if status != 0 {
		t.Fatalf("typed main failed [%d]: %s %s", status, stdout, stderr)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByReference("1")
	fatal(t, problem)
	if row == nil || row.State != "succeeded" {
		t.Fatalf("missing completed typed script: %+v", row)
	}
	if !strings.Contains(stdout, "image/png") {
		t.Fatalf("typed image result missing: %s", stdout)
	}
	failing := strings.Replace(code, `return out.save_image(ImageFrame(2, 2, b"\xff\x00\x00" * 4), format="png")`, `raise ValueError("image-output-canary")`, 1)
	if failing == code {
		t.Fatal("failure fixture did not replace its image writer")
	}
	must(t, os.WriteFile(script, []byte(failing), 0600))
	status, stdout, stderr = runCozyStreams(t, root, "run", script, "--await", "--json")
	if status == 0 || !strings.Contains(stdout+stderr, "image-output-canary") || strings.Contains(stdout+stderr, "panic:") {
		t.Fatalf("declared image failure was not rendered as a typed refusal [%d]: %s %s", status, stdout, stderr)
	}
}

// The model class lives in a captured editable library, outside the script tree.
// Both modules refuse execution; only the static source reader can describe it.
func TestPrivateScriptDescribesCapturedModelWithoutImports(t *testing.T) {
	integration(t)
	root, err := os.MkdirTemp("", "cozy-script-model-description-")
	must(t, err)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		if t.Failed() {
			t.Log("captured model description evidence retained", root)
		} else {
			_ = os.RemoveAll(root)
		}
	})
	project := t.TempDir()
	library := filepath.Join(project, "model_types")
	must(t, os.MkdirAll(library, 0700))
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(`[project]
name="private-script-model-types"
version="0.0.1"
requires-python=">=3.12,<3.13"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["model_types.py"]
`), 0600))
	must(t, os.WriteFile(filepath.Join(library, "model_types.py"), []byte(`raise RuntimeError("model dependency imported during description")
from cozy_runtime.author import Model, uses_components
class Probe(Model[object], encoded_leaves="accept"):
    @uses_components("video_vae")
    def decode(self): pass
`), 0600))
	sources := ""
	if wheel := *privateScriptRuntimeWheel; wheel != "" {
		sources = fmt.Sprintf("# cozy-runtime={path=%q}\n", wheel)
	}
	script := filepath.Join(project, "prepare.py")
	must(t, os.WriteFile(script, []byte(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime>=`+runtimeFixtureVersion(t, *privateScriptRuntimeWheel)+`","private-script-model-types>=0.0.1"]
# [tool.uv.sources]
# private-script-model-types={path="./model_types",editable=true}
`+sources+`# ///
raise RuntimeError("client script imported during description")
from model_types import Probe

def main(*, model: Probe): pass
`), 0600))
	code, out := runCozy(t, root, "run", script, "--describe", "--json")
	var described struct {
		Fields []json.RawMessage `json:"fields"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &described) != nil || described.Fields == nil {
		t.Fatalf("captured model source was not described without imports [%d]: %s", code, out)
	}
	if len(described.Fields) != 0 {
		t.Fatalf("injected model became an ordinary payload field: %s", out)
	}
}
