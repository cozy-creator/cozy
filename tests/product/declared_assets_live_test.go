package producttest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"image"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestDeclaredAssetsActualCallable(t *testing.T) {
	integration(t)
	if *privateScriptRuntimeWheel == "" {
		t.Skip("supply -script-runtime-wheel for the composed Runtime Assets proof")
	}
	root, err := os.MkdirTemp("", "cozy-assets-")
	must(t, err)
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all"); _ = os.RemoveAll(root) })
	project := t.TempDir()
	version := runtimeFixtureVersion(t, *privateScriptRuntimeWheel)
	script := filepath.Join(project, "assets_app.py")
	code := `
from typing import Annotated
import msgspec
from cozy_runtime.author import App, Assets, AssetBound, AssetLimits, Image, Context, invocable
Pictures = Annotated[Assets[Annotated[Image, AssetBound(max_bytes=1024, max_decoded_bytes=4096)]], AssetLimits(images=2,total=3), msgspec.Meta(min_length=1)]
OptionalPictures = Annotated[Assets[Annotated[Image, AssetBound(max_bytes=1024, max_decoded_bytes=4096)]], AssetLimits(images=2,total=3)]
class Result(msgspec.Struct):
    labels: list[str]
    ids: list[str]
    positions: list[int]
    sizes: list[int]
    fidelities: list[str]
app = App()
def result(assets: Pictures) -> Result:
    assert assets.info("艾丽丝").position == 0
    assert assets["艾丽丝"].size == (2, 2)
    return Result([assets.info(i).label for i in range(len(assets))],[assets.info(i).id for i in range(len(assets))],[assets.info(i).position for i in range(len(assets))],[assets.info(i).size_bytes for i in range(len(assets))],[assets.info(i).fidelity for i in range(len(assets))])
@invocable(memoize=False)
async def main(ctx: Context, *, assets: Pictures, fail: bool = False) -> Result:
    ctx.raise_if_cancelled()
    got = result(assets)
    if fail:
        raise ValueError("deliberate asset retry")
    return got
app.job(main)
class Request(msgspec.Struct):
    prompt: str
@app.entrypoint
async def collect(payload: Request, assets: Pictures) -> Result:
    assert payload.prompt == "unchanged"
    return result(assets)
@app.entrypoint
async def empty(payload: Request, assets: OptionalPictures) -> Result:
    assert payload.prompt == "text only"
    assert len(assets) == 0
    return Result([], [], [], [], [])
`
	must(t, os.WriteFile(script, []byte(code), 0600))
	metadata := `[project]
name = "cozy-assets-proof"
version = "1.0.0"
requires-python = ">=3.12,<3.13"
dependencies = ["cozy-runtime[media]>=` + version + `"]
[project.entry-points."cozy.application"]
default = "assets_app:app"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["assets_app.py"]
[tool.uv.sources]
cozy-runtime = {path = ` + strconv.Quote(*privateScriptRuntimeWheel) + `}
`
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte(`[application]
object = "assets_app:app"
`), 0600))
	if output, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("lock Assets fixture: %v %s", err, output)
	}
	if status, stdout, stderr := runCozyStreams(t, root, "--json", "package", "install", project, "--editable"); status != 0 {
		t.Fatalf("Assets package install failed: %d %s %s", status, stdout, stderr)
	}

	photo := filepath.Join(project, "same #100%?.png")
	file, err := os.Create(photo)
	must(t, err)
	must(t, png.Encode(file, image.NewRGBA(image.Rect(0, 0, 2, 2))))
	must(t, file.Close())

	for _, function := range []string{"collect", "main"} {
		t.Run(function, func(t *testing.T) {
			args := []string{"--json", "run", "local/cozy-assets-proof/" + function}
			if function == "collect" {
				args = append(args, "prompt=unchanged")
			}
			args = append(args, "--asset", "艾丽丝="+photo, "--asset", photo, "--asset-fidelity", "艾丽丝=high", "--asset-fidelity", "1=low", "--await")
			status, stdout, stderr := runCozyStreams(t, root, args...)
			var result struct {
				Status string `json:"status"`
				Result struct {
					Labels     []string `json:"labels"`
					IDs        []string `json:"ids"`
					Positions  []int    `json:"positions"`
					Sizes      []int    `json:"sizes"`
					Fidelities []string `json:"fidelities"`
				} `json:"result"`
			}
			if status != 0 || json.Unmarshal([]byte(stdout), &result) != nil || result.Status != "completed" {
				t.Fatalf("actual Assets job failed: code=%d stdout=%s stderr=%s", status, stdout, stderr)
			}
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			reference := "1"
			if function == "main" {
				reference = "2"
			}
			row, problem := store.RequestByReference(reference)
			fatal(t, problem)
			if row == nil || len(row.Assets) != 2 {
				t.Fatal("completed request lost its input identity")
			}
			photoBytes, err := os.ReadFile(photo)
			must(t, err)
			photoDigest := sha256.Sum256(photoBytes)
			// Serving and jobs alike freeze their inputs into a request-owned stage at
			// admission, so later edits of the original cannot reach the machine.
			for _, asset := range row.Assets {
				if asset.Snapshot == nil || !strings.HasPrefix(asset.LocalPath, filepath.Join(root, "tmp", row.ID)+string(filepath.Separator)) ||
					asset.Digest != "sha256:"+hex.EncodeToString(photoDigest[:]) {
					t.Fatalf("%s input was not frozen in its request stage: %+v", function, asset)
				}
			}
			inputs, problem := store.MachineInputs(row.ID)
			fatal(t, problem)
			if len(inputs) != 2 {
				t.Fatalf("%s recorded %d machine input receipts, want 2", function, len(inputs))
			}
			if _, err := os.Stat(filepath.Join(root, "inputs")); !os.IsNotExist(err) {
				t.Fatalf("submission created an input store: %v", err)
			}
			if _, err := os.Stat(photo); err != nil {
				t.Fatalf("terminal cleanup removed original: %v", err)
			}
			got := result.Result
			if len(got.Labels) != 2 || got.Labels[0] != "艾丽丝" || got.Labels[1] != "" || got.IDs[0] != "assets.0.asset" || got.IDs[1] != "assets.1.asset" || got.Positions[0] != 0 || got.Positions[1] != 1 || got.Sizes[0] <= 0 || got.Sizes[0] != got.Sizes[1] || len(got.Fidelities) != 2 || got.Fidelities[0] != "high" || got.Fidelities[1] != "low" {
				t.Fatalf("actual author received changed assets: %+v", got)
			}
		})
	}
	t.Run("kind-count", func(t *testing.T) {
		code, out, stderr := runCozyStreams(t, root, "--json", "run", "local/cozy-assets-proof/main", "--asset", photo, "--asset", photo, "--asset", photo)
		if code == 0 || !strings.Contains(out+stderr, "maximum") {
			t.Fatalf("authored image limit was ignored: %d %s %s", code, out, stderr)
		}
		st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
		fatal(t, problem)
		defer st.Close()
		row, problem := st.RequestByReference("3")
		fatal(t, problem)
		if row != nil {
			t.Fatal("over-limit Assets invocation queued a request")
		}
	})
	t.Run("retry", func(t *testing.T) {
		code, out, _ := runCozyStreams(t, root, "--json", "run", "local/cozy-assets-proof/main", "fail=true", "--asset", "艾丽丝="+photo, "--asset", photo, "--await")
		if code == 0 || !strings.Contains(out, "deliberate asset retry") {
			t.Fatalf("initial asset job did not fail intentionally: %d %s", code, out)
		}
		st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
		fatal(t, problem)
		defer st.Close()
		prior, problem := st.RequestByReference("3")
		fatal(t, problem)
		if prior == nil || prior.State != "failed" || !st.RetainedRetryAvailable(*prior) || len(prior.Assets) != 2 {
			t.Fatalf("failed job lost retained assets: %+v", prior)
		}
		code, out, _ = runCozyStreams(t, root, "--json", "run", "local/cozy-assets-proof/main", "fail=false", "--retry", prior.ID)
		if code == 0 || !strings.Contains(out, "assets") {
			t.Fatalf("retry silently dropped required asset inputs: %d %s", code, out)
		}
		missing, problem := st.RequestByReference("4")
		fatal(t, problem)
		if missing != nil {
			t.Fatal("missing-assets retry queued a request")
		}
		code, out, stderr := runCozyStreams(t, root, "--json", "run", "local/cozy-assets-proof/main", "fail=false", "--retry", prior.ID, "--asset", "艾丽丝="+photo, "--asset", photo, "--await")
		if code != 0 {
			t.Fatalf("explicit asset retry failed: %d %s %s", code, out, stderr)
		}
		fresh, problem := st.RequestByReference("4")
		fatal(t, problem)
		if fresh == nil || fresh.State != "succeeded" || fresh.RetryOf != prior.ID || len(fresh.Assets) != 2 {
			t.Fatalf("retry lost its input lineage: %+v", fresh)
		}
		for i, asset := range fresh.Assets {
			if asset.Digest != prior.Assets[i].Digest || asset.FieldPath != prior.Assets[i].FieldPath || asset.Order != prior.Assets[i].Order || asset.MediaType != prior.Assets[i].MediaType {
				t.Fatalf("retry changed retained asset identity: %+v", fresh.Assets)
			}
		}
	})
	t.Run("empty", func(t *testing.T) {
		code, out, stderr := runCozyStreams(t, root, "--json", "run", "local/cozy-assets-proof/empty", "prompt=text only", "--await")
		var answer struct {
			Status string `json:"status"`
			Result struct {
				Labels []string `json:"labels"`
			} `json:"result"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &answer) != nil || answer.Status != "completed" || answer.Result.Labels == nil || len(answer.Result.Labels) != 0 {
			t.Fatalf("zero-reference Assets invocation failed: %d %s %s", code, out, stderr)
		}
	})

}
