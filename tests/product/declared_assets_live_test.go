package producttest

import (
	"encoding/json"
	"flag"
	"image"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
)

var assetsRuntimeWheel = flag.String("assets-runtime-wheel", "", "Exact Runtime wheel with the explicit Assets input contract for composed CLI proof")

func TestDeclaredAssetsActualCallable(t *testing.T) {
	if *assetsRuntimeWheel == "" {
		t.Skip("supply -assets-runtime-wheel for the composed Runtime Assets proof")
	}
	root, err := os.MkdirTemp("", "cozy-assets-")
	must(t, err)
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all"); _ = os.RemoveAll(root) })
	project := t.TempDir()
	script := filepath.Join(project, "assets_app.py")
	code := `
from typing import Annotated
import msgspec
from cozy_runtime.author import App, Assets, AssetBound, ImageAsset, Context, invocable
Pictures = Annotated[Assets[Annotated[ImageAsset, AssetBound(max_bytes=1024, max_decoded_bytes=4096)]], msgspec.Meta(min_length=1,max_length=3)]
class Result(msgspec.Struct):
    labels: list[str]
    ids: list[str]
    positions: list[int]
    sizes: list[int]
app = App()
def result(assets: Pictures) -> Result:
    assert assets.by_label("艾丽丝").position == 0
    return Result([a.label for a in assets],[a.id for a in assets],[a.position for a in assets],[len(a.read_bytes()) for a in assets])
@invocable(memoize=False)
async def main(ctx: Context, *, assets: Pictures) -> Result:
    ctx.raise_if_cancelled()
    return result(assets)
app.job(main)
class Request(msgspec.Struct):
    prompt: str
@app.entrypoint
async def collect(payload: Request, assets: Pictures) -> Result:
    assert payload.prompt == "unchanged"
    return result(assets)
`
	must(t, os.WriteFile(script, []byte(code), 0600))
	metadata := `[project]
name = "cozy-assets-proof"
version = "1.0.0"
requires-python = ">=3.12,<3.13"
dependencies = ["cozy-runtime"]
[project.entry-points."cozy.application"]
default = "assets_app:app"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["assets_app.py"]
[tool.uv.sources]
cozy-runtime = {path = ` + strconv.Quote(*assetsRuntimeWheel) + `}
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

	photo := filepath.Join(project, "same.png")
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
			args = append(args, "--asset", "艾丽丝="+photo, "--asset", photo, "--await")
			status, stdout, stderr := runCozyStreams(t, root, args...)
			var result struct {
				Status string `json:"status"`
				Result struct {
					Labels    []string `json:"labels"`
					IDs       []string `json:"ids"`
					Positions []int    `json:"positions"`
					Sizes     []int    `json:"sizes"`
				} `json:"result"`
			}
			if status != 0 || json.Unmarshal([]byte(stdout), &result) != nil || result.Status != "completed" {
				t.Fatalf("actual Assets job failed: code=%d stdout=%s stderr=%s", status, stdout, stderr)
			}
			got := result.Result
			if len(got.Labels) != 2 || got.Labels[0] != "艾丽丝" || got.Labels[1] != "" || got.IDs[0] != "assets.0.asset" || got.IDs[1] != "assets.1.asset" || got.Positions[0] != 0 || got.Positions[1] != 1 || got.Sizes[0] <= 0 || got.Sizes[0] != got.Sizes[1] {
				t.Fatalf("actual author received changed assets: %+v", got)
			}
		})
	}
}
