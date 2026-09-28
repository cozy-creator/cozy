package producttest

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A real parent forwards a reordered Assets collection to an independent child
// executor. Labels participate in memo identity without changing content hashes.
func TestDeclaredAssetsManagedLabelsAndMemo(t *testing.T) {
	integration(t)
	if *privateScriptRuntimeWheel == "" {
		t.Skip("supply -script-runtime-wheel for the composed Runtime Assets proof")
	}
	wheel, err := filepath.Abs(*privateScriptRuntimeWheel)
	must(t, err)
	version := runtimeFixtureVersion(t, wheel)
	root, err := os.MkdirTemp("", "cozy-assets-child-")
	must(t, err)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		if !t.Failed() {
			_ = os.RemoveAll(root)
		} else {
			t.Log("Assets child proof retained", root)
		}
	})
	project := t.TempDir()
	child := filepath.Join(project, "child")
	must(t, os.MkdirAll(child, 0700))
	write := func(dir, name, body string) {
		t.Helper()
		must(t, os.WriteFile(filepath.Join(dir, name), []byte(body), 0600))
	}
	metadata := func(name, module, extra, sources string) string {
		return fmt.Sprintf(`[project]
name=%q
version="0.1.0"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime[media]>=%s"%s]
[tool.uv.sources]
cozy-runtime={path=%q}
%s
[project.entry-points."cozy.application"]
default=%q
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=[%q]
`, name, version, extra, wheel, sources, module+":app", module+".py")
	}
	write(child, "pyproject.toml", metadata("labelled-assets-child", "label_child", "", ""))
	write(child, "package.toml", "[application]\nobject=\"label_child:app\"\n")
	write(child, "label_child.py", `from typing import Annotated
import msgspec
from cozy_runtime.author import App, Assets, AssetBound, AssetLimits, Image, Context, invocable
Pictures=Annotated[Assets[Annotated[Image,AssetBound(max_bytes=1024,max_decoded_bytes=4096)]],AssetLimits(images=2,total=3),msgspec.Meta(min_length=1)]
class Result(msgspec.Struct):
    labels: list[str]
    ids: list[str]
    rgb: list[str]
    fidelities: list[str]
@invocable(memoize=True)
async def inspect_assets(ctx: Context, *, assets: Pictures) -> Result:
    ctx.raise_if_cancelled()
    return Result([assets.info(i).label for i in range(len(assets))],[assets.info(i).id for i in range(len(assets))],[image.tobytes().hex() for image in assets],[assets.info(i).fidelity for i in range(len(assets))])
app=App()
app.job(inspect_assets)
`)
	write(project, "pyproject.toml", metadata("labelled-assets-parent", "label_parent", `,"labelled-assets-child>=0.1.0"`, `labelled-assets-child={path="./child"}`))
	write(project, "package.toml", "[application]\nobject=\"label_parent:app\"\n")
	write(project, "label_parent.py", `from typing import Annotated
import msgspec
from cozy_runtime.author import App, Assets, AssetBound, AssetLimits, Image, Context, invocable
from label_child import Result, inspect_assets
Pictures=Annotated[Assets[Annotated[Image,AssetBound(max_bytes=1024,max_decoded_bytes=4096)]],AssetLimits(images=2,total=3),msgspec.Meta(min_length=1)]
app=App()
@invocable(memoize=False)
async def run(ctx: Context, *, assets: Pictures) -> Result:
    return await inspect_assets(assets=assets.select(1,0))
app.job(run)
`)
	for _, dir := range []string{child, project} {
		if out, err := exec.Command("uv", "lock", "--project", dir).CombinedOutput(); err != nil {
			t.Fatalf("lock fixture: %v %s", err, out)
		}
	}
	if code, out, stderr := runCozyStreams(t, root, "--json", "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("install parent: %d %s %s", code, out, stderr)
	}
	photo := filepath.Join(t.TempDir(), "photo.png")
	f, err := os.Create(photo)
	must(t, err)
	pixels := image.NewRGBA(image.Rect(0, 0, 2, 2))
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			pixels.Set(x, y, color.RGBA{R: 255, A: 255})
		}
	}
	must(t, png.Encode(f, pixels))
	must(t, f.Close())
	st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer st.Close()
	var previous machineChildProof
	for index, item := range []struct{ label, fidelity string }{{"alice", "high"}, {"alice", "high"}, {"carol", "high"}, {"carol", "low"}} {
		label := item.label
		code, out, stderr := runCozyStreams(t, root, "--json", "run", "local/labelled-assets-parent/run", "--asset", label+"="+photo, "--asset", "bob="+photo, "--asset-fidelity", label+"="+item.fidelity, "--asset-fidelity", "bob=medium", "--await")
		var answer struct {
			Status string `json:"status"`
			Result struct {
				Labels     []string `json:"labels"`
				IDs        []string `json:"ids"`
				RGB        []string `json:"rgb"`
				Fidelities []string `json:"fidelities"`
			} `json:"result"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &answer) != nil || answer.Status != "completed" {
			t.Fatalf("parent invocation: %d %s %s", code, out, stderr)
		}
		if len(answer.Result.Labels) != 2 || answer.Result.Labels[0] != "bob" || answer.Result.Labels[1] != label || answer.Result.IDs[0] != "assets.0.asset" || answer.Result.IDs[1] != "assets.1.asset" || answer.Result.RGB[0] != strings.Repeat("ff0000", 4) || answer.Result.RGB[1] != answer.Result.RGB[0] || len(answer.Result.Fidelities) != 2 || answer.Result.Fidelities[0] != "medium" || answer.Result.Fidelities[1] != item.fidelity {
			t.Fatalf("child lost labels/order/decode: %+v", answer.Result)
		}
		// The parent is a machine execution; its managed child lives in Runtime's journal.
		children := machineChildren(t, root, st, strconv.Itoa(index+1))
		if len(children) != 1 || children[0].State != "succeeded" || children[0].Computation == "" {
			t.Fatalf("child custody missing: %+v", children)
		}
		current := children[0]
		switch {
		case index == 0 && current.Executions != 1:
			t.Fatalf("first labelled child did not execute: %+v", current)
		case index == 1 && (current.Executions != 0 || current.Computation != previous.Computation):
			t.Fatalf("same labelled inputs recomputed: %+v", current)
		case index == 2 && (current.Executions != 1 || current.Computation == previous.Computation):
			t.Fatalf("changed label reused prior child: %+v", current)
		case index == 3 && (current.Executions != 1 || current.Computation == previous.Computation):
			t.Fatalf("changed fidelity reused prior child: %+v", current)
		}
		previous = current
	}
}
