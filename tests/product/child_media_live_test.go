package producttest

import (
	"bytes"
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

// Real Creator, independent executors, generated interface wheel, staged input
// custody and completed memo lookup; no synthetic control peer.
func TestPrivateChildMediaUsesExistingInputGrantsAndMemo(t *testing.T) {
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires exact candidate Runtime wheel with child media forwarding")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	version := strings.Split(filepath.Base(wheel), "-")[1]
	control := filepath.Join(t.TempDir(), "control")
	uv := func(args ...string) {
		t.Helper()
		out, err := exec.Command("uv", args...).CombinedOutput()
		if err != nil {
			t.Fatalf("uv: %v\n%s", err, out)
		}
	}
	uv("venv", control, "--python", "3.12")
	uv("pip", "install", "--python", filepath.Join(control, "bin", "python"), wheel, "av>=18.1,<19")
	root, err := os.MkdirTemp("", "cozy-child-media-")
	must(t, err)
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
		} else {
			t.Log("media proof retained", root)
		}
	})
	defer tracePrivateChildWait(t, root)()
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
dependencies=["cozy-runtime[media]==%s"%s]
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
	write(child, "pyproject.toml", metadata("private-media-scorer", "scorer", "", ""))
	write(child, "package.toml", "[application]\nobject=\"scorer:app\"\n")
	write(child, "scorer.py", `import hashlib
import msgspec
from typing import Annotated
from cozy_runtime.author import App,Context,ImageAsset,AssetBound,MediaDecoder,invocable
class Input(msgspec.Struct,tag="image"):
    image: Annotated[ImageAsset,AssetBound(max_bytes=4096,max_decoded_bytes=4096)]
class Result(msgspec.Struct):
    digest: str
@invocable(memoize=True)
async def score(ctx:Context,*,media:Input,decoder:MediaDecoder)->Result:
    image=decoder.decode_image(media.image)
    return Result(hashlib.sha256(image.rgb).hexdigest())
app=App()
app.job(score)
`)
	write(project, "pyproject.toml", metadata("private-media-parent", "parent", `,"private-media-scorer==0.1.0"`, "private-media-scorer={path=\"./child\"}"))
	write(project, "package.toml", "[application]\nobject=\"parent:app\"\n")
	write(project, "parent.py", `import msgspec
from typing import Annotated
from cozy_runtime.author import App,Context,ImageAsset,AssetBound
from scorer import score,Input,Result
class Request(msgspec.Struct):
    image: Annotated[ImageAsset,AssetBound(max_bytes=4096,max_decoded_bytes=4096)]
app=App()
@app.job
async def run(ctx:Context,payload:Request)->Result:
    return await score(media=Input(payload.image))
`)
	run := func(args ...string) {
		t.Helper()
		code, out := runCozyPath(t, root, path, args...)
		if code != 0 {
			t.Fatalf("cozy %s [%d]: %s", strings.Join(args, " "), code, out)
		}
	}
	uv("lock", "--project", child)
	uv("lock", "--project", project)
	run("package", "install", project, "--editable", "--json")
	imagePath := filepath.Join(t.TempDir(), "input.png")
	pixels := image.NewRGBA(image.Rect(0, 0, 16, 16))
	pixels.Set(2, 2, color.RGBA{R: 255, A: 255})
	var buffer bytes.Buffer
	must(t, png.Encode(&buffer, pixels))
	must(t, os.WriteFile(imagePath, buffer.Bytes(), 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	var original records.Request
	for index := 0; index < 2; index++ {
		run("run", "local/private-media-parent/run", "--asset", "image="+imagePath, "--await", "--json")
		parent, problem := store.RequestByReference(strconv.Itoa(index*2 + 1))
		fatal(t, problem)
		children, problem := store.Children(parent.ID)
		fatal(t, problem)
		if len(children) != 1 || children[0].State != "succeeded" || len(children[0].Assets) != 1 {
			t.Fatalf("child missing retained media: %+v", children)
		}
		if index == 0 {
			original = children[0]
		} else if children[0].ReusedFrom != original.ID || children[0].Ordinal != 0 {
			t.Fatalf("new parent rescored identical media: %+v", children[0])
		}
	}
}
