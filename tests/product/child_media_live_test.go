package producttest

import (
	"bytes"
	"flag"
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

	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/pelletier/go-toml/v2"
)

var privateChildEvalWheel = flag.String("child-eval-wheel", "", "exact Cozy Eval wheel for managed scoring transport")

var privateChildEvalSource = flag.String("child-eval-source", "", "actual editable Cozy Eval project for selected managed-extra capture")

func TestUnpublishedChildEvalManagedExtraUsesDirectEditableLibrary(t *testing.T) {
	if *privateChildEvalSource == "" {
		t.Skip("requires actual editable Cozy Eval project")
	}
	privateChildMediaProof(t, *privateChildEvalSource)
}

// Real Creator, independent executors, generated interface wheel, borrowed input
// custody and completed memo lookup; no synthetic control peer.
func TestUnpublishedChildMediaUsesExistingInputGrantsAndMemo(t *testing.T) {
	privateChildMediaProof(t, "")
}

func privateChildMediaProof(t *testing.T, directSource string) {
	if *privateChildRuntimeWheel == "" || (directSource == "" && *privateChildEvalWheel == "") {
		t.Skip("requires candidate Runtime and Cozy Eval wheels")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	version := runtimeFixtureVersion(t, wheel)
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
	var sourceMetadata, sourceLock []byte
	if directSource != "" {
		_, files, problem := packagepublish.LibrarySourceTree(directSource)
		fatal(t, problem)
		for name, source := range files {
			destination := filepath.Join(child, filepath.FromSlash(name))
			must(t, os.MkdirAll(filepath.Dir(destination), 0700))
			raw, err := os.ReadFile(source)
			must(t, err)
			must(t, os.WriteFile(destination, raw, 0600))
		}
		sourceMetadata, err = os.ReadFile(filepath.Join(child, "pyproject.toml"))
		must(t, err)
		sourceLock, err = os.ReadFile(filepath.Join(child, "uv.lock"))
		must(t, err)
		var parsed struct {
			Project struct {
				Version string `toml:"version"`
			} `toml:"project"`
		}
		must(t, toml.Unmarshal(sourceMetadata, &parsed))
		write(project, "pyproject.toml", metadata("private-media-parent", "parent", `,"cozy-eval[managed]>=`+strings.SplitN(parsed.Project.Version, "+", 2)[0]+`"`, `cozy-eval={path="./child"}`))
		base := filepath.Join(t.TempDir(), "base")
		uv("venv", base, "--python", "3.12")
		uv("pip", "install", "--python", filepath.Join(base, "bin", "python"), child)
		check := exec.Command(filepath.Join(base, "bin", "python"), "-I", "-c", `import sys;from importlib.metadata import distributions;import cozy_eval;assert 'cozy_runtime' not in sys.modules;assert not any(d.metadata['Name']=='cozy-runtime' for d in distributions())`)
		if out, err := check.CombinedOutput(); err != nil {
			t.Fatalf("base library imported Runtime: %v %s", err, out)
		}
	} else {
		evalWheel, err := filepath.Abs(*privateChildEvalWheel)
		must(t, err)
		evalVersion := strings.SplitN(strings.Split(filepath.Base(evalWheel), "-")[1], "+", 2)[0]
		write(child, "pyproject.toml", metadata("private-media-scorer", "scorer", `,"cozy-eval>=`+evalVersion+`"`, "cozy-eval={path="+strconv.Quote(evalWheel)+"}"))
		write(child, "package.toml", "[application]\nobject=\"scorer:app\"\n")
		write(child, "scorer.py", "from cozy_eval.operations import app\n")
		write(project, "pyproject.toml", metadata("private-media-parent", "parent", `,"private-media-scorer>=0.1.0"`, "private-media-scorer={path=\"./child\"}"))
	}
	write(project, "package.toml", "[application]\nobject=\"parent:app\"\n")
	write(project, "parent.py", `import msgspec
from typing import Annotated
from cozy_runtime.author import App,Context,ImageAsset,AssetBound
from cozy_eval.operations import score_pair,ImageInput,ScoreResult
class Request(msgspec.Struct):
    image: Annotated[ImageAsset,AssetBound(max_bytes=4096,max_decoded_bytes=4096)]
app=App()
@app.job
async def run(ctx:Context,payload:Request)->ScoreResult:
    return await score_pair(baseline=ImageInput(image=payload.image),candidate=ImageInput(image=payload.image),metrics=["psnr","ssim"])
`)
	run := func(args ...string) {
		t.Helper()
		if code, out := runCozyPath(t, root, path, args...); code != 0 {
			t.Fatalf("cozy %s [%d]: %s", strings.Join(args, " "), code, out)
		}
	}
	if directSource == "" {
		uv("lock", "--project", child)
	}
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
	var original, previous records.Request
	runs := 3
	if directSource != "" {
		runs = 4
	}
	for index := 0; index < runs; index++ {
		if index == 3 {
			helper := filepath.Join(child, "src", "cozy_eval", "facts.py")
			code, err := os.ReadFile(helper)
			must(t, err)
			must(t, os.WriteFile(helper, append(code, []byte("\n# Same-version editable source capture proof.\n")...), 0600))
		}
		if index == 2 {
			pixels.Set(3, 3, color.RGBA{G: 255, A: 255})
			buffer.Reset()
			must(t, png.Encode(&buffer, pixels))
			must(t, os.WriteFile(imagePath, buffer.Bytes(), 0600))
		}
		run("run", "local/private-media-parent/run", "--asset", "image="+imagePath, "--await", "--json")
		parent, problem := store.RequestByReference(strconv.Itoa(index*2 + 1))
		fatal(t, problem)
		children, problem := store.Children(parent.ID)
		fatal(t, problem)
		if len(children) != 1 || children[0].State != "succeeded" || len(children[0].Assets) != 2 {
			t.Fatalf("child missing retained media: %+v", children)
		}
		if index == 0 {
			original = children[0]
		} else if index == 1 && (children[0].ReusedFrom != original.ID || children[0].Ordinal != 0) {
			t.Fatalf("new parent rescored identical media: %+v", children[0])
		}
		if index >= 2 && (children[0].ReusedFrom != "" || children[0].Ordinal != 1) {
			t.Fatalf("changed media reused an old score: %+v", children[0])
		}
		if index == 3 && children[0].ChildTargetDigest == previous.ChildTargetDigest {
			t.Fatal("same-version metric source edit preserved old implementation identity")
		}
		previous = children[0]
	}
	if directSource != "" {
		actual, err := os.ReadFile(filepath.Join(child, "pyproject.toml"))
		must(t, err)
		if !bytes.Equal(actual, sourceMetadata) {
			t.Fatal("managed capture mutated original editable pyproject")
		}
		actual, err = os.ReadFile(filepath.Join(child, "uv.lock"))
		must(t, err)
		if !bytes.Equal(actual, sourceLock) {
			t.Fatal("managed capture mutated original editable lock")
		}

	}

}
