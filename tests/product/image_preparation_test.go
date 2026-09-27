package producttest

import (
	"bytes"
	"encoding/json"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// Client-side preparation is optional work: a policy this host cannot apply exactly keeps
// the interface usable and sends raw bytes. Only a mistyped member refuses the document.
func TestImagePreparationDescriptorAppliesOnlyExactDecodedPolicies(t *testing.T) {
	decoded := strings.Replace(declaredAssetsInterface, `"parameter":"assets"`, `"parameter":"assets","view":"decoded"`, 1)
	for _, test := range []struct {
		policy, outcome string
	}{
		{`{"profile":"image-fit/1","max_edge":8192,"max_pixels":16777216}`, "prepared"},
		{`{"profile":"image-fit/1","max_edge":64}`, "prepared"},
		{`{"profile":"image-fit/1","max_pixels":256}`, "prepared"},
		{`{"profile":"image-fit/2","max_edge":64}`, "prepared"},
		{`{"profile":"image-fit/1"}`, "raw"},
		{`{"profile":"image-fit/1","max_edge":0}`, "raw"},
		{`{"profile":"image-fit/1","max_pixels":null}`, "raw"},
		{`{"profile":"image-fit/1","max_edge":64,"crop":true}`, "raw"},
		{`{"profile":"image-fit/1","max_edge":true}`, "refused"},
		{`{"profile":"image-fit/1","max_pixels":1.5}`, "refused"},
	} {
		raw := strings.Replace(decoded, `"kind":"image"`, `"kind":"image","prepare":`+test.policy, 1)
		iface, problem := launch.DecodePackageInterface([]byte(raw))
		if (problem != nil) != (test.outcome == "refused") {
			t.Fatalf("policy %s admission: %v", test.policy, problem)
		}
		if problem != nil {
			continue
		}
		ep, problem := iface.Function("run")
		fatal(t, problem)
		if prepared := ep.Assets.Kinds[0].Preparation != nil; prepared != (test.outcome == "prepared") {
			t.Fatalf("policy %s: prepared=%v, want %s", test.policy, prepared, test.outcome)
		}
	}
	policy := `"prepare":{"profile":"image-fit/1","max_edge":64},`
	for name, raw := range map[string]string{
		"raw view":   strings.Replace(declaredAssetsInterface, `"kind":"image"`, policy+`"kind":"image"`, 1),
		"video kind": strings.Replace(strings.Replace(decoded, `"kind":"image"`, policy+`"kind":"image"`, 1), `"kind":"image"`, `"kind":"video"`, 1),
	} {
		iface, problem := launch.DecodePackageInterface([]byte(raw))
		fatal(t, problem)
		ep, problem := iface.Function("run")
		fatal(t, problem)
		if ep.Assets.Kinds[0].Preparation != nil {
			t.Fatalf("%s applied image preparation", name)
		}
	}
}

func TestImagePreparationActualCLI(t *testing.T) {
	if *assetsRuntimeWheel == "" {
		t.Skip("supply -assets-runtime-wheel for the actual image preparation product proof")
	}
	root, err := os.MkdirTemp("", "cozy-image-preparation-")
	must(t, err)
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all"); _ = os.RemoveAll(root) })
	prefix := filepath.Join(t.TempDir(), "host-runtime")
	for _, command := range [][]string{
		{"uv", "venv", "--python", "3.12", prefix},
		{"uv", "pip", "install", "--python", filepath.Join(prefix, "bin", "python"), *assetsRuntimeWheel + "[media]"},
	} {
		if out, err := exec.Command(command[0], command[1:]...).CombinedOutput(); err != nil {
			t.Fatalf("qualified helper prefix: %v %s", err, out)
		}
	}
	runtimeBin := filepath.Join(prefix, "bin", "cozy-runtime") //cozy:allow exact Runtime wheel in the isolated product fixture
	run := func(args ...string) (int, string, string) {
		cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
		cmd.Env = launch.InstallToolEnv(records.PackageInstall{Runtime: runtimeBin}, childEnv(t, root))
		var stdout, stderr bytes.Buffer
		cmd.Stdout, cmd.Stderr = &stdout, &stderr
		_ = cmd.Run()
		return cmd.ProcessState.ExitCode(), stdout.String(), stderr.String()
	}
	project := t.TempDir()
	must(t, os.WriteFile(filepath.Join(project, "prepared_input.py"), []byte(`from typing import Annotated
import msgspec
from cozy_runtime.author import App, AssetBound, AssetLimits, Assets, Image, ImagePreparation
app = App()
class Request(msgspec.Struct):
    prompt: str
class Result(msgspec.Struct):
    width: int
    height: int
    media_type: str
    size_bytes: int
Pictures = Annotated[Assets[Annotated[Image, AssetBound(max_bytes=16384, max_decoded_bytes=1048576)]], AssetLimits(images=1), ImagePreparation(max_edge=64)]
@app.entrypoint
def inspect(payload: Request, assets: Pictures) -> Result:
    assert payload.prompt == "unchanged"
    info = assets.info("reference")
    return Result(assets[0].width, assets[0].height, info.media_type, info.size_bytes)
`), 0600))
	version := runtimeFixtureVersion(t, *assetsRuntimeWheel)
	metadata := `[project]
name = "prepared-input"
version = "1.0.0"
requires-python = ">=3.12,<3.13"
dependencies = ["cozy-runtime[media]>=` + version + `"]
[project.entry-points."cozy.application"]
default = "prepared_input:app"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["prepared_input.py"]
[tool.uv.sources]
cozy-runtime = {path = ` + strconv.Quote(*assetsRuntimeWheel) + `}
`
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject = \"prepared_input:app\"\n"), 0600))
	if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("lock preparation fixture: %v %s", err, out)
	}
	if code, out, err := run("--json", "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("install preparation fixture: %d %s %s", code, out, err)
	}
	photo := filepath.Join(project, "reference.jpg")
	pixels := image.NewRGBA(image.Rect(0, 0, 512, 512))
	random := rand.New(rand.NewPCG(7, 11))
	for y := 0; y < 512; y++ {
		for x := 0; x < 512; x++ {
			pixels.SetRGBA(x, y, color.RGBA{byte(random.Uint32()), byte(random.Uint32()), byte(random.Uint32()), 255})
		}
	}
	file, err := os.Create(photo)
	must(t, err)
	must(t, jpeg.Encode(file, pixels, &jpeg.Options{Quality: 95}))
	must(t, file.Close())
	before, err := os.ReadFile(photo)
	must(t, err)
	if len(before) <= 16384 {
		t.Fatal("fixture does not require preparation before byte admission")
	}
	code, out, errout := run("--json", "run", "local/prepared-input/inspect", "prompt=unchanged", "--asset", "reference="+photo, "--await")
	var result struct {
		Status string `json:"status"`
		Result struct {
			Width, Height int
			MediaType     string `json:"media_type"`
			SizeBytes     int64  `json:"size_bytes"`
		} `json:"result"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &result) != nil || result.Status != "completed" {
		t.Fatalf("actual prepared input failed: %d %s %s", code, out, errout)
	}
	if result.Result.Width != 64 || result.Result.Height != 64 || result.Result.MediaType != "image/webp" || result.Result.SizeBytes > 16384 {
		t.Fatalf("worker did not receive the prepared carrier: %+v", result)
	}
	after, err := os.ReadFile(photo)
	must(t, err)
	if !bytes.Equal(before, after) {
		t.Fatal("client preparation changed the original")
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByReference("1")
	fatal(t, problem)
	if request == nil || len(request.Assets) != 1 {
		t.Fatal("prepared request lost input")
	}
	asset := request.Assets[0]
	if filepath.Dir(asset.LocalPath) != filepath.Join(os.TempDir(), "cozy") ||
		filepath.Base(asset.LocalPath) != strings.TrimPrefix(asset.Digest, "sha256:")+".webp" {
		t.Fatalf("derivative is not Runtime's hash-addressed temporary file: %s", asset.LocalPath)
	}
	if _, err := os.Stat(asset.LocalPath); err != nil {
		t.Fatalf("temporary derivative was removed while request could retry: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "inputs")); !os.IsNotExist(err) {
		t.Fatalf("prepared image was copied into an input store: %v", err)
	}
	remaining, err := filepath.Glob(filepath.Join(root, "tmp", "image-preparation-*"))
	must(t, err)
	if len(remaining) != 0 {
		t.Fatalf("image preparation created request scratch: %v", remaining)
	}
}

func TestImageSourceProbeDoesNotHashTheLargeSource(t *testing.T) {
	path := filepath.Join(t.TempDir(), "source.png")
	file, err := os.Create(path)
	must(t, err)
	must(t, png.Encode(file, image.NewRGBA(image.Rect(0, 0, 2, 2))))
	const sourceLength = int64(5) << 30
	must(t, file.Truncate(sourceLength))
	must(t, file.Close())
	length, mime, problem := inputasset.Probe(path)
	fatal(t, problem)
	if length != sourceLength || mime != "image/png" {
		t.Fatalf("header/stat probe changed facts: %d %s", length, mime)
	}
	if _, problem := inputasset.Fingerprint(path, 1024); problem == nil {
		t.Fatal("admitted byte bound was widened by the header probe")
	}
}
