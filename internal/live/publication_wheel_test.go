package live

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy-creator/internal/endpointprofile"
	"github.com/cozy-creator/cozy-creator/internal/endpointpublish"
	"github.com/cozy-creator/cozy-creator/internal/wheel"
)

// TestEndpointPublicationWheels is the release-border matrix: one deterministic
// pure project wheel, planted native/build-input refusals, and one separately
// prebuilt native custom wheel whose exact tags/roots/digest are declared without a
// build or retag step.
func TestEndpointPublicationWheels(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, filepath.Join(root, "pyproject.toml"), `[project]
name = "marco-polo"
version = "1.0.0"
`)
	mustWrite(t, filepath.Join(root, "endpoint.toml"), `[application]
object = "marco_polo:app"
`)
	mustWrite(t, filepath.Join(root, "marco_polo.py"), "app = object()\n")
	mustWrite(t, filepath.Join(root, "uv.lock"), "version = 1\n")
	mustWrite(t, filepath.Join(root, "endpoint.release.json"), `{"compatible_accelerator_models":["NVIDIA GeForce RTX 4090"]}`)
	packed, problem := wheel.Pack(wheel.Request{Tree: root, OutDir: filepath.Join(root, "dist")})
	if problem != nil {
		t.Fatalf("pure project pack refused: %s", problem)
	}
	if packed.Fact.Digest != packed.Digest || packed.Fact.Length != packed.Bytes ||
		strings.Join(packed.Fact.Tags, ",") != wheel.Tag ||
		strings.Join(packed.Fact.ImportRoots, ",") != "marco_polo" {
		t.Fatalf("project WheelFact disagrees with packed bytes: %+v / %+v", packed, packed.Fact)
	}
	for _, forbidden := range []string{"pyproject.toml", "uv.lock", "endpoint.release.json"} {
		if slices.Contains(packed.Entries, forbidden) {
			t.Fatalf("publication-only metadata entered the runtime project wheel: %s", forbidden)
		}
	}

	mustWrite(t, filepath.Join(root, "marco_polo.so"), "native")
	if _, problem := wheel.Pack(wheel.Request{Tree: root, OutDir: filepath.Join(root, "dist2")}); problem == nil || problem.ErrName() != "project_wheel_native_file" {
		t.Fatalf("native project file was not refused by name: %v", problem)
	}
	must(t, os.Remove(filepath.Join(root, "marco_polo.so")))
	mustWrite(t, filepath.Join(root, "setup.py"), "raise SystemExit('must not run')\n")
	if _, problem := wheel.Pack(wheel.Request{Tree: root, OutDir: filepath.Join(root, "dist3")}); problem == nil || problem.ErrName() != "project_wheel_build_input" {
		t.Fatalf("project build input was not refused by name: %v", problem)
	}

	custom := filepath.Join(t.TempDir(), "custom_op-1.2.3-cp312-cp312-manylinux_2_28_x86_64.whl")
	writeTestWheel(t, custom, "custom_op", "1.2.3", false,
		[]string{"cp312-cp312-manylinux_2_28_x86_64"}, map[string][]byte{
			"custom_op/__init__.py": []byte("from ._native import run\n"),
			"custom_op/_native.so":  []byte("prebuilt-native-fixture"),
		})
	fact, problem := wheel.Inspect(custom, wheel.CustomWheel)
	if problem != nil {
		t.Fatalf("prebuilt custom wheel refused: %s", problem)
	}
	if fact.Distribution != "custom-op" || fact.Version != "1.2.3" ||
		strings.Join(fact.ImportRoots, ",") != "custom_op" || fact.Length <= 0 {
		t.Fatalf("wrong custom WheelFact: %+v", fact)
	}
	if _, problem := wheel.Inspect(custom, wheel.ProjectWheel); problem == nil ||
		problem.ErrName() != "project_wheel_native_file" {
		t.Fatalf("native custom bytes masqueraded as a project wheel: %v", problem)
	}

	nonpure := filepath.Join(t.TempDir(), "marco_polo-1.0.0-cp312-cp312-linux_x86_64.whl")
	writeTestWheel(t, nonpure, "marco_polo", "1.0.0", false,
		[]string{"cp312-cp312-linux_x86_64"}, map[string][]byte{"marco_polo.py": []byte("app=1\n")})
	if _, problem := wheel.Inspect(nonpure, wheel.ProjectWheel); problem == nil ||
		problem.ErrName() != "project_wheel_not_pure" {
		t.Fatalf("non-pure project tag was not refused: %v", problem)
	}
	mislabeled := filepath.Join(t.TempDir(), "custom_op-1.2.3-py3-none-any.whl")
	writeTestWheel(t, mislabeled, "custom_op", "1.2.3", true,
		[]string{"py3-none-any"}, map[string][]byte{"custom_op/_native.so": []byte("native")})
	if _, problem := wheel.Inspect(mislabeled, wheel.CustomWheel); problem == nil ||
		problem.ErrName() != "custom_wheel_native_mislabeled" {
		t.Fatalf("native custom wheel mislabeled pure was not refused: %v", problem)
	}
	oddPure := filepath.Join(t.TempDir(), "helper-1.0.0-cp312-none-linux_x86_64.whl")
	writeTestWheel(t, oddPure, "helper", "1.0.0", true,
		[]string{"cp312-none-linux_x86_64"}, map[string][]byte{"helper.py": []byte("value=1\n")})
	if fact, problem := wheel.Inspect(oddPure, wheel.CustomWheel); problem != nil || fact.Native {
		t.Fatalf("platform-tagged but measured-pure custom wheel was misclassified: %+v %v", fact, problem)
	}

	profiles, problem := endpointprofile.NormalizeSet([]string{endpointprofile.CU130, endpointprofile.CU126, endpointprofile.CU130})
	if problem != nil || strings.Join(profiles, ",") != endpointprofile.CU126+","+endpointprofile.CU130 {
		t.Fatalf("profile set is not sorted/deduplicated: %v %v", profiles, problem)
	}
	if _, problem := endpointprofile.NormalizeSet(nil); problem == nil || problem.ErrName() != "usage" {
		t.Fatalf("profile default unexpectedly exists: %v", problem)
	}
}

func TestEndpointPublishPackage(t *testing.T) {
	repo := t.TempDir()
	mustWrite(t, filepath.Join(repo, "pyproject.toml"), `[project]
name = "marco-polo"
version = "1.0.0"
dependencies = []
`)
	mustWrite(t, filepath.Join(repo, "endpoint.toml"), `[application]
object = "marco_polo:app"
`)
	mustWrite(t, filepath.Join(repo, "marco_polo.py"), "app = object()\n")
	mustWrite(t, filepath.Join(repo, "uv.lock"), "version = 1\n")
	mustWrite(t, filepath.Join(repo, "endpoint.descriptor.json"), `{
  "jobs": [],
  "format": "cozy.endpoint.descriptor/1",
  "entrypoints": [],
  "application": "marco_polo:app"
}`)
	mustWrite(t, filepath.Join(repo, "endpoint.release.json"), `{
  "model_roots": [],
  "compatible_accelerator_models": ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"],
  "native_wheel_proof": {"fixture":"custom_op:run","expected_result_digest":"sha256:`+strings.Repeat("c", 64)+`"}
}`)
	git(t, repo, "init", "-q")
	git(t, repo, "config", "user.email", "fixture@example.invalid")
	git(t, repo, "config", "user.name", "Fixture")
	git(t, repo, "add", ".")
	git(t, repo, "commit", "-qm", "fixture")

	custom := filepath.Join(t.TempDir(), "custom_op-1.2.3-cp312-cp312-manylinux_2_28_x86_64.whl")
	writeTestWheel(t, custom, "custom_op", "1.2.3", false,
		[]string{"cp312-cp312-manylinux_2_28_x86_64"}, map[string][]byte{
			"custom_op/__init__.py": []byte("from ._native import run\n"),
			"custom_op/_native.so":  []byte("prebuilt-native-fixture"),
		})
	request := endpointpublish.Request{Tree: repo, Release: "1.0.0",
		Profiles: []string{endpointprofile.CU130, endpointprofile.CU126},
		CustomWheels: []string{
			endpointprofile.CU130 + "=" + custom,
			endpointprofile.CU126 + "=" + custom,
		},
	}
	a, problem := endpointpublish.Prepare(request)
	if problem != nil {
		t.Fatalf("first package refused: %s", problem)
	}
	defer a.Close()
	if a.Declaration.Format != "tensorhub.endpoint_release_declaration/1" ||
		len(a.Declaration.ModelRoots) != 0 {
		t.Fatalf("weightless declaration format/root set is wrong: %+v", a.Declaration)
	}
	request.Release = "prod_2026-08-28.a" // release grammar is opaque, never a wheel version
	request.Profiles = []string{endpointprofile.CU126, endpointprofile.CU130, endpointprofile.CU126}
	b, problem := endpointpublish.Prepare(request)
	if problem != nil {
		t.Fatalf("second package refused: %s", problem)
	}
	defer b.Close()
	aBytes, _ := a.Declaration.CanonicalBytes()
	bBytes, _ := b.Declaration.CanonicalBytes()
	if !bytes.Equal(aBytes, bBytes) || a.Declaration.SourceArchive != b.Declaration.SourceArchive ||
		a.Declaration.ProjectWheel.Digest != b.Declaration.ProjectWheel.Digest {
		t.Fatalf("same committed tree/profile set did not reproduce\n%s\n%s", aBytes, bBytes)
	}
	if len(a.Declaration.CustomWheels) != 1 || len(a.Declaration.CustomWheels[0].Profiles) != 2 ||
		len(a.Files) != 6 || string(mustRead(t, a.Files["evaluated_config"])) != "{}" {
		t.Fatalf("incomplete declaration or upload role set: %+v files=%v", a.Declaration, a.Files)
	}

	checkpointA := "sha256:" + strings.Repeat("a", 64)
	checkpointZ := "sha256:" + strings.Repeat("f", 64)
	objectRef := `{"digest":"sha256:` + strings.Repeat("b", 64) + `","length":10}`
	mustWrite(t, filepath.Join(repo, "endpoint.descriptor.json"), `{"application":"marco_polo:app","entrypoints":[{"models":[{"class":"Model"}],"name":"marco"}],"format":"cozy.endpoint.descriptor/1","jobs":[]}`)
	git(t, repo, "add", "endpoint.descriptor.json")
	git(t, repo, "commit", "-qm", "declare model input")
	if _, problem := endpointpublish.Prepare(request); problem == nil || problem.ErrName() != "endpoint_model_bindings_absent" {
		t.Fatalf("model-bearing descriptor without bindings did not refuse: %v", problem)
	}
	mustWrite(t, filepath.Join(repo, "endpoint.release.json"), `{
  "compatible_accelerator_models": ["NVIDIA H200"],
  "model_roots": [
    {"org":"cozy","name":"z-model","checkpoint_id":"`+checkpointZ+`"},
    {"org":"cozy","name":"a-model","checkpoint_id":"`+checkpointA+`"},
    {"org":"cozy","name":"z-model","checkpoint_id":"`+checkpointZ+`"}
  ],
  "model_bindings": [
    {"path":"z.path","checkpoint":{"org":"cozy","name":"z-model","checkpoint_id":"`+checkpointZ+`"},"config":{"assets":[],"document":`+objectRef+`},"execution_layout":[{"component":"transformer","root":{"org":"cozy","name":"z-model","checkpoint_id":"`+checkpointZ+`"}}],"hardware_variant":"sm90"},
    {"path":"a.path","checkpoint":{"org":"cozy","name":"a-model","checkpoint_id":"`+checkpointA+`"},"config":{"assets":[{"name":"tokenizer","ref":`+objectRef+`}],"document":`+objectRef+`},"execution_layout":[{"component":"encoder","root":{"org":"cozy","name":"a-model","checkpoint_id":"`+checkpointA+`"}}],"hardware_variant":"sm90"}
  ],
  "native_wheel_proof": {"fixture":"custom_op:run","expected_result_digest":"sha256:`+strings.Repeat("c", 64)+`"}
}`)
	git(t, repo, "add", "endpoint.release.json")
	git(t, repo, "commit", "-qm", "bind models")
	modelPackage, problem := endpointpublish.Prepare(request)
	if problem != nil {
		t.Fatalf("model-bearing package refused: %s", problem)
	}
	defer modelPackage.Close()
	if len(modelPackage.Declaration.ModelRoots) != 2 ||
		modelPackage.Declaration.ModelRoots[0].Name != "a-model" ||
		len(modelPackage.Declaration.ModelBindings) != 2 ||
		modelPackage.Declaration.ModelBindings[0].Path != "a.path" {
		t.Fatalf("model roots/bindings are not canonical: %+v %+v",
			modelPackage.Declaration.ModelRoots, modelPackage.Declaration.ModelBindings)
	}

	mustWrite(t, filepath.Join(repo, "uncommitted.py"), "x=1\n")
	if _, problem := endpointpublish.Prepare(request); problem == nil || problem.ErrName() != "endpoint_tree_dirty" {
		t.Fatalf("dirty source tree did not refuse: %v", problem)
	}
	must(t, os.Remove(filepath.Join(repo, "uncommitted.py")))
	mustWrite(t, filepath.Join(repo, "native.so"), "native")
	git(t, repo, "add", "native.so")
	git(t, repo, "commit", "-qm", "plant native")
	if _, problem := endpointpublish.Prepare(request); problem == nil || problem.ErrName() != "endpoint_source_native_input" {
		t.Fatalf("committed native source did not refuse before packaging: %v", problem)
	}
}

func TestEndpointPublicationRequiresAcceleratorModel(t *testing.T) {
	repo := trackedEndpointFixture(t)
	mustWrite(t, filepath.Join(repo, "endpoint.release.json"),
		`{"compatible_accelerator_models":[],"model_bindings":[],"model_roots":[]}`)
	git(t, repo, "add", "endpoint.release.json")
	git(t, repo, "commit", "-qm", "remove execution compatibility")
	if _, problem := endpointpublish.Prepare(endpointpublish.Request{Tree: repo, Release: "weightless",
		Profiles: []string{endpointprofile.CU130}}); problem == nil ||
		problem.ErrName() != "endpoint_compatible_accelerator_models_absent" {
		t.Fatalf("empty execution compatibility set did not refuse locally: %v", problem)
	}
}

func mustWrite(t *testing.T, path, body string) {
	t.Helper()
	must(t, os.MkdirAll(filepath.Dir(path), 0o755))
	must(t, os.WriteFile(path, []byte(body), 0o644))
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	body, err := os.ReadFile(path)
	must(t, err)
	return body
}

func git(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %s: %s\n%s", strings.Join(args, " "), err, out)
	}
}

func writeTestWheel(t *testing.T, filename, distribution, version string, pure bool,
	tags []string, payload map[string][]byte,
) {
	t.Helper()
	distInfo := distribution + "-" + version + ".dist-info"
	members := map[string][]byte{}
	for name, body := range payload {
		members[name] = body
	}
	members[distInfo+"/METADATA"] = []byte(fmt.Sprintf(
		"Metadata-Version: 2.1\nName: %s\nVersion: %s\n", distribution, version))
	var wheelBody strings.Builder
	fmt.Fprintf(&wheelBody, "Wheel-Version: 1.0\nRoot-Is-Purelib: %t\n", pure)
	for _, tag := range tags {
		fmt.Fprintf(&wheelBody, "Tag: %s\n", tag)
	}
	members[distInfo+"/WHEEL"] = []byte(wheelBody.String())
	recordName := distInfo + "/RECORD"
	var names []string
	for name := range members {
		names = append(names, name)
	}
	sort.Strings(names)
	var record strings.Builder
	for _, name := range names {
		sum := sha256.Sum256(members[name])
		fmt.Fprintf(&record, "%s,sha256=%s,%d\n", name,
			base64.RawURLEncoding.EncodeToString(sum[:]), len(members[name]))
	}
	fmt.Fprintf(&record, "%s,,\n", recordName)
	members[recordName] = []byte(record.String())
	names = append(names, recordName)

	var body bytes.Buffer
	zw := zip.NewWriter(&body)
	for _, name := range names {
		w, err := zw.Create(name)
		must(t, err)
		_, err = w.Write(members[name])
		must(t, err)
	}
	must(t, zw.Close())
	must(t, os.WriteFile(filename, body.Bytes(), 0o644))
}
