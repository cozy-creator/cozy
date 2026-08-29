package live

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/endpointpublish"
	"github.com/cozy-creator/cozy-creator/internal/launch"
	"github.com/cozy-creator/cozy-creator/internal/wheel"
)

// TestEndpointPublicationWheels is the release-border matrix: one deterministic
// pure project wheel and planted native/build-input refusals.
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
	packed, problem := wheel.Pack(wheel.Request{Tree: root, OutDir: filepath.Join(root, "dist")})
	if problem != nil {
		t.Fatalf("pure project pack refused: %s", problem)
	}
	if packed.Fact.Digest != packed.Digest || packed.Fact.Length != packed.Bytes ||
		strings.Join(packed.Fact.Tags, ",") != wheel.Tag ||
		strings.Join(packed.Fact.ImportRoots, ",") != "marco_polo" {
		t.Fatalf("project WheelFact disagrees with packed bytes: %+v / %+v", packed, packed.Fact)
	}
	for _, forbidden := range []string{"pyproject.toml", "uv.lock"} {
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
}

func TestEndpointPublishPackage(t *testing.T) {
	repo := t.TempDir()
	mustWrite(t, filepath.Join(repo, "pyproject.toml"), `[project]
name = "marco-polo"
version = "1.0.0"
requires-python = ">=3.12,<3.13"
dependencies = ["torch>=2.13,<2.14", "cozy-runtime==2.0.0"]
`)
	mustWrite(t, filepath.Join(repo, "endpoint.toml"), `[application]
object = "marco_polo:app"
`)
	mustWrite(t, filepath.Join(repo, "marco_polo.py"), "app = object()\n")
	mustWrite(t, filepath.Join(repo, "uv.lock"), "version = 1\n")
	git(t, repo, "init", "-q")
	git(t, repo, "config", "user.email", "fixture@example.invalid")
	git(t, repo, "config", "user.name", "Fixture")
	git(t, repo, "add", ".")
	git(t, repo, "commit", "-qm", "fixture")

	weightlessDescriptor := `{"application":"marco_polo:app","entrypoints":[],"format":"cozy.endpoint.descriptor/1","jobs":[]}`
	request := endpointpublish.Request{Tree: repo, Release: "1.0.0",
		Runtime: fakeDescriptorRuntime(t, weightlessDescriptor)}
	a, problem := endpointpublish.Prepare(request)
	if problem != nil {
		t.Fatalf("first package refused: %s", problem)
	}
	defer a.Close()
	if a.Declaration.Format != "tensorhub.endpoint_release_declaration/1" {
		t.Fatalf("weightless declaration format is wrong: %+v", a.Declaration)
	}
	request.Release = "prod_2026-08-28.a" // release grammar is opaque, never a wheel version
	b, problem := endpointpublish.Prepare(request)
	if problem != nil {
		t.Fatalf("second package refused: %s", problem)
	}
	defer b.Close()
	aBytes, _ := a.Declaration.CanonicalBytes()
	bBytes, _ := b.Declaration.CanonicalBytes()
	if !bytes.Equal(aBytes, bBytes) || a.Declaration.SourceArchive != b.Declaration.SourceArchive ||
		a.Declaration.ProjectWheel.Digest != b.Declaration.ProjectWheel.Digest {
		t.Fatalf("same committed tree did not reproduce\n%s\n%s", aBytes, bBytes)
	}
	if len(a.Files) != 4 {
		t.Fatalf("incomplete declaration or upload role set: %+v files=%v", a.Declaration, a.Files)
	}
	for _, retired := range []string{"profiles", "custom_wheels", "evaluated_config", "compatible_accelerator_models", "model_roots", "model_bindings", "native_wheel_proof"} {
		if bytes.Contains(aBytes, []byte(`"`+retired+`"`)) {
			t.Fatalf("retired declaration field %q remains in %s", retired, aBytes)
		}
	}
	if metadata := wheelMetadata(t, a.Files["project_wheel"]); !strings.Contains(metadata, "Requires-Python: >=3.12,<3.13") ||
		!strings.Contains(metadata, "Requires-Dist: torch>=2.13,<2.14") ||
		!strings.Contains(metadata, "Requires-Dist: cozy-runtime==2.0.0") {
		t.Fatalf("project metadata did not carry declared compatibility requirements:\n%s", metadata)
	}

	request.Runtime = fakeDescriptorRuntime(t,
		`{"application":"marco_polo:app","entrypoints":[{"models":[{"class":"Model","component_use":{},"path":"marco.models.model","stamps":{}}],"name":"marco","request":{"fields":[]},"result":{"fields":[]}}],"format":"cozy.endpoint.descriptor/1","jobs":[]}`)
	if _, problem := endpointpublish.Prepare(request); problem == nil || problem.ErrName() != "endpoint_model_binding_deferred" {
		t.Fatalf("model-bearing publication did not report its explicit deferral: %v", problem)
	}
	request.Runtime = fakeDescriptorRuntime(t, weightlessDescriptor)
	mustWrite(t, filepath.Join(repo, "endpoint.release.json"), `{}`)
	git(t, repo, "add", ".")
	git(t, repo, "commit", "-qm", "plant retired metadata")
	if _, problem := endpointpublish.Prepare(request); problem == nil || problem.ErrName() != "endpoint_metadata_retired" {
		t.Fatalf("retired release metadata did not refuse: %v", problem)
	}
	must(t, os.Remove(filepath.Join(repo, "endpoint.release.json")))
	git(t, repo, "add", "-u")
	git(t, repo, "commit", "-qm", "remove retired metadata")
	mustWrite(t, filepath.Join(repo, "endpoint.descriptor.json"), weightlessDescriptor)
	git(t, repo, "add", "endpoint.descriptor.json")
	git(t, repo, "commit", "-qm", "plant retired descriptor")
	if _, problem := endpointpublish.Prepare(request); problem == nil || problem.ErrName() != "endpoint_metadata_retired" {
		t.Fatalf("committed generated descriptor did not refuse: %v", problem)
	}
	must(t, os.Remove(filepath.Join(repo, "endpoint.descriptor.json")))
	git(t, repo, "add", "-u")
	git(t, repo, "commit", "-qm", "remove retired descriptor")

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

func TestEndpointPublicationNeedsNoPublisherProfile(t *testing.T) {
	repo := trackedEndpointFixture(t)
	pack, problem := endpointpublish.Prepare(endpointpublish.Request{Tree: repo, Release: "weightless",
		Runtime: filepath.Join(repo, ".test-bin", "uv")})
	if problem != nil {
		t.Fatalf("weightless pure-Python publication required publisher compatibility input: %v", problem)
	}
	defer pack.Close()
}

func TestEndpointPublishDerivesDescriptorWithLockedRuntime(t *testing.T) {
	homeDir, err := os.UserHomeDir()
	must(t, err)
	fixture := filepath.Join(homeDir, "cozy_v2", "cozy-runtime", "proofs", "fixtures", "marco-polo-endpoint") //cozy:allow peer Runtime repository fixture
	for _, required := range []string{"pyproject.toml", "uv.lock", "endpoint.toml"} {
		if _, err := os.Stat(filepath.Join(fixture, required)); err != nil {
			t.Skipf("Runtime Marco fixture is not present: %v", err)
		}
	}
	for _, retired := range []string{"endpoint.descriptor.json", "endpoint.release.json", "endpoint.evaluated-config.json"} {
		if _, err := os.Stat(filepath.Join(fixture, retired)); err == nil {
			t.Skipf("Runtime Marco fixture has not landed the source-only hardcut yet: %s", retired)
		}
	}
	pyproject := string(mustRead(t, filepath.Join(fixture, "pyproject.toml")))
	lock := string(mustRead(t, filepath.Join(fixture, "uv.lock")))
	if !strings.Contains(pyproject, "cozy-runtime==") || !strings.Contains(lock, `name = "cozy-runtime"`) { //cozy:allow assertion over peer fixture metadata
		t.Fatal("Marco fixture does not carry cozy-runtime in both pyproject.toml and uv.lock")
	}
	_, problem := config.Load()
	fatal(t, problem)
	pack, problem := endpointpublish.Prepare(endpointpublish.Request{Tree: fixture, Release: "1.0.0"})
	if problem != nil {
		t.Fatalf("real locked Runtime derivation refused: %s", problem.Message)
	}
	defer pack.Close()
	descriptor, problem := launch.DecodeDescriptor(mustRead(t, pack.Files["descriptor"]))
	fatal(t, problem)
	if _, problem := descriptor.Function("marco"); problem != nil {
		t.Fatalf("derived Marco descriptor has no marco function: %s", problem.Message)
	}
}

func fakeDescriptorRuntime(t *testing.T, descriptor string) string {
	t.Helper()
	file := filepath.Join(t.TempDir(), "cozy-runtime") //cozy:allow independent Runtime descriptor fixture
	mustWrite(t, file, "#!/bin/sh\nprintf '%s\\n' '"+strings.ReplaceAll(descriptor, "'", "'\"'\"'")+"'\n")
	must(t, os.Chmod(file, 0o755))
	return file
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

func wheelMetadata(t *testing.T, file string) string {
	t.Helper()
	zr, err := zip.OpenReader(file)
	must(t, err)
	defer zr.Close()
	for _, member := range zr.File {
		if !strings.HasSuffix(member.Name, ".dist-info/METADATA") {
			continue
		}
		reader, err := member.Open()
		must(t, err)
		body, err := io.ReadAll(reader)
		_ = reader.Close()
		must(t, err)
		return string(body)
	}
	t.Fatal("wheel has no METADATA")
	return ""
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
