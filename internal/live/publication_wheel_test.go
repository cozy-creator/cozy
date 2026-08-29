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
	mustWrite(t, filepath.Join(root, "pyproject.toml"), `[build-system]
requires = ["uv_build>=0.9.18,<0.10"]
build-backend = "uv_build"

[project]
name = "marco-polo"
version = "1.0.0"

[tool.uv.build-backend]
module-root = ""
`)
	mustWrite(t, filepath.Join(root, "endpoint.toml"), `[application]
object = "marco_polo:app"
`)
	mustWrite(t, filepath.Join(root, "marco_polo", "__init__.py"), "app = 'committed'\n")
	mustWrite(t, filepath.Join(root, "uv.lock"), "version = 1\n")
	git(t, root, "init", "-q")
	git(t, root, "config", "user.email", "fixture@example.invalid")
	git(t, root, "config", "user.name", "Fixture")
	git(t, root, "add", ".")
	git(t, root, "commit", "-qm", "baseline")
	mustWrite(t, filepath.Join(root, "marco_polo", "__init__.py"), "app = 'modified'\n")
	mustWrite(t, filepath.Join(root, "marco_polo", "untracked.py"), "value = 'untracked'\n")
	packed, problem := wheel.Build(wheel.Request{Tree: root, OutDir: filepath.Join(root, "wheel-out")})
	if problem != nil {
		t.Fatalf("current working tree build refused: %s", problem)
	}
	if strings.Join(packed.Fact.Tags, ",") != wheel.Tag ||
		strings.Join(packed.Fact.ImportRoots, ",") != "marco_polo" {
		t.Fatalf("project WheelFact disagrees with built bytes: %+v", packed.Fact)
	}
	entries := wheelEntries(t, packed.Path)
	for _, required := range []string{"marco_polo/__init__.py", "marco_polo/untracked.py"} {
		if !slices.Contains(entries, required) {
			t.Fatalf("current working-tree member %s is absent from %v", required, entries)
		}
	}

	mustWrite(t, filepath.Join(root, "marco_polo", "native.so"), "native")
	if _, problem := wheel.Build(wheel.Request{Tree: root, OutDir: filepath.Join(root, "native-out")}); problem == nil || problem.ErrName() != "project_wheel_native_file" {
		t.Fatalf("native project file was not refused by name: %v", problem)
	}
	must(t, os.Remove(filepath.Join(root, "marco_polo", "native.so")))
	mustWrite(t, filepath.Join(root, "marco_polo", ".env.local"), "TOKEN=secret\n")
	if _, problem := wheel.Build(wheel.Request{Tree: root, OutDir: filepath.Join(root, "secret-out")}); problem == nil || problem.ErrName() != "project_wheel_credential" {
		t.Fatalf("project wheel credential was not refused: %v", problem)
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

func TestEndpointPublishSourceRefusals(t *testing.T) {
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
	request := endpointpublish.Request{Tree: repo, Release: "1.0.0"}
	mustWrite(t, filepath.Join(repo, "endpoint.release.json"), `{}`)
	if _, problem := endpointpublish.Prepare(request); problem == nil || problem.ErrName() != "endpoint_metadata_retired" {
		t.Fatalf("retired release metadata did not refuse: %v", problem)
	}
	must(t, os.Remove(filepath.Join(repo, "endpoint.release.json")))
	mustWrite(t, filepath.Join(repo, "endpoint.descriptor.json"), `{}`)
	if _, problem := endpointpublish.Prepare(request); problem == nil || problem.ErrName() != "endpoint_metadata_retired" {
		t.Fatalf("generated descriptor did not refuse: %v", problem)
	}
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
	replay, problem := endpointpublish.Prepare(endpointpublish.Request{Tree: fixture, Release: "opaque-release-id"})
	if problem != nil {
		t.Fatalf("real locked Runtime replay refused: %s", problem.Message)
	}
	defer replay.Close()
	declaration, _ := pack.Declaration.CanonicalBytes()
	replayed, _ := replay.Declaration.CanonicalBytes()
	if !bytes.Equal(declaration, replayed) || len(pack.Files) != 4 {
		t.Fatalf("same source did not reproduce its four-role declaration:\n%s\n%s", declaration, replayed)
	}
	for _, retired := range []string{"profiles", "custom_wheels", "evaluated_config", "compatible_accelerator_models", "model_roots", "model_bindings", "native_wheel_proof"} {
		if bytes.Contains(declaration, []byte(`"`+retired+`"`)) {
			t.Fatalf("retired declaration field %q remains in %s", retired, declaration)
		}
	}
	if metadata := wheelMetadata(t, pack.Files["project_wheel"]); !strings.Contains(metadata, "Requires-Dist: cozy-runtime==0.0.3") {
		t.Fatalf("project wheel lost its Runtime compatibility requirement:\n%s", metadata)
	}
	descriptor, problem := launch.DecodeDescriptor(mustRead(t, pack.Files["descriptor"]))
	fatal(t, problem)
	if _, problem := descriptor.Function("marco"); problem != nil {
		t.Fatalf("derived Marco descriptor has no marco function: %s", problem.Message)
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

func wheelEntries(t *testing.T, file string) []string {
	t.Helper()
	zr, err := zip.OpenReader(file)
	must(t, err)
	defer zr.Close()
	entries := make([]string, 0, len(zr.File))
	for _, member := range zr.File {
		if !member.FileInfo().IsDir() {
			entries = append(entries, member.Name)
		}
	}
	sort.Strings(entries)
	return entries
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
