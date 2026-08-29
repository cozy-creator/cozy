package live

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy-creator/internal/endpointprofile"
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
	packed, problem := wheel.Pack(wheel.Request{Tree: root, OutDir: filepath.Join(root, "dist")})
	if problem != nil {
		t.Fatalf("pure project pack refused: %s", problem)
	}
	if packed.Fact.Digest != packed.Digest || packed.Fact.Length != packed.Bytes ||
		strings.Join(packed.Fact.Tags, ",") != wheel.Tag ||
		strings.Join(packed.Fact.ImportRoots, ",") != "marco_polo" {
		t.Fatalf("project WheelFact disagrees with packed bytes: %+v / %+v", packed, packed.Fact)
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

	profiles, problem := endpointprofile.NormalizeSet([]string{endpointprofile.CU130, endpointprofile.CU126, endpointprofile.CU130})
	if problem != nil || strings.Join(profiles, ",") != endpointprofile.CU126+","+endpointprofile.CU130 {
		t.Fatalf("profile set is not sorted/deduplicated: %v %v", profiles, problem)
	}
	if _, problem := endpointprofile.NormalizeSet(nil); problem == nil || problem.ErrName() != "usage" {
		t.Fatalf("profile default unexpectedly exists: %v", problem)
	}
}

func mustWrite(t *testing.T, path, body string) {
	t.Helper()
	must(t, os.MkdirAll(filepath.Dir(path), 0o755))
	must(t, os.WriteFile(path, []byte(body), 0o644))
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
