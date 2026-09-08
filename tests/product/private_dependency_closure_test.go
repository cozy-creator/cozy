package producttest

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/csv"
	"encoding/json"
	"flag"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/wheel"
)

var privateRecordedInstall = flag.String("private-recorded-install", "", "captured PackageInstall JSON for actual private closure qualification")
var privateStageDirectory = flag.String("private-stage-directory", "", "owned output directory for private closure qualification")

func TestPrivateRecordedRegistryStage(t *testing.T) {
	if *privateRecordedInstall == "" {
		t.Skip("requires an actual retained install fixture")
	}
	data, err := os.ReadFile(*privateRecordedInstall)
	must(t, err)
	var install records.PackageInstall
	must(t, json.Unmarshal(data, &install))
	directory := *privateStageDirectory
	if directory == "" {
		directory = t.TempDir()
	}
	revision, problem := localpackage.Stage(t.Context(), home.Layout{LocalPackages: directory}, install)
	fatal(t, problem)
	versions := map[string]string{}
	for _, file := range revision.Files {
		identity, problem := wheel.InspectIdentity(file.Path)
		fatal(t, problem)
		versions[identity.Distribution] = identity.Version
	}
	for _, pin := range strings.Split(install.Closure, "\n") {
		name, version, _ := strings.Cut(pin, "==")
		if !packagepublish.ImageOwnedDistribution(name) && versions[name] != version {
			t.Fatalf("captured dependency omitted or changed: %s; supplied=%v", pin, versions)
		}
	}
	encoded, err := json.MarshalIndent(revision, "", "  ")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(directory, "qualification.json"), encoded, 0o600))
	t.Logf("captured %s with %d exact wheels", revision.Digest, len(revision.Files))
}

func privateClosureLock() []byte {
	return []byte(`version = 1
[[package]]
name = "fixture"
version = "1.0"
source = { editable = "." }
[[package]]
name = "numpy"
version = "2.5.3"
source = { registry = "https://pypi.org/simple" }
[[package]]
name = "scipy"
version = "1.18.1"
source = { registry = "https://pypi.org/simple" }
wheels = [{url = "https://files.pythonhosted.org/packages/ab/cd/scipy-1.18.1-py3-none-any.whl", size = 1024, hash = "sha256:` + strings.Repeat("a", 64) + `"}]
[[package]]
name = "unused-extra"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
`)
}

func TestPrivateRegistryClosureIncludesSelectedExtrasAndPinsBase(t *testing.T) {
	closure := "fixture==1.0\nnumpy==2.5.3\nscipy==1.18.1"
	rows, pins, problem := packagepublish.PrivateRegistryRows(privateClosureLock(), closure, "fixture", "1.0", nil)
	fatal(t, problem)
	if len(rows) != 1 || rows[0].Name != "scipy" || rows[0].SHA256 != strings.Repeat("a", 64) || !reflect.DeepEqual(pins, []string{"numpy==2.5.3", "scipy==1.18.1"}) {
		t.Fatalf("selected extra omitted or base silently substituted: rows=%+v pins=%v", rows, pins)
	}
	for _, candidate := range []struct {
		label, closure string
		lock           []byte
	}{
		{"missing extra", closure, bytes.Replace(privateClosureLock(), []byte(`name = "scipy"`), []byte(`name = "other"`), 1)},
		{"base version drift", "fixture==1.0\nnumpy==2.5.2\nscipy==1.18.1", privateClosureLock()},
		{"incompatible wheel", closure, bytes.Replace(privateClosureLock(), []byte("py3-none-any"), []byte("cp311-cp311-win_amd64"), 1)},
		{"unlocked installed package", closure + "\nmissing==1.0", privateClosureLock()},
		{"changed index", closure, bytes.ReplaceAll(privateClosureLock(), []byte("https://pypi.org/simple"), []byte("https://another.invalid/simple"))},
		{"duplicate source", closure, append(privateClosureLock(), []byte("\n[[package]]\nname=\"scipy\"\nversion=\"1.18.1\"\n")...)},
	} {
		t.Run(candidate.label, func(t *testing.T) {
			if _, _, problem := packagepublish.PrivateRegistryRows(candidate.lock, candidate.closure, "fixture", "1.0", nil); problem == nil {
				t.Fatal("incomplete or ambiguous private closure accepted")
			}
		})
	}
}

func TestPrivateWheelPinsPreserveImplementationAndVerifyRecord(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "fixture-1.0-py3-none-any.whl")
	file, err := os.Create(source)
	must(t, err)
	writer := zip.NewWriter(file)
	members := map[string]string{"fixture.py": "VALUE = 123\n", "fixture-1.0.dist-info/METADATA": "Metadata-Version: 2.3\nName: fixture\nVersion: 1.0\nRequires-Dist: numpy>=1.26\nProvides-Extra: audio\n\nOriginal description.\n", "fixture-1.0.dist-info/RECORD": "", "fixture-1.0.dist-info/package-interface.json": "{\"unchanged\":true}"}
	for _, name := range []string{"fixture.py", "fixture-1.0.dist-info/METADATA", "fixture-1.0.dist-info/RECORD", "fixture-1.0.dist-info/package-interface.json"} {
		out, err := writer.Create(name)
		must(t, err)
		_, err = io.WriteString(out, members[name])
		must(t, err)
	}
	must(t, writer.Close())
	must(t, file.Close())
	target := filepath.Join(root, "sealed", "fixture-1.0-py3-none-any.whl")
	fatal(t, wheel.PinDependencies(source, target, []string{"numpy==2.5.3", "scipy==1.18.1"}))
	archive, err := zip.OpenReader(target)
	must(t, err)
	defer archive.Close()
	actual := map[string][]byte{}
	for _, member := range archive.File {
		in, err := member.Open()
		must(t, err)
		data, err := io.ReadAll(in)
		must(t, err)
		must(t, in.Close())
		actual[member.Name] = data
	}
	if string(actual["fixture.py"]) != members["fixture.py"] || string(actual["fixture-1.0.dist-info/package-interface.json"]) != members["fixture-1.0.dist-info/package-interface.json"] {
		t.Fatal("pinning changed executable or interface bytes")
	}
	metadata := string(actual["fixture-1.0.dist-info/METADATA"])
	if !strings.Contains(metadata, "Requires-Dist: numpy==2.5.3\n") || !strings.Contains(metadata, "Requires-Dist: scipy==1.18.1\n") || strings.Contains(metadata, "Provides-Extra:") || strings.Contains(metadata, ">=") {
		t.Fatal("private wheel did not freeze its selected closure")
	}
	record, err := csv.NewReader(bytes.NewReader(actual["fixture-1.0.dist-info/RECORD"])).ReadAll()
	must(t, err)
	if len(record) != len(actual) {
		t.Fatal("RECORD omitted a wheel member")
	}
	for _, row := range record {
		if strings.HasSuffix(row[0], "/RECORD") {
			continue
		}
		sum := sha256.Sum256(actual[row[0]])
		if row[1] != "sha256="+base64.RawURLEncoding.EncodeToString(sum[:]) || row[2] != strconv.Itoa(len(actual[row[0]])) {
			t.Fatal("RECORD no longer verifies pinned wheel bytes")
		}
	}
	second := filepath.Join(root, "again", "fixture-1.0-py3-none-any.whl")
	fatal(t, wheel.PinDependencies(source, second, []string{"numpy==2.5.3", "scipy==1.18.1"}))
	a, err := os.ReadFile(target)
	must(t, err)
	b, err := os.ReadFile(second)
	must(t, err)
	if !bytes.Equal(a, b) {
		t.Fatal("private metadata pinning is not deterministic")
	}
}
