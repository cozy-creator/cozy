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
	"sort"
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

func TestCapturedRegistryStage(t *testing.T) {
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

func capturedClosureLock() []byte {
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

func TestCapturedRegistryClosureIncludesSelectedExtrasAndLeavesImageBaseToWorker(t *testing.T) {
	closure := "fixture==1.0\nnumpy==2.5.3\nscipy==1.18.1"
	rows, pins, problem := packagepublish.CapturedRegistryRows(capturedClosureLock(), closure, "fixture", "1.0", nil)
	fatal(t, problem)
	if len(rows) != 1 || rows[0].Name != "scipy" || rows[0].SHA256 != strings.Repeat("a", 64) || !reflect.DeepEqual(pins, []string{"scipy==1.18.1"}) {
		t.Fatalf("selected extra omitted or image-owned base was repinned: rows=%+v pins=%v", rows, pins)
	}
	for _, candidate := range []struct {
		label, closure string
		lock           []byte
	}{
		{"missing extra", closure, bytes.Replace(capturedClosureLock(), []byte(`name = "scipy"`), []byte(`name = "other"`), 1)},
		{"base version drift", "fixture==1.0\nnumpy==2.5.2\nscipy==1.18.1", capturedClosureLock()},
		{"incompatible wheel", closure, bytes.Replace(capturedClosureLock(), []byte("py3-none-any"), []byte("cp311-cp311-win_amd64"), 1)},
		{"unlocked installed package", closure + "\nmissing==1.0", capturedClosureLock()},
		{"changed index", closure, bytes.ReplaceAll(capturedClosureLock(), []byte("https://pypi.org/simple"), []byte("https://another.invalid/simple"))},
		{"duplicate source", closure, append(capturedClosureLock(), []byte("\n[[package]]\nname=\"scipy\"\nversion=\"1.18.1\"\n")...)},
	} {
		t.Run(candidate.label, func(t *testing.T) {
			if _, _, problem := packagepublish.CapturedRegistryRows(candidate.lock, candidate.closure, "fixture", "1.0", nil); problem == nil {
				t.Fatal("incomplete or ambiguous private closure accepted")
			}
		})
	}
}

func TestUnpublishedWheelPinsPreserveImplementationAndVerifyRecord(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "fixture-1.0-py3-none-any.whl")
	file, err := os.Create(source)
	must(t, err)
	writer := zip.NewWriter(file)
	members := map[string]string{
		"fixture.py":                                     "VALUE = 123\n",
		"fixture-1.0.dist-info/METADATA":                 "Metadata-Version: 2.3\nName: fixture\nVersion: 1.0\nRequires-Dist: numpy>=1.26\nProvides-Extra: audio\n\nOriginal description.\n",
		"fixture-1.0.dist-info/RECORD":                   "",
		"fixture-1.0.dist-info/package-interface.json":   "{\"unchanged\":true}",
		"fixture/_vendor/other-2.0.dist-info/METADATA":   "Metadata-Version: 2.3\nName: other\nVersion: 2.0\nRequires-Dist: original>=1.0\n\n",
		"fixture/_vendor/other-2.0.dist-info/RECORD":     "original nested record\n",
		"fixture/_vendor/other-2.0.dist-info/RECORD.jws": "original nested signature\n",
	}
	names := make([]string, 0, len(members))
	for name := range members {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
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
	for name, original := range members {
		if name != "fixture-1.0.dist-info/METADATA" && name != "fixture-1.0.dist-info/RECORD" && string(actual[name]) != original {
			t.Fatalf("pinning changed implementation or vendored metadata: %s", name)
		}
	}
	metadata := string(actual["fixture-1.0.dist-info/METADATA"])
	if !strings.Contains(metadata, "Requires-Dist: numpy==2.5.3\n") || !strings.Contains(metadata, "Requires-Dist: scipy==1.18.1\n") || strings.Contains(metadata, "Provides-Extra:") || strings.Contains(metadata, ">=") {
		t.Fatal("captured wheel did not freeze its selected closure")
	}
	record, err := csv.NewReader(bytes.NewReader(actual["fixture-1.0.dist-info/RECORD"])).ReadAll()
	must(t, err)
	if len(record) != len(actual) {
		t.Fatal("RECORD omitted a wheel member")
	}
	for _, row := range record {
		if row[0] == "fixture-1.0.dist-info/RECORD" {
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

func TestUnpublishedWheelPreservesImageOwnedRequirement(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "fixture-1.0-py3-none-any.whl")
	file, err := os.Create(source)
	must(t, err)
	writer := zip.NewWriter(file)
	for name, body := range map[string]string{
		"fixture.py":                     "VALUE = 123\n",
		"fixture-1.0.dist-info/METADATA": "Metadata-Version: 2.3\nName: fixture\nVersion: 1.0\nProvides-Extra: gpu\nRequires-Dist: torch>=2.13,<3; extra == 'gpu'\nRequires-Dist: scipy>=1.0\n\n",
		"fixture-1.0.dist-info/RECORD":   "",
	} {
		out, err := writer.Create(name)
		must(t, err)
		_, err = io.WriteString(out, body)
		must(t, err)
	}
	must(t, writer.Close())
	must(t, file.Close())
	target := filepath.Join(root, "sealed", "fixture-1.0-py3-none-any.whl")
	fatal(t, wheel.PinDependenciesPreserving(source, target, []string{"scipy==1.18.1"}, func(raw string) bool {
		return strings.HasPrefix(raw, "torch")
	}))
	archive, err := zip.OpenReader(target)
	must(t, err)
	defer archive.Close()
	for _, member := range archive.File {
		if !strings.HasSuffix(member.Name, ".dist-info/METADATA") {
			continue
		}
		in, err := member.Open()
		must(t, err)
		body, err := io.ReadAll(in)
		must(t, err)
		must(t, in.Close())
		metadata := string(body)
		if !strings.Contains(metadata, "Requires-Dist: torch>=2.13,<3; extra == 'gpu'\n") || !strings.Contains(metadata, "Requires-Dist: scipy==1.18.1\n") || strings.Contains(metadata, "scipy>=1.0") || strings.Contains(metadata, "Provides-Extra:") {
			t.Fatalf("captured wheel did not preserve only the image-owned compatibility range: %s", metadata)
		}
		return
	}
	t.Fatal("sealed wheel metadata missing")
}
