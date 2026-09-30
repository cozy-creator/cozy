package producttest

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/csv"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/wheel"
)

func TestSourceBuiltWheelCarriesInterfaceWithoutImportingPackage(t *testing.T) {
	project := weightlessProject(t)
	// Neither static discovery nor the build may import this application.
	file, err := os.OpenFile(filepath.Join(project, "weightless.py"), os.O_APPEND|os.O_WRONLY, 0)
	must(t, err)
	_, err = file.WriteString("\nraise RuntimeError('package imported during preparation')\n")
	must(t, err)
	must(t, file.Close())
	pack, problem := packagepublish.PrepareFrom(project)
	fatal(t, problem)
	t.Cleanup(pack.Close)
	fatal(t, pack.Build(t.Context()))
	members := readInterfaceWheel(t, pack.Wheel)
	var embedded []byte
	for name, body := range members {
		if strings.Count(name, "/") == 1 && strings.HasSuffix(name, ".dist-info/package-interface.json") {
			embedded = body
		}
	}
	described, err := os.ReadFile(pack.PackageInterface)
	must(t, err)
	if len(embedded) == 0 || !bytes.Equal(embedded, described) {
		t.Fatal("source-built wheel did not carry its freshly described interface")
	}
	// uv installs the standard wheel metadata into the selected environment.
	venv := filepath.Join(t.TempDir(), "venv")
	python := filepath.Join(venv, "bin", "python")
	for _, args := range [][]string{
		{"venv", "--python", "3.12", venv},
		{"pip", "install", "--python", python, "--no-deps", pack.Wheel},
	} {
		if out, err := exec.CommandContext(t.Context(), "uv", args...).CombinedOutput(); err != nil {
			t.Fatalf("uv %v: %v\n%s", args, err, out)
		}
	}
	code := "import importlib.metadata; d=importlib.metadata.distribution('cozy-weightless-package'); print(d.read_text('package-interface.json'),end='')"
	installed, err := exec.CommandContext(t.Context(), python, "-c", code).CombinedOutput()
	must(t, err)
	if !bytes.Equal(installed, embedded) {
		t.Fatal("installed distribution metadata differs from the sealed wheel")
	}
}

func TestEmbeddedInterfacePreservesWheelTagsMembersAndExistingMetadata(t *testing.T) {
	root := t.TempDir()
	source := prebuiltProjectWheel(t, root, "cozy-weightless-package", "cp312-cp312-manylinux_2_28_x86_64", "weightless:app")
	original := readInterfaceWheel(t, source)
	document := []byte(`{"application":"weightless:app","entrypoints":[],"jobs":[],"future_field":true}`)
	target := filepath.Join(root, "described", filepath.Base(source))
	result, problem := wheel.EmbedInterface(source, target, document)
	fatal(t, problem)
	if result != target || filepath.Base(result) != filepath.Base(source) {
		t.Fatal("embedding changed the wheel's platform tag")
	}
	actual := readInterfaceWheel(t, result)
	for name, body := range original {
		if !strings.HasSuffix(name, ".dist-info/RECORD") && !bytes.Equal(body, actual[name]) {
			t.Fatalf("embedding changed original wheel member %s", name)
		}
	}
	if len(actual) != len(original)+1 {
		t.Fatal("embedding changed members besides interface and RECORD")
	}
	second := filepath.Join(root, "second", filepath.Base(source))
	result, problem = wheel.EmbedInterface(target, second, []byte(`{"different":true}`))
	fatal(t, problem)
	if result != target {
		t.Fatal("an existing backend interface was replaced")
	}
	if _, err := os.Stat(second); !os.IsNotExist(err) {
		t.Fatal("an existing interface created a rewritten wheel")
	}
	if got := readInterfaceWheel(t, source); !bytes.Equal(got["cozy_weightless_package-1.0.0.dist-info/RECORD"], original["cozy_weightless_package-1.0.0.dist-info/RECORD"]) {
		t.Fatal("embedding modified the original build artifact")
	}
}

// Verify RECORD independently, including the added metadata and unchanged payloads.
func readInterfaceWheel(t *testing.T, path string) map[string][]byte {
	t.Helper()
	archive, err := zip.OpenReader(path)
	must(t, err)
	defer archive.Close()
	members := map[string][]byte{}
	var record string
	for _, member := range archive.File {
		reader, err := member.Open()
		must(t, err)
		members[member.Name], err = io.ReadAll(reader)
		must(t, err)
		must(t, reader.Close())
		if strings.Count(member.Name, "/") == 1 && strings.HasSuffix(member.Name, ".dist-info/RECORD") {
			record = member.Name
		}
	}
	rows, err := csv.NewReader(bytes.NewReader(members[record])).ReadAll()
	must(t, err)
	if record == "" || len(rows) != len(members) {
		t.Fatal("RECORD does not cover every member")
	}
	for _, row := range rows {
		if len(row) != 3 {
			t.Fatal("invalid RECORD row")
		}
		if row[0] == record {
			if row[1] != "" || row[2] != "" {
				t.Fatal("RECORD must not hash itself")
			}
			continue
		}
		body, found := members[row[0]]
		digest := sha256.Sum256(body)
		if !found || row[1] != "sha256="+base64.RawURLEncoding.EncodeToString(digest[:]) || row[2] != strconv.Itoa(len(body)) {
			t.Fatalf("RECORD does not verify %s", row[0])
		}
	}
	return members
}
