package producttest

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/csv"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/runtimeoperation"
	"github.com/cozy-creator/cozy/internal/wheel"
)

func TestRuntimeOperationCarrierContainsOnlyExactMetadata(t *testing.T) {
	digest := "sha256:" + strings.Repeat("a", 64)
	name, raw, problem := runtimeoperation.Carrier("0.16.8", digest)
	fatal(t, problem)
	_, repeated, problem := runtimeoperation.Carrier("0.16.8", digest)
	fatal(t, problem)
	if !bytes.Equal(raw, repeated) {
		t.Fatal("same builtin identity changed its carrier")
	}
	path := filepath.Join(t.TempDir(), name)
	must(t, os.WriteFile(path, raw, 0600))
	identity, problem := wheel.InspectIdentity(path)
	fatal(t, problem)
	if identity.Distribution != runtimeoperation.Name || identity.Version != "0.16.8" {
		t.Fatal("carrier changed distribution identity")
	}
	contents, problem := wheel.InspectContents(path)
	fatal(t, problem)
	if len(contents.ImportRoots) != 0 {
		t.Fatal("metadata carrier installed executable code")
	}
	archive, err := zip.NewReader(bytes.NewReader(raw), int64(len(raw)))
	must(t, err)
	files := map[string][]byte{}
	for _, file := range archive.File {
		stream, err := file.Open()
		must(t, err)
		body, err := io.ReadAll(stream)
		must(t, err)
		must(t, stream.Close())
		files[filepath.Base(file.Name)] = body
	}
	if len(files) != 4 || !strings.Contains(string(files["entry_points.txt"]), runtimeoperation.Application) ||
		!strings.Contains(string(files["METADATA"]), digest) || strings.Contains(string(files["METADATA"]), "numpy==") {
		t.Fatal("carrier omitted exact descriptor or invented observed dependency pins")
	}
	rows, err := csv.NewReader(bytes.NewReader(files["RECORD"])).ReadAll()
	must(t, err)
	for _, row := range rows {
		if filepath.Base(row[0]) == "RECORD" {
			continue
		}
		sum := sha256.Sum256(files[filepath.Base(row[0])])
		if row[1] != "sha256="+base64.RawURLEncoding.EncodeToString(sum[:]) {
			t.Fatal("carrier RECORD did not bind bytes")
		}
	}
	for _, version := range []string{"0.16.7", "../../../evil", "invalid"} {
		if _, _, problem := runtimeoperation.Carrier(version, digest); problem == nil {
			t.Fatal("invalid Runtime version admitted")
		}
	}
	if _, _, problem := runtimeoperation.Carrier("0.16.8", "not-a-digest"); problem == nil {
		t.Fatal("unbound interface admitted")
	}
}
