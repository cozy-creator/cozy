// Package runtimeoperation identifies the fixed App supplied by the worker Runtime.
// Its private carrier contains metadata only; Runtime implementation bytes remain
// owned by the selected base, independently of any calling package.
package runtimeoperation

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/csv"
	"fmt"
	"sort"
	"strings"
	"time"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

const (
	Name        = "cozy-runtime-operations"
	Module      = "cozy_runtime.derive.operations"
	Application = Module + ":app"
	Floor       = "0.16.8"
)

// Carrier is deterministic for the selected Runtime release and exact builtin
// interface. It carries no source, imports, startup hooks or numerical libraries.
// Actual implementation/numerical identity is qualified by the selected worker.
func Carrier(version, interfaceDigest string) (string, []byte, *exit.Error) {
	parsed, err := pep440.Parse(version)
	if err != nil || parsed.LessThan(pep440.MustParse(Floor)) || parsed.String() != version {
		return "", nil, exit.New(exit.Validation, "Runtime operations require a canonical compatible release")
	}
	if _, err := canonical.Raw(interfaceDigest); err != nil {
		return "", nil, exit.New(exit.Validation, "Runtime operations require an exact builtin interface")
	}
	dist := "cozy_runtime_operations-" + version + ".dist-info"
	files := map[string][]byte{
		dist + "/METADATA": []byte(fmt.Sprintf(
			"Metadata-Version: 2.4\nName: %s\nVersion: %s\nRequires-Python: >=3.12\n"+
				"Requires-Dist: cozy-runtime>=%s,<1\nRequires-Dist: numpy>=1.26\n"+
				"Cozy-Builtin: operations\nCozy-Builtin-Interface: %s\n",
			Name, version, Floor, interfaceDigest)),
		dist + "/WHEEL":            []byte("Wheel-Version: 1.0\nGenerator: cozy-runtime-builtin\nRoot-Is-Purelib: true\nTag: py3-none-any\n"),
		dist + "/entry_points.txt": []byte("[cozy.application]\ndefault = " + Application + "\n"),
	}
	names := make([]string, 0, len(files)+1)
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)
	var record bytes.Buffer
	csvWriter := csv.NewWriter(&record)
	for _, name := range names {
		raw := files[name]
		digest := sha256.Sum256(raw)
		_ = csvWriter.Write([]string{name, "sha256=" + base64.RawURLEncoding.EncodeToString(digest[:]), fmt.Sprint(len(raw))})
	}
	_ = csvWriter.Write([]string{dist + "/RECORD", "", ""})
	csvWriter.Flush()
	if err := csvWriter.Error(); err != nil {
		return "", nil, exit.Internalf("cannot encode Runtime carrier record: %s", err)
	}
	files[dist+"/RECORD"] = record.Bytes()
	names = append(names, dist+"/RECORD")
	sort.Strings(names)
	var result bytes.Buffer
	archive := zip.NewWriter(&result)
	for _, name := range names {
		header := &zip.FileHeader{Name: name, Method: zip.Deflate}
		header.SetModTime(time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC))
		header.SetMode(0o644)
		file, err := archive.CreateHeader(header)
		if err != nil {
			return "", nil, exit.Internalf("cannot create Runtime carrier: %s", err)
		}
		if _, err := file.Write(files[name]); err != nil {
			return "", nil, exit.Internalf("cannot write Runtime carrier: %s", err)
		}
	}
	if err := archive.Close(); err != nil {
		return "", nil, exit.Internalf("cannot close Runtime carrier: %s", err)
	}
	return strings.ReplaceAll(Name, "-", "_") + "-" + version + "-py3-none-any.whl", result.Bytes(), nil
}

// Export names the fixed Runtime managed derivation surface.
func Export(name string) bool { return name == "quantize" || name == "prepare_model" }
