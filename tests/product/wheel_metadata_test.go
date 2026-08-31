package producttest

import (
	"archive/zip"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/wheel"
)

func writeMetadataWheel(t *testing.T, metadata string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "quality_judge-1.0.7-py3-none-any.whl")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	archive := zip.NewWriter(file)
	member, err := archive.Create("quality_judge-1.0.7.dist-info/METADATA")
	if err == nil {
		_, err = member.Write([]byte(metadata))
	}
	if closeErr := archive.Close(); err == nil {
		err = closeErr
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	return path
}

func TestWheelMetadataIdentityStopsAtDescription(t *testing.T) {
	path := writeMetadataWheel(t, "Metadata-Version: 2.4\r\n"+
		"Name: quality-judge\r\n"+
		"Version: 1.0.7\r\n"+
		"Description-Content-Type: text/markdown\r\n"+
		"\r\n"+
		"Quality judge documentation\r\n"+
		"Name: this is description text, not a second header\r\n"+
		"Version: neither is this\r\n")

	identity, problem := wheel.InspectIdentity(path)
	if problem != nil || identity.Distribution != "quality-judge" || identity.Version != "1.0.7" {
		t.Fatalf("wheel identity = %+v, %v", identity, problem)
	}
}

func TestWheelMetadataIdentityRejectsDuplicateHeader(t *testing.T) {
	for _, duplicate := range []string{"Name: other\n", "Version: 2.0.0\n"} {
		t.Run(strings.TrimSuffix(duplicate, "\n"), func(t *testing.T) {
			path := writeMetadataWheel(t, "Metadata-Version: 2.4\n"+
				"Name: quality-judge\n"+
				"Version: 1.0.7\n"+duplicate+"\n")
			_, problem := wheel.InspectIdentity(path)
			if problem == nil || problem.Name != "wheel_structure_invalid" ||
				!strings.Contains(problem.Message, "METADATA repeats") {
				t.Fatalf("duplicate identity header problem = %#v", problem)
			}
		})
	}
}

func TestWheelMetadataIdentityAcceptsFoldedUnrelatedHeader(t *testing.T) {
	path := writeMetadataWheel(t, "Metadata-Version: 2.4\n"+
		"Project-URL: Documentation,\n"+
		" https://example.test/quality-judge\n"+
		"Name: quality-judge\n"+
		"Version: 1.0.7\n")

	identity, problem := wheel.InspectIdentity(path)
	if problem != nil || identity.Distribution != "quality-judge" || identity.Version != "1.0.7" {
		t.Fatalf("wheel identity = %+v, %v", identity, problem)
	}
}
