package producttest

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

// Publication inspects native wheels without importing their platform-specific
// bytes. Executing a real native extension is covered by Runtime's install proof.
func TestPrebuiltProjectWheelsPublication(t *testing.T) {
	if code, out := runCozy(t, t.TempDir(), "package", "publish", "--help"); code != 0 || !strings.Contains(out, "--wheel") {
		t.Fatalf("prebuilt wheel CLI input is unavailable [exit %d]: %s", code, out)
	}
	project := weightlessProject(t)
	dir := t.TempDir()
	first := prebuiltProjectWheel(t, dir, "cozy-weightless-package", "cp312-cp312-manylinux_2_28_x86_64", "weightless:app")
	second := prebuiltProjectWheel(t, dir, "cozy-weightless-package", "cp313-cp313-manylinux_2_28_x86_64", "weightless:app")
	pack, problem := packagepublish.PrepareFrom(project)
	fatal(t, problem)
	t.Cleanup(pack.Close)
	fatal(t, pack.BuildForPublish(t.Context(), packagepublish.Namespace{Hub: "http://127.0.0.1:1", Account: "proof"}, second, first))
	if len(pack.ProjectWheels) != 2 || pack.SourceArchive == "" || pack.PackageInterface == "" {
		t.Fatalf("prebuilt publication lost artifacts: %#v", pack)
	}
	names := []string{filepath.Base(pack.ProjectWheels[0]), filepath.Base(pack.ProjectWheels[1])}
	if !slices.Equal(names, []string{filepath.Base(first), filepath.Base(second)}) {
		t.Fatalf("publication rebuilt/replaced supplied wheels: %v", names)
	}
	for i, source := range []string{first, second} {
		original, err := os.ReadFile(source)
		must(t, err)
		staged, err := os.ReadFile(pack.ProjectWheels[i])
		must(t, err)
		if !slices.Equal(original, staged) {
			t.Fatal("wheel changed during snapshot")
		}
	}
	pack.Close()
	for _, source := range []string{first, second} {
		if _, err := os.Stat(source); err != nil {
			t.Fatalf("cleanup removed user artifact: %v", err)
		}
	}
	for _, tc := range []struct {
		name    string
		wheels  []string
		refusal string
	}{
		{"duplicate", []string{first, first}, "project_wheel_duplicate"},
		{"wrong distribution", []string{prebuiltProjectWheel(t, dir, "wrong-project", "cp312-cp312-linux_x86_64", "weightless:app")}, "project_metadata_mismatch"},
		{"wrong entrypoint", []string{prebuiltProjectWheel(t, dir, "cozy-weightless-package", "cp314-cp314-linux_x86_64", "weightless:other")}, "project_wheel_application_entrypoint_mismatch"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pack, problem := packagepublish.PrepareFrom(project)
			fatal(t, problem)
			defer pack.Close()
			if problem := pack.BuildForPublish(t.Context(), packagepublish.Namespace{Hub: "http://127.0.0.1:1", Account: "proof"}, tc.wheels...); problem == nil || problem.Name != tc.refusal {
				t.Fatalf("bad prebuilt wheel result: %v, want %s", problem, tc.refusal)
			}
		})
	}
}

func prebuiltProjectWheel(t *testing.T, dir, name, tag, entrypoint string) string {
	t.Helper()
	normalized := strings.ReplaceAll(name, "-", "_")
	path := filepath.Join(dir, normalized+"-1.0.0-"+tag+".whl")
	file, err := os.Create(path)
	must(t, err)
	archive := zip.NewWriter(file)
	metadata := normalized + "-1.0.0.dist-info/"
	record := ""
	for member, value := range map[string]string{
		"weightless.py":               "app = object()\n",
		metadata + "METADATA":         fmt.Sprintf("Metadata-Version: 2.3\nName: %s\nVersion: 1.0.0\n", name),
		metadata + "WHEEL":            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: " + tag + "\n",
		metadata + "entry_points.txt": "[cozy.application]\ndefault = " + entrypoint + "\n",
	} {
		writer, err := archive.Create(member)
		must(t, err)
		_, err = writer.Write([]byte(value))
		must(t, err)
		digest := sha256.Sum256([]byte(value))
		record += fmt.Sprintf("%s,sha256=%s,%d\n", member, base64.RawURLEncoding.EncodeToString(digest[:]), len(value))
	}
	writer, err := archive.Create(metadata + "RECORD")
	must(t, err)
	_, err = writer.Write([]byte(record + metadata + "RECORD,,\n"))
	must(t, err)
	must(t, archive.Close())
	must(t, file.Close())
	return path
}
