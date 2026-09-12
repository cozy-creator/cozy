package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func TestUnpublishedRentalPreservesSelectedFunctionDegrees(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	venv := filepath.Join(t.TempDir(), "venv")
	if out, err := exec.Command("uv", "venv", "--python", "3.12", venv).CombinedOutput(); err != nil {
		t.Fatalf("uv venv: %v\n%s", err, out)
	}
	metadata := filepath.Join(venv, "lib", "python3.12", "site-packages", "minimax_h3-1.14.2.dist-info")
	must(t, os.MkdirAll(metadata, 0700))
	must(t, os.WriteFile(filepath.Join(metadata, "METADATA"), []byte("Metadata-Version: 2.3\nName: minimax-h3\nVersion: 1.14.2\nRequires-Python: >=3.12,<3.13\n"), 0600))
	raw, err := os.ReadFile("testdata/h3-rental-residency/package-interface.json")
	must(t, err)
	for index, arm := range []struct {
		name, function string
		want           []int
	}{
		{"actual H3 serving", "fl2va", []int{2, 4}},
		{"undeclared selected slot", "fl2va", nil},
		{"independent slot intersection", "fl2va", []int{4}},
		{"job remains ungrouped", "segment", nil},
		{"missing requested function", "absent", nil},
	} {
		t.Run(arm.name, func(t *testing.T) {
			var document map[string]any
			must(t, json.Unmarshal(raw, &document))
			for _, value := range document["entrypoints"].([]any) {
				entry := value.(map[string]any)
				if entry["name"] != "fl2va" {
					continue
				}
				models := entry["models"].([]any)
				if arm.name == "undeclared selected slot" {
					delete(models[0].(map[string]any), "sequence_parallel")
				}
				if arm.name == "independent slot intersection" {
					other := map[string]any{"class": "Other", "path": "fl2va.models.other",
						"component_use": map[string]any{}, "sequence_parallel": map[string]any{"degrees": []int{4, 8}}}
					entry["models"] = append(models, other)
				}
			}
			body, err := json.Marshal(document)
			must(t, err)
			body, err = canonical.NormalizeJCS(body)
			must(t, err)
			installed := cleanupTestInstall(layout, fmt.Sprintf("%016x", index+1), "1.14.2")
			installed.Package, installed.PackageInterface = "local/minimax-h3", assessmentDigest(body)
			must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(installed.Dir)), 0700))
			must(t, os.Symlink(venv, filepath.Join(installed.Dir, "venv")))
			must(t, os.WriteFile(launch.PackageInterfacePath(installed.Dir), body, 0444))
			fatal(t, store.RecordInstall(installed))
			constraints, problem := cli.RentalConstraints(&cli.Context{Cfg: config.Config{Home: layout.Root}},
				records.Request{InstallID: installed.ID, Package: installed.Package, Release: installed.Version, Entrypoint: arm.function})
			fatal(t, problem)
			if !slices.Equal(constraints.Degrees, arm.want) {
				t.Fatalf("installed %s declares %v; rental constraints retained %v", arm.function, arm.want, constraints.Degrees)
			}
			if constraints.RequiresPython != ">=3.12,<3.13" {
				t.Fatalf("Python requirements lost during degree extraction: %+v", constraints)
			}
			for _, width := range []int{2, 4, 8} {
				verdict := rental.WidthUnusable(width, arm.function == "segment", constraints)
				if (verdict == "") != slices.Contains(arm.want, width) {
					t.Fatalf("width %d got %q for authored degrees %v", width, verdict, arm.want)
				}
			}
		})
	}
}
