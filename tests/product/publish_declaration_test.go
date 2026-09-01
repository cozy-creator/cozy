package producttest

// cl-078: the client declares registry rows instead of proxying PyPI bytes, and
// derives evidence only for resolvable declared pairs. Both arms drive the real
// publication code over real documents — the pylock below is a frozen `uv
// export --locked --format pylock.toml` for a project depending on one pure
// registry wheel.

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

const frozenPylock = `lock-version = "1.0"
created-by = "uv"

[[packages]]
name = "annotated-doc"
version = "0.0.3"
index = "https://pypi.org/simple"

[[packages.wheels]]
url = "https://files.pythonhosted.org/packages/d1/23/annotated_doc-0.0.3-py3-none-any.whl"
size = 6118

[packages.wheels.hashes]
sha256 = "b09a2fe63e5e2249a4d0b5c086acc4372e1d44de2b76a4c72cebbdbef7231e67"
`

func TestRegistryLockRowsReplaceTheProxiedDownload(t *testing.T) {
	rows, problem := packagepublish.RegistryRowsFromLock([]byte(frozenPylock), nil)
	if problem != nil {
		t.Fatalf("frozen pylock refused: %v", problem)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rows))
	}
	row := rows[0]
	if row.Name != "annotated-doc" || row.Version != "0.0.3" || row.Size != 6118 ||
		row.SHA256 != "b09a2fe63e5e2249a4d0b5c086acc4372e1d44de2b76a4c72cebbdbef7231e67" ||
		!strings.HasPrefix(row.URL, "https://files.pythonhosted.org/") {
		t.Fatalf("row = %+v", row)
	}

	// The whole download discipline survives as row validation.
	foreign := strings.Replace(frozenPylock, "files.pythonhosted.org", "evil.example.com", 1)
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(foreign), nil); problem == nil ||
		problem.Name != "registry_dependency_origin_refused" {
		t.Fatalf("foreign origin answered %v", problem)
	}
	otherIndex := strings.Replace(frozenPylock, "https://pypi.org/simple", "https://mirror.example/simple", 1)
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(otherIndex), nil); problem == nil ||
		problem.Name != "registry_dependency_index_refused" {
		t.Fatalf("unpinned index answered %v", problem)
	}
	rooted := strings.Replace(frozenPylock, `name = "annotated-doc"`, `name = "numpy"`, 1)
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(rooted), nil); problem == nil ||
		problem.Name != "registry_dependency_platform_root_present" {
		t.Fatalf("platform root answered %v", problem)
	}
	shortHash := strings.Replace(frozenPylock,
		"b09a2fe63e5e2249a4d0b5c086acc4372e1d44de2b76a4c72cebbdbef7231e67", "b09a", 1)
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(shortHash), nil); problem == nil ||
		problem.Name != "registry_dependency_identity_invalid" {
		t.Fatalf("unbounded identity answered %v", problem)
	}
}

func TestUnresolvableSourceProfilesRefuseBeforeAnyDerivation(t *testing.T) {
	root := t.TempDir()
	descriptor := filepath.Join(root, "descriptor.json")
	body := `{"format":"cozy.package.descriptor/1","application":"a:b",` +
		`"entrypoints":[{"name":"generate","request":{},"result":{},` +
		`"models":[{"class":"Denoiser","path":"generate.models.denoiser",` +
		`"stamps":{},"component_use":{},"source_profile":"just-a-token"}]}],"jobs":[]}`
	if err := os.WriteFile(descriptor, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	pack := &packagepublish.Package{Root: root, Descriptor: descriptor,
		Wheel: descriptor /* never reached */, Tree: root, Name: "thing", Release: "1.0.0"}
	problem := pack.DeriveEvidence(t.Context(), "acme",
		func(_ context.Context, model, lane string) (packagepublish.DeriveInput, *exit.Error) {
			t.Fatalf("resolver reached for %s/%s", model, lane)
			return packagepublish.DeriveInput{}, nil
		})
	if problem == nil || problem.Name != "source_profile_unresolvable" {
		t.Fatalf("two-segment profile answered %v", problem)
	}
}
