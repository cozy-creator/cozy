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

// th-107: a native-only registry dependency selects its platform-target wheel
// — pure stays first choice, the newest manylinux with the most specific
// python tag wins among natives, and a foreign-platform-only package refuses
// typed. The hf-xet rows are the real quality-judge uv.lock rows.
const nativePylock = `lock-version = "1.0"
created-by = "uv"

[[packages]]
name = "hf-xet"
version = "1.6.0"
index = "https://pypi.org/simple"

[[packages.wheels]]
url = "https://files.pythonhosted.org/packages/67/4e/a28359bf1c1ecf11eba22123168c138698f7cb576ac678f5a2e16cd5da08/hf_xet-1.6.0-cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
size = 4464663

[packages.wheels.hashes]
sha256 = "d62671bb130879cef0ee4c9ebe47a14af6c66ec53e6d84dc15936e5ffdfac82f"

[[packages.wheels]]
url = "https://files.pythonhosted.org/packages/ab/5f/311725e2a905534dfee2dcb5b08414f249147f1f12252bfc2bd24caa075c/hf_xet-1.6.0-cp38-abi3-musllinux_1_2_x86_64.whl"
size = 4675937

[packages.wheels.hashes]
sha256 = "8fb4f71cba6129110c3374a33f919001ff130488fc23553698e34cc1c2a1198c"
`

func TestNativeRegistryWheelsAdmitThePlatformTarget(t *testing.T) {
	rows, problem := packagepublish.RegistryRowsFromLock([]byte(nativePylock), nil)
	if problem != nil {
		t.Fatalf("native-only pylock refused: %v", problem)
	}
	if len(rows) != 1 || !strings.HasSuffix(rows[0].URL,
		"hf_xet-1.6.0-cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl") ||
		rows[0].SHA256 != "d62671bb130879cef0ee4c9ebe47a14af6c66ec53e6d84dc15936e5ffdfac82f" ||
		rows[0].Size != 4464663 {
		t.Fatalf("rows = %+v, want the manylinux cp38-abi3 wheel", rows)
	}

	// A wheel set for the wrong platforms only refuses with a typed reason.
	wrong := strings.ReplaceAll(nativePylock,
		"cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64", "cp38-abi3-win_amd64")
	wrong = strings.ReplaceAll(wrong, "cp38-abi3-musllinux_1_2_x86_64", "cp311-cp311-macosx_11_0_arm64")
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(wrong), nil); problem == nil ||
		problem.Name != "registry_dependency_platform_mismatch" {
		t.Fatalf("wrong-platform wheels answered %v", problem)
	}

	// A pure wheel stays first choice over any native wheel.
	pure := strings.Replace(nativePylock, "cp38-abi3-musllinux_1_2_x86_64", "py3-none-any", 1)
	rows, problem = packagepublish.RegistryRowsFromLock([]byte(pure), nil)
	if problem != nil || len(rows) != 1 || !strings.HasSuffix(rows[0].URL, "-py3-none-any.whl") {
		t.Fatalf("pure preference answered rows=%+v problem=%v", rows, problem)
	}

	// Among natives the most specific python tag wins, then the newest manylinux.
	specific := strings.Replace(nativePylock, "cp38-abi3-musllinux_1_2_x86_64",
		"cp312-cp312-manylinux_2_17_x86_64", 1)
	rows, problem = packagepublish.RegistryRowsFromLock([]byte(specific), nil)
	if problem != nil || len(rows) != 1 || !strings.HasSuffix(rows[0].URL,
		"hf_xet-1.6.0-cp312-cp312-manylinux_2_17_x86_64.whl") {
		t.Fatalf("python specificity answered rows=%+v problem=%v", rows, problem)
	}
	newest := strings.Replace(nativePylock, "cp38-abi3-musllinux_1_2_x86_64",
		"cp38-abi3-manylinux_2_28_x86_64", 1)
	rows, problem = packagepublish.RegistryRowsFromLock([]byte(newest), nil)
	if problem != nil || len(rows) != 1 || !strings.HasSuffix(rows[0].URL,
		"hf_xet-1.6.0-cp38-abi3-manylinux_2_28_x86_64.whl") {
		t.Fatalf("newest manylinux answered rows=%+v problem=%v", rows, problem)
	}

	// A manylinux floor above the fleet glibc cap is not admissible.
	tooNew := strings.ReplaceAll(nativePylock,
		"cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64", "cp38-abi3-manylinux_2_39_x86_64")
	tooNew = strings.ReplaceAll(tooNew, "cp38-abi3-musllinux_1_2_x86_64", "cp38-abi3-win_amd64")
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(tooNew), nil); problem == nil ||
		problem.Name != "registry_dependency_platform_mismatch" {
		t.Fatalf("over-cap manylinux answered %v", problem)
	}
}

func TestUnresolvableSourceProfilesRefuseBeforeAnyDerivation(t *testing.T) {
	profiled := func(t *testing.T, profile string) *packagepublish.Package {
		root := t.TempDir()
		descriptor := filepath.Join(root, "descriptor.json")
		body := `{"format":"cozy.package.descriptor/1","application":"a:b",` +
			`"entrypoints":[{"name":"generate","request":{},"result":{},` +
			`"models":[{"class":"Denoiser","path":"generate.models.denoiser",` +
			`"stamps":{},"component_use":{},"source_profile":"` + profile + `"}]}],"jobs":[]}`
		if err := os.WriteFile(descriptor, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		return &packagepublish.Package{Root: root, Descriptor: descriptor,
			Wheel: descriptor /* never reached */, Tree: root, Name: "thing", Release: "1.0.0"}
	}
	// Outside the four-or-five segment grammar the resolver is never reached.
	for _, profile := range []string{"just-a-token", "a/b/c/d/e/f", "a/b/c/d/"} {
		problem := profiled(t, profile).DeriveEvidence(t.Context(), "acme",
			func(_ context.Context, model, release, lane, config string) (packagepublish.DeriveInput, *exit.Error) {
				t.Fatalf("resolver reached for %s@%s/%s config %q", model, release, lane, config)
				return packagepublish.DeriveInput{}, nil
			})
		if problem == nil || problem.Name != "source_profile_unresolvable" {
			t.Fatalf("profile %q answered %v", profile, problem)
		}
	}
	// The optional fifth segment is the config selector (th-114): the resolver
	// receives it exactly, and empty for a four-segment profile.
	for profile, config := range map[string]string{
		"acme/sd-turbo/1.0.0/bf16":      "",
		"acme/sd-turbo/1.0.0/bf16/unet": "unet",
	} {
		stop := exit.Named(exit.Unavailable, "resolver_stopped", "recorded the parse")
		var got [4]string
		problem := profiled(t, profile).DeriveEvidence(t.Context(), "acme",
			func(_ context.Context, model, release, lane, config string) (packagepublish.DeriveInput, *exit.Error) {
				got = [4]string{model, release, lane, config}
				return packagepublish.DeriveInput{}, stop
			})
		if problem == nil || problem.Name != "resolver_stopped" {
			t.Fatalf("profile %q answered %v", profile, problem)
		}
		if got != [4]string{"acme/sd-turbo", "1.0.0", "bf16", config} {
			t.Fatalf("profile %q resolved as %v", profile, got)
		}
	}
}
