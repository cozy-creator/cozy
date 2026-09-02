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
	rows, problem := packagepublish.RegistryRowsFromLock([]byte(frozenPylock), nil, "")
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
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(foreign), nil, ""); problem == nil ||
		problem.Name != "registry_dependency_origin_refused" {
		t.Fatalf("foreign origin answered %v", problem)
	}
	otherIndex := strings.Replace(frozenPylock, "https://pypi.org/simple", "https://mirror.example/simple", 1)
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(otherIndex), nil, ""); problem == nil ||
		problem.Name != "registry_dependency_index_refused" {
		t.Fatalf("unpinned index answered %v", problem)
	}
	rooted := strings.Replace(frozenPylock, `name = "annotated-doc"`, `name = "numpy"`, 1)
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(rooted), nil, ""); problem == nil ||
		problem.Name != "registry_dependency_platform_root_present" {
		t.Fatalf("platform root answered %v", problem)
	}
	shortHash := strings.Replace(frozenPylock,
		"b09a2fe63e5e2249a4d0b5c086acc4372e1d44de2b76a4c72cebbdbef7231e67", "b09a", 1)
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(shortHash), nil, ""); problem == nil ||
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
	rows, problem := packagepublish.RegistryRowsFromLock([]byte(nativePylock), nil, "")
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
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(wrong), nil, ""); problem == nil ||
		problem.Name != "registry_dependency_platform_mismatch" {
		t.Fatalf("wrong-platform wheels answered %v", problem)
	}

	// A pure wheel stays first choice over any native wheel.
	pure := strings.Replace(nativePylock, "cp38-abi3-musllinux_1_2_x86_64", "py3-none-any", 1)
	rows, problem = packagepublish.RegistryRowsFromLock([]byte(pure), nil, "")
	if problem != nil || len(rows) != 1 || !strings.HasSuffix(rows[0].URL, "-py3-none-any.whl") {
		t.Fatalf("pure preference answered rows=%+v problem=%v", rows, problem)
	}

	// Among natives the most specific python tag wins, then the newest manylinux.
	specific := strings.Replace(nativePylock, "cp38-abi3-musllinux_1_2_x86_64",
		"cp312-cp312-manylinux_2_17_x86_64", 1)
	rows, problem = packagepublish.RegistryRowsFromLock([]byte(specific), nil, "")
	if problem != nil || len(rows) != 1 || !strings.HasSuffix(rows[0].URL,
		"hf_xet-1.6.0-cp312-cp312-manylinux_2_17_x86_64.whl") {
		t.Fatalf("python specificity answered rows=%+v problem=%v", rows, problem)
	}
	newest := strings.Replace(nativePylock, "cp38-abi3-musllinux_1_2_x86_64",
		"cp38-abi3-manylinux_2_28_x86_64", 1)
	rows, problem = packagepublish.RegistryRowsFromLock([]byte(newest), nil, "")
	if problem != nil || len(rows) != 1 || !strings.HasSuffix(rows[0].URL,
		"hf_xet-1.6.0-cp38-abi3-manylinux_2_28_x86_64.whl") {
		t.Fatalf("newest manylinux answered rows=%+v problem=%v", rows, problem)
	}

	// A manylinux floor above the fleet glibc cap is not admissible.
	tooNew := strings.ReplaceAll(nativePylock,
		"cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64", "cp38-abi3-manylinux_2_39_x86_64")
	tooNew = strings.ReplaceAll(tooNew, "cp38-abi3-musllinux_1_2_x86_64", "cp38-abi3-win_amd64")
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(tooNew), nil, ""); problem == nil ||
		problem.Name != "registry_dependency_platform_mismatch" {
		t.Fatalf("over-cap manylinux answered %v", problem)
	}
}

func TestSlotSeedResolutionFromDefaultBindings(t *testing.T) {
	staged := func(t *testing.T, bindings string) *packagepublish.Package {
		root := t.TempDir()
		descriptor := filepath.Join(root, "descriptor.json")
		body := `{"format":"cozy.package.descriptor/1","application":"a:b",` +
			`"entrypoints":[{"name":"generate","request":{},"result":{},` +
			`"models":[{"class":"Denoiser","path":"generate.models.denoiser",` +
			`"stamps":{},"component_use":{}},` +
			`{"class":"Denoiser","path":"generate.models.refiner",` +
			`"stamps":{},"component_use":{}}]}],"jobs":[]}`
		if err := os.WriteFile(descriptor, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "package.toml"),
			[]byte("[application]\nobject = \"a:b\"\n"+bindings), 0o600); err != nil {
			t.Fatal(err)
		}
		return &packagepublish.Package{Root: root, Descriptor: descriptor,
			Wheel: descriptor /* never reached */, Tree: root, Name: "thing", Release: "1.0.0"}
	}
	unreachable := func(t *testing.T) packagepublish.SeedResolver {
		return func(_ context.Context, model, release, lane string) (packagepublish.DeriveInput, *exit.Error) {
			t.Fatalf("resolver reached for %s@%s/%s", model, release, lane)
			return packagepublish.DeriveInput{}, nil
		}
	}
	// A bindings key naming no declared slot path or class refuses before resolution.
	problem := staged(t, "[bindings.\"NoSuchClass\"]\nmodel = \"acme/sd\"\nrelease = \"1.0.0\"\nlane = \"bf16\"\n").
		DeriveEvidence(t.Context(), "acme", unreachable(t))
	if problem == nil || problem.Name != "slot_binding_unknown" {
		t.Fatalf("unknown bindings key answered %v", problem)
	}
	// Two slots of one class bound to different seeds refuse: one class, one topology fact.
	problem = staged(t, "[bindings.\"generate.models.denoiser\"]\nmodel = \"acme/sd\"\nrelease = \"1.0.0\"\nlane = \"bf16\"\n"+
		"[bindings.\"generate.models.refiner\"]\nmodel = \"acme/other\"\nrelease = \"2.0.0\"\nlane = \"bf16\"\n").
		DeriveEvidence(t.Context(), "acme", unreachable(t))
	if problem == nil || problem.Name != "slot_binding_divergent" {
		t.Fatalf("divergent class seeds answered %v", problem)
	}
	// An incomplete default (no release/lane) is not a seed: the class is skipped,
	// visibly, and nothing derives.
	pack := staged(t, "[bindings.\"Denoiser\"]\nmodel = \"acme/sd\"\n")
	if problem := pack.DeriveEvidence(t.Context(), "acme", unreachable(t)); problem != nil {
		t.Fatalf("incomplete default must skip, not refuse: %v", problem)
	}
	if len(pack.Evidence) != 0 || len(pack.SlotFacts) != 0 ||
		len(pack.SlotFactsSkipped) != 1 || pack.SlotFactsSkipped[0] != "Denoiser" {
		t.Fatalf("skip was not recorded: facts=%v skipped=%v", pack.SlotFacts, pack.SlotFactsSkipped)
	}
	// A complete class-keyed default resolves exactly once as (model, release, lane).
	stop := exit.Named(exit.Unavailable, "resolver_stopped", "recorded the seed")
	var got [3]string
	calls := 0
	problem = staged(t, "[bindings.\"Denoiser\"]\nmodel = \"acme/sd\"\nrelease = \"1.0.0\"\nlane = \"bf16\"\n").
		DeriveEvidence(t.Context(), "acme",
			func(_ context.Context, model, release, lane string) (packagepublish.DeriveInput, *exit.Error) {
				calls++
				got = [3]string{model, release, lane}
				return packagepublish.DeriveInput{}, stop
			})
	if problem == nil || problem.Name != "resolver_stopped" {
		t.Fatalf("seed resolution answered %v", problem)
	}
	if calls != 1 || got != [3]string{"acme/sd", "1.0.0", "bf16"} {
		t.Fatalf("seed resolved %d times as %v", calls, got)
	}
}

// th-113 client half: a row locked to the publisher's OWN org index references
// a wheel already in the hub's custody — publish declares nothing and ships
// nothing for it; the published uv.lock carries its URL and hash and install
// resolves it from the hub exactly like PyPI. Any other index still refuses.
// The sdxl rows are the real quantize uv.lock rows against the dev hub.
const orgIndexPylock = `lock-version = "1.0"
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

[[packages]]
name = "sdxl"
version = "2.0.16"
index = "http://127.0.0.1:8819/v1/index/paul/simple/"

[[packages.wheels]]
url = "http://127.0.0.1:8819/v1/index/paul/files/f2926e8dd87777ed74e041fe2dfba89731df8af9ed1b2e47d80fc90694fadfe7/sdxl-2.0.16-py3-none-any.whl"

[packages.wheels.hashes]
sha256 = "f2926e8dd87777ed74e041fe2dfba89731df8af9ed1b2e47d80fc90694fadfe7"
`

func TestSameOrgIndexRowsRideTheLockAndDeclareNothing(t *testing.T) {
	rows, problem := packagepublish.RegistryRowsFromLock([]byte(orgIndexPylock), nil, "paul")
	if problem != nil {
		t.Fatalf("same-org index row refused: %v", problem)
	}
	if len(rows) != 1 || rows[0].Name != "annotated-doc" {
		t.Fatalf("rows = %+v, want only the PyPI row", rows)
	}

	// Another org's namespace is not this publisher's to link.
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(orgIndexPylock), nil, "acme"); problem == nil ||
		problem.Name != "registry_dependency_index_refused" {
		t.Fatalf("foreign-org index answered %v", problem)
	}
	// No declared organization admits no org index at all.
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(orgIndexPylock), nil, ""); problem == nil ||
		problem.Name != "registry_dependency_index_refused" {
		t.Fatalf("org-less publish answered %v", problem)
	}
	// The namespace is read from the one hub index shape, never a lookalike path.
	lookalike := strings.Replace(orgIndexPylock,
		"http://127.0.0.1:8819/v1/index/paul/simple/", "http://127.0.0.1:8819/paul/simple/", 1)
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(lookalike), nil, "paul"); problem == nil ||
		problem.Name != "registry_dependency_index_refused" {
		t.Fatalf("lookalike index path answered %v", problem)
	}
}
