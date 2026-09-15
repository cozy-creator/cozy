package producttest

// cl-078: the client declares registry rows instead of proxying PyPI bytes. The
// pylock below is a frozen `uv export --locked --format pylock.toml` for a project
// depending on one pure registry wheel.

import (
	"fmt"
	"strings"
	"testing"

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

// th-113 client half: a row locked to the publisher's OWN org index references
// a wheel already in the hub's custody — publish DECLARES the row and ships
// nothing for it. The hub custody-shares its committed org-index claim into
// the release, and install serves the wheel from the plan like any registry
// dependency. Size rides as 0 because index pages advertise none; the hub's
// claim is the length authority. Any other index still refuses. The sdxl rows
// are the real quantize uv.lock rows against the dev hub.
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

func TestSameOrgIndexRowsAreDeclaredForCustodyShare(t *testing.T) {
	rows, problem := packagepublish.RegistryRowsFromLock([]byte(orgIndexPylock), nil, "paul")
	if problem != nil {
		t.Fatalf("same-org index row refused: %v", problem)
	}
	if len(rows) != 2 || rows[0].Name != "annotated-doc" || rows[1].Name != "sdxl" {
		t.Fatalf("rows = %+v, want the PyPI row and the org row", rows)
	}
	sdxl := rows[1]
	if sdxl.Version != "2.0.16" || sdxl.Size != 0 ||
		sdxl.SHA256 != "f2926e8dd87777ed74e041fe2dfba89731df8af9ed1b2e47d80fc90694fadfe7" ||
		sdxl.URL != "http://127.0.0.1:8819/v1/index/paul/files/f2926e8dd87777ed74e041fe2dfba89731df8af9ed1b2e47d80fc90694fadfe7/sdxl-2.0.16-py3-none-any.whl" {
		t.Fatalf("org row = %+v, want the exact lock facts with size 0", sdxl)
	}

	// The URL must be the org's own file door for the locked sha256.
	swapped := strings.Replace(orgIndexPylock,
		"/v1/index/paul/files/f2926e8dd87777ed74e041fe2dfba89731df8af9ed1b2e47d80fc90694fadfe7/",
		"/v1/index/paul/files/"+strings.Repeat("0", 64)+"/", 1)
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(swapped), nil, "paul"); problem == nil ||
		problem.Name != "registry_dependency_origin_refused" {
		t.Fatalf("digest-swapped file URL answered %v", problem)
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

func TestRegistryPublicationKeepsCompleteLargeFrameworkClosure(t *testing.T) {
	lock := "lock-version = \"1.0\"\n"
	for _, name := range []string{"torch", "nvidia-cublas-cu13", "numpy"} {
		lock += fmt.Sprintf("[[packages]]\nname=%q\nversion=\"1.0\"\nindex=\"https://pypi.org/simple\"\nwheels=[{url=%q,size=1073741824,hashes={sha256=%q}}]\n", name, "https://files.pythonhosted.org/packages/"+strings.ReplaceAll(name, "-", "_")+"-1.0-py3-none-any.whl", strings.Repeat("a", 64))
	}
	rows, problem := packagepublish.RegistryRowsFromLock([]byte(lock), nil, "")
	fatal(t, problem)
	if len(rows) != 3 {
		t.Fatalf("published framework closure was pruned: %+v", rows)
	}
	tooLarge := strings.Replace(lock, "size=1073741824", "size=2147483649", 1)
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(tooLarge), nil, ""); problem == nil || problem.Name != "registry_dependency_identity_invalid" {
		t.Fatalf("unbounded public artifact admitted: %v", problem)
	}
	official := strings.ReplaceAll(strings.ReplaceAll(strings.ReplaceAll(lock, "name=\"nvidia-cublas-cu13\"", "name=\"torchaudio\""), "nvidia_cublas_cu13-", "torchaudio-"), "name=\"numpy\"", "name=\"torchvision\"")
	official = strings.ReplaceAll(official, "numpy-", "torchvision-")
	official = strings.ReplaceAll(official, "https://pypi.org/simple", "https://download.pytorch.org/whl/cpu")
	official = strings.ReplaceAll(official, "https://files.pythonhosted.org/packages/", "https://download-r2.pytorch.org/whl/cpu/")
	rows, problem = packagepublish.RegistryRowsFromLock([]byte(official), nil, "")
	fatal(t, problem)
	for _, row := range rows {
		if !strings.HasPrefix(row.URL, "https://download.pytorch.org/whl/cpu/") || row.Size != 1<<30 || row.SHA256 != strings.Repeat("a", 64) {
			t.Fatalf("official artifact identity changed: %+v", row)
		}
	}
}

// Universal PyTorch locks contain separate macOS and Linux distributions. The
// publication target must select the Linux branch without losing its closure.
func TestRegistryPublicationSelectsTargetMarkers(t *testing.T) {
	foreign := strings.Replace(frozenPylock, "created-by = \"uv\"", "", 1)
	foreign = strings.TrimPrefix(foreign, "lock-version = \"1.0\"\n")
	foreign = strings.ReplaceAll(foreign, "0.0.3", "0.0.2")
	foreign = strings.Replace(foreign, "[[packages]]", "[[packages]]\nmarker=\"sys_platform == 'darwin'\"", 1)
	foreign = strings.Replace(foreign, "py3-none-any", "cp312-cp312-macosx_14_0_arm64", 1)
	local := strings.Replace(frozenPylock, "[[packages]]", "[[packages]]\nmarker=\"sys_platform == 'linux' and python_version == '3.12'\"", 1)
	rows, problem := packagepublish.RegistryRowsFromLock([]byte(local+foreign), nil, "")
	fatal(t, problem)
	if len(rows) != 1 || rows[0].Version != "0.0.3" {
		t.Fatalf("universal lock lost the selected target: %+v", rows)
	}
}
