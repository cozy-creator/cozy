// The companion store is the local twin of the base worker image's CUDA bake
// (cr-086 arm 0). Base worker images opt into `cozy-runtime-cuda-kernels` only on
// their exact supported NVIDIA CUDA profile, and the wheel carries a local version
// (`+torch2.13cu130`), lives on no index, and is built into the image from the
// sibling checkout — so a local venv materialized from the very same release lock
// silently ran eager and forfeited the fused plan. Locally this host IS the image
// layer, and `<home>/companions/*.whl` is where its profile wheels live. A wheel
// joins a freshly materialized venv exactly when host and venv match the profile
// the wheel's own name declares — platform tag, cp/abi tag, and the
// `+torchM.NcuXYZ` local version — and never otherwise. Only worker-image-owned
// distributions (cozy-runtime's exported roster) ride here: this is image parity,
// not a side-load channel.
package install

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

var (
	// name-version-python-abi-platform.whl; wheels with build tags do not parse
	// and do not join.
	companionFilename = regexp.MustCompile(
		`^([A-Za-z0-9_.]+)-([A-Za-z0-9_.!+]+)-([a-z0-9_]+)-([a-z0-9_]+)-([a-z0-9_.]+)\.whl$`)
	// The declared CUDA profile, spelled exactly as the image bake spells it:
	// +torch<major>.<minor>cu<XYZ>, where XYZ is major*10+minor (cu130 == CUDA 13.0).
	companionProfile = regexp.MustCompile(`\+torch([0-9]+)\.([0-9]+)cu([0-9]{2,4})$`)
	companionPython  = regexp.MustCompile(`^cp([0-9])([0-9]{1,2})$`)
	nameRuns         = regexp.MustCompile(`[-_.]+`)
	torchVersionLine = regexp.MustCompile(`(?m)^__version__ = '([^']+)'`)
	torchCUDALine    = regexp.MustCompile(`(?m)^cuda[^=]*= '([^']+)'`)
	pythonMinorOf    = regexp.MustCompile(`^3\.([0-9]+)`)
)

type companionWheel struct {
	path         string
	filename     string
	distribution string // normalized
	torchMajor   int
	torchMinor   int
	cuda         int // cuXYZ digits: 130 == CUDA 13.0
	pythonMinor  int
	abi3         bool
	platform     string
}

// CompanionsStale says whether this host's companion store now provides a wheel
// the install's venv lacks. A pinned install is reused on an unchanged source
// digest, but that reuse is honest only while the venv still holds everything the
// host's profile provides — a store or driver that gained the profile after the
// install would otherwise stay silently eager until the next release bump.
func CompanionsStale(store, venvDir string) bool {
	return len(matchingCompanions(store, venvDir)) > 0
}

// joinCompanions installs every store wheel whose declared profile this host and
// the materialized venv both match, then proves the joined environment is still a
// consistent closure. It runs only after the locked closure stands, changes
// nothing when the store is absent, empty, or all mismatched, and never touches a
// distribution the lock already installed.
func joinCompanions(store, venvDir string, warn *[]string) *exit.Error {
	wheels := matchingCompanions(store, venvDir)
	if len(wheels) == 0 {
		return nil
	}
	args := []string{"pip", "install", "--offline", "--no-index", "--no-deps", "--no-build",
		"--python", home.VenvPython(venvDir)}
	files := make([]string, 0, len(wheels))
	for _, wheel := range wheels {
		args = append(args, wheel.path)
		files = append(files, wheel.filename)
	}
	joined := strings.Join(files, ", ")
	if out, problem := runCompanionUV(args...); problem != nil {
		return companionRefusal(store, joined, out)
	}
	if out, problem := runCompanionUV("pip", "check", "--python", home.VenvPython(venvDir)); problem != nil {
		return companionRefusal(store, joined, out)
	}
	*warn = append(*warn, "joined CUDA-profile companion wheel(s): "+joined)
	return nil
}

func companionRefusal(store, joined, detail string) *exit.Error {
	return exit.Named(exit.Validation, "companion_wheel_incompatible",
		"companion wheel(s) %s declare this host's profile but do not join the release's locked closure",
		joined).
		WithRemedy("uv said: %s — remove or rebuild the wheel under %s", condense(detail), store)
}

func runCompanionUV(args ...string) (string, *exit.Error) {
	cmd := exec.Command("uv", args...)
	cmd.Env = config.Frozen().Tool()
	var out strings.Builder
	cmd.Stdout, cmd.Stderr = &out, &out
	if err := cmd.Run(); err != nil {
		return out.String(), exit.Internalf("uv refused: %s", condense(out.String()))
	}
	return out.String(), nil
}

func matchingCompanions(store, venvDir string) []companionWheel {
	entries, err := os.ReadDir(store)
	if err != nil || runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		return nil
	}
	torchPublic, torchCUDA := venvTorch(venvDir)
	if torchPublic == "" || torchCUDA == "" {
		return nil
	}
	pyMinor := venvPythonMinor(venvDir)
	driver := hostCUDA()
	var out []companionWheel
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		wheel, ok := parseCompanion(store, entry.Name())
		if !ok || !packagepublish.ImageOwnedDistribution(wheel.distribution) {
			continue
		}
		if !linuxX86Platform(wheel.platform) {
			continue
		}
		if pyMinor == 0 || wheel.pythonMinor > pyMinor || (!wheel.abi3 && wheel.pythonMinor != pyMinor) {
			continue
		}
		if !strings.HasPrefix(torchPublic, fmt.Sprintf("%d.%d.", wheel.torchMajor, wheel.torchMinor)) {
			continue
		}
		if torchCUDA != fmt.Sprintf("%d.%d", wheel.cuda/10, wheel.cuda%10) || driver < wheel.cuda {
			continue
		}
		if installedInVenv(venvDir, wheel.distribution) {
			continue
		}
		out = append(out, wheel)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].filename < out[j].filename })
	return out
}

func parseCompanion(store, filename string) (companionWheel, bool) {
	empty := companionWheel{}
	name := companionFilename.FindStringSubmatch(filename)
	if name == nil {
		return empty, false
	}
	profile := companionProfile.FindStringSubmatch(name[2])
	if profile == nil {
		return empty, false
	}
	python := companionPython.FindStringSubmatch(name[3])
	if python == nil || python[1] != "3" {
		return empty, false
	}
	abi := name[4]
	if abi != "abi3" && abi != name[3] {
		return empty, false
	}
	torchMajor, _ := strconv.Atoi(profile[1])
	torchMinor, _ := strconv.Atoi(profile[2])
	cuda, _ := strconv.Atoi(profile[3])
	pythonMinor, _ := strconv.Atoi(python[2])
	if cuda == 0 || pythonMinor == 0 {
		return empty, false
	}
	return companionWheel{
		path:         filepath.Join(store, filename),
		filename:     filename,
		distribution: nameRuns.ReplaceAllString(strings.ToLower(name[1]), "-"),
		torchMajor:   torchMajor,
		torchMinor:   torchMinor,
		cuda:         cuda,
		pythonMinor:  pythonMinor,
		abi3:         abi == "abi3",
		platform:     name[5],
	}, true
}

func linuxX86Platform(tag string) bool {
	return tag == "linux_x86_64" ||
		(strings.HasPrefix(tag, "manylinux") && strings.HasSuffix(tag, "_x86_64"))
}

// venvTorch reads the materialized Torch build facts from the venv's own
// `torch/version.py` — the accelerator build identity (`2.13.0+cu130`, cuda
// `'13.0'`) without importing anything. An absent or CPU Torch matches no
// CUDA-profile wheel.
func venvTorch(venvDir string) (public, cuda string) {
	matches, _ := filepath.Glob(filepath.Join(venvDir, "lib", "python3.*",
		"site-packages", "torch", "version.py"))
	if len(matches) != 1 {
		return "", ""
	}
	raw, err := os.ReadFile(matches[0])
	if err != nil {
		return "", ""
	}
	version := torchVersionLine.FindSubmatch(raw)
	build := torchCUDALine.FindSubmatch(raw)
	if version == nil || build == nil {
		return "", ""
	}
	public, _, _ = strings.Cut(string(version[1]), "+")
	return public, string(build[1])
}

func venvPythonMinor(venvDir string) int {
	m := pythonMinorOf.FindStringSubmatch(pythonVersion(venvDir))
	if m == nil {
		return 0
	}
	minor, _ := strconv.Atoi(m[1])
	return minor
}

func installedInVenv(venvDir, distribution string) bool {
	escaped := strings.ReplaceAll(distribution, "-", "_")
	matches, _ := filepath.Glob(filepath.Join(venvDir, "lib", "python3.*",
		"site-packages", escaped+"-*.dist-info"))
	return len(matches) > 0
}
