package install

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"net/textproto"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/wheel"
)

func venvMetadata(venv string) map[string]string {
	path := filepath.Join(venv, "pyvenv.cfg")
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Size() > 1<<20 {
		return nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	values := map[string]string{}
	for _, line := range strings.Split(string(raw), "\n") {
		key, value, ok := strings.Cut(line, "=")
		if ok {
			values[strings.TrimSpace(key)] = strings.TrimSpace(value)
		}
	}
	return values
}

// BasePython names the venv's actual external interpreter. Dependency-bearing
// venv Python startup may execute .pth hooks, even under uv's isolated probe.
func BasePython(venv string) (string, *exit.Error) {
	metadata := venvMetadata(venv)
	if metadata == nil {
		return "", exit.New(exit.Structural, "captured environment has no regular pyvenv.cfg")
	}
	prefix, err := filepath.Abs(venv)
	if err != nil {
		return "", exit.New(exit.Structural, "captured environment path is invalid")
	}
	prefix, err = filepath.EvalSymlinks(prefix)
	if err != nil {
		return "", exit.New(exit.Structural, "captured environment path is unavailable")
	}
	base, err := filepath.EvalSymlinks(home.VenvPython(venv))
	if err != nil {
		return "", exit.New(exit.Structural, "captured environment base interpreter is unavailable")
	}
	base, err = filepath.Abs(base)
	if err != nil {
		return "", exit.New(exit.Structural, "captured environment base interpreter path is invalid")
	}
	inside := func(path string) bool {
		rel, err := filepath.Rel(prefix, path)
		return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
	}
	if inside(base) && runtime.GOOS == "windows" && filepath.IsAbs(metadata["home"]) {
		base = filepath.Join(metadata["home"], "python.exe")
	}
	info, err := os.Stat(base)
	if err != nil || !info.Mode().IsRegular() || inside(base) {
		return "", exit.New(exit.Structural, "captured environment does not name an external base interpreter")
	}
	return base, nil
}

func pythonVersion(venv string) string {
	base, problem := BasePython(venv)
	if problem != nil {
		return ""
	}
	// -S suppresses startup hooks even in an operator-owned base prefix. The
	// pyvenv declaration and actual base must agree; some uv releases store only
	// the major/minor pair, so query the trusted base for the exact patch version.
	version := strings.TrimSpace(strings.TrimPrefix(runOut(base, "-I", "-S", "-V"), "Python "))
	metadata := venvMetadata(venv)
	declared := metadata["version_info"]
	if declared == "" {
		declared = metadata["version"]
	}
	if declared == "" || !(version == declared || strings.HasPrefix(version, declared+".")) {
		return ""
	}
	return version
}

// closure observes installed metadata directly. `uv pip list` can execute .pth
// startup hooks to discover additional import paths; those paths are not part of
// the exact installed-wheel roster and cannot enter captured dependency capture.
func closure(venv string) (int, string) {
	distributions, ok := installedDistributions(venv)
	if !ok {
		return 0, ""
	}
	lines := make([]string, 0, len(distributions))
	for name, distribution := range distributions {
		lines = append(lines, name+"=="+distribution.version)
	}
	sort.Strings(lines)
	return len(lines), strings.Join(lines, "\n")
}

type installedDistribution struct {
	version  string
	headers  textproto.MIMEHeader
	metadata []byte
}

func installedDistributions(venv string) (map[string]installedDistribution, bool) {
	metadata := venvMetadata(venv)
	version := metadata["version_info"]
	if version == "" {
		version = metadata["version"]
	}
	parts := strings.Split(version, ".")
	if len(parts) < 2 {
		return nil, false
	}
	if _, err := strconv.Atoi(parts[0]); err != nil {
		return nil, false
	}
	if _, err := strconv.Atoi(parts[1]); err != nil {
		return nil, false
	}
	if parts[0] == "" || parts[1] == "" {
		return nil, false
	}
	roots := []string{filepath.Join(venv, "lib", "python"+parts[0]+"."+parts[1], "site-packages")}
	if runtime.GOOS == "windows" {
		roots = []string{filepath.Join(venv, "Lib", "site-packages")}
	}
	seen := map[string]installedDistribution{}
	for _, root := range roots {
		entries, err := os.ReadDir(root)
		if err != nil {
			return nil, false
		}
		for _, entry := range entries {
			if !entry.IsDir() || !strings.HasSuffix(entry.Name(), ".dist-info") {
				continue
			}
			path := filepath.Join(root, entry.Name(), "METADATA")
			info, err := os.Lstat(path)
			if err != nil || !info.Mode().IsRegular() || info.Size() > 1<<20 {
				return nil, false
			}
			raw, err := os.ReadFile(path)
			if err != nil {
				return nil, false
			}
			name, version, problem := wheel.MetadataIdentity(raw)
			name = normalizedRequirementName(name)
			if problem != nil || seen[name].version != "" || len(seen) >= 4096 {
				return nil, false
			}
			headers, err := textproto.NewReader(bufio.NewReader(bytes.NewReader(raw))).ReadMIMEHeader()
			if err != nil && err != io.EOF {
				return nil, false
			}
			seen[name] = installedDistribution{version: version, headers: headers, metadata: raw}
		}
	}
	return seen, true
}

// ExecutionRequirements reads the selected package graph, stopping at image-owned
// distributions. Local installed versions remain separate capture facts.
func ExecutionRequirements(ctx context.Context, venv, project string, extras []string) (packagepublish.RequirementSelection, *exit.Error) {
	selected, _, problem := executionRequirements(ctx, venv, project, extras)
	return selected, problem
}

func executionRequirements(ctx context.Context, venv, project string, extras []string) (packagepublish.RequirementSelection, string, *exit.Error) {
	distributions, ok := installedDistributions(venv)
	root, found := distributions[normalizedRequirementName(project)]
	if !ok || !found {
		return packagepublish.RequirementSelection{}, "", exit.New(exit.Structural, "captured package requirements are unavailable")
	}
	metadata := map[string]string{}
	for name, distribution := range distributions {
		if !packagepublish.ImageOwnedDistribution(name) {
			metadata[name] = string(distribution.metadata)
		}
	}
	selected, problem := packagepublish.ActiveRequirements(ctx, project, extras, metadata, pythonVersion(venv))
	return selected, root.headers.Get("Requires-Python"), problem
}

// ImageRequirements preserves the authored Python bound and selected extras while
// using the same requirement graph as private wheel sealing.
func ImageRequirements(venv, project string, extras ...string) ([]string, string, *exit.Error) {
	selected, python, problem := executionRequirements(context.Background(), venv, project, extras)
	return selected.ImageRequirements(), python, problem
}
