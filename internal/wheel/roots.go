package wheel

// A project wheel that installs no Python module is a wheel the worker cannot
// import from. cozy-runtime measures import roots before it installs a package
// (package_environment._import_roots); this is the same measurement, taken here
// so a backend that silently packaged nothing is refused on the author's
// machine instead of on a rented pod minutes later.

import (
	"archive/zip"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Contents is what one wheel installs: the Python import roots it would put on
// sys.path, plus what the build backend wrote so a refusal can say what it saw.
type Contents struct {
	ImportRoots []string // sorted unique roots, the runtime's measurement
	TopLevel    []string // the backend's top_level.txt lines, when it wrote one
	Members     []string // sorted member paths outside .dist-info
	EntryPoints []EntryPoint
}

// EntryPoint is one entry_points.txt row: what importlib.metadata reports for
// the installed wheel, and so what a worker discovers an application from.
type EntryPoint struct {
	Group, Name, Object string
}

// Group returns the entry points registered under one group, in file order.
func (c Contents) Group(group string) []EntryPoint {
	out := []EntryPoint{}
	for _, entry := range c.EntryPoints {
		if entry.Group == group {
			out = append(out, entry)
		}
	}
	return out
}

var (
	reImportRoot     = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	importableSuffix = []string{".py", ".pyi", ".dll", ".dylib", ".pyd", ".so"}
)

// InspectContents measures one bounded wheel's import roots the way cozy-runtime
// does: a top-level filename counts only when it is a Python or native module, a
// directory only when an importable member lives below it, and .dist-info,
// .egg-info, .libs, __pycache__ and non-purelib/platlib .data trees never count.
func InspectContents(file string) (Contents, *exit.Error) {
	var out Contents
	abs, err := filepath.Abs(file)
	if err != nil {
		return out, exit.Usagef("wheel path %q is not resolvable: %s", file, err)
	}
	info, err := os.Stat(abs)
	if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > MaxWheelBytes {
		return out, exit.Named(exit.Validation, "wheel_size_invalid",
			"%s is not a non-empty regular wheel at or below %d B", abs, MaxWheelBytes)
	}
	f, err := os.Open(abs)
	if err != nil {
		return out, exit.Named(exit.Structural, "wheel_unreadable", "%s: %v", abs, err)
	}
	defer f.Close()
	zr, err := zip.NewReader(f, info.Size())
	if err != nil || len(zr.File) == 0 || len(zr.File) > maxZipEntries {
		return out, exit.Named(exit.Validation, "wheel_zip_invalid", "%s is not a bounded ZIP wheel", filepath.Base(abs))
	}
	roots := []string{}
	for _, member := range zr.File {
		name := member.Name
		if strings.HasSuffix(name, ".dist-info/top_level.txt") && strings.Count(name, "/") == 1 {
			body, problem := wheelMember(member)
			if problem != nil {
				return out, problem
			}
			for _, line := range strings.Split(string(body), "\n") {
				if line = strings.TrimSpace(line); line != "" {
					out.TopLevel = append(out.TopLevel, line)
				}
			}
		}
		if strings.HasSuffix(name, ".dist-info/entry_points.txt") && strings.Count(name, "/") == 1 {
			body, problem := wheelMember(member)
			if problem != nil {
				return out, problem
			}
			out.EntryPoints = append(out.EntryPoints, parseEntryPoints(string(body))...)
		}
		if !strings.HasSuffix(name, "/") && !strings.HasSuffix(firstSegment(name), ".dist-info") {
			out.Members = append(out.Members, name)
		}
		if root, ok := importRoot(name); ok {
			roots = append(roots, root)
		}
	}
	out.ImportRoots = sortedUnique(roots)
	out.Members = sortedUnique(out.Members)
	return out, nil
}

// Describe names what the backend produced, for a refusal a human can act on.
func (c Contents) Describe() string {
	parts := []string{}
	if len(c.TopLevel) != 0 {
		parts = append(parts, "top_level.txt names "+strings.Join(c.TopLevel, ", "))
	}
	switch n := len(c.Members); {
	case n == 0:
		parts = append(parts, "the wheel carries only its .dist-info metadata")
	case n <= 8:
		parts = append(parts, "the wheel installs only "+strings.Join(c.Members, ", "))
	default:
		parts = append(parts, fmt.Sprintf("the wheel installs only %s and %d more",
			strings.Join(c.Members[:8], ", "), n-8))
	}
	return strings.Join(parts, "; ")
}

func importRoot(name string) (string, bool) {
	installed, ok := installPath(name)
	if !ok {
		return "", false
	}
	parts := strings.Split(installed, "/")
	first, last := parts[0], parts[len(parts)-1]
	if !hasAnySuffix(last, importableSuffix) {
		return "", false
	}
	if strings.HasSuffix(first, ".dist-info") || strings.HasSuffix(first, ".egg-info") ||
		strings.HasSuffix(first, ".libs") || first == "__pycache__" {
		return "", false
	}
	root := strings.TrimSuffix(strings.TrimSuffix(first, ".py"), ".pyi")
	root, _, _ = strings.Cut(root, ".")
	if !reImportRoot.MatchString(root) {
		return "", false
	}
	return root, true
}

// installPath maps a member to where it lands on sys.path: .data/purelib and
// .data/platlib trees install at their root, other .data trees install nowhere.
func installPath(name string) (string, bool) {
	parts := strings.Split(name, "/")
	if !strings.HasSuffix(parts[0], ".data") {
		return name, true
	}
	if len(parts) < 3 || (parts[1] != "purelib" && parts[1] != "platlib") {
		return "", false
	}
	return strings.Join(parts[2:], "/"), true
}

func firstSegment(name string) string {
	first, _, _ := strings.Cut(name, "/")
	return first
}

func hasAnySuffix(value string, suffixes []string) bool {
	for _, suffix := range suffixes {
		if strings.HasSuffix(value, suffix) {
			return true
		}
	}
	return false
}

// parseEntryPoints reads the entry_points.txt grammar: `[group]` headers, then
// `name = module:object [extras]` rows; blank lines and `#` comments are skipped.
func parseEntryPoints(body string) []EntryPoint {
	out := []EntryPoint{}
	group := ""
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			group = strings.TrimSpace(line[1 : len(line)-1])
			continue
		}
		name, object, ok := strings.Cut(line, "=")
		if !ok || group == "" {
			continue
		}
		out = append(out, EntryPoint{Group: group, Name: strings.TrimSpace(name),
			Object: strings.TrimSpace(object)})
	}
	return out
}
