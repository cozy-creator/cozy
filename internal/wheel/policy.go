package wheel

import (
	"path"
	"regexp"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

var (
	reName      = regexp.MustCompile(`^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$`)
	reNormalize = regexp.MustCompile(`[-_.]+`)
)

var compiledExt = map[string]bool{
	".c": true, ".h": true, ".cc": true, ".cpp": true, ".cxx": true, ".hpp": true,
	".pyx": true, ".pxd": true, ".pxi": true,
	".so": true, ".pyd": true, ".dylib": true, ".dll": true, ".a": true, ".o": true,
	".rs": true, ".f90": true, ".cu": true, ".cuh": true,
}

var nativeSourceExt = map[string]bool{
	".c": true, ".h": true, ".cc": true, ".cpp": true, ".cxx": true, ".hpp": true,
	".pyx": true, ".pxd": true, ".pxi": true, ".rs": true, ".f90": true, ".cu": true, ".cuh": true,
}

var buildInput = map[string]bool{
	"cargo.lock": true, "cargo.toml": true, "cmakelists.txt": true,
	"gnumakefile": true, "makefile": true, "manifest.in": true,
	"meson.build": true, "meson_options.txt": true, "setup.cfg": true, "setup.py": true,
}

func normalize(name string) string {
	return strings.ToLower(reNormalize.ReplaceAllString(name, "-"))
}

func refuseCompiled(name string) *exit.Error {
	return exit.Named(exit.Validation, "project_wheel_native_file",
		"%s is native source or a native binary; the project wheel must be %s", name, Tag).
		WithRemedy("remove it from the project wheel; native dependencies use separately declared prebuilt wheels")
}

func buildInputName(name string) bool {
	base := strings.ToLower(path.Base(name))
	return buildInput[base] || strings.HasPrefix(base, "dockerfile") || strings.HasSuffix(base, ".cmake")
}

func forbiddenProjectMember(name string) (code, class string) {
	lower := strings.ToLower(name)
	parts := strings.Split(lower, "/")
	for index, part := range parts[:len(parts)-1] {
		switch part {
		case ".aws", ".ssh":
			return "project_wheel_credential", "credential directory"
		case ".git", ".hg", ".svn", ".jj", "__pycache__", ".mypy_cache",
			".ruff_cache", ".pytest_cache", ".tox":
			return "project_wheel_local_artifact", "local VCS, environment, or cache material"
		}
		if index == 0 && (part == "credentials" || part == "secrets") {
			return "project_wheel_credential", "credential directory"
		}
		if index == 0 && (part == ".venv" || part == "venv" || part == "node_modules") {
			return "project_wheel_local_artifact", "local environment or dependency cache"
		}
	}
	base := path.Base(lower)
	switch {
	case base == ".env", strings.HasPrefix(base, ".env."), base == ".netrc",
		base == ".npmrc", base == ".pypirc", strings.HasPrefix(base, "id_rsa"),
		strings.HasSuffix(base, ".pem"), strings.HasSuffix(base, ".key"):
		return "project_wheel_credential", "credential or key material"
	case strings.HasSuffix(base, ".pyc"), strings.HasSuffix(base, ".pyo"):
		return "project_wheel_local_artifact", "local bytecode cache"
	}
	return "", ""
}

func checkName(name string) *exit.Error {
	bad := func(why string) *exit.Error {
		return exit.Named(exit.Validation, "unsafe_entry", "%q %s", name, why)
	}
	switch {
	case name == "" || name == ".":
		return bad("is not a path")
	case path.IsAbs(name) || strings.HasPrefix(name, "/"):
		return bad("is absolute")
	case strings.Contains(name, "\\"):
		return bad("contains a backslash")
	case name != path.Clean(name):
		return bad("is not normalized")
	}
	for _, part := range strings.Split(name, "/") {
		if part == "" || part == "." || part == ".." {
			return bad("escapes or repeats a path separator")
		}
	}
	for i := 0; i < len(name); i++ {
		if name[i] < 0x20 || name[i] == 0x7f {
			return bad("contains a control character")
		}
	}
	return nil
}
