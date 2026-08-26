package wheel

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// MaxTreeBytes caps what one endpoint project may become. An endpoint's wheel is CODE and
// small config: weights, datasets and checkpoints reach a pod through the CAS plane, never
// through the importable unit.
const MaxTreeBytes = 256 << 20

// excludedDir names never enter a wheel: version control, caches, virtual environments,
// editor state, and build output. The list is FIXED — a per-tree ignore file would make
// the wheel's contents a function of a file the tree can change without changing the
// files it ships, and `project_wheel_digest` would stop being a function of the code.
var excludedDir = map[string]bool{
	".git": true, ".hg": true, ".svn": true, ".jj": true,
	"__pycache__": true, ".mypy_cache": true, ".ruff_cache": true, ".pytest_cache": true,
	".venv": true, "venv": true, "node_modules": true,
	".idea": true, ".vscode": true, ".tox": true,
	"build": true, "dist": true,
}

// excludedFile is the same rule for leaves.
var excludedFile = map[string]bool{
	".DS_Store": true, "Thumbs.db": true, ".gitignore": true, ".gitattributes": true,
}

// compiledExt is what a project that needs to COMPILE something carries. Any of these is
// a typed refusal: the launch packer emits py3-none-any and nothing else.
var compiledExt = map[string]bool{
	".c": true, ".h": true, ".cc": true, ".cpp": true, ".cxx": true, ".hpp": true,
	".pyx": true, ".pxd": true, ".pxi": true,
	".so": true, ".pyd": true, ".dylib": true, ".dll": true, ".a": true, ".o": true,
	".rs": true, ".f90": true, ".cu": true,
}

// entry is one file that will land in the wheel, at exactly `Path` under site-packages.
type entry struct {
	Path string // slash-separated, relative, normalized
	Data []byte
	Hash string // hex sha256, for the tree digest
}

// walk produces the CANONICAL path-sorted tree: the build input and the identity source
// (tensorhub-build.md §1 stage 1). Everything it refuses, it refuses by name.
func walk(root string) ([]entry, *exit.Error) {
	info, err := os.Stat(root)
	if err != nil {
		return nil, exit.Named(exit.NotFound, "tree_missing", "%s is not readable: %v", root, err)
	}
	if !info.IsDir() {
		return nil, exit.Named(exit.Usage, "tree_not_a_directory", "%s is not a directory", root)
	}

	var out []entry
	var total int64
	seen := map[string]string{} // case-folded path -> the path that claimed it

	err = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if p == root {
			return nil
		}
		rel, rerr := filepath.Rel(root, p)
		if rerr != nil {
			return rerr
		}
		rel = filepath.ToSlash(rel)

		if d.IsDir() {
			if excludedDir[d.Name()] || strings.HasSuffix(d.Name(), ".egg-info") {
				return fs.SkipDir
			}
			return nil
		}
		if excludedFile[d.Name()] || strings.HasSuffix(d.Name(), ".pyc") || strings.HasSuffix(d.Name(), ".pyo") {
			return nil
		}

		if e := checkName(rel); e != nil {
			return e
		}
		// A symlink is a second name for bytes that may live outside the tree; a device,
		// socket or FIFO is not content at all. Neither is silently skipped.
		mode := d.Type()
		if mode&fs.ModeSymlink != 0 {
			return exit.Named(exit.Validation, "unsafe_entry",
				"%s is a symlink; the canonical tree holds regular files only", rel).
				WithRemedy("replace the link with the file it names, or drop it")
		}
		if !mode.IsRegular() {
			return exit.Named(exit.Validation, "unsafe_entry",
				"%s is not a regular file (%s)", rel, mode.String()).
				WithRemedy("a wheel carries file bytes; nothing else has content to carry")
		}
		if prior, dup := seen[strings.ToLower(rel)]; dup {
			return exit.Named(exit.Validation, "unsafe_entry",
				"%s and %s differ only by case; the wheel must install on a case-insensitive filesystem too", prior, rel)
		}
		seen[strings.ToLower(rel)] = rel

		if ext := strings.ToLower(path.Ext(rel)); compiledExt[ext] {
			return refuseCompiled(rel)
		}
		if base := path.Base(rel); base == "setup.py" {
			return exit.Named(exit.Validation, "project_code_execution",
				"%s is a packaging step that RUNS project code", rel).
				WithRemedy("the env lane executes no tenant code at assembly (tensorhub-build.md §0); " +
					"declare static metadata in pyproject.toml's `[project]` instead").
				WithNext("packaging that must execute project code is the FUTURE sandboxed class: " +
					"it runs as hostile input under the VM-class sandbox posture (tensorhub-build.md §1.1), " +
					"never as an unremarked exception")
		}

		st, serr := d.Info()
		if serr != nil {
			return serr
		}
		total += st.Size()
		if total > MaxTreeBytes {
			return exit.Named(exit.Validation, "tree_too_large",
				"the project tree exceeds %d B; weights and datasets travel as CAS blobs, not inside the importable unit", MaxTreeBytes)
		}
		body, berr := os.ReadFile(p)
		if berr != nil {
			return berr
		}
		sum := sha256.Sum256(body)
		out = append(out, entry{Path: rel, Data: body, Hash: hex.EncodeToString(sum[:])})
		return nil
	})
	if err != nil {
		if e := exit.As(err); e != nil {
			return nil, e
		}
		return nil, exit.Named(exit.Structural, "tree_unreadable", "reading %s: %v", root, err)
	}
	if len(out) == 0 {
		return nil, exit.Named(exit.Validation, "tree_empty", "%s holds no packable file", root)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, nil
}

func refuseCompiled(rel string) *exit.Error {
	return exit.Named(exit.Validation, "compiled_extension",
		"%s is a compiled-extension source or binary; this packer emits %s and nothing else", rel, Tag).
		WithRemedy("a pure-python endpoint has no such file in its tree").
		WithNext("compiled project wheels are the FUTURE sandboxed class: they execute a toolchain " +
			"at assembly and therefore land on the VM-class sandbox posture (tensorhub-build.md §1.1), " +
			"never as an unremarked exception to \"assembly executes no tenant code\"")
}

// checkName holds the rule that nothing inside a wheel is an absolute path, a traversal,
// or a name a zip reader has to interpret.
func checkName(rel string) *exit.Error {
	bad := func(why string) *exit.Error {
		return exit.Named(exit.Validation, "unsafe_entry", "%q %s", rel, why).
			WithRemedy("wheel entry names are relative, normalized, printable and slash-separated")
	}
	switch {
	case rel == "" || rel == ".":
		return bad("is not a path")
	case path.IsAbs(rel) || strings.HasPrefix(rel, "/"):
		return bad("is absolute")
	case strings.Contains(rel, "\\"):
		return bad("contains a backslash")
	case rel != path.Clean(rel):
		return bad("is not normalized")
	}
	for _, part := range strings.Split(rel, "/") {
		if part == "" || part == "." || part == ".." {
			return bad("escapes or repeats a path separator")
		}
	}
	for i := 0; i < len(rel); i++ {
		if rel[i] < 0x20 || rel[i] == 0x7f {
			return bad("holds a control character")
		}
	}
	return nil
}

// treeDigest is the identity of the canonical tree itself: path and content, in sorted
// order, and nothing about the machine that read it. It is recorded beside the wheel
// digest so a two-seat comparison can say WHICH half diverged when they disagree.
func treeDigest(entries []entry) string {
	h := sha256.New()
	for _, e := range entries {
		fmt.Fprintf(h, "%s\x00%s\n", e.Path, e.Hash)
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

// checkBackend is the ONE thing `[build-system]` is read for. The packer does not consult
// it to decide how to build — there is one way, and this is the refusal that says so.
func checkBackend(d declaration) *exit.Error {
	if !d.hasBuildSy || d.backend == "" {
		return nil
	}
	return exit.Named(exit.Validation, "build_backend_unsupported",
		"%s declares `[build-system] build-backend = %q`; the Cozy packer implements no backend and fires no PEP 517 hook",
		pyprojectName, d.backend).
		WithRemedy("remove `[build-system]`: a pure-python endpoint is packed from its declared tree, " +
			"and its dependencies are resolved from the lock, not by a backend").
		WithNext("custom PEP 517 backends are the FUTURE sandboxed class: a backend is tenant code, " +
			"so it runs under the VM-class sandbox posture (tensorhub-build.md §1.1) or it does not run")
}

// checkApplication answers the only structural question the packer can answer about the
// endpoint's own contract: whether the module `endpoint.toml` points the runtime at is
// actually IN the wheel. Everything at serve time imports the installed wheel, so a
// module that is not in it is unservable — said here, not at first invoke.
func checkApplication(d declaration, entries []entry) *exit.Error {
	if d.appModule == "" {
		return nil
	}
	want := strings.ReplaceAll(d.appModule, ".", "/")
	for _, e := range entries {
		if e.Path == want+".py" || strings.HasPrefix(e.Path, want+"/") {
			return nil
		}
	}
	return exit.Named(exit.Validation, "application_module_absent",
		"%s points the runtime at `%s`, which this tree does not contain", endpointName, d.appModule).
		WithRemedy("describe, conformance and serving all import the INSTALLED wheel; a module outside it is unreachable")
}
