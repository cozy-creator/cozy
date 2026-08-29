// Package packagepublish prepares one portable package release declaration.
// It owns local packaging and nothing server-authoritative: Tensorhub chooses keys,
// profiles/base revisions, proof seats, execution identities, and serving state.
package packagepublish

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/wheel"
)

const (
	PackageDescriptorName = "package.descriptor.json"
	LockName              = "uv.lock"
	MaxLockBytes          = 16 << 20
	MaxSourceBytes        = 512 << 20
)

type ObjectRef struct {
	Digest string `json:"digest"`
	Length int64  `json:"length"`
}

type Declaration struct {
	Format            string     `json:"format"`
	SourceArchive     ObjectRef  `json:"source_archive"`
	SourceLock        ObjectRef  `json:"source_lock"`
	ProjectWheel      wheel.Fact `json:"project_wheel"`
	PackageDescriptor ObjectRef  `json:"package_descriptor"`
}

// Package retains the exact local bytes for a foreground begin/upload/finalize walk.
// Paths never enter Declaration or any Tensorhub request.
type Package struct {
	Declaration Declaration
	Files       map[string]string // exact upload role -> local path
	Root        string            // disposable staging root
}

func (p *Package) Close() { _ = os.RemoveAll(p.Root) }

type Request struct {
	Tree    string
	Release string
}

// Prepare requires a clean committed source subtree, deterministically packs its
// tracked files and pure project wheel, and canonicalizes the descriptor. Package
// requirements stay in wheel metadata and uv.lock. No project command executes.
func Prepare(req Request) (*Package, *exit.Error) {
	release := strings.TrimSpace(req.Release)
	if release == "" || strings.ContainsAny(release, `/\`) || release == "." || release == ".." {
		return nil, exit.Usagef("--release needs one safe immutable release id")
	}
	tree, files, e := trackedTree(req.Tree)
	if e != nil {
		return nil, e
	}
	if e := auditSource(tree, files); e != nil {
		return nil, e
	}
	root, err := os.MkdirTemp("", "cozy-package-publish-")
	if err != nil {
		return nil, exit.Internalf("cannot create package publication staging: %s", err)
	}
	fail := func(problem *exit.Error) (*Package, *exit.Error) {
		_ = os.RemoveAll(root)
		return nil, problem
	}

	archive := filepath.Join(root, "source.tar.gz")
	if e := sourceArchive(tree, files, archive); e != nil {
		return fail(e)
	}
	lockSource := filepath.Join(tree, LockName)
	lockPath := filepath.Join(root, LockName)
	lock, e := copyBounded(lockSource, lockPath, MaxLockBytes, "source_lock")
	if e != nil {
		return fail(e)
	}
	descriptor, e := deriveDescriptor(tree, root)
	if e != nil {
		return fail(e)
	}
	needsBindings, e := descriptorNeedsBindings(descriptor)
	if e != nil {
		return fail(e)
	}
	if needsBindings {
		return fail(exit.Named(exit.Validation, "package_model_binding_deferred",
			"model-bearing package publication is not available yet").
			WithRemedy("keep binding intent in package.toml; Tensorhub will resolve model releases there when the model-binding lane lands"))
	}

	project, e := wheel.Pack(wheel.Request{Tree: tree,
		OutDir: filepath.Join(root, "project")})
	if e != nil {
		return fail(e)
	}
	archiveRef, e := fileRef(archive, MaxSourceBytes, "source_archive")
	if e != nil {
		return fail(e)
	}

	result := &Package{Root: root, Files: map[string]string{
		"source_archive":     archive,
		"source_lock":        lockPath,
		"project_wheel":      project.Path,
		"package_descriptor": descriptor,
	}}
	result.Declaration = Declaration{
		Format:        "tensorhub.package_release_declaration/1",
		SourceArchive: archiveRef, SourceLock: lock,
		ProjectWheel: project.Fact, PackageDescriptor: descriptorRef(descriptor),
	}
	return result, nil
}

func trackedTree(tree string) (string, []string, *exit.Error) {
	abs, err := filepath.Abs(tree)
	if err != nil {
		return "", nil, exit.Usagef("--dir %q is not resolvable: %s", tree, err)
	}
	info, err := os.Stat(abs)
	if err != nil || !info.IsDir() {
		return "", nil, exit.Named(exit.NotFound, "package_tree_absent", "%s is not a directory", abs)
	}
	topRaw, err := exec.Command("git", "-C", abs, "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return "", nil, exit.Named(exit.Validation, "package_tree_untracked",
			"%s is not inside a Git working tree", abs).
			WithRemedy("package publication snapshots committed tracked files, never an unrestricted directory walk")
	}
	top := strings.TrimSpace(string(topRaw))
	rel, err := filepath.Rel(top, abs)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", nil, exit.Internalf("package tree %s escaped Git root %s", abs, top)
	}
	scope := "."
	if rel != "." {
		scope = filepath.ToSlash(rel)
	}
	status := exec.Command("git", "-C", top, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", scope)
	body, err := status.Output()
	if err != nil {
		return "", nil, exit.Named(exit.Structural, "package_tree_unreadable", "git status failed: %v", err)
	}
	if len(body) != 0 {
		return "", nil, exit.Named(exit.Conflict, "package_tree_dirty",
			"%s has changed or untracked files", abs).
			WithRemedy("commit the exact package source first; a normal release is one immutable tracked tree")
	}
	listing := exec.Command("git", "-C", top, "ls-files", "-z", "--", scope)
	body, err = listing.Output()
	if err != nil {
		return "", nil, exit.Named(exit.Structural, "package_tree_unreadable", "git ls-files failed: %v", err)
	}
	var files []string
	prefix := ""
	if scope != "." {
		prefix = scope + "/"
	}
	for _, raw := range bytes.Split(body, []byte{0}) {
		if len(raw) == 0 {
			continue
		}
		name := filepath.ToSlash(string(raw))
		name = strings.TrimPrefix(name, prefix)
		if name == "" || strings.HasPrefix(name, "../") {
			return "", nil, exit.Internalf("git returned out-of-scope package path %q", name)
		}
		files = append(files, name)
	}
	if len(files) == 0 {
		return "", nil, exit.Named(exit.Validation, "package_tree_empty", "%s has no tracked files", abs)
	}
	sort.Strings(files)
	return abs, files, nil
}

var weightExt = map[string]bool{
	".bin": true, ".ckpt": true, ".gguf": true, ".onnx": true, ".pickle": true,
	".pkl": true, ".pt": true, ".pth": true, ".safetensors": true,
}

var nativeSource = map[string]bool{
	".a": true, ".c": true, ".cc": true, ".cpp": true, ".cu": true, ".cxx": true,
	".dll": true, ".dylib": true, ".f90": true, ".h": true, ".hpp": true,
	".o": true, ".pxd": true, ".pxi": true, ".pyd": true, ".pyx": true,
	".rs": true, ".so": true,
}

func auditSource(root string, files []string) *exit.Error {
	var total int64
	for _, name := range files {
		lower, base := strings.ToLower(name), strings.ToLower(path.Base(name))
		if base == "package.descriptor.json" || base == "package.release.json" ||
			base == "package.evaluated-config.json" {
			return exit.Named(exit.Validation, "package_metadata_retired",
				"%s is retired package publication metadata", name).
				WithRemedy("delete it; package.toml is the only author configuration, and Tensorhub derives compatibility from pyproject.toml and uv.lock")
		}
		parts := strings.Split(lower, "/")
		for _, part := range parts[:len(parts)-1] {
			if part == ".aws" || part == ".ssh" || part == "credentials" || part == "secrets" {
				return sourceRefusal("package_source_credential", name, "credential directory")
			}
		}
		if base == ".env" || strings.HasPrefix(base, ".env.") || base == ".netrc" ||
			base == ".npmrc" || base == ".pypirc" || strings.HasPrefix(base, "id_rsa") ||
			strings.HasSuffix(base, ".pem") || strings.HasSuffix(base, ".key") {
			return sourceRefusal("package_source_credential", name, "credential/key material")
		}
		ext := strings.ToLower(path.Ext(base))
		if weightExt[ext] {
			return sourceRefusal("package_source_model_bytes", name, "model weight/pickle bytes")
		}
		if nativeSource[ext] {
			return sourceRefusal("package_source_native_input", name, "native source or binary")
		}
		if sourceBuildInput(base) {
			return sourceRefusal("package_source_build_input", name, "build recipe")
		}
		full := filepath.Join(root, filepath.FromSlash(name))
		info, err := os.Lstat(full)
		if err != nil || !info.Mode().IsRegular() {
			return sourceRefusal("package_source_entry_invalid", name, "non-regular or unreadable entry")
		}
		total += info.Size()
		if total > MaxSourceBytes {
			return exit.Named(exit.Validation, "package_source_too_large",
				"tracked package source exceeds %d B", MaxSourceBytes)
		}
	}
	return nil
}

func deriveDescriptor(tree, root string) (string, *exit.Error) {
	output := filepath.Join(root, PackageDescriptorName)
	environment := filepath.Join(root, "descriptor-venv")
	toolEnv := config.Frozen().Tool(
		"UV_PROJECT_ENVIRONMENT="+environment,
		"UV_LINK_MODE=hardlink",
	)
	sync := exec.Command("uv", "sync", "--locked", "--no-progress", "--no-install-project",
		"--project", tree)
	sync.Env = toolEnv
	var syncStderr bytes.Buffer
	sync.Stdout, sync.Stderr = io.Discard, &syncStderr
	err := sync.Run()
	if sync.ProcessState == nil {
		return "", exit.Named(exit.Structural, "package_descriptor_environment_missing",
			"cannot run uv for the package's locked environment: %v", err)
	}
	if sync.ProcessState.ExitCode() != 0 {
		return "", exit.Named(exit.Structural, "package_descriptor_environment_refused",
			"uv sync --locked --no-install-project refused: %s",
			strings.Join(strings.Fields(syncStderr.String()), " ")).
			WithRemedy("make pyproject.toml and uv.lock an exact portable dependency closure")
	}
	runtime := home.VenvTool(environment, "cozy-runtime")
	cmd := exec.Command(runtime, "--json", "--dir", tree, "describe")
	cmd.Env = toolEnv
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err = cmd.Run()
	if cmd.ProcessState == nil {
		return "", exit.Named(exit.Structural, "package_descriptor_runtime_missing",
			"cannot run the package's locked cozy-runtime: %v", err).
			WithRemedy("declare cozy-runtime in pyproject.toml and lock it in uv.lock")
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		return "", descriptorRefusal(code, stderr.Bytes())
	}
	if stdout.Len() == 0 || stdout.Len() > canonical.DocMax {
		return "", exit.Named(exit.Validation, "package_descriptor_invalid",
			"cozy-runtime describe returned %d bytes; expected 1..%d", stdout.Len(), canonical.DocMax)
	}
	canonicalBytes, err := canonical.NormalizeJCS(stdout.Bytes())
	if err != nil {
		return "", exit.Named(exit.Validation, "package_descriptor_invalid",
			"cozy-runtime describe returned invalid canonical JSON: %v", err)
	}
	if err := os.WriteFile(output, canonicalBytes, 0o600); err != nil {
		return "", exit.Internalf("cannot stage derived descriptor: %s", err)
	}
	return output, nil
}

func descriptorRefusal(code int, body []byte) *exit.Error {
	c := exit.Code(code)
	if !c.Valid() {
		c = exit.Internal
	}
	var doc struct {
		Error struct {
			Name    string `json:"name"`
			Message string `json:"message"`
			Remedy  string `json:"remedy"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &doc) == nil && doc.Error.Message != "" {
		name := doc.Error.Name
		if name == "" {
			name = "package_descriptor_refused"
		}
		problem := exit.Named(c, name, "%s", doc.Error.Message)
		if doc.Error.Remedy != "" {
			problem.WithRemedy("%s", doc.Error.Remedy)
		}
		return problem
	}
	message := strings.Join(strings.Fields(string(body)), " ")
	if message == "" {
		message = fmt.Sprintf("cozy-runtime exited %d", code)
	}
	return exit.Named(c, "package_descriptor_refused", "%s", message)
}

func sourceBuildInput(base string) bool {
	switch base {
	case "cargo.lock", "cargo.toml", "cmakelists.txt", "gnumakefile", "makefile",
		"manifest.in", "meson.build", "meson_options.txt", "setup.cfg", "setup.py":
		return true
	}
	return strings.HasPrefix(base, "dockerfile") || strings.HasSuffix(base, ".cmake")
}

func sourceRefusal(code, name, class string) *exit.Error {
	return exit.Named(exit.Validation, code, "%s is %s and cannot enter a package release", name, class).
		WithRemedy("publish pure Python code only; model weights live in model repositories, and source/native builds are not admitted")
}

func sourceArchive(root string, files []string, output string) *exit.Error {
	f, err := os.OpenFile(output, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return exit.Internalf("cannot create source archive: %s", err)
	}
	defer f.Close()
	gz, err := gzip.NewWriterLevel(f, gzip.BestCompression)
	if err != nil {
		return exit.Internalf("cannot create source compressor: %s", err)
	}
	gz.Header = gzip.Header{ModTime: time.Unix(0, 0), OS: 255}
	tw := tar.NewWriter(gz)
	for _, name := range files {
		full := filepath.Join(root, filepath.FromSlash(name))
		info, err := os.Stat(full)
		if err != nil {
			return exit.Internalf("tracked source %s disappeared: %s", name, err)
		}
		header := &tar.Header{Name: name, Mode: 0o644, Size: info.Size(),
			ModTime: time.Unix(0, 0), AccessTime: time.Unix(0, 0), ChangeTime: time.Unix(0, 0),
			Typeflag: tar.TypeReg, Format: tar.FormatPAX}
		if err := tw.WriteHeader(header); err != nil {
			return exit.Internalf("cannot archive %s: %s", name, err)
		}
		in, err := os.Open(full)
		if err != nil {
			return exit.Internalf("cannot read %s: %s", name, err)
		}
		_, copyErr := io.Copy(tw, in)
		in.Close()
		if copyErr != nil {
			return exit.Internalf("cannot archive %s: %s", name, copyErr)
		}
	}
	if err := tw.Close(); err != nil {
		return exit.Internalf("cannot finish source tar: %s", err)
	}
	if err := gz.Close(); err != nil {
		return exit.Internalf("cannot finish source gzip: %s", err)
	}
	return nil
}

func descriptorNeedsBindings(file string) (bool, *exit.Error) {
	body, err := os.ReadFile(file)
	if err != nil {
		return false, exit.Internalf("cannot reread canonical descriptor: %s", err)
	}
	var descriptor struct {
		Entrypoints []struct {
			Models []json.RawMessage `json:"models"`
		} `json:"entrypoints"`
		Jobs []struct {
			Models []json.RawMessage `json:"models"`
		} `json:"jobs"`
	}
	if err := json.Unmarshal(body, &descriptor); err != nil {
		return false, exit.Named(exit.Validation, "package_descriptor_invalid", "%s: %v", PackageDescriptorName, err)
	}
	for _, entrypoint := range descriptor.Entrypoints {
		if len(entrypoint.Models) > 0 {
			return true, nil
		}
	}
	for _, job := range descriptor.Jobs {
		if len(job.Models) > 0 {
			return true, nil
		}
	}
	return false, nil
}

func copyBounded(source, target string, limit int64, role string) (ObjectRef, *exit.Error) {
	ref, e := fileRef(source, limit, role)
	if e != nil {
		return ref, e
	}
	in, err := os.Open(source)
	if err != nil {
		return ref, exit.Named(exit.Structural, role+"_unreadable", "%s: %v", source, err)
	}
	defer in.Close()
	out, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return ref, exit.Internalf("cannot stage %s: %s", role, err)
	}
	_, err = io.Copy(out, in)
	closeErr := out.Close()
	if err != nil || closeErr != nil {
		return ref, exit.Internalf("cannot stage %s: %v %v", role, err, closeErr)
	}
	return ref, nil
}

func fileRef(file string, limit int64, role string) (ObjectRef, *exit.Error) {
	var out ObjectRef
	f, err := os.Open(file)
	if err != nil {
		return out, exit.Named(exit.NotFound, role+"_absent", "%s: %v", file, err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > limit {
		return out, exit.Named(exit.Validation, role+"_size_invalid", "%s is outside 1..%d B", file, limit)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return out, exit.Named(exit.Structural, role+"_unreadable", "%s: %v", file, err)
	}
	return ObjectRef{Digest: "sha256:" + hex.EncodeToString(h.Sum(nil)), Length: info.Size()}, nil
}

func descriptorRef(file string) ObjectRef {
	ref, _ := fileRef(file, canonical.DocMax, "canonical_document")
	return ref
}

func (d Declaration) CanonicalBytes() ([]byte, *exit.Error) {
	body, err := json.Marshal(d)
	if err != nil {
		return nil, exit.Internalf("cannot encode package declaration: %s", err)
	}
	canonicalBytes, err := canonical.NormalizeJCS(body)
	if err != nil {
		return nil, exit.Internalf("package declaration is outside canonical JSON: %s", err)
	}
	return canonicalBytes, nil
}

func (d Declaration) Digest() string {
	body, _ := d.CanonicalBytes()
	sum := sha256.Sum256(body)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func (d Declaration) String() string {
	return fmt.Sprintf("%s (%s)", d.Digest(), d.ProjectWheel.Filename)
}
