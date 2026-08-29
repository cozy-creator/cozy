// Package endpointpublish prepares one portable endpoint release declaration.
// It owns local packaging and nothing server-authoritative: Tensorhub chooses keys,
// profiles/base revisions, proof seats, execution identities, and serving state.
package endpointpublish

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
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/endpointprofile"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/wheel"
)

const (
	DescriptorName = "endpoint.descriptor.json"
	EvaluatedName  = "endpoint.evaluated-config.json"
	ReleaseName    = "endpoint.release.json"
	LockName       = "uv.lock"
	MaxLockBytes   = 16 << 20
	MaxSourceBytes = 512 << 20
)

type ObjectRef struct {
	Digest string `json:"digest"`
	Length int64  `json:"length"`
}

type CustomWheel struct {
	Wheel    wheel.Fact `json:"wheel"`
	Profiles []string   `json:"profiles"`
}

type RootRef struct {
	Org          string `json:"org"`
	Name         string `json:"name"`
	CheckpointID string `json:"checkpoint_id"`
}

type NamedRef struct {
	Name string    `json:"name"`
	Ref  ObjectRef `json:"ref"`
}

type ModelConfig struct {
	Assets   []NamedRef `json:"assets"`
	Document ObjectRef  `json:"document"`
}

type ExecutionComponent struct {
	Component string  `json:"component"`
	Root      RootRef `json:"root"`
}

type ModelBinding struct {
	Checkpoint      RootRef              `json:"checkpoint"`
	Config          ModelConfig          `json:"config"`
	ExecutionLayout []ExecutionComponent `json:"execution_layout"`
	HardwareVariant string               `json:"hardware_variant"`
	Path            string               `json:"path"`
}

type NativeWheelProof struct {
	ExpectedResultDigest string `json:"expected_result_digest"`
	Fixture              string `json:"fixture"`
}

type Declaration struct {
	Format                      string            `json:"format"`
	SourceArchive               ObjectRef         `json:"source_archive"`
	SourceLock                  ObjectRef         `json:"source_lock"`
	ProjectWheel                wheel.Fact        `json:"project_wheel"`
	Profiles                    []string          `json:"profiles"`
	CustomWheels                []CustomWheel     `json:"custom_wheels"`
	Descriptor                  ObjectRef         `json:"descriptor"`
	EvaluatedConfig             ObjectRef         `json:"evaluated_config"`
	CompatibleAcceleratorModels []string          `json:"compatible_accelerator_models"`
	ModelRoots                  []RootRef         `json:"model_roots"`
	ModelBindings               []ModelBinding    `json:"model_bindings"`
	NativeWheelProof            *NativeWheelProof `json:"native_wheel_proof,omitempty"`
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
	Tree         string
	Release      string
	Profiles     []string
	CustomWheels []string // repeatable <profile>=<wheel path>
}

type releaseConfig struct {
	CompatibleAcceleratorModels []string          `json:"compatible_accelerator_models"`
	ModelRoots                  []RootRef         `json:"model_roots"`
	ModelBindings               []ModelBinding    `json:"model_bindings"`
	NativeWheelProof            *NativeWheelProof `json:"native_wheel_proof"`
}

// Prepare requires a clean committed source subtree, deterministically packs its
// tracked files and pure project wheel, canonicalizes semantic documents, and
// inspects optional exact prebuilt custom wheels. No project command executes.
func Prepare(req Request) (*Package, *exit.Error) {
	profiles, e := endpointprofile.NormalizeSet(req.Profiles)
	if e != nil {
		return nil, e
	}
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
	root, err := os.MkdirTemp("", "cozy-endpoint-publish-")
	if err != nil {
		return nil, exit.Internalf("cannot create endpoint publication staging: %s", err)
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
	descriptorSource := filepath.Join(tree, DescriptorName)
	descriptor, e := canonicalDocument(descriptorSource, filepath.Join(root, DescriptorName), false)
	if e != nil {
		return fail(e)
	}
	evaluated, e := evaluatedConfig(tree, filepath.Join(root, EvaluatedName))
	if e != nil {
		return fail(e)
	}
	config, e := readReleaseConfig(tree)
	if e != nil {
		return fail(e)
	}
	compatibleModels := sortedUnique(config.CompatibleAcceleratorModels)
	if len(compatibleModels) == 0 {
		return fail(exit.Named(exit.Validation, "endpoint_compatible_accelerator_models_absent",
			"endpoint.release.json names no compatible accelerator model").
			WithRemedy("declare the exact provider-neutral GPU model set this release may qualify; qualification cannot mutate release compatibility"))
	}
	needsBindings, e := descriptorNeedsBindings(descriptor)
	if e != nil {
		return fail(e)
	}
	if needsBindings && len(config.ModelBindings) == 0 {
		return fail(exit.Named(exit.Validation, "endpoint_model_bindings_absent",
			"the descriptor declares model inputs but endpoint.release.json carries no model_bindings").
			WithRemedy("declare the exact checkpoint, config refs, execution layout, hardware variant, and binding path"))
	}

	project, e := wheel.Pack(wheel.Request{Tree: tree,
		OutDir: filepath.Join(root, "project")})
	if e != nil {
		return fail(e)
	}
	custom, customFiles, e := inspectCustom(req.CustomWheels, profiles, filepath.Join(root, "custom"))
	if e != nil {
		return fail(e)
	}
	native := false
	for _, item := range custom {
		native = native || item.Wheel.Native
	}
	if native {
		if config.NativeWheelProof == nil || !digest(config.NativeWheelProof.ExpectedResultDigest) ||
			!fixtureName(config.NativeWheelProof.Fixture) {
			return fail(exit.Named(exit.Validation, "native_wheel_proof_absent",
				"native custom wheels require a banked fixture and expected canonical result digest in endpoint.release.json"))
		}
	} else if config.NativeWheelProof != nil {
		return fail(exit.Named(exit.Validation, "native_wheel_proof_unexpected",
			"endpoint.release.json declares native proof but no custom wheel contains native bytes"))
	}
	archiveRef, e := fileRef(archive, MaxSourceBytes, "source_archive")
	if e != nil {
		return fail(e)
	}

	result := &Package{Root: root, Files: map[string]string{
		"source_archive":   archive,
		"source_lock":      lockPath,
		"project_wheel":    project.Path,
		"descriptor":       descriptor,
		"evaluated_config": evaluated,
	}}
	for role, file := range customFiles {
		result.Files[role] = file
	}
	result.Declaration = Declaration{
		Format:        "tensorhub.endpoint_release_declaration/1",
		SourceArchive: archiveRef, SourceLock: lock,
		ProjectWheel: project.Fact, Profiles: profiles, CustomWheels: custom,
		Descriptor: descriptorRef(descriptor), EvaluatedConfig: descriptorRef(evaluated),
		CompatibleAcceleratorModels: compatibleModels,
		ModelRoots:                  sortedRoots(config.ModelRoots),
		ModelBindings:               config.ModelBindings,
		NativeWheelProof:            config.NativeWheelProof,
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
		return "", nil, exit.Named(exit.NotFound, "endpoint_tree_absent", "%s is not a directory", abs)
	}
	topRaw, err := exec.Command("git", "-C", abs, "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return "", nil, exit.Named(exit.Validation, "endpoint_tree_untracked",
			"%s is not inside a Git working tree", abs).
			WithRemedy("endpoint publication snapshots committed tracked files, never an unrestricted directory walk")
	}
	top := strings.TrimSpace(string(topRaw))
	rel, err := filepath.Rel(top, abs)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", nil, exit.Internalf("endpoint tree %s escaped Git root %s", abs, top)
	}
	scope := "."
	if rel != "." {
		scope = filepath.ToSlash(rel)
	}
	status := exec.Command("git", "-C", top, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", scope)
	body, err := status.Output()
	if err != nil {
		return "", nil, exit.Named(exit.Structural, "endpoint_tree_unreadable", "git status failed: %v", err)
	}
	if len(body) != 0 {
		return "", nil, exit.Named(exit.Conflict, "endpoint_tree_dirty",
			"%s has changed or untracked files", abs).
			WithRemedy("commit the exact endpoint source first; a normal release is one immutable tracked tree")
	}
	listing := exec.Command("git", "-C", top, "ls-files", "-z", "--", scope)
	body, err = listing.Output()
	if err != nil {
		return "", nil, exit.Named(exit.Structural, "endpoint_tree_unreadable", "git ls-files failed: %v", err)
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
			return "", nil, exit.Internalf("git returned out-of-scope endpoint path %q", name)
		}
		files = append(files, name)
	}
	if len(files) == 0 {
		return "", nil, exit.Named(exit.Validation, "endpoint_tree_empty", "%s has no tracked files", abs)
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
		parts := strings.Split(lower, "/")
		for _, part := range parts[:len(parts)-1] {
			if part == ".aws" || part == ".ssh" || part == "credentials" || part == "secrets" {
				return sourceRefusal("endpoint_source_credential", name, "credential directory")
			}
		}
		if base == ".env" || strings.HasPrefix(base, ".env.") || base == ".netrc" ||
			base == ".npmrc" || base == ".pypirc" || strings.HasPrefix(base, "id_rsa") ||
			strings.HasSuffix(base, ".pem") || strings.HasSuffix(base, ".key") {
			return sourceRefusal("endpoint_source_credential", name, "credential/key material")
		}
		ext := strings.ToLower(path.Ext(base))
		if weightExt[ext] {
			return sourceRefusal("endpoint_source_model_bytes", name, "model weight/pickle bytes")
		}
		if nativeSource[ext] {
			return sourceRefusal("endpoint_source_native_input", name, "native source or binary")
		}
		if sourceBuildInput(base) {
			return sourceRefusal("endpoint_source_build_input", name, "build recipe")
		}
		full := filepath.Join(root, filepath.FromSlash(name))
		info, err := os.Lstat(full)
		if err != nil || !info.Mode().IsRegular() {
			return sourceRefusal("endpoint_source_entry_invalid", name, "non-regular or unreadable entry")
		}
		total += info.Size()
		if total > MaxSourceBytes {
			return exit.Named(exit.Validation, "endpoint_source_too_large",
				"tracked endpoint source exceeds %d B", MaxSourceBytes)
		}
	}
	return nil
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
	return exit.Named(exit.Validation, code, "%s is %s and cannot enter an endpoint release", name, class).
		WithRemedy("publish code only; weights use model roots, and native dependencies use separately prebuilt exact custom wheels")
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

func canonicalDocument(source, output string, allowAbsent bool) (string, *exit.Error) {
	body, err := os.ReadFile(source)
	if os.IsNotExist(err) && allowAbsent {
		body = []byte("{}")
	} else if err != nil {
		return "", exit.Named(exit.NotFound, "endpoint_document_absent", "%s is required: %v", source, err)
	}
	canonicalBytes, err := canonical.NormalizeJCS(body)
	if err != nil {
		return "", exit.Named(exit.Validation, "endpoint_document_invalid", "%s: %v", filepath.Base(source), err)
	}
	if err := os.WriteFile(output, canonicalBytes, 0o600); err != nil {
		return "", exit.Internalf("cannot stage %s: %s", filepath.Base(source), err)
	}
	return output, nil
}

func evaluatedConfig(tree, output string) (string, *exit.Error) {
	return canonicalDocument(filepath.Join(tree, EvaluatedName), output, true)
}

func readReleaseConfig(tree string) (releaseConfig, *exit.Error) {
	var out releaseConfig
	file := filepath.Join(tree, ReleaseName)
	body, err := os.ReadFile(file)
	if os.IsNotExist(err) {
		out.CompatibleAcceleratorModels = []string{}
		out.ModelRoots = []RootRef{}
		out.ModelBindings = []ModelBinding{}
		return out, nil
	}
	if err != nil {
		return out, exit.Named(exit.Structural, "endpoint_release_config_unreadable", "%s: %v", file, err)
	}
	canonicalBytes, err := canonical.NormalizeJCS(body)
	if err != nil {
		return out, exit.Named(exit.Validation, "endpoint_release_config_invalid", "%s: %v", file, err)
	}
	decoder := json.NewDecoder(bytes.NewReader(canonicalBytes))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&out); err != nil {
		return out, exit.Named(exit.Validation, "endpoint_release_config_invalid", "%s: %v", file, err)
	}
	out.ModelRoots = sortedRoots(out.ModelRoots)
	for i, root := range out.ModelRoots {
		if !resourceSlug(root.Org) || !resourceSlug(root.Name) || !digest(root.CheckpointID) {
			return out, exit.Named(exit.Validation, "endpoint_model_root_invalid",
				"model_roots[%d] needs lowercase org/name and checkpoint_id sha256:<64 hex>", i)
		}
	}
	bindings, problem := normalizeBindings(out.ModelBindings)
	if problem != nil {
		return out, problem
	}
	out.ModelBindings = bindings
	boundRoots := map[string]bool{}
	for _, binding := range bindings {
		boundRoots[rootKey(binding.Checkpoint)] = true
		for _, component := range binding.ExecutionLayout {
			boundRoots[rootKey(component.Root)] = true
		}
	}
	declaredRoots := map[string]bool{}
	for _, root := range out.ModelRoots {
		declaredRoots[rootKey(root)] = true
	}
	if len(boundRoots) != len(declaredRoots) {
		return out, exit.Named(exit.Validation, "endpoint_model_roots_mismatch",
			"model_roots and model_bindings do not name the same exact checkpoint set")
	}
	for root := range boundRoots {
		if !declaredRoots[root] {
			return out, exit.Named(exit.Validation, "endpoint_model_roots_mismatch",
				"model binding root %s is absent from model_roots", strings.ReplaceAll(root, "\x00", "/"))
		}
	}
	return out, nil
}

var bindingIdentifier = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.-]{0,127}$`)
var hardwareVariant = regexp.MustCompile(`^sm[0-9]{2,}$`)
var proofFixture = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$`)

func fixtureName(value string) bool { return proofFixture.MatchString(value) }

func normalizeBindings(values []ModelBinding) ([]ModelBinding, *exit.Error) {
	out := append([]ModelBinding{}, values...)
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	prior := ""
	for i := range out {
		binding := &out[i]
		if !bindingIdentifier.MatchString(binding.Path) || binding.Path == prior ||
			!hardwareVariant.MatchString(binding.HardwareVariant) ||
			!validRoot(binding.Checkpoint) || len(binding.ExecutionLayout) == 0 ||
			!validObjectRef(binding.Config.Document) {
			return nil, exit.Named(exit.Validation, "endpoint_model_binding_invalid",
				"model_bindings[%d] has an invalid path, checkpoint, config, layout, or hardware_variant", i)
		}
		prior = binding.Path
		binding.Config.Assets = append([]NamedRef{}, binding.Config.Assets...)
		sort.Slice(binding.Config.Assets, func(i, j int) bool {
			return binding.Config.Assets[i].Name < binding.Config.Assets[j].Name
		})
		priorAsset := ""
		for j, asset := range binding.Config.Assets {
			if !bindingIdentifier.MatchString(asset.Name) || asset.Name == priorAsset || !validObjectRef(asset.Ref) {
				return nil, exit.Named(exit.Validation, "endpoint_model_binding_invalid",
					"model_bindings[%d].config.assets[%d] is invalid or duplicated", i, j)
			}
			priorAsset = asset.Name
		}
		seenComponent := map[string]bool{}
		for j, component := range binding.ExecutionLayout {
			if !bindingIdentifier.MatchString(component.Component) || seenComponent[component.Component] ||
				!validRoot(component.Root) {
				return nil, exit.Named(exit.Validation, "endpoint_model_binding_invalid",
					"model_bindings[%d].execution_layout[%d] is invalid or duplicated", i, j)
			}
			seenComponent[component.Component] = true
		}
	}
	return out, nil
}

func validRoot(root RootRef) bool {
	return resourceSlug(root.Org) && resourceSlug(root.Name) && digest(root.CheckpointID)
}

func validObjectRef(ref ObjectRef) bool { return digest(ref.Digest) && ref.Length > 0 }

func rootKey(root RootRef) string { return root.Org + "\x00" + root.Name + "\x00" + root.CheckpointID }

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
		return false, exit.Named(exit.Validation, "endpoint_descriptor_invalid", "%s: %v", DescriptorName, err)
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

func inspectCustom(specs, releaseProfiles []string, stageRoot string) ([]CustomWheel, map[string]string, *exit.Error) {
	type grouped struct {
		fact     wheel.Fact
		file     string
		profiles map[string]bool
	}
	byDigest := map[string]*grouped{}
	profileAllowed := map[string]bool{}
	for _, profile := range releaseProfiles {
		profileAllowed[profile] = true
	}
	for _, spec := range specs {
		profile, file, ok := strings.Cut(spec, "=")
		if !ok || strings.TrimSpace(file) == "" {
			return nil, nil, exit.Usagef("--custom-wheel needs <profile>=<wheel-path>, not %q", spec)
		}
		profiles, e := endpointprofile.NormalizeSet([]string{profile})
		if e != nil {
			return nil, nil, e
		}
		profile = profiles[0]
		if !profileAllowed[profile] {
			return nil, nil, exit.Named(exit.Validation, "custom_wheel_profile_outside_release",
				"custom wheel profile %s is not in this publication's --profile set", profile)
		}
		fact, e := wheel.Inspect(strings.TrimSpace(file), wheel.CustomWheel)
		if e != nil {
			return nil, nil, e
		}
		abs, _ := filepath.Abs(strings.TrimSpace(file))
		group := byDigest[fact.Digest]
		if group == nil {
			group = &grouped{fact: fact, file: abs, profiles: map[string]bool{}}
			byDigest[fact.Digest] = group
		} else if !factEqual(group.fact, fact) || group.file != abs {
			return nil, nil, exit.Named(exit.Conflict, "custom_wheel_identity_changed",
				"one digest was declared with changed WheelFact or pathname")
		}
		group.profiles[profile] = true
	}
	out := []CustomWheel{}
	files := map[string]string{}
	seenDistProfile := map[string]string{}
	for _, group := range byDigest {
		profiles := make([]string, 0, len(group.profiles))
		for profile := range group.profiles {
			key := group.fact.Distribution + "\x00" + profile
			if prior := seenDistProfile[key]; prior != "" && prior != group.fact.Digest {
				return nil, nil, exit.Named(exit.Conflict, "custom_wheel_profile_conflict",
					"%s profile %s names multiple wheel digests", group.fact.Distribution, profile)
			}
			seenDistProfile[key] = group.fact.Digest
			profiles = append(profiles, profile)
		}
		sort.Strings(profiles)
		out = append(out, CustomWheel{Wheel: group.fact, Profiles: profiles})
		role := "custom_wheel:" + group.fact.Distribution + ":" + strings.TrimPrefix(group.fact.Digest, "sha256:")
		staged := filepath.Join(stageRoot, strings.TrimPrefix(group.fact.Digest, "sha256:"), group.fact.Filename)
		if err := os.MkdirAll(filepath.Dir(staged), 0o700); err != nil {
			return nil, nil, exit.Internalf("cannot stage custom wheel: %s", err)
		}
		if _, problem := copyBounded(group.file, staged, wheel.MaxWheelBytes, "custom_wheel"); problem != nil {
			return nil, nil, problem
		}
		files[role] = staged
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Wheel.Distribution != out[j].Wheel.Distribution {
			return out[i].Wheel.Distribution < out[j].Wheel.Distribution
		}
		return out[i].Wheel.Digest < out[j].Wheel.Digest
	})
	return out, files, nil
}

func factEqual(a, b wheel.Fact) bool {
	return a.Digest == b.Digest && a.Distribution == b.Distribution &&
		a.Filename == b.Filename && a.Length == b.Length && a.Version == b.Version &&
		slices.Equal(a.ImportRoots, b.ImportRoots) && slices.Equal(a.Tags, b.Tags)
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

func sortedUnique(values []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" && !seen[value] {
			seen[value] = true
			out = append(out, value)
		}
	}
	sort.Strings(out)
	return out
}

func sortedRoots(values []RootRef) []RootRef {
	seen := map[string]bool{}
	out := []RootRef{}
	for _, value := range values {
		key := value.Org + "\x00" + value.Name + "\x00" + value.CheckpointID
		if !seen[key] {
			seen[key] = true
			out = append(out, value)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Org != out[j].Org {
			return out[i].Org < out[j].Org
		}
		if out[i].Name != out[j].Name {
			return out[i].Name < out[j].Name
		}
		return out[i].CheckpointID < out[j].CheckpointID
	})
	return out
}

func resourceSlug(value string) bool {
	if value == "" {
		return false
	}
	for i, r := range value {
		if !(r >= 'a' && r <= 'z' || i > 0 && r >= '0' && r <= '9' ||
			i > 0 && (r == '-' || r == '_')) {
			return false
		}
	}
	return true
}

func digest(value string) bool {
	if len(value) != 71 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:"))
	return err == nil && strings.ToLower(value) == value
}

func (d Declaration) CanonicalBytes() ([]byte, *exit.Error) {
	body, err := json.Marshal(d)
	if err != nil {
		return nil, exit.Internalf("cannot encode endpoint declaration: %s", err)
	}
	canonicalBytes, err := canonical.NormalizeJCS(body)
	if err != nil {
		return nil, exit.Internalf("endpoint declaration is outside canonical JSON: %s", err)
	}
	return canonicalBytes, nil
}

func (d Declaration) Digest() string {
	body, _ := d.CanonicalBytes()
	sum := sha256.Sum256(body)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func (d Declaration) String() string {
	return fmt.Sprintf("%s (%d profile(s), %d custom wheel(s))", d.Digest(), len(d.Profiles), len(d.CustomWheels))
}
