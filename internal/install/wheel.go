package install

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
)

// HasInstalledApplications is only a discovery filter over the selected venv.
// Full metadata, source ownership and interface admission follow on exact wheel
// bytes. Ordinary libraries and already-bound source overlays need no new path.
func HasInstalledApplications(python, installed string, ignored map[string]string) (bool, *exit.Error) {
	pins := map[string]bool{}
	for _, row := range strings.Split(installed, "\n") {
		name, _, ok := strings.Cut(row, "==")
		if ok && ignored[name] == "" && !packagepublish.ImageOwnedDistribution(name) {
			pins[name] = true
		}
	}
	site := filepath.Join(filepath.Dir(filepath.Dir(python)), "lib", "python3.12", "site-packages")
	entries, err := os.ReadDir(site)
	if err != nil {
		return false, exit.New(exit.Conflict, "selected Python environment metadata is unavailable")
	}
	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasSuffix(entry.Name(), ".dist-info") {
			continue
		}
		stem := strings.TrimSuffix(entry.Name(), ".dist-info")
		index := strings.LastIndexByte(stem, '-')
		if index < 0 || !pins[normalizedRequirementName(stem[:index])] {
			continue
		}
		path := filepath.Join(site, entry.Name(), "entry_points.txt")
		info, err := os.Lstat(path)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil || !info.Mode().IsRegular() || info.Size() > 1<<20 {
			return false, exit.New(exit.Validation, "selected distribution entry points are not bounded regular metadata")
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return false, exit.New(exit.Conflict, "selected distribution entry points changed")
		}
		for _, line := range strings.Split(string(raw), "\n") {
			if strings.TrimSpace(line) == "[cozy.application]" {
				return true, nil
			}
		}
	}
	return false, nil
}

// ReadInstalledInterface asks the host's static parser about one explicitly
// selected distribution. Neither its Python nor its package imports execute.
func ReadInstalledInterface(ctx context.Context, python, distribution string) (*launch.PackageInterface, *exit.Error) {
	env := config.Frozen().Tool("COZY_HOME=" + runtimeScratchHome())
	bin, problem := hostruntime.Path(env)
	if problem != nil {
		return nil, problem
	}
	command := exec.CommandContext(ctx, bin, "--json", "describe", "--distribution", distribution, "--environment-python", python)
	command.Env = env
	var stdout, stderr strings.Builder
	command.Stdout, command.Stderr = &stdout, &stderr
	if err := command.Run(); err != nil {
		if command.ProcessState == nil {
			return nil, exit.Internalf("cannot start installed interface reader")
		}
		return nil, hostruntime.RuntimeExit(command.ProcessState.ExitCode(), "describe", "private_wheel_interface_refused", stdout.String(), stderr.String())
	}
	return launch.DecodePackageInterface([]byte(stdout.String()))
}

// CaptureWheel installs an exact wheel closure into an independent private
// environment. It runs no resolver and records no fake source project or pin.
func CaptureWheel(ctx context.Context, layout home.Layout, store *records.Store, parentPython, project string, dependencies map[string]packagepublish.CapturedDependency, surface *launch.PackageInterface) (*Result, *exit.Error) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		return nil, exit.New(exit.Validation, "private callable wheels require Linux amd64")
	}
	id, problem := newInstallID()
	if problem != nil {
		return nil, problem
	}
	dir := layout.InstallDir(id)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, exit.Internalf("cannot create private wheel install")
	}
	fail := func(problem *exit.Error) (*Result, *exit.Error) { _ = os.RemoveAll(dir); return nil, problem }
	root, ok := dependencies[project]
	if !ok || root.Path == "" || root.Digest == "" || !root.Application {
		return fail(exit.New(exit.Validation, "private wheel install requires its exact App wheel"))
	}
	names := make([]string, 0, len(dependencies))
	for name := range dependencies {
		names = append(names, name)
	}
	sort.Strings(names)
	var requirements []string
	type identity struct{ Name, Version, Wheel, BaseRequirement string }
	var identities []identity
	pins := map[string]string{}
	for _, name := range names {
		dependency := dependencies[name]
		if dependency.Name != name || dependency.Version == "" || dependency.Requirement == "" {
			return fail(exit.New(exit.Validation, "private wheel dependency is incomplete"))
		}
		requirements = append(requirements, dependency.Requirement)
		item := identity{Name: name, Version: dependency.Version, Wheel: dependency.Digest}
		if dependency.Path == "" {
			item.BaseRequirement = dependency.Requirement
		}
		identities = append(identities, item)
		pins[name] = dependency.Version
	}
	raw, _ := json.Marshal(identities)
	raw, err := canonical.NormalizeJCS(raw)
	if err != nil {
		return fail(exit.Internalf("cannot identify private wheel closure"))
	}
	digest, _ := canonical.Spell(canonical.Digest(raw))
	lockPath := filepath.Join(dir, "requirements.txt")
	if err := os.WriteFile(lockPath, []byte(strings.Join(requirements, "\n")+"\n"), 0o600); err != nil {
		return fail(exit.Internalf("cannot retain private wheel requirements"))
	}
	venv := filepath.Join(dir, "venv")
	env := config.Frozen().Tool()
	if problem := runUV(dir, env, "private_wheel_python_refused", "cannot select private wheel Python", "venv", "--python", parentPython, "--no-project", venv); problem != nil {
		return fail(problem)
	}
	if problem := runUV(dir, env, "private_wheel_install_refused", "cannot install exact private wheel closure", "pip", "install", "--no-deps", "--require-hashes", "--python", home.VenvPython(venv), "--requirements", lockPath); problem != nil {
		return fail(problem)
	}
	if problem := runUV(dir, env, "private_wheel_requirements_refused", "private wheel requirements are not satisfied", "pip", "check", "--python", home.VenvPython(venv)); problem != nil {
		return fail(problem)
	}
	count, installed := closure(venv)
	if installed != packagepublish.PinnedClosure(pins) {
		return fail(exit.New(exit.Conflict, "installed callable wheel environment differs from its exact selected closure"))
	}
	path := launch.PackageInterfacePath(dir)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fail(exit.Internalf("cannot retain private wheel interface"))
	}
	if err := os.WriteFile(path, surface.Raw, 0o600); err != nil {
		return fail(exit.Internalf("cannot retain private wheel interface"))
	}
	major, problem := MajorOf(root.Version)
	if problem != nil {
		return fail(problem)
	}
	inst := records.PackageInstall{ID: id, Package: "local/" + project, Version: root.Version, Major: major,
		SourceKind: "wheel", SourceRef: dir, SourceDigest: digest, ProjectDir: dir, Dir: dir,
		Python: pythonVersion(venv), UV: toolVersion("uv", "--version"), LockDigest: digest,
		Platform: "linux/amd64", Packages: count, Closure: installed, PackageInterface: surface.Digest}
	inst.BytesExcl, inst.BytesShared = Disk(dir)
	if problem := store.RecordInstall(inst); problem != nil {
		return fail(problem)
	}
	return &Result{Install: inst}, nil
}
