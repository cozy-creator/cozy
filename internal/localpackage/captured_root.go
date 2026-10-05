package localpackage

import (
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/wheel"
)

// CapturedRoot sends one already-captured environment, keeping each callable's package identity.
func CapturedRoot(capture ExecutionCapture, rootID string) (Installation, *exit.Error) {
	var root Installation
	for _, installation := range capture.Installations {
		if installation.ID == rootID {
			root = installation
		}
	}
	if root.ID == "" {
		return Installation{}, exit.New(exit.Conflict, "captured root installation is absent")
	}
	if len(capture.Installations) == 1 {
		return root, nil
	}
	rootHasWheel := false
	var source *File
	for i := range root.Files {
		file := &root.Files[i]
		rootHasWheel = rootHasWheel || file.Kind == "project"
		if file.Kind == "source" {
			source = file
		}
	}
	if !rootHasWheel && source != nil {
		// Earlier source captures already contain the relocated local dependency
		// trees and frozen lock. Replay that immutable archive through uv sync;
		// mixing a new wheel closure into that lock would change its installation.
		root.Files = []File{*source}
		root.DependencyRequirements = nil
		return root, nil
	}
	root.Callees = map[string]string{}
	wheels := map[string]File{}
	versions := map[string]string{}
	projects := map[string]bool{}
	// Start with the retained callable wheels. The caller's already-selected wheel
	// overrides one below when present, so an independent child lock cannot repin it.
	for _, installation := range capture.Installations {
		for name, pkg := range installation.Callees {
			if previous := root.Callees[name]; previous != "" && previous != pkg {
				return Installation{}, exit.New(exit.Conflict, "captured dependency has conflicting package identities: %s", name)
			}
			root.Callees[name] = pkg
		}
		for _, file := range installation.Files {
			if file.Kind != "project" {
				continue
			}
			fact, problem := wheel.InspectIdentity(file.Path)
			if problem != nil {
				return Installation{}, problem
			}
			if !wheel.SameVersion(fact.Version, installation.Release) || fact.Distribution != packageDistribution(installation.Package) {
				return Installation{}, exit.New(exit.Conflict, "captured project wheel differs from its installation")
			}
			if previous := wheels[fact.Distribution]; previous.Path != "" && previous.Digest != file.Digest {
				return Installation{}, exit.New(exit.Conflict, "captured callable selects two different executables: %s", fact.Distribution)
			}
			wheels[fact.Distribution], versions[fact.Distribution], projects[fact.Distribution] = file, fact.Version, true
			if root.Callees[fact.Distribution] == "" {
				root.Callees[fact.Distribution] = installation.Package
			}
		}
	}
	if !projects[packageDistribution(root.Package)] {
		return Installation{}, exit.New(exit.Conflict, "multi-package capture has no retained root wheel; capture its source at intake")
	}
	for _, file := range root.Files {
		if file.Kind == "source" || file.Kind == "project" {
			continue
		}
		fact, problem := wheel.InspectIdentity(file.Path)
		if problem != nil {
			return Installation{}, problem
		}
		if selected := versions[fact.Distribution]; selected != "" && !wheel.SameVersion(selected, fact.Version) {
			return Installation{}, exit.New(exit.Conflict, "captured environment selects conflicting versions: %s", fact.Distribution)
		}
		wheels[fact.Distribution] = file
	}
	// A supplied wheel and a registry URL for the same distribution are competing
	// sources to uv. Keep the captured caller's selected wheel, with its version checked above.
	var requirements []string
	for _, line := range strings.Split(string(root.DependencyRequirements), "\n") {
		fields := strings.Fields(line)
		if len(fields) > 0 && wheels[fields[0]].Path == "" {
			requirements = append(requirements, line)
		}
	}
	root.DependencyRequirements = nil
	if len(requirements) > 0 {
		root.DependencyRequirements = []byte(strings.Join(requirements, "\n") + "\n")
	}
	root.Files, root.SourceArchive = nil, ""
	for _, file := range wheels {
		root.Files = append(root.Files, file)
	}
	sort.Slice(root.Files, func(i, j int) bool { return root.Files[i].Filename < root.Files[j].Filename })
	return root, nil
}

func packageDistribution(pkg string) string {
	if _, name, ok := strings.Cut(pkg, "/"); ok {
		return name
	}
	return pkg
}
