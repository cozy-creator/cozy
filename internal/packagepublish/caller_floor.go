package packagepublish

import (
	"strings"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/wheel"
)

// ValidateCallerRuntime keeps the generated SDK's consumer requirement in the
// authored metadata and exact selected lock, before any publication is committed.
func ValidateCallerRuntime(p *Package, floor string) *exit.Error {
	minimum, err := pep440.Parse(floor)
	if err != nil {
		return exit.Internalf("invalid generated caller Runtime floor")
	}
	document, problem := readProjectDocument(p.Files["pyproject.toml"])
	if problem != nil {
		return problem
	}
	bounded := false
	for _, raw := range document.Project.Dependencies {
		requirement, problem := parseRequirement(raw)
		if problem != nil {
			return problem
		}
		if requirement.name != hostruntime.Distribution || !requirement.hasSpec {
			continue
		}
		for _, term := range strings.Split(requirement.specifier.String(), ",") {
			term = strings.TrimSpace(term)
			for _, operator := range []string{">=", ">", "~=", "=="} {
				value, matched := strings.CutPrefix(term, operator)
				if !matched {
					continue
				}
				if operator == "==" {
					if !strings.HasSuffix(value, ".*") {
						continue
					}
					value = strings.TrimSuffix(value, ".*")
				}
				version, parseError := pep440.Parse(value)
				if parseError == nil && !version.LessThan(minimum) {
					bounded = true
				}
				break
			}
		}
	}
	locked := false
	for _, row := range p.Registry {
		if row.Name == hostruntime.Distribution {
			version, err := pep440.Parse(row.Version)
			locked = err == nil && !version.LessThan(minimum)
		}
	}
	for _, dependency := range p.DependencyWheels {
		identity, problem := wheel.InspectIdentity(dependency.Path)
		if problem != nil {
			return problem
		}
		if identity.Distribution == hostruntime.Distribution {
			version, err := pep440.Parse(identity.Version)
			locked = err == nil && !version.LessThan(minimum)
		}
	}
	if !bounded || !locked {
		return exit.Named(exit.Structural, "package_publish.caller_runtime_floor",
			"generated Python callers require cozy-runtime %s or newer in both the declared dependency range and selected lock", floor).
			WithRemedy("set cozy-runtime>=%s in pyproject.toml and run uv lock --upgrade-package cozy-runtime before publishing", floor)
	}
	return nil
}
