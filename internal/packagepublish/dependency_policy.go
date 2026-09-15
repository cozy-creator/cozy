package packagepublish

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"net/textproto"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/wheel"
)

var dependencyRelease = regexp.MustCompile(`(?i)^v?([0-9]+!)?([0-9]+(?:\.[0-9]+)*)(.*)$`)

// Authored requirements express compatibility. Lockfiles and captured wheel
// identities still select exact bytes, and third-party metadata is not rewritten.
func validateDependencyPolicy(raw, source string) *exit.Error {
	req, problem := parseRequirement(raw)
	if problem != nil || !req.hasSpec {
		return problem
	}
	for _, spec := range strings.Split(req.specifier.String(), ",") {
		spec = strings.TrimSpace(spec)
		operator := ""
		for _, candidate := range []string{"===", "~=", "==", "!=", ">=", "<=", ">", "<", "="} {
			if strings.HasPrefix(spec, candidate) {
				operator = candidate
				break
			}
		}
		version := strings.TrimSpace(strings.TrimPrefix(spec, operator))
		release := dependencyRelease.FindStringSubmatch(version)
		allowed := false
		if len(release) == 4 {
			parts := strings.Split(release[2], ".")
			switch operator {
			case ">=", ">":
				allowed = !strings.Contains(version, "||")
			case "==":
				allowed = release[3] == ".*" && len(parts) <= 2
			case "~=":
				// ~=2.46.4 means >=2.46.4,<2.47; a fourth component
				// would constrain patches instead of the major/minor series.
				allowed = len(parts) <= 3
			case "<":
				// <2.47.0 and <3.0.0 are ordinary minor/major boundaries.
				allowed = release[3] == ""
				for _, part := range parts[min(2, len(parts)):] {
					allowed = allowed && strings.Trim(part, "0") == ""
				}
			}
		}
		if !allowed {
			return exit.Named(exit.Validation, "project_dependency_version_policy",
				"%s dependency %q restricts dependency patches; exact versions and patch upper bounds are not allowed", source, raw).
				WithRemedy("use a minimum version and optional major or minor upper bound, for example %s; keep exact versions in uv.lock", dependencyRangeSuggestion(req.name, release))
		}
	}
	return nil
}

func dependencyRangeSuggestion(name string, release []string) string {
	if len(release) != 4 {
		return name + ">=2.46.4,<3"
	}
	parts := strings.Split(release[2], ".")
	major, err := strconv.ParseUint(parts[0], 10, 63)
	if err != nil {
		return name + ">=" + release[1] + release[2]
	}
	return fmt.Sprintf("%s>=%s%s,<%s%d", name, release[1], release[2], release[1], major+1)
}

func validateProjectDependencyPolicy(document projectMetadata) *exit.Error {
	for _, raw := range document.Project.Dependencies {
		if problem := validateDependencyPolicy(raw, "[project].dependencies"); problem != nil {
			return problem
		}
	}
	groups := make([]string, 0, len(document.Project.OptionalDependencies))
	for name := range document.Project.OptionalDependencies {
		groups = append(groups, name)
	}
	slices.Sort(groups)
	for _, name := range groups {
		for _, raw := range document.Project.OptionalDependencies[name] {
			if problem := validateDependencyPolicy(raw, "[project.optional-dependencies]."+name); problem != nil {
				return problem
			}
		}
	}
	return nil
}

// Check the backend's authored metadata before a private capture derives its
// exact dependency closure. Dynamic requirements cannot bypass the source check.
func validateProjectWheelDependencies(path string) *exit.Error {
	raw, problem := wheel.Metadata(path)
	if problem != nil {
		return problem
	}
	headers, err := textproto.NewReader(bufio.NewReader(bytes.NewReader(raw))).ReadMIMEHeader()
	if err != nil && err != io.EOF {
		return exit.Named(exit.Validation, "project_metadata_invalid", "project wheel METADATA header is malformed")
	}
	for _, raw := range headers.Values("Requires-Dist") {
		if problem := validateDependencyPolicy(raw, "project wheel Requires-Dist"); problem != nil {
			return problem
		}
	}
	return nil
}
