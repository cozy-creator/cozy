package launch

import (
	"strings"

	pep440 "github.com/aquasecurity/go-pep440-version"
)

type RuntimeRequirementConflict struct {
	Distribution string `json:"distribution"`
	Installed    string `json:"installed"`
	Required     string `json:"required"`
}

// RuntimeRequirementMismatch compares the actual updatable worker pair, which
// is deliberately absent from the immutable heavy image inventory. Callers
// evaluate PEP 508 markers for the target interpreter before this check.
func RuntimeRequirementMismatch(requirements []string, runtime, tensorfs string) *RuntimeRequirementConflict {
	for _, raw := range requirements {
		name, tail := requirementParts(raw)
		if name != "cozy-runtime" && name != "tensorfs" { //cozy:allow distribution metadata, not a binary invocation
			continue
		}
		if strings.HasPrefix(tail, "[") {
			if end := strings.Index(tail, "]"); end >= 0 {
				tail = strings.TrimSpace(tail[end+1:])
			}
		}
		if strings.ContainsAny(tail, ";@") {
			continue
		}
		installed := runtime
		if name == "tensorfs" {
			installed = tensorfs
		}
		version, err := pep440.Parse(installed)
		bounds, boundsErr := pep440.NewSpecifiers(tail)
		if err != nil || boundsErr == nil && !bounds.Check(version) {
			return &RuntimeRequirementConflict{Distribution: name, Installed: installed, Required: raw}
		}
	}
	return nil
}
