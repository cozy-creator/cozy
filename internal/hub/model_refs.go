package hub

import (
	"regexp"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

var modelLabelPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+!-]{0,63}$`)

func ValidModelLabel(value string) bool { return modelLabelPattern.MatchString(value) }

// ParseModelRef reads the ONE model-ref grammar (cl-109):
//
//	org/model[@release[/lane]][#lane|#sha256:<hex>]
//
// It serves the `model.<param>=` run key and `package bind` alike; a lane narrows to
// one encoding and a manifest to one exact release artifact.
func ParseModelRef(raw string) (model, release, lane, manifest string, problem *exit.Error) {
	spec := strings.TrimSpace(raw)
	if strings.Count(spec, "#") > 1 {
		return "", "", "", "", exit.Usagef("%q carries more than one manifest", raw)
	}
	rest, digest, hasManifest := strings.Cut(spec, "#")
	if hasManifest {
		if digest == "" {
			return "", "", "", "", exit.Usagef("%q carries an empty manifest", raw)
		}
		if strings.HasPrefix(digest, "sha256:") {
			if _, err := canonical.Raw(digest); err != nil {
				return "", "", "", "", exit.Usagef("%q is not a sha256 model manifest", digest)
			}
			manifest = digest
		} else if ValidModelLabel(digest) {
			lane = digest
		} else {
			return "", "", "", "", exit.Usagef("%q is not a model lane", digest)
		}
	}
	if strings.Count(rest, "@") > 1 {
		return "", "", "", "", exit.Usagef("%q carries more than one release", rest)
	}
	model, versioned, pinned := strings.Cut(rest, "@")
	if _, e := ParseRef(model); e != nil {
		return "", "", "", "", e
	}
	if pinned {
		var sliced bool
		var explicitLane string
		release, explicitLane, sliced = strings.Cut(versioned, "/")
		if sliced {
			if lane != "" && lane != explicitLane {
				return "", "", "", "", exit.Usagef("%q selects conflicting lanes", raw)
			}
			lane = explicitLane
		}
		if release == "" || sliced && lane == "" || strings.ContainsAny(lane, " \t") {
			return "", "", "", "", exit.Usagef("%q is not org/model[@release[/lane]][#lane|#sha256:<hex>]", raw)
		}
	}
	return model, release, lane, manifest, nil
}
