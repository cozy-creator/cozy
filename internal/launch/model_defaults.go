package launch

import (
	"regexp"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

var authoredLanePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+!-]{0,127}$`)
var digestShapedRelease = regexp.MustCompile(`^[0-9a-f]{64}$`)

// relativeOwner stands in for the package's own account while an org-relative lane is
// parsed; it never leaves this file.
const relativeOwner = "owner"

// defaultModelBinding validates and lowers immutable metadata once at descriptor
// admission. Resolution reuses this value; it never invents a persisted Hub row. An
// org-relative lane (`model@release/lane`) lowers to a model with no org, which
// Slot.Default qualifies with the package's owner.
func defaultModelBinding(slot Slot) (*hub.PackageBindingRow, *exit.Error) {
	if len(slot.DefaultLadder) == 0 {
		return nil, nil
	}
	row := &hub.PackageBindingRow{Slot: slot.Path}
	for _, rung := range slot.DefaultLadder {
		if rung.GPUs > 1 && !slot.sequenceParallelDegrees()[rung.GPUs] {
			return nil, exit.Named(exit.Validation, "package_model_default_invalid", "%s default count %d is not a declared sequence-parallel degree", slot.Path, rung.GPUs)
		}
		spelled := rung.Lane
		name, _, _ := strings.Cut(spelled, "@")
		relative := !strings.Contains(name, "/")
		if relative {
			spelled = relativeOwner + "/" + spelled
		}
		model, release, lane, manifest, problem := hub.ParseModelRef(spelled)
		if problem != nil || manifest != "" || strings.TrimSpace(rung.Lane) != rung.Lane || !hub.ValidModelLabel(release) || digestShapedRelease.MatchString(release) || !authoredLanePattern.MatchString(lane) {
			return nil, exit.Named(exit.Validation, "package_model_default_invalid", "%s default lane %q must be [org/]model@release/lane", slot.Path, rung.Lane)
		}
		if relative {
			model = strings.TrimPrefix(model, relativeOwner+"/")
		}
		if row.Model != "" && (row.Model != model || row.Release != release) {
			return nil, exit.Named(exit.Validation, "package_model_default_invalid", "%s default ladder must use the same model and release in every rung", slot.Path)
		}
		row.Model, row.Release = model, release
		row.Ladder = append(row.Ladder, hub.BindingRung{GPU: rung.GPU, GPUs: rung.GPUs, Lane: lane})
	}
	if problem := hub.ValidateLadder(row.Ladder); problem != nil {
		return nil, exit.Named(exit.Validation, "package_model_default_invalid", "%s default ladder: %s", slot.Path, problem.Message)
	}
	return row, nil
}

// RelativeDefault reports whether the slot's authored default names a model of the
// package's own account instead of an explicit org.
func (s Slot) RelativeDefault() bool {
	return s.DefaultBinding != nil && !strings.Contains(s.DefaultBinding.Model, "/")
}

// Default is the slot's authored default with an org-relative model resolved to owner,
// the account that owns the package. It is nil when the slot has no default, or when the
// default is org-relative and the owner is unknown.
func (s Slot) Default(owner string) *hub.PackageBindingRow {
	if s.DefaultBinding == nil || s.RelativeDefault() && owner == "" {
		return nil
	}
	row := *s.DefaultBinding
	if s.RelativeDefault() {
		row.Model = owner + "/" + row.Model
	}
	return &row
}
