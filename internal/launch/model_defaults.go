package launch

import (
	"regexp"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

var authoredLanePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+!-]{0,127}$`)
var digestShapedRelease = regexp.MustCompile(`^[0-9a-f]{64}$`)

// defaultModelBinding validates and lowers immutable metadata once at descriptor
// admission. Resolution reuses this value; it never invents a persisted Hub row.
func defaultModelBinding(slot Slot) (*hub.PackageBindingRow, *exit.Error) {
	if len(slot.DefaultLadder) == 0 {
		return nil, nil
	}
	row := &hub.PackageBindingRow{Slot: slot.Path}
	for _, rung := range slot.DefaultLadder {
		model, release, lane, manifest, problem := hub.ParseModelRef(rung.Lane)
		if problem != nil || manifest != "" || strings.TrimSpace(rung.Lane) != rung.Lane || !hub.ValidModelLabel(release) || digestShapedRelease.MatchString(release) || !authoredLanePattern.MatchString(lane) {
			return nil, exit.Named(exit.Validation, "package_model_default_invalid", "%s default lane %q must be org/model@release/lane", slot.Path, rung.Lane)
		}
		if row.Model != "" && (row.Model != model || row.Release != release) {
			return nil, exit.Named(exit.Validation, "package_model_default_invalid", "%s default ladder must use the same model and release in every rung", slot.Path)
		}
		row.Model, row.Release = model, release
		row.Ladder = append(row.Ladder, hub.BindingRung{GPU: rung.GPU, Lane: lane})
	}
	if problem := hub.ValidateLadder(row.Ladder); problem != nil {
		return nil, exit.Named(exit.Validation, "package_model_default_invalid", "%s default ladder: %s", slot.Path, problem.Message)
	}
	return row, nil
}
