package launch

import (
	"regexp"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

var digestShapedRelease = regexp.MustCompile(`^[0-9a-f]{64}$`)

// validateDefault admits a slot's authored default once, at descriptor admission (xs-042 C1):
// `[org/]model@release`, an explicit release and never a digest or a lane.
func validateDefault(slot Slot) *exit.Error {
	if slot.Default == "" {
		return nil
	}
	spelled := slot.Default
	if slot.RelativeDefault() {
		spelled = "owner/" + spelled
	}
	_, release, lane, manifest, problem := hub.ParseModelRef(spelled)
	if problem != nil || lane != "" || manifest != "" || strings.TrimSpace(slot.Default) != slot.Default ||
		!hub.ValidModelLabel(release) || digestShapedRelease.MatchString(release) {
		return exit.Named(exit.Validation, "package_model_default_invalid",
			"%s default %q must be [org/]model@release", slot.Path, slot.Default)
	}
	return nil
}

// RelativeDefault reports whether the slot's authored default names a model of the
// package's own account instead of an explicit org.
func (s Slot) RelativeDefault() bool {
	name, _, _ := strings.Cut(s.Default, "@")
	return s.Default != "" && !strings.Contains(name, "/")
}

// DefaultRef is the slot's authored default, `org/model@release`, an org-relative model
// qualified by owner, the account that owns the package. It is "" when the slot has no
// default, or when the default is org-relative and the owner is unknown.
func (s Slot) DefaultRef(owner string) string {
	if !s.RelativeDefault() {
		return s.Default
	}
	if owner == "" {
		return ""
	}
	return owner + "/" + s.Default
}

// Bits is the weight precision the machine's rules give the slot when a run names none,
// before any memory budget (xs-042 §2.2): 16 for a slot that allows no less, 4 for a decode
// slot that allows 4, else 8.
func (s Slot) Bits() int {
	switch {
	case s.MinBits == 16:
		return 16
	case s.MinBits == 4 && s.Phase == "decode":
		return 4
	}
	return 8
}
