package launch

import (
	"regexp"
	"strings"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"

	pep440 "github.com/aquasecurity/go-pep440-version"
)

// Pre-spend admission checks interpreter/platform facts. Package distributions
// are resolved into private environments; an image's preinstalled versions are
// cache opportunities, not constraints on the package's selected closure.

// BaseProfile is the label read back into its four coordinates.
type BaseProfile struct {
	AcceleratorBuild string
	OSCPU            string
	PythonABI        string
	TorchRelease     string
}

// TorchFree answers whether this base carries no Torch at all.
func (p BaseProfile) TorchFree() bool { return p.TorchRelease == "none" }

var (
	torchProfile = regexp.MustCompile(`^torch([0-9]+\.[0-9]+\.[0-9]+)-([a-z0-9]+)-(cp[0-9]{3})-([a-z0-9-]+)$`)
	cpuProfile   = regexp.MustCompile(`^python([0-9]+)\.([0-9]+)-cpu-([a-z0-9-]+)$`)
)

// ParseBaseProfile reads a Tensorhub profile label. Both spellings are the hub's own:
// `torch2.13.0-cu130-cp312-linux-x86`, and `python3.12-cpu-linux-x86` for a torch-free
// base. An unrecognized label is not an error here — it is a coordinate this client
// cannot read, so nothing is refused on it.
func ParseBaseProfile(label string) (BaseProfile, bool) {
	if m := torchProfile.FindStringSubmatch(label); m != nil {
		return BaseProfile{TorchRelease: m[1], AcceleratorBuild: m[2], PythonABI: m[3], OSCPU: m[4]}, true
	}
	if m := cpuProfile.FindStringSubmatch(label); m != nil {
		abi := "cp" + m[1] + m[2]
		return BaseProfile{TorchRelease: "none", AcceleratorBuild: "cpu", PythonABI: abi, OSCPU: m[3]}, true
	}
	return BaseProfile{}, false
}

// BaseMismatch names the coordinate this base cannot satisfy, or "" when it can or when
// the answer is undecidable from what the label and the release say.
func BaseMismatch(profile BaseProfile, _ []string, requiresPython string) string {
	return pythonMismatch(profile, requiresPython)
}

func pythonMismatch(profile BaseProfile, requiresPython string) string {
	requiresPython = strings.TrimSpace(requiresPython)
	if requiresPython == "" || len(profile.PythonABI) != 5 ||
		!strings.HasPrefix(profile.PythonABI, "cp") {
		return ""
	}
	// The same target the worker evaluates markers against: cp312 -> 3.12.0.
	target, err := pep440.Parse(profile.PythonABI[2:3] + "." + profile.PythonABI[3:] + ".0")
	if err != nil {
		return ""
	}
	specifiers, err := pep440.NewSpecifiers(requiresPython)
	if err != nil || specifiers.Check(target) {
		return ""
	}
	return "the package requires Python " + requiresPython +
		" and this base is " + profile.PythonABI
}

// InventoryMismatch validates the interpreter that will create the private
// environment. Distribution and worker-control compatibility are separate:
// package wheels may differ from the base; protocol negotiation guards control.
func InventoryMismatch(inventory *pb.ImageInventory, _ []string, requiresPython string, _ ...bool) string {
	if inventory == nil {
		return "the rental image inventory is absent"
	}
	if requiresPython != "" {
		current, err := pep440.Parse(inventory.Python)
		bounds, boundsErr := pep440.NewSpecifiers(requiresPython)
		if err != nil {
			return "the rental image reports an invalid Python version"
		}
		if boundsErr == nil && !bounds.Check(current) {
			return "Python " + inventory.Python + " does not satisfy " + requiresPython
		}
	}
	return ""
}
