package launch

import (
	"regexp"
	"strings"

	"github.com/cozy-creator/cozy/internal/packagepublish"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"

	pep440 "github.com/aquasecurity/go-pep440-version"
)

// The PRE-SPEND compatibility filter (th-075 leaves the verdict to the pod; this only
// declines to pay for a machine that cannot possibly run the release).
//
// Tensorhub publishes one profile label per rental SKU — the base image's small public
// matching vocabulary, four coordinates, already on GET /v1/rental-skus. A release's
// stored Requirements and RequiresPython are equally public. Where those two disagree on
// a coordinate the label actually names, renting spends an hour's money to reach a pod
// refusal that was legible before the ask.
//
// It is a FILTER, never a selector: the SKU still chooses the machine, and every arm here
// fails open. An unreadable label, an unparsable specifier, a marker this does not
// evaluate — all mean "cannot decide", which means "do not refuse". Ordinary-dependency
// mismatches (a pillow floor, a numpy cap) are deliberately not preflighted at all: the
// worker compares the package against the image it actually booted and refuses typed.

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
func BaseMismatch(profile BaseProfile, requirements []string, requiresPython string) string {
	if mismatch := torchMismatch(profile, requirements); mismatch != "" {
		return mismatch
	}
	return pythonMismatch(profile, requiresPython)
}

func torchMismatch(profile BaseProfile, requirements []string) string {
	for _, requirement := range requirements {
		name := requirementName(requirement)
		switch name {
		case "torch", "torchaudio", "torchvision", "triton":
		default:
			continue
		}
		if profile.TorchFree() {
			return "the package needs " + name + " and this base carries no Torch"
		}
		// Only torch's own release is a profile coordinate. The rest of the family
		// version independently, so their specifiers are the pod's business.
		if name != "torch" {
			continue
		}
		specifiers, ok := requirementSpecifiers(requirement)
		if !ok {
			continue
		}
		carried, err := pep440.Parse(profile.TorchRelease)
		if err != nil || specifiers.Check(carried) {
			continue
		}
		return "the package requires " + strings.TrimSpace(requirement) +
			" and this base carries torch " + profile.TorchRelease
	}
	return ""
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

// requirementSpecifiers reads the version specifier off a PEP 508 requirement. A
// requirement carrying extras, an environment marker, or a direct URL is reported
// undecidable rather than guessed at.
func requirementSpecifiers(requirement string) (pep440.Specifiers, bool) {
	_, rest := requirementParts(requirement)
	if rest == "" || strings.ContainsAny(rest, "[;@") {
		return pep440.Specifiers{}, false
	}
	specifiers, err := pep440.NewSpecifiers(rest)
	if err != nil {
		return pep440.Specifiers{}, false
	}
	return specifiers, true
}

// InventoryMismatch checks only protected base distributions. Incidental image
// libraries may be replaced by captured wheels. Marked requirements stay undecided here.
func InventoryMismatch(inventory *pb.ImageInventory, requirements []string, requiresPython string) string {
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
	installed := map[string]string{}
	for _, row := range inventory.Distributions {
		installed[requirementName(row.Distribution)] = row.Version
	}
	for _, requirement := range requirements {
		name, tail := requirementParts(requirement)
		if !packagepublish.ImageOwnedDistribution(name) {
			continue
		}
		if strings.HasPrefix(tail, "[") {
			if end := strings.Index(tail, "]"); end >= 0 {
				tail = strings.TrimSpace(tail[end+1:])
			}
		}
		// Markers and direct references remain the worker's exact resolver's decision.
		if strings.ContainsAny(tail, ";@") {
			continue
		}
		carried, present := installed[name]
		if !present {
			return "the rental image does not provide required base distribution " + name
		}
		bounds, err := pep440.NewSpecifiers(tail)
		version, versionErr := pep440.Parse(carried)
		if versionErr != nil {
			return "the rental image reports an invalid version for " + name
		}
		if err == nil && !bounds.Check(version) {
			return name + " " + carried + " does not satisfy " + strings.TrimSpace(requirement)
		}
	}
	return ""
}
