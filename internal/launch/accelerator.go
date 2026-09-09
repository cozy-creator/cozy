package launch

import "strings"

// AcceleratorRequired derives the machine class from the package's immutable
// dependency facts. Model parameters describe inputs, not where their consumer
// computes: a TensorFS/NumPy transform can process a model entirely on CPU.
//
// These distributions are the accelerator half of the base-owned Python stack.
// Requiring one selects the GPU image; an ordinary package requirement does not.
func AcceleratorRequired(requirements []string) bool {
	for _, requirement := range requirements {
		switch requirementName(requirement) {
		case "torch", "torchaudio", "torchvision", "triton":
			return true
		}
	}
	return false
}

func requirementName(requirement string) string {
	name, _ := requirementParts(requirement)
	return name
}

func requirementParts(requirement string) (string, string) {
	requirement = strings.TrimSpace(requirement)
	end := 0
	for end < len(requirement) {
		c := requirement[end]
		if !('a' <= c && c <= 'z') && !('A' <= c && c <= 'Z') &&
			!('0' <= c && c <= '9') && c != '-' && c != '_' && c != '.' {
			break
		}
		end++
	}
	name := strings.ToLower(requirement[:end])
	return strings.Join(strings.FieldsFunc(name, func(r rune) bool { return r == '-' || r == '_' || r == '.' }), "-"), strings.TrimSpace(requirement[end:])
}
