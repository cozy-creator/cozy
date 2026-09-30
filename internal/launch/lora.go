package launch

import (
	"math"
	"regexp"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// LoRAOverride is one ordered CLI selection. Resolution pins Ref before submission.
type LoRAOverride struct{ Slot, Component, Ref, Scale string }

var decimalScale = regexp.MustCompile(`^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$`)
var capturedModelPath = regexp.MustCompile(`^(?:[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*/)?[A-Za-z_][A-Za-z0-9_]*\.models\.[A-Za-z_][A-Za-z0-9_]*$`)

// CapturedModelSlot recognizes an exact callable slot selector. Runtime validates
// that the named callable and slot belong to the frozen execution graph.
func CapturedModelSlot(selector string) (pkg, path string, ok bool) {
	if !capturedModelPath.MatchString(selector) {
		return "", "", false
	}
	index := strings.LastIndex(selector, "/")
	if index < 0 {
		return "", selector, true
	}
	return selector[:index], selector[index+1:], true
}

func CanonicalLoRAScale(raw string) (string, *exit.Error) {
	if !decimalScale.MatchString(raw) {
		return "", exit.Usagef("LoRA strength %q must be a finite decimal number", raw)
	}
	value, err := strconv.ParseFloat(raw, 64)
	if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
		return "", exit.Usagef("LoRA strength %q must be finite", raw)
	}
	if value == 0 {
		return "0", nil
	}
	return strconv.FormatFloat(value, 'g', -1, 64), nil
}

func ParseLoRAs(ep *Entrypoint, values []string) ([]LoRAOverride, *exit.Error) {
	out := make([]LoRAOverride, 0, len(values))
	for _, raw := range values {
		target, ref, ok := strings.Cut(raw, "=")
		slotName, component, named := strings.Cut(target, ":")
		if !ok || !named || component == "" || strings.ContainsAny(component, "=,: \t\n\r") {
			return nil, exit.Usagef("--lora expects model-parameter:component=reference[,strength]")
		}
		slot, problem := modelOverrideSlot(ep, slotName)
		if problem != nil {
			return nil, problem
		}
		reference, scale, provided := strings.Cut(ref, ",")
		if reference == "" {
			return nil, exit.Usagef("LoRA reference is empty")
		}
		if !provided {
			scale = "1"
		}
		scale, problem = CanonicalLoRAScale(scale)
		if problem != nil {
			return nil, problem
		}
		out = append(out, LoRAOverride{Slot: slot.Path, Component: component, Ref: reference, Scale: scale})
	}
	return out, nil
}
