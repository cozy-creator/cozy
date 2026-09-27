package launch

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// ConversionSlot is the primary input of a conversion job: the first declared model input
// of a job with at least one weights output. Such a job reads `cozy run <job> <input>
// [<org/model>]`: the input is any model reference that slot accepts, the second positional
// is the checkpoint destination `--upload-to` also names, and any further model input is
// bound with `model.<param>=`.
func ConversionSlot(ep *Entrypoint) *Slot {
	if len(ep.Models) == 0 || len(ep.WeightsOutputs) == 0 {
		return nil
	}
	return &ep.Models[0]
}

// ConversionTerms rewrites a conversion job's bare positionals into the ordinary
// `model.<param>=<ref>` term and returns the destination, if one was given.
func ConversionTerms(target string, ep *Entrypoint, terms []string) ([]string, string, *exit.Error) {
	slot := ConversionSlot(ep)
	if slot == nil {
		return terms, "", nil
	}
	out := make([]string, 0, len(terms))
	var positional []string
	for _, term := range terms {
		if strings.Contains(term, "=") || strings.HasPrefix(term, "kernel.") {
			out = append(out, term)
			continue
		}
		positional = append(positional, term)
	}
	if len(positional) > 2 {
		return nil, "", exit.Usagef("%s takes <%s> and an optional <org/model> destination; got %d positional values",
			ep.Name, slot.Param, len(positional)).WithRemedy("%s", UsageLine(target, ep))
	}
	if len(positional) > 0 {
		out = append(out, "model."+slot.Param+"="+positional[0])
	}
	destination := ""
	if len(positional) == 2 {
		destination = positional[1]
	}
	return out, destination, nil
}
