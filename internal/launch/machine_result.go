package launch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// MachineResultDrift is how a machine's result differs from the captured result schema.
// The machine is an independently upgraded peer: a field it returns that the package does
// not declare is ignored, and a declared output it does not return, or returns unusably,
// fails alone while the rest of the result is read.
type MachineResultDrift struct {
	Ignored []string          // undeclared result field paths
	Failed  map[string]string // declared top-level output -> why it cannot be used
	// bare is a result declared as one type rather than a struct of outputs; its one
	// output is named "result".
	bare bool
}

// Warnings renders the drift as the one-line warnings a run records.
func (d MachineResultDrift) Warnings() []string {
	out := make([]string, 0, len(d.Ignored)+len(d.Failed))
	for _, path := range d.Ignored {
		out = append(out, fmt.Sprintf("the machine returned result field %q, which the package does not declare; ignored", path))
	}
	for _, name := range slices.Sorted(maps.Keys(d.Failed)) {
		out = append(out, fmt.Sprintf("output %q failed: %s", name, d.Failed[name]))
	}
	return out
}

// Usable answers whether the declared output holding this path was read.
func (d MachineResultDrift) Usable(path string) bool {
	top, _, _ := strings.Cut(path, ".")
	if d.bare {
		top = "result"
	}
	_, failed := d.Failed[top]
	return !failed
}

// ValidateMachineResult checks final result metadata against the captured schema. The
// worker's own schema digest and byte spelling are another Runtime version's facts, so
// the result is judged by the values this host consumes. Asset results carry their
// immutable facts, while the ordinary payload validator accepts only an input reference
// at those positions. Only an unreadable result refuses as a whole; every declared output
// is judged on its own.
func ValidateMachineResult(schema json.RawMessage, envelope *pb.ResultEnvelope) (MachineResultDrift, *exit.Error) {
	drift := MachineResultDrift{Failed: map[string]string{}}
	if envelope == nil || envelope.ResultBlob != nil || len(envelope.InlineResult) == 0 || len(envelope.InlineResult) > pb.MaxInlineControlBytes {
		return drift, exit.New(exit.Conflict, "machine result requires its bounded inline metadata")
	}
	var declared Struct
	if json.Unmarshal(schema, &declared) != nil {
		return drift, exit.New(exit.Conflict, "machine result schema is unreadable")
	}
	var value any
	decoder := json.NewDecoder(bytes.NewReader(envelope.InlineResult))
	decoder.UseNumber()
	if decoder.Decode(&value) != nil {
		return drift, exit.New(exit.Conflict, "machine result is unreadable")
	}
	var shape map[string]json.RawMessage
	if json.Unmarshal(schema, &shape) != nil || shape["fields"] == nil {
		// One declared type is one output.
		drift.bare = true
		var ignored []string
		if problem := validateRendered(schema, value, "result", &reading{ignored: &ignored}); problem != nil {
			drift.Failed["result"] = problem.Message
		}
		drift.Ignored = ignored
		return drift, nil
	}
	object, ok := value.(map[string]any)
	if !ok {
		return drift, exit.New(exit.Conflict, "machine result is not an object")
	}
	fields := map[string]Field{}
	for _, field := range declared.Fields {
		fields[field.Name] = field
	}
	for name := range object {
		if _, known := fields[name]; !known && name != declared.TagField {
			drift.Ignored = append(drift.Ignored, name)
			delete(object, name)
		}
	}
	for _, field := range declared.Fields {
		if _, present := object[field.Name]; !present && field.Wire == "required" {
			drift.Failed[field.Name] = "the machine did not return it"
		}
	}
	entrypoint := &Entrypoint{Result: declared}
	for _, path := range AssetPaths(declared) {
		if !drift.Usable(path) {
			continue
		}
		if reason := projectResultAsset(object, path, entrypoint); reason != "" {
			top, _, _ := strings.Cut(path, ".")
			drift.Failed[top] = reason
		}
	}
	for _, field := range declared.Fields {
		element, present := object[field.Name]
		if !present || !drift.Usable(field.Name) {
			continue
		}
		var ignored []string
		if problem := validateFieldReading(field, element, field.Name, &reading{ignored: &ignored}); problem != nil {
			drift.Failed[field.Name] = problem.Message
			continue
		}
		drift.Ignored = append(drift.Ignored, ignored...)
	}
	sort.Strings(drift.Ignored)
	return drift, nil
}

// projectResultAsset replaces one asset result's typed metadata with its reference, the
// form the payload validator reads, after checking it against the declared media
// contract. An absent position is left to the type check of its output.
func projectResultAsset(object map[string]any, path string, entrypoint *Entrypoint) string {
	parts := strings.Split(path, ".")
	parent := object
	for _, part := range parts[:len(parts)-1] {
		child, ok := parent[part].(map[string]any)
		if !ok {
			return ""
		}
		parent = child
	}
	last := parts[len(parts)-1]
	child, present := parent[last]
	if !present {
		return ""
	}
	asset, ok := child.(map[string]any)
	spec, declared := ResultAssetSpec(entrypoint, path)
	if !ok || !declared || asset["kind"] != spec.Kind {
		return "its asset result has no typed metadata"
	}
	ref, ok := asset["asset_ref"].(string)
	if !ok || asset["digest"] != ref {
		return "its asset result changed its content identity"
	}
	if _, err := canonical.Raw(ref); err != nil {
		return "its asset result digest is invalid"
	}
	size, ok := asset["size_bytes"].(json.Number)
	length, err := size.Int64()
	media, mediaOK := asset["media_type"].(string)
	if !ok || err != nil || length < 0 || spec.MaxBytes > 0 && length > spec.MaxBytes || spec.Kind != "tree" && (!mediaOK || !spec.AcceptsMediaType(media)) {
		return "its asset result exceeds its declared media contract"
	}
	parent[last] = ref
	return ""
}
