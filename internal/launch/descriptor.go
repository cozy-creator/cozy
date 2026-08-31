// Package launch turns an INSTALLED GENERATION into the facts needed to serve it: the
// package's verified surface, the artifact its binding selects, and the PackageSpec the
// orchestrator launches. It is what replaces cl-006's `--dev-package` document (cl-010).
//
// Nothing here re-derives a fact its owner already produced:
//
//   - THE SURFACE is a generation-private descriptor derived once by the release's own
//     Runtime at install. Reading it back costs microseconds; re-running `describe` per
//     invocation would import the package's module graph to learn a fact already frozen.
//     The recorded semantic digest is checked on every read.
//   - THE PLACEMENT FACTS come from the exact PlacementSet retained at install. Runtime owns
//     no local model-ref index, and cozy-creator never composes a TensorFS store path.
//   - THE SELECTION is `package.toml`'s `[bindings]` table — the author's declared
//     default, in the runtime's own grammar and vocabulary.
//
// For a wholly weightless package the installed runtime is the sole canonical plan writer:
// `bindings --json` reports exact ArtifactSubjects before spawn and
// `serve --weightless-package` privately stages the same bytes. Cozy consumes the
// identities as record-owner intent and never reconstructs the documents. The older local
// pinned-binding writer below remains only for the modeled path that still carries it.
package launch

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// DescriptorFile is Runtime's derived document inside an immutable generation. It is
// never committed in package source.
const DescriptorFile = "descriptor.json"
const descriptorFormat = "cozy.package.descriptor/1"

// PackageDescriptor is the closed PackageDescriptor/1 this host reads. Unknown fields refuse;
// Raw is normalized canonical JSON for control-plane transport and semantic identity.
type PackageDescriptor struct {
	Format      string            `json:"format"`
	Application string            `json:"application"`
	Entrypoints []Entrypoint      `json:"entrypoints"`
	Jobs        []Entrypoint      `json:"jobs"`
	Productions []ModelProduction `json:"model_productions"`
	Digest      string            `json:"-"`
	Raw         json.RawMessage   `json:"-"`
}

// Entrypoint is one callable surface: its request schema, its declared model slots, and
// its result shape.
type Entrypoint struct {
	Name         string `json:"name"`
	Kind         string `json:"-"`
	DescriptorID string `json:"-"`
	Models       []Slot `json:"models"`
	Request      Struct `json:"request"`
	Result       Struct `json:"result"`
	Publishes    bool   `json:"publishes"`
	// ArtifactOutputs is the job's explicit ArtifactSink slot set. It is separate from
	// result asset fields because worker-protocol rev5 OutputBinding has no kind.
	ArtifactOutputs []ArtifactOutput     `json:"artifact_outputs"`
	Resources       ResourceRequirements `json:"resources"`
}

type ArtifactOutput struct {
	OutputID string `json:"output_id"`
	MimeType string `json:"mime_type"`
	MaxBytes uint64 `json:"max_bytes"`
}

// ModelProduction is one bounded, static graph authored by a package. Creator
// schedules it; Runtime never executes the declaration itself.
type ModelProduction struct {
	Name    string                  `json:"name"`
	Sources map[string]string       `json:"sources"`
	Steps   []ModelProductionStep   `json:"steps"`
	Outputs []ModelProductionOutput `json:"outputs"`
}

type ModelProductionStep struct {
	Name      string               `json:"name"`
	Callable  string               `json:"callable"`
	Models    map[string]string    `json:"models"`
	Outputs   []string             `json:"outputs"`
	Resources ResourceRequirements `json:"resources"`
}

type ResourceRequirements struct {
	GPUCount  int64  `json:"gpu_count"`
	Placement string `json:"placement"`
	Requires  string `json:"requires"`
}

type ModelProductionOutput struct {
	Name             string                  `json:"name"`
	Source           string                  `json:"source"`
	LaneKey          string                  `json:"lane_key"`
	RequiredContract ModelProductionContract `json:"required_contract"`
}

type ModelProductionContract struct {
	TopologyDigest string   `json:"topology_digest"`
	Encodings      []string `json:"encodings"`
}

// Slot is one declared model binding path — capability, never selection.
type Slot struct {
	Class        string              `json:"class"`
	Path         string              `json:"path"`
	Param        string              `json:"-"`
	Stamps       map[string]string   `json:"stamps"`
	ComponentUse map[string][]string `json:"component_use"`
}

// Struct is a rendered msgspec struct.
type Struct struct {
	Fields   []Field         `json:"fields"`
	TagField string          `json:"tag_field"`
	Tag      json.RawMessage `json:"tag"`
}

// Field is one declared field and its rendered type. `Type` is a string for a scalar
// (`int`, `str`) and an object for an asset (`{"asset":"image"}`), a list
// (`{"list":"float"}`) or a nested struct (`{"struct":…,"fields":[…]}`).
type Field struct {
	Name        string           `json:"name"`
	Type        json.RawMessage  `json:"type"`
	Wire        string           `json:"wire"`
	Constraints FieldConstraints `json:"constraints"`
	AssetBound  struct {
		MaxBytes   int64    `json:"max_bytes"`
		MediaTypes []string `json:"media_types"`
	} `json:"asset_bound"`
}

func (f *Field) UnmarshalJSON(data []byte) error {
	type plain Field
	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*f = Field(decoded)
	if f.Wire == "" {
		f.Wire = "required"
	}
	return nil
}

type FieldConstraints struct {
	MinLength *int64   `json:"min_length"`
	MaxLength *int64   `json:"max_length"`
	GT        *float64 `json:"gt"`
	GE        *float64 `json:"ge"`
	LE        *float64 `json:"le"`
	Unknown   []string `json:"-"`
}

func (c *FieldConstraints) UnmarshalJSON(data []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	for key := range raw {
		switch key {
		case "min_length", "max_length", "gt", "ge", "le":
		default:
			c.Unknown = append(c.Unknown, key)
		}
	}
	sort.Strings(c.Unknown)
	type plain FieldConstraints
	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	unknown := c.Unknown
	*c = FieldConstraints(decoded)
	c.Unknown = unknown
	return nil
}

func exactKeys(raw json.RawMessage, required, optional []string) (map[string]json.RawMessage, error) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		return nil, err
	}
	allowed := map[string]bool{}
	for _, key := range required {
		allowed[key] = true
		if _, ok := object[key]; !ok {
			return nil, fmt.Errorf("missing field %q", key)
		}
	}
	for _, key := range optional {
		allowed[key] = true
	}
	for key := range object {
		if !allowed[key] {
			return nil, fmt.Errorf("unknown field %q", key)
		}
	}
	return object, nil
}

func validateClosedDescriptor(data []byte) error {
	root, err := exactKeys(data,
		[]string{"application", "entrypoints", "format", "jobs", "model_productions"}, nil)
	if err != nil {
		return err
	}
	for collection, kind := range map[string]string{"entrypoints": "entrypoint", "jobs": "job"} {
		var rows []json.RawMessage
		if err := json.Unmarshal(root[collection], &rows); err != nil {
			return err
		}
		for _, row := range rows {
			required := []string{"name", "request", "result"}
			optional := []string{"models"}
			if kind == "job" {
				required = append(required, "publishes")
				optional = append(optional, "resources", "artifact_outputs")
			}
			callable, err := exactKeys(row, required, optional)
			if err != nil {
				return err
			}
			for _, name := range []string{"request", "result"} {
				if err := validateStructRaw(callable[name]); err != nil {
					return err
				}
			}
			if models := callable["models"]; models != nil {
				var slots []json.RawMessage
				if err := json.Unmarshal(models, &slots); err != nil {
					return err
				}
				for _, slot := range slots {
					if _, err := exactKeys(slot,
						[]string{"class", "component_use", "path", "stamps"}, nil); err != nil {
						return err
					}
				}
			}
			if resources := callable["resources"]; resources != nil {
				if err := validateResourceKeys(resources); err != nil {
					return err
				}
			}
			if outputs := callable["artifact_outputs"]; outputs != nil {
				var rows []json.RawMessage
				if err := json.Unmarshal(outputs, &rows); err != nil {
					return err
				}
				for _, output := range rows {
					if _, err := exactKeys(output,
						[]string{"max_bytes", "mime_type", "output_id"}, nil); err != nil {
						return err
					}
				}
			}
		}
	}
	return validateProductionKeys(root["model_productions"])
}

func validateResourceKeys(raw json.RawMessage) error {
	object, err := exactKeys(raw, nil, []string{"gpu_count", "placement", "requires"})
	if err != nil {
		return err
	}
	if len(object) == 0 {
		return fmt.Errorf("resources must not be empty")
	}
	if value := object["gpu_count"]; value != nil {
		var count int64
		if json.Unmarshal(value, &count) != nil || count < 1 {
			return fmt.Errorf("resources.gpu_count must be a positive integer")
		}
	}
	if value := object["placement"]; value != nil {
		var placement string
		if json.Unmarshal(value, &placement) != nil ||
			(placement != "single_node" && placement != "any") {
			return fmt.Errorf("resources.placement must be single_node or any")
		}
	}
	if value := object["requires"]; value != nil {
		var requires string
		if json.Unmarshal(value, &requires) != nil || strings.TrimSpace(requires) == "" {
			return fmt.Errorf("resources.requires must be a non-empty string")
		}
	}
	return nil
}

func validateProductionKeys(raw json.RawMessage) error {
	var productions []json.RawMessage
	if err := json.Unmarshal(raw, &productions); err != nil || len(productions) > 16 {
		return fmt.Errorf("model_productions must be an array of at most 16 entries")
	}
	for _, rawProduction := range productions {
		production, err := exactKeys(rawProduction,
			[]string{"name", "sources", "steps", "outputs"}, nil)
		if err != nil {
			return err
		}
		var steps []json.RawMessage
		if err := json.Unmarshal(production["steps"], &steps); err != nil {
			return err
		}
		for _, rawStep := range steps {
			step, err := exactKeys(rawStep,
				[]string{"callable", "models", "name", "outputs"}, []string{"resources"})
			if err != nil {
				return err
			}
			if step["resources"] != nil {
				if err := validateResourceKeys(step["resources"]); err != nil {
					return err
				}
			}
		}
		var outputs []json.RawMessage
		if err := json.Unmarshal(production["outputs"], &outputs); err != nil {
			return err
		}
		for _, rawOutput := range outputs {
			output, err := exactKeys(rawOutput,
				[]string{"lane_key", "name", "required_contract", "source"}, nil)
			if err != nil {
				return err
			}
			if _, err := exactKeys(output["required_contract"],
				[]string{"encodings", "topology_digest"}, nil); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateStructRaw(raw json.RawMessage) error {
	object, err := exactKeys(raw, []string{"fields"}, []string{"tag", "tag_field"})
	if err != nil {
		return err
	}
	tagged := object["tag_field"] != nil || object["tag"] != nil
	var tagField string
	if tagged {
		if object["tag_field"] == nil || object["tag"] == nil ||
			json.Unmarshal(object["tag_field"], &tagField) != nil || tagField == "" {
			return fmt.Errorf("tagged struct must carry one non-empty tag_field and tag")
		}
	}
	var fields []json.RawMessage
	if err := json.Unmarshal(object["fields"], &fields); err != nil {
		return err
	}
	for _, rawField := range fields {
		field, err := exactKeys(rawField, []string{"name", "type"},
			[]string{"asset_bound", "constraints", "wire"})
		if err != nil {
			return err
		}
		if tagged {
			var name string
			if json.Unmarshal(field["name"], &name) != nil || name == tagField {
				return fmt.Errorf("tagged struct repeats its synthetic discriminator field")
			}
		}
		if rawWire := field["wire"]; rawWire != nil {
			var wire string
			if json.Unmarshal(rawWire, &wire) != nil || (wire != "optional" && wire != "omissible") {
				return fmt.Errorf("wire must be absent for required fields or spell optional|omissible")
			}
		}
		if err := validateTypeRaw(field["type"]); err != nil {
			return err
		}
		if bound := field["asset_bound"]; bound != nil {
			if _, err := exactKeys(bound, nil, []string{"max_bytes", "media_types"}); err != nil {
				return err
			}
		}
		if constraints := field["constraints"]; constraints != nil {
			if _, err := exactKeys(constraints, nil,
				[]string{"ge", "gt", "le", "max_length", "min_length"}); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateTypeRaw(raw json.RawMessage) error {
	var scalar string
	if json.Unmarshal(raw, &scalar) == nil {
		switch scalar {
		case "bool", "float", "int", "null", "str":
			return nil
		}
		return fmt.Errorf("unsupported descriptor scalar %q", scalar)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		return err
	}
	switch {
	case object["asset"] != nil:
		_, err := exactKeys(raw, []string{"asset"}, nil)
		return err
	case object["input"] != nil:
		_, err := exactKeys(raw, []string{"input"}, nil)
		return err
	case object["literal"] != nil:
		_, err := exactKeys(raw, []string{"literal"}, nil)
		return err
	case object["list"] != nil:
		if _, err := exactKeys(raw, []string{"list"}, nil); err != nil {
			return err
		}
		return validateTypeRaw(object["list"])
	case object["union"] != nil:
		union, err := exactKeys(raw, []string{"union"}, []string{"tag_field"})
		if err != nil {
			return err
		}
		var branches []json.RawMessage
		if err := json.Unmarshal(object["union"], &branches); err != nil || len(branches) == 0 {
			return fmt.Errorf("empty descriptor union")
		}
		for _, branch := range branches {
			if union["tag_field"] != nil {
				member, err := exactKeys(branch, []string{"fields", "tag"}, nil)
				if err != nil {
					return err
				}
				member["tag_field"] = union["tag_field"]
				expanded, err := json.Marshal(member)
				if err != nil {
					return err
				}
				if err := validateStructRaw(expanded); err != nil {
					return err
				}
				continue
			}
			if err := validateTypeRaw(branch); err != nil {
				return err
			}
		}
		return nil
	case object["fields"] != nil:
		return validateStructRaw(raw)
	}
	return fmt.Errorf("unsupported descriptor type")
}

func validateEntrypoint(ep *Entrypoint) *exit.Error {
	if ep.Name == "" {
		return exit.New(exit.Validation, "descriptor carries an unnamed %s", ep.Kind)
	}
	for i := range ep.Models {
		slot := &ep.Models[i]
		prefix := ep.Name + ".models."
		param := strings.TrimPrefix(slot.Path, prefix)
		if slot.Class == "" || param == slot.Path || param == "" || strings.Contains(param, ".") {
			return exit.New(exit.Validation,
				"%s has invalid model path %q; it must be %s<parameter>", ep.Name, slot.Path, prefix)
		}
		slot.Param = param
	}
	for _, pair := range []struct {
		name string
		body Struct
	}{{"request", ep.Request}, {"result", ep.Result}} {
		for _, field := range pair.body.Fields {
			if field.Name == "" || (field.Wire != "required" && field.Wire != "optional" &&
				field.Wire != "omissible") {
				return exit.New(exit.Validation, "%s.%s has an invalid field", ep.Name, pair.name)
			}
			if len(field.Constraints.Unknown) > 0 {
				return exit.New(exit.Validation, "%s.%s uses unsupported constraints: %s",
					ep.Name, field.Name, strings.Join(field.Constraints.Unknown, ", "))
			}
		}
	}
	if ep.Kind != "job" && len(ep.ArtifactOutputs) > 0 {
		return exit.New(exit.Validation, "%s declares artifact outputs outside the job surface", ep.Name)
	}
	if ep.Resources.GPUCount < 0 ||
		(ep.Resources.Placement != "" && ep.Resources.Placement != "single_node" &&
			ep.Resources.Placement != "any") {
		return exit.New(exit.Validation, "%s has invalid resource requirements", ep.Name)
	}
	seenArtifacts := map[string]bool{}
	for _, output := range ep.ArtifactOutputs {
		if output.OutputID == "" || seenArtifacts[output.OutputID] || output.MaxBytes == 0 ||
			output.MaxBytes > (uint64(1)<<53)-1 || output.MimeType != orchestrator.ArtifactManifestMime {
			return exit.New(exit.Validation,
				"%s has an invalid artifact output %q: slots are unique snapshot MIME rows with a 1..2^53-1 byte cap",
				ep.Name, output.OutputID)
		}
		seenArtifacts[output.OutputID] = true
	}
	return nil
}

var (
	productionNamePattern   = regexp.MustCompile(`^[a-z][a-z0-9_-]{0,63}$`)
	callablePattern         = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,63}/[a-z0-9][a-z0-9._-]{0,63}/[a-z][a-z0-9_-]{0,63}$`)
	contractPattern         = regexp.MustCompile(`^[a-z0-9][a-z0-9._+-]{0,63}(?:/[a-z0-9][a-z0-9._+-]{0,63})*$`)
	productionDigestPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
)

func validateProductions(productions []ModelProduction) *exit.Error {
	if len(productions) > 16 {
		return exit.New(exit.Validation, "descriptor carries more than 16 model productions")
	}
	seen := map[string]bool{}
	for i := range productions {
		production := &productions[i]
		if !productionNamePattern.MatchString(production.Name) || seen[production.Name] {
			return exit.New(exit.Validation, "descriptor has invalid or duplicate model production %q", production.Name)
		}
		seen[production.Name] = true
		if problem := validateProduction(production); problem != nil {
			return problem
		}
	}
	return nil
}

func validateProduction(production *ModelProduction) *exit.Error {
	if len(production.Sources) < 1 || len(production.Sources) > 16 {
		return badProduction(production, "must declare 1 through 16 named source profiles")
	}
	for slot, profile := range production.Sources {
		if !productionNamePattern.MatchString(slot) || !contractPattern.MatchString(profile) {
			return badProduction(production, "has invalid source slot %q or profile %q", slot, profile)
		}
	}
	if len(production.Steps) < 1 || len(production.Steps) > 64 {
		return badProduction(production, "must declare 1 through 64 steps")
	}
	steps := map[string]ModelProductionStep{}
	declaredOutputs := map[string]bool{}
	for _, step := range production.Steps {
		if !productionNamePattern.MatchString(step.Name) || steps[step.Name].Name != "" ||
			!callablePattern.MatchString(step.Callable) {
			return badProduction(production, "has an invalid step name or callable")
		}
		if len(step.Models) < 1 || len(step.Models) > 16 || len(step.Outputs) < 1 ||
			len(step.Outputs) > 8 {
			return badProduction(production, "step %s exceeds its input/output bounds", step.Name)
		}
		for input, reference := range step.Models {
			if !productionNamePattern.MatchString(input) || reference == "" {
				return badProduction(production, "step %s has an invalid model edge", step.Name)
			}
		}
		for _, output := range step.Outputs {
			key := step.Name + "." + output
			if !productionNamePattern.MatchString(output) || declaredOutputs[key] {
				return badProduction(production, "step %s has an invalid or duplicate output", step.Name)
			}
			declaredOutputs[key] = true
		}
		steps[step.Name] = step
	}
	if len(production.Outputs) < 1 || len(production.Outputs) > 16 {
		return badProduction(production, "must declare 1 through 16 required outputs")
	}
	outputNames, lanes := map[string]bool{}, map[string]bool{}
	for _, output := range production.Outputs {
		if !productionNamePattern.MatchString(output.Name) || outputNames[output.Name] ||
			!productionNamePattern.MatchString(output.LaneKey) || lanes[output.LaneKey] ||
			!declaredOutputs[output.Source] {
			return badProduction(production, "has an invalid, duplicate, or unresolved required output")
		}
		outputNames[output.Name], lanes[output.LaneKey] = true, true
		contract := output.RequiredContract
		if !productionDigestPattern.MatchString(contract.TopologyDigest) ||
			!validProductionDigestList(contract.Encodings, 32) {
			return badProduction(production, "output %s has an invalid required contract", output.Name)
		}
	}
	_, problem := production.OrderedSteps()
	return problem
}

func validProductionDigestList(values []string, cap int) bool {
	if len(values) < 1 || len(values) > cap {
		return false
	}
	for i, value := range values {
		if !productionDigestPattern.MatchString(value) || i > 0 && values[i-1] >= value {
			return false
		}
	}
	return true
}

func badProduction(production *ModelProduction, format string, values ...any) *exit.Error {
	return exit.New(exit.Validation, "model production %s: %s", production.Name,
		fmt.Sprintf(format, values...))
}

// OrderedSteps returns the declaration's one deterministic topological order.
// Declaration order carries no control-flow meaning.
func (production *ModelProduction) OrderedSteps() ([]ModelProductionStep, *exit.Error) {
	byName := make(map[string]ModelProductionStep, len(production.Steps))
	dependencies := make(map[string]map[string]bool, len(production.Steps))
	for _, step := range production.Steps {
		byName[step.Name] = step
		dependencies[step.Name] = map[string]bool{}
	}
	for _, step := range production.Steps {
		for _, reference := range step.Models {
			if _, source := production.Sources[reference]; source {
				continue
			}
			owner, _, _ := strings.Cut(reference, ".")
			if dependency, ok := byName[owner]; !ok || !contains(dependency.Outputs,
				strings.TrimPrefix(reference, owner+".")) {
				return nil, badProduction(production, "step %s references unknown edge %q", step.Name, reference)
			}
			dependencies[step.Name][owner] = true
		}
	}
	ordered := make([]ModelProductionStep, 0, len(byName))
	placed := map[string]bool{}
	for len(ordered) < len(byName) {
		ready := make([]string, 0)
		for name := range byName {
			if placed[name] {
				continue
			}
			all := true
			for dependency := range dependencies[name] {
				all = all && placed[dependency]
			}
			if all {
				ready = append(ready, name)
			}
		}
		if len(ready) == 0 {
			return nil, badProduction(production, "contains a cycle")
		}
		sort.Strings(ready)
		for _, name := range ready {
			ordered = append(ordered, byName[name])
			placed[name] = true
		}
	}
	required := map[string]bool{}
	var retain func(string)
	retain = func(name string) {
		if required[name] {
			return
		}
		required[name] = true
		for dependency := range dependencies[name] {
			retain(dependency)
		}
	}
	for _, output := range production.Outputs {
		owner, _, _ := strings.Cut(output.Source, ".")
		retain(owner)
	}
	if len(required) != len(byName) {
		return nil, badProduction(production, "contains a disconnected step")
	}
	return ordered, nil
}

func contains(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}

// DescriptorPath is the one generation-private location for Runtime-derived bytes.
func DescriptorPath(generationDir string) string {
	return filepath.Join(generationDir, "documents", DescriptorFile)
}

// ReadDescriptor reads the private descriptor and joins it to the install record.
func ReadDescriptor(path, expectDigest string) (*PackageDescriptor, *exit.Error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, exit.Named(exit.Structural, "descriptor_absent",
			"this generation carries no private %s", DescriptorFile).
			WithRemedy("reinstall from the original source so its Runtime can derive the descriptor")
	}
	d, problem := DecodeDescriptor(data)
	if problem != nil {
		return nil, problem
	}
	if expectDigest != "" && d.Digest != expectDigest {
		return nil, exit.Named(exit.Conflict, "descriptor_stale",
			"the private descriptor content digests to %s and this install recorded %s", d.Digest, expectDigest).
			WithRemedy("the immutable generation is corrupt; reinstall it from its original source")
	}
	return d, nil
}

// DecodeDescriptor reads the one closed descriptor/1 grammar and derives its canonical
// semantic identity. Collection membership supplies callable kind; the document does not
// repeat it.
func DecodeDescriptor(data []byte) (*PackageDescriptor, *exit.Error) {
	if len(data) > canonical.DocMax {
		return nil, exit.New(exit.Validation, "%s exceeds the %d-byte cap", DescriptorFile, canonical.DocMax)
	}
	normalized, err := canonical.NormalizeJCS(data)
	if err != nil {
		return nil, exit.New(exit.Validation, "%s violates descriptor/1: %s", DescriptorFile, err)
	}
	if err := validateClosedDescriptor(normalized); err != nil {
		return nil, exit.New(exit.Validation, "%s violates descriptor/1: %s", DescriptorFile, err)
	}
	decoder := json.NewDecoder(bytes.NewReader(normalized))
	decoder.DisallowUnknownFields()
	var d PackageDescriptor
	if err := decoder.Decode(&d); err != nil {
		return nil, exit.New(exit.Validation, "%s is not a descriptor document: %s", DescriptorFile, err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return nil, exit.New(exit.Validation, "%s carries trailing JSON", DescriptorFile)
	}
	if d.Format != descriptorFormat || d.Application == "" {
		return nil, exit.New(exit.Validation, "%s format/application is invalid", DescriptorFile)
	}
	var raw struct {
		Jobs []json.RawMessage `json:"jobs"`
	}
	if err := json.Unmarshal(normalized, &raw); err != nil || len(raw.Jobs) != len(d.Jobs) {
		return nil, exit.New(exit.Validation, "%s carries invalid job rows", DescriptorFile)
	}
	for i := range d.Entrypoints {
		d.Entrypoints[i].Kind = "entrypoint"
		if problem := validateEntrypoint(&d.Entrypoints[i]); problem != nil {
			return nil, problem
		}
	}
	for i := range d.Jobs {
		d.Jobs[i].Kind = "job"
		digest := sha256.Sum256(append([]byte("cozy.runtime.job-descriptor\x00"), raw.Jobs[i]...))
		d.Jobs[i].DescriptorID, err = canonical.Spell(digest[:])
		if err != nil {
			return nil, exit.Internalf("cannot spell job descriptor id: %s", err)
		}
		if problem := validateEntrypoint(&d.Jobs[i]); problem != nil {
			return nil, problem
		}
	}
	if problem := validateProductions(d.Productions); problem != nil {
		return nil, problem
	}
	digest, err := canonical.Spell(canonical.Digest(normalized))
	if err != nil {
		return nil, exit.Internalf("cannot spell descriptor digest: %s", err)
	}
	d.Digest = digest
	d.Raw = normalized
	return &d, nil
}

// Production finds one reviewed static model-production graph by name.
func (d *PackageDescriptor) Production(name string) (*ModelProduction, *exit.Error) {
	for i := range d.Productions {
		if d.Productions[i].Name == name {
			return &d.Productions[i], nil
		}
	}
	available := make([]string, 0, len(d.Productions))
	for _, production := range d.Productions {
		available = append(available, production.Name)
	}
	sort.Strings(available)
	return nil, exit.New(exit.NotFound, "this release registers no model production %q", name).
		WithRemedy("it registers: %s", strings.Join(available, ", "))
}

// Function finds one entrypoint or job by name.
func (d *PackageDescriptor) Function(name string) (*Entrypoint, *exit.Error) {
	for i := range d.Entrypoints {
		if d.Entrypoints[i].Name == name {
			return &d.Entrypoints[i], nil
		}
	}
	for i := range d.Jobs {
		if d.Jobs[i].Name == name {
			return &d.Jobs[i], nil
		}
	}
	return nil, exit.New(exit.NotFound, "this release registers no function %q", name).
		WithRemedy("it registers: %s", strings.Join(d.Names(), ", "))
}

// Names is every callable this release registers.
func (d *PackageDescriptor) Names() []string {
	out := []string{}
	for _, e := range d.Entrypoints {
		out = append(out, e.Name)
	}
	for _, j := range d.Jobs {
		out = append(out, j.Name)
	}
	sort.Strings(out)
	return out
}

// AssetPaths is every asset-typed field path of a result struct, dotted for nesting.
// These ARE the output ids: the orchestrator grants one destination per asset field path,
// so `--out` is a consequence of the package's declared result rather than a convention
// the CLI and the package each have to remember. Same walk cr-016's `payload.asset_paths`
// makes over the same document.
func AssetPaths(s Struct) []string {
	return assetPaths(s, "")
}

func assetPaths(s Struct, prefix string) []string {
	out := []string{}
	for _, f := range s.Fields {
		name := prefix + f.Name
		kind, nested := typeOf(f.Type)
		switch kind {
		case "asset":
			out = append(out, name)
		case "struct":
			out = append(out, assetPaths(nested, name+".")...)
		}
	}
	return out
}

// typeOf classifies one rendered type: "scalar" (with the scalar's name in `scalar`),
// "literal", "asset", "list", or "struct" (with the nested struct).
func typeOf(raw json.RawMessage) (kind string, nested Struct) {
	if len(raw) == 0 {
		return "unknown", Struct{}
	}
	var scalar string
	if json.Unmarshal(raw, &scalar) == nil {
		return "scalar:" + scalar, Struct{}
	}
	var object map[string]json.RawMessage
	if json.Unmarshal(raw, &object) != nil {
		return "unknown", Struct{}
	}
	if _, ok := object["asset"]; ok {
		return "asset", Struct{}
	}
	if _, ok := object["literal"]; ok {
		return "literal", Struct{}
	}
	// A job's typed INPUT tree (cr-009). It is not an asset and not a scalar struct: its
	// wire value is a REF, and the grant is what turns that ref into a readable path.
	if kind, ok := object["input"]; ok && string(kind) == `"tree"` {
		return "tree", Struct{}
	}
	if _, ok := object["fields"]; ok {
		var s Struct
		if json.Unmarshal(raw, &s) == nil {
			return "struct", s
		}
	}
	if _, ok := object["list"]; ok {
		return "list", Struct{}
	}
	return "unknown", Struct{}
}

// TypeOfField is the rendered type of one request field, for the payload grammar.
func (e *Entrypoint) TypeOfField(name string) (json.RawMessage, bool) {
	for _, f := range e.Request.Fields {
		if f.Name == name {
			return f.Type, true
		}
	}
	return nil, false
}

// InputKind returns the compact descriptor kind for one request field.
func (e *Entrypoint) InputKind(name string) (string, bool) {
	for _, field := range e.Request.Fields {
		if field.Name == name {
			kind, _ := typeOf(field.Type)
			return kind, true
		}
	}
	return "", false
}

// RequestFields names every declared request field, in the author's declaration order.
func (e *Entrypoint) RequestFields() []string {
	out := make([]string, 0, len(e.Request.Fields))
	for _, f := range e.Request.Fields {
		out = append(out, f.Name)
	}
	return out
}
