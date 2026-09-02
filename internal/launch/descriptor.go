// Package launch turns an INSTALLED PACKAGE into the facts needed to serve it: the
// package's verified surface, the artifact its binding selects, and the PackageSpec the
// orchestrator launches. It is what replaces cl-006's `--dev-package` document (cl-010).
//
// Nothing here re-derives a fact its owner already produced:
//
//   - THE SURFACE is an install-private descriptor derived once by the release's own
//     Runtime at install. Reading it back costs microseconds; re-running `describe` per
//     invocation would import the package's module graph to learn a fact already frozen.
//     The recorded semantic digest is checked on every read.
//   - THE PLACEMENT FACTS come from the exact PlacementSet retained at install. Runtime owns
//     no local model-ref index, and cozy-creator never composes a TensorFS store path.
//   - THE SELECTION is `package.toml`'s `[bindings]` table — the author's declared
//     default, in the runtime's own grammar and vocabulary.
//
// For a wholly weightless package the installed runtime is the sole canonical plan writer:
// `bindings --json` reports exact WeightsSubjects before spawn and
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
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// DescriptorFile is Runtime's derived document inside an immutable install. It is
// never committed in package source.
const DescriptorFile = "descriptor.json"
const descriptorFormat = "cozy.package.descriptor/1"

// PackageDescriptor is the closed PackageDescriptor/1 this host reads. Unknown fields refuse;
// Raw is normalized canonical JSON for control-plane transport and semantic identity.
type PackageDescriptor struct {
	Format      string          `json:"format"`
	Application string          `json:"application"`
	Entrypoints []Entrypoint    `json:"entrypoints"`
	Jobs        []Entrypoint    `json:"jobs"`
	Digest      string          `json:"-"`
	Raw         json.RawMessage `json:"-"`
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
	// WeightsOutputs is the job's explicit WeightsSink slot set. It is separate from
	// result asset fields because worker-protocol rev5 OutputBinding has no kind.
	WeightsOutputs []WeightsOutput `json:"weights_outputs"`
}

type WeightsOutput struct {
	OutputID string `json:"output_id"`
	MimeType string `json:"mime_type"`
	MaxBytes uint64 `json:"max_bytes"`
}

// Slot is one declared model binding path — capability, never selection. Its members
// mirror cozy-runtime's `internal/descriptor.py` slot (cr-078a): `{class, component_use,
// path}` plus the class's one keyword `encoded_leaves` and an optional `sequence_parallel`
// document. `stamps` is a RETIRED member: every release published before the cut ships an
// empty map, which reads as nothing declared, and a value in it refuses by name below.
type Slot struct {
	Class            string              `json:"class"`
	Path             string              `json:"path"`
	Param            string              `json:"-"`
	ComponentUse     map[string][]string `json:"component_use"`
	EncodedLeaves    string              `json:"encoded_leaves,omitempty"`
	SequenceParallel json.RawMessage     `json:"sequence_parallel,omitempty"`
	Stamps           map[string]string   `json:"stamps,omitempty"`
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

// refuseRetiredSlotMembers is the reader-side half of cr-078a's hard cut, the same rule
// cozy-runtime's `_refuse_retired` keeps: an EMPTY `stamps` map is what every release
// published before the cut carries and reads as nothing declared; a stamp in it, or any
// `source_profile` (cr-077), is a member the grammar deleted and refuses by name.
func refuseRetiredSlotMembers(members map[string]json.RawMessage) error {
	if raw, ok := members["stamps"]; ok {
		var stamps map[string]json.RawMessage
		if err := json.Unmarshal(raw, &stamps); err != nil {
			return fmt.Errorf("stamps must be an object")
		}
		if len(stamps) != 0 {
			return fmt.Errorf("stamps are retired (model-code-fit D1); a semantic twin is a " +
				"component name or a config fact")
		}
	}
	if _, ok := members["source_profile"]; ok {
		return fmt.Errorf("source_profile is retired (cr-077); capability is the slot's class annotation")
	}
	return nil
}

func validateClosedDescriptor(data []byte) error {
	root, err := exactKeys(data,
		[]string{"application", "entrypoints", "format", "jobs"}, nil)
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
				optional = append(optional, "weights_outputs")
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
					members, err := exactKeys(slot,
						[]string{"class", "component_use", "path"},
						[]string{"encoded_leaves", "sequence_parallel", "stamps", "source_profile"})
					if err != nil {
						return err
					}
					if err := refuseRetiredSlotMembers(members); err != nil {
						return err
					}
					if leaves, ok := members["encoded_leaves"]; ok {
						var value string
						if json.Unmarshal(leaves, &value) != nil || (value != "refuse" && value != "accept") {
							return fmt.Errorf("encoded_leaves must be \"refuse\" or \"accept\"")
						}
					}
				}
			}
			if outputs := callable["weights_outputs"]; outputs != nil {
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
	if ep.Kind != "job" && len(ep.WeightsOutputs) > 0 {
		return exit.New(exit.Validation, "%s declares weights outputs outside the job surface", ep.Name)
	}
	seenWeights := map[string]bool{}
	for _, output := range ep.WeightsOutputs {
		if output.OutputID == "" || seenWeights[output.OutputID] || output.MaxBytes == 0 ||
			output.MaxBytes > (uint64(1)<<53)-1 || output.MimeType != orchestrator.WeightsManifestMime {
			return exit.New(exit.Validation,
				"%s has an invalid weights output %q: slots are unique snapshot MIME rows with a 1..2^53-1 byte cap",
				ep.Name, output.OutputID)
		}
		seenWeights[output.OutputID] = true
	}
	return nil
}

// DescriptorPath is the one install-private location for Runtime-derived bytes.
func DescriptorPath(installDir string) string {
	return filepath.Join(installDir, "documents", DescriptorFile)
}

// ReadDescriptor reads the private descriptor and joins it to the install record.
func ReadDescriptor(path, expectDigest string) (*PackageDescriptor, *exit.Error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, exit.Named(exit.Structural, "descriptor_absent",
			"this install carries no private %s", DescriptorFile).
			WithRemedy("reinstall from the original source so its Runtime can derive the descriptor")
	}
	d, problem := DecodeDescriptor(data)
	if problem != nil {
		return nil, problem
	}
	if expectDigest != "" && d.Digest != expectDigest {
		return nil, exit.Named(exit.Conflict, "descriptor_stale",
			"the private descriptor content digests to %s and this install recorded %s", d.Digest, expectDigest).
			WithRemedy("the immutable install is corrupt; reinstall it from its original source")
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
	digest, err := canonical.Spell(canonical.Digest(normalized))
	if err != nil {
		return nil, exit.Internalf("cannot spell descriptor digest: %s", err)
	}
	d.Digest = digest
	d.Raw = normalized
	return &d, nil
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
