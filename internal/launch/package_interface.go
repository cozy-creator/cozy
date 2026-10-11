// Package launch turns an INSTALLED PACKAGE into the facts needed to serve it: the
// package's verified surface, the artifact its binding selects, and the PackageSpec the
// orchestrator launches. It is what replaces cl-006's `--dev-package` document (cl-010).
//
// Nothing here re-derives a fact its owner already produced:
//
//   - THE SURFACE is an install-scoped package interface read once at install by THIS host's
//     Runtime — a static reading of the source that imports nothing (cl-175) — and compared
//     with the committed release. Reading it back costs microseconds; the recorded semantic
//     metadata is read without a content fingerprint admission gate.
//   - THE PLACEMENT FACTS come from the exact PlacementSet retained at install. Runtime owns
//     no local model-ref index, and cozy-creator never composes a TensorFS store path.
//   - THE SELECTION is the request's own: the hub binding's rung for the machine, or a
//     `model.<param>=` run key, with immutable authored defaults when no owner override exists.
//
// For a wholly weightless package the installed runtime is the sole canonical plan writer:
// `bindings --json` reports exact WeightsSubjects before spawn and
// `serve --weightless-package` privately stages the same bytes. Cozy consumes the
// identities as record-owner intent and never reconstructs the documents. The older local
// pinned-binding writer below remains only for the modeled path that still carries it.
package launch

import (
	"crypto/sha256"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// PackageInterfaceFile is Runtime's derived document inside an immutable install. It is
// never committed in package source.
const PackageInterfaceFile = "package-interface.json"
const packageInterfaceFormat = "cozy.package.interface/"

// PackageInterface is the package-interface subset this host consumes. Members it does not
// read are ignored so documents from older and newer Runtimes keep loading, and a callable it
// cannot read is Unavailable alone, with why; Raw is the whole normalized canonical JSON for
// control-plane transport and semantic identity.
type PackageInterface struct {
	Format      string
	Application string
	Entrypoints []Entrypoint
	Jobs        []Entrypoint
	Unavailable map[string]*exit.Error
	Raw         json.RawMessage
}

// Entrypoint is one callable surface: its request schema, its declared model slots, and
// its result shape.
type Entrypoint struct {
	Name         string `json:"name"`
	Internal     bool   `json:"internal,omitempty"`
	Kind         string `json:"-"`
	DescriptorID string `json:"-"`
	Models       []Slot `json:"models"`
	Request      Struct `json:"request"`
	Result       Struct `json:"result"`
	Publishes    bool   `json:"publishes"`
	// WeightsOutputs is the job's explicit WeightsSink slot set. It is separate from
	// result asset fields because worker-protocol rev5 OutputBinding has no kind.
	WeightsOutputs []WeightsOutput `json:"weights_outputs"`
	Invocable      *Invocable      `json:"invocable,omitempty"`
	Assets         *AssetsSlot     `json:"assets,omitempty"`
	// Accelerator is a job's own execution-device declaration; nil when undeclared.
	Accelerator *bool `json:"accelerator,omitempty"`
}

type Invocable struct {
	OperationIdentity            string                     `json:"operation_identity,omitempty"`
	OperationIdentityUnavailable string                     `json:"operation_identity_unavailable,omitempty"`
	Memoize                      bool                       `json:"memoize"`
	Capabilities                 []string                   `json:"capabilities"`
	Context                      string                     `json:"context"`
	Module                       string                     `json:"module"`
	Export                       string                     `json:"export"`
	Parameters                   []string                   `json:"parameters"`
	Defaults                     map[string]json.RawMessage `json:"defaults"`
	TypeNames                    map[string]string          `json:"type_names"`
	EnumMembers                  map[string]json.RawMessage `json:"enum_members"`
}

type WeightsOutput struct {
	OutputID string `json:"output_id"`
	MimeType string `json:"mime_type"`
	MaxBytes uint64 `json:"max_bytes"`
}

// Slot is one declared model binding path with optional authored selection defaults. Its members
// mirror cozy-runtime's `internal/package_interface.py` slot (cr-078a): `{class, component_use,
// path}` plus the class's two closed consents `encoded_leaves` and `fusion` (h3a-015) and an
// optional `sequence_parallel` document. `stamps` is a RETIRED member: every release published before the cut ships an
// empty map, which reads as nothing declared, and a value in it refuses by name below.
type Slot struct {
	Class            string                 `json:"class"`
	Path             string                 `json:"path"`
	Param            string                 `json:"-"`
	ComponentUse     map[string][]string    `json:"component_use"`
	EncodedLeaves    string                 `json:"encoded_leaves,omitempty"`
	Fusion           string                 `json:"fusion,omitempty"`
	SequenceParallel json.RawMessage        `json:"sequence_parallel,omitempty"`
	DefaultLadder    []ModelDefaultRung     `json:"default_ladder,omitempty"`
	DefaultBinding   *hub.PackageBindingRow `json:"-"`
}

// ModelDefaultRung binds a GPU pattern to a full org/model@release/lane reference.
// This immutable metadata is a fallback; it is never a mutable Hub binding row.
type ModelDefaultRung struct {
	GPU  string `json:"gpu"`
	GPUs int    `json:"gpus,omitempty"`
	Lane string `json:"lane"`
}

// SequenceParallelDegrees intersects the selected entrypoint's model slots. Other
// functions are separate constructions and cannot restrict this request's GPU group.
// A slot that declares nothing makes the selected construction unshardable.
//
// This is CAPABILITY, never selection: it says a group of this degree can be built, not
// that one will be. What decides the actual degree is the width of the machine the renter
// bought (cl-179).
func (entrypoint *Entrypoint) SequenceParallelDegrees() []int {
	folded, first := map[int]bool{}, true
	for _, slot := range entrypoint.Models {
		declared := slot.sequenceParallelDegrees()
		if first {
			folded, first = declared, false
			continue
		}
		for degree := range folded {
			if !declared[degree] {
				delete(folded, degree)
			}
		}
	}
	out := make([]int, 0, len(folded))
	for degree := range folded {
		out = append(out, degree)
	}
	sort.Ints(out)
	return out
}

// sequenceParallelDegrees reads one slot's `{"degrees": [K, ...]}`. An absent, unreadable
// or empty document declares nothing, which is the same answer as a slot that cannot be
// sharded — this side never infers a degree an author did not write.
func (s Slot) sequenceParallelDegrees() map[int]bool {
	if len(s.SequenceParallel) == 0 {
		return nil
	}
	var declared struct {
		Degrees []int `json:"degrees"`
	}
	if json.Unmarshal(s.SequenceParallel, &declared) != nil {
		return nil
	}
	out := make(map[int]bool, len(declared.Degrees))
	for _, degree := range declared.Degrees {
		if degree >= 2 {
			out[degree] = true
		}
	}
	return out
}

// Struct is a rendered msgspec struct.
type Struct struct {
	Input    string          `json:"input,omitempty"`
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
	Default     json.RawMessage  `json:"default,omitempty"`
	Constraints FieldConstraints `json:"constraints"`
	AssetBound  struct {
		MaxBytes        int64    `json:"max_bytes"`
		MaxDecodedBytes int64    `json:"max_decoded_bytes,omitempty"`
		MediaTypes      []string `json:"media_types"`
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

// FieldConstraints are the bounds this host checks early. Runtime validates every declared
// constraint at execution, so one this host cannot read is left to it.
type FieldConstraints struct {
	MultipleOf *json.Number `json:"multiple_of"`
	MinLength  *int64       `json:"min_length"`
	MaxLength  *int64       `json:"max_length"`
	GT         *float64     `json:"gt"`
	GE         *float64     `json:"ge"`
	LE         *float64     `json:"le"`
}

// MissingComponents returns the package-declared lower bound that a checkpoint
// cannot supply. ComponentUse is intentionally conservative: it unions only
// explicitly declared names, so an undeclared method can never cause a false
// incompatibility verdict.
func MissingComponents(slot Slot, available []string) []string {
	have := make(map[string]struct{}, len(available))
	for _, component := range available {
		have[component] = struct{}{}
	}
	required := make(map[string]struct{})
	for _, components := range slot.ComponentUse {
		for _, component := range components {
			required[component] = struct{}{}
		}
	}
	missing := make([]string, 0, len(required))
	for component := range required {
		if _, ok := have[component]; !ok {
			missing = append(missing, component)
		}
	}
	sort.Strings(missing)
	return missing
}

func validateEntrypoint(ep *Entrypoint) *exit.Error {
	if ep.Name == "" {
		return exit.New(exit.Validation, "package interface carries an unnamed %s", ep.Kind)
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
		var problem *exit.Error
		slot.DefaultBinding, problem = defaultModelBinding(*slot)
		if problem != nil {
			return problem
		}
	}
	for _, pair := range []struct {
		name string
		body Struct
	}{{"request", ep.Request}, {"result", ep.Result}} {
		for _, field := range pair.body.Fields {
			if field.Name == "" {
				return exit.New(exit.Validation, "%s.%s has an unnamed field", ep.Name, pair.name)
			}
		}
	}
	for _, field := range ep.Result.Fields {
		if path := ungrantedCollectionOutput(field.Type, field.Name, false); path != "" {
			return exit.Named(exit.Validation, "output_collection_unsupported",
				"%s result %s contains a native output in a collection without fixed destination grants", ep.Name, path).
				WithRemedy("return one FileAsset or Tree, or a fixed msgspec.Struct with named output fields")
		}
	}
	if ep.Kind != "job" && len(ep.WeightsOutputs) > 0 {
		return exit.New(exit.Validation, "%s declares weights outputs outside the job surface", ep.Name)
	}
	seenWeights := map[string]bool{}
	for _, output := range ep.WeightsOutputs {
		if output.OutputID == "" || seenWeights[output.OutputID] ||
			output.MaxBytes > (uint64(1)<<53)-1 || output.MimeType != orchestrator.WeightsManifestMime {
			return exit.New(exit.Validation,
				"%s has an invalid weights output %q: slots are unique snapshot MIME rows with a 0..2^53-1 new-byte cap",
				ep.Name, output.OutputID)
		}
		seenWeights[output.OutputID] = true
	}
	return nil
}

// Result grants name exact field paths. Collections and unions cannot supply those
// paths before execution; scalar collections, fixed native fields and a list of assets at a
// fixed path (a list output) remain valid.
// The closed type grammar has already been validated before this semantic walk.
func ungrantedCollectionOutput(raw json.RawMessage, path string, dynamic bool) string {
	var object map[string]json.RawMessage
	if json.Unmarshal(raw, &object) != nil {
		return ""
	}
	if dynamic && (object["asset"] != nil || string(object["input"]) == `"tree"`) {
		return path
	}
	if fields := object["fields"]; fields != nil {
		var rows []Field
		_ = json.Unmarshal(fields, &rows)
		for _, field := range rows {
			if found := ungrantedCollectionOutput(field.Type, path+"."+field.Name, dynamic); found != "" {
				return found
			}
		}
	}
	if item := object["list"]; item != nil {
		// A list of assets at a fixed path is a list output: each product the function
		// publishes into it is appended, and its one `path.*` grant covers every item.
		if kind, _ := typeOf(item); kind == "asset" && !dynamic {
			return ""
		}
		return ungrantedCollectionOutput(item, path+"[]", true)
	}
	if mapping := object["map"]; mapping != nil {
		var pair map[string]json.RawMessage
		_ = json.Unmarshal(mapping, &pair)
		for _, side := range []string{"key", "value"} {
			if found := ungrantedCollectionOutput(pair[side], path+"["+side+"]", true); found != "" {
				return found
			}
		}
	}
	if union := object["union"]; union != nil {
		var branches []json.RawMessage
		_ = json.Unmarshal(union, &branches)
		for _, branch := range branches {
			if found := ungrantedCollectionOutput(branch, path, true); found != "" {
				return found
			}
		}
	}
	return ""
}

// PackageInterfacePath is the one install-scoped location for Runtime-derived bytes.
func PackageInterfacePath(installDir string) string {
	return filepath.Join(installDir, "documents", PackageInterfaceFile)
}

// ReadPackageInterface reads the stored package interface and joins it to the install record.
func ReadPackageInterface(path string) (*PackageInterface, *exit.Error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, exit.Named(exit.Structural, "package_interface_absent",
			"this install carries no stored %s", PackageInterfaceFile).
			WithRemedy("reinstall from the original source so its Runtime can derive the package interface")
	}
	d, problem := DecodePackageInterface(data)
	if problem != nil {
		return nil, problem
	}
	return d, nil
}

// DecodePackageInterface reads the package-interface/1 members this host consumes and derives
// its canonical interface facts. Collection membership supplies callable kind; the document
// does not repeat it.
func DecodePackageInterface(data []byte) (*PackageInterface, *exit.Error) {
	if len(data) > canonical.DocMax {
		return nil, exit.New(exit.Validation, "%s exceeds the %d-byte cap", PackageInterfaceFile, canonical.DocMax)
	}
	normalized, err := canonical.NormalizeJCS(data)
	if err != nil {
		return nil, exit.New(exit.Validation, "%s is not canonical JSON: %s", PackageInterfaceFile, err)
	}
	var doc struct {
		Format      string            `json:"format"`
		Application string            `json:"application"`
		Entrypoints []json.RawMessage `json:"entrypoints"`
		Jobs        []json.RawMessage `json:"jobs"`
	}
	if err := json.Unmarshal(normalized, &doc); err != nil {
		return nil, exit.New(exit.Validation, "%s is not a package interface document: %s", PackageInterfaceFile, err)
	}
	if !strings.HasPrefix(doc.Format, packageInterfaceFormat) || doc.Application == "" {
		return nil, exit.New(exit.Validation, "%s format/application is invalid", PackageInterfaceFile)
	}
	d := &PackageInterface{Format: doc.Format, Application: doc.Application, Unavailable: map[string]*exit.Error{}, Raw: normalized}
	for kind, rows := range map[string][]json.RawMessage{"entrypoint": doc.Entrypoints, "job": doc.Jobs} {
		for _, raw := range rows {
			callable, problem := decodeCallable(kind, raw)
			switch {
			case problem != nil:
				if problem.Remedy == "" {
					problem.Remedy = "update cozy, or call another function of this package"
				}
				d.Unavailable[callable.Name] = problem
			case kind == "job":
				d.Jobs = append(d.Jobs, callable)
			default:
				d.Entrypoints = append(d.Entrypoints, callable)
			}
		}
	}
	return d, nil
}

// decodeCallable reads one entrypoint or job. One this host cannot read is refused alone: its
// problem answers every call of it.
func decodeCallable(kind string, raw json.RawMessage) (Entrypoint, *exit.Error) {
	var extra struct {
		Name           string                       `json:"name"`
		Internal       json.RawMessage              `json:"internal"`
		Assets         json.RawMessage              `json:"assets"`
		WeightsOutputs []map[string]json.RawMessage `json:"weights_outputs"`
	}
	_ = json.Unmarshal(raw, &extra)
	callable := Entrypoint{Name: extra.Name, Kind: kind}
	if err := json.Unmarshal(raw, &callable); err != nil {
		return Entrypoint{Name: extra.Name}, exit.New(exit.Validation, "%s is not readable here: %s", extra.Name, err)
	}
	// Decoding null as false would publish a callable its author hid.
	if len(extra.Internal) > 0 && string(extra.Internal) != "true" && string(extra.Internal) != "false" {
		return callable, exit.New(exit.Validation, "%s internal must be a boolean", callable.Name)
	}
	admitAssetsSlot(&callable, extra.Assets)
	if kind == "job" {
		// A new-byte budget has no safe default: zero would refuse the job's own output.
		for _, output := range extra.WeightsOutputs {
			if budget, present := output["max_bytes"]; !present || string(budget) == "null" {
				return callable, exit.New(exit.Validation, "%s weights outputs need an explicit max_bytes", callable.Name)
			}
		}
		digest := sha256.Sum256(append([]byte("cozy.runtime.job-descriptor\x00"), raw...))
		callable.DescriptorID, _ = canonical.Spell(digest[:])
	}
	problem := validateEntrypoint(&callable)
	return callable, problem
}

// Function finds one entrypoint or job by name.
func (d *PackageInterface) Function(name string) (*Entrypoint, *exit.Error) {
	if problem, unavailable := d.Unavailable[name]; unavailable {
		return nil, problem
	}
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
		WithRemedy("it registers: %s", strings.Join(d.PublicNames(), ", "))
}

// RequirePublic guards user-facing discovery and root submission. Managed child
// resolution reads the complete interface and checks its admitted parent instead.
func (e *Entrypoint) RequirePublic() *exit.Error {
	if e.Internal {
		return exit.Named(exit.NotFound, "callable_internal", "%s is an internal package function", e.Name).
			WithRemedy("invoke a public function from this package")
	}
	return nil
}

// ModelSlotPaths is every entrypoint and job model-slot path, sorted unique: the
// wire's model_slot_paths.
func (d *PackageInterface) ModelSlotPaths() []string {
	paths := []string{}
	for _, entry := range append(append([]Entrypoint(nil), d.Entrypoints...), d.Jobs...) {
		for _, model := range entry.Models {
			paths = append(paths, model.Path)
		}
	}
	slices.Sort(paths)
	return slices.Compact(paths)
}

// Names lists all registered callables for execution and dependency capture.
func (d *PackageInterface) Names() []string { return d.names(false) }

// PublicNames lists only externally callable entrypoints and jobs.
func (d *PackageInterface) PublicNames() []string { return d.names(true) }

func (d *PackageInterface) names(publicOnly bool) []string {
	out := []string{}
	for _, e := range d.Entrypoints {
		if !publicOnly || !e.Internal {
			out = append(out, e.Name)
		}
	}
	for _, j := range d.Jobs {
		if !publicOnly || !j.Internal {
			out = append(out, j.Name)
		}
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
		case "asset", "tree":
			out = append(out, name)
		case "struct":
			out = append(out, assetPaths(nested, name+".")...)
		}
	}
	return out
}

// OutputSlots are a result's output grants: each asset field's path, and `<field>.*` for a
// list of assets, whose one grant covers every item the function publishes into it.
func OutputSlots(s Struct) []string {
	out := AssetPaths(s)
	for _, name := range AssetLists(s) {
		out = append(out, name+".*")
	}
	return out
}

// AssetLists names every result field that is a list of assets: an output that grows by one
// product each time the function publishes into it.
func AssetLists(s Struct) []string {
	var out []string
	for _, f := range s.Fields {
		var object struct {
			List json.RawMessage `json:"list"`
		}
		if json.Unmarshal(f.Type, &object) == nil && len(object.List) > 0 {
			if kind, _ := typeOf(object.List); kind == "asset" {
				out = append(out, f.Name)
			}
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
	if kind, ok := object["input"]; ok && string(kind) == `"model"` {
		return "model", Struct{Input: "model"}
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

// InputKind returns the compact package-interface kind for one request field.
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

// AllowsGPUCount checks an explicit execution group against the authored model capability.
func (s Slot) AllowsGPUCount(count int) bool {
	return count >= 0 && (count <= 1 || s.sequenceParallelDegrees()[count])
}

// SupportedGPUCounts is package-code capability, including the unpartitioned path.
func (s Slot) SupportedGPUCounts() []int {
	counts := []int{1}
	for degree := range s.sequenceParallelDegrees() {
		if degree > 1 {
			counts = append(counts, degree)
		}
	}
	sort.Ints(counts)
	return counts
}
