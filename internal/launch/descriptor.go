// Package launch turns an INSTALLED GENERATION into the facts needed to serve it: the
// endpoint's verified surface, the artifact its binding selects, and the EndpointSpec the
// coordinator launches. It is what replaces cl-006's `--dev-endpoint` document (cl-010).
//
// Nothing here re-derives a fact its owner already produced:
//
//   - THE SURFACE is `endpoint.descriptor.json`, committed in the release's own source and
//     PROVEN at install by the release's own runtime (`cozy-runtime describe --check`, cl-009).
//     Reading it back costs microseconds; re-running `describe` per invocation would import
//     the endpoint's whole module graph to learn a fact already vouched for. The recorded
//     `surface_digest` is checked on every read, so an edited source tree is a refusal.
//   - THE ARTIFACT FACTS (store root, per-component snapshots, immutable config, variant,
//     physical floor) come from the runtime's own local artifact index, read through
//     `cozy-runtime list --json`. cozy-creator never composes a store path.
//   - THE SELECTION is `endpoint.toml`'s `[bindings]` table — the author's declared
//     default, in the runtime's own grammar and vocabulary.
//
// What cozy-creator DOES own is the LOCAL PINNED-BINDING RECORD it mints from those three
// (see coord.Binding): th-004 owns the real EntrypointBindingPlan document, this is the
// named seam, and the identity rule — the plan id is the digest of the record's canonical
// bytes — does not move when the document does.
package launch

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// DescriptorFile is the name cr-003 froze. It is committed in the endpoint's source and
// `describe --check` is what proves it current.
const DescriptorFile = "endpoint.descriptor.json"

// Descriptor is the part of the EndpointDescriptor this host reads. Unknown keys are
// carried in `Raw` and rendered untouched — a client that adds a field does not need this
// reader to change, and this reader never claims to understand one it does not.
type Descriptor struct {
	Application string          `json:"application"`
	Digest      string          `json:"surface_digest"`
	Entrypoints []Entrypoint    `json:"entrypoints"`
	Jobs        []Entrypoint    `json:"jobs"`
	Raw         json.RawMessage `json:"-"`
}

// Entrypoint is one callable surface: its request schema, its declared model slots, and
// its result shape.
type Entrypoint struct {
	Name    string   `json:"name"`
	Kind    string   `json:"kind"`
	GPU     bool     `json:"gpu"`
	Models  []Slot   `json:"models"`
	Request Struct   `json:"request"`
	Result  Struct   `json:"result"`
	Caps    []string `json:"capabilities"`
}

// Slot is one declared model binding path — capability, never selection.
type Slot struct {
	Class        string              `json:"class"`
	Path         string              `json:"path"`
	Param        string              `json:"param"`
	ComponentUse map[string][]string `json:"component_use"`
}

// Struct is a rendered msgspec struct.
type Struct struct {
	Name   string  `json:"struct"`
	Fields []Field `json:"fields"`
}

// Field is one declared field and its rendered type. `Type` is a string for a scalar
// (`int`, `str`) and an object for an asset (`{"asset":"image"}`), a list
// (`{"list":"float"}`) or a nested struct (`{"struct":…,"fields":[…]}`).
type Field struct {
	Name string          `json:"name"`
	Type json.RawMessage `json:"type"`
	Wire string          `json:"wire"`
}

// ReadDescriptor reads the committed descriptor out of a generation's source tree and
// checks it against the digest the install recorded.
//
// The check is the whole point of reading it here rather than re-deriving: cl-009 ran the
// release's OWN runtime over this file and recorded what it vouched for. If the two
// disagree now, the source tree moved under an install (the `--dir` editable door is
// exactly how), and serving a stale surface would be worse than refusing.
func ReadDescriptor(sourceDir, expectDigest string) (*Descriptor, *exit.Error) {
	path := filepath.Join(sourceDir, DescriptorFile)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, exit.Named(exit.Structural, "descriptor_absent",
			"this generation's source carries no %s", DescriptorFile).
			WithRemedy("an installed release commits its descriptor; `cozy-runtime describe --write-descriptor` is what writes one").
			WithNext("cozy install <org/endpoint> --force")
	}
	var d Descriptor
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, exit.New(exit.Validation, "%s is not a descriptor document: %s", path, err)
	}
	d.Raw = data
	if expectDigest != "" && d.Digest != expectDigest {
		return nil, exit.Named(exit.Conflict, "descriptor_stale",
			"the committed descriptor is %s and this install recorded %s", d.Digest, expectDigest).
			WithRemedy("the source tree changed after the install; reinstall so the surface and the record are one document").
			WithNext("cozy install <org/endpoint> --force")
	}
	return &d, nil
}

// Function finds one entrypoint or job by name.
func (d *Descriptor) Function(name string) (*Entrypoint, *exit.Error) {
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
func (d *Descriptor) Names() []string {
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
// These ARE the output ids: the coordinator grants one destination per asset field path,
// so `--out` is a consequence of the endpoint's declared result rather than a convention
// the CLI and the endpoint each have to remember. Same walk cr-016's `payload.asset_paths`
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
// "asset", "list", or "struct" (with the nested struct).
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

// RequestFields names every declared request field, in the author's declaration order.
func (e *Entrypoint) RequestFields() []string {
	out := make([]string, 0, len(e.Request.Fields))
	for _, f := range e.Request.Fields {
		out = append(out, f.Name)
	}
	return out
}
