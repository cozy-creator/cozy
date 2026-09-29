package producttest

import (
	"encoding/json"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
)

// This is the actual published H3 1.1.2 interface, read from the live catalog.
// ParseAssets must retain the tag while inserting the exact out-of-band file ref;
// the ordinary payload validator must admit that same selected union branch.
func TestH3ReferenceImageUsesPublishedTaggedUnion(t *testing.T) {
	raw, err := os.ReadFile("testdata/h3/package-interface-1.1.2.json")
	must(t, err)
	iface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	if assessmentDigest(iface.Raw) != "sha256:94e2be3a0661184a49a66d31dd3dd79e5d259fe7a53561aadfc890497cc88222" {
		t.Fatal("fixture no longer names the published H3 interface")
	}
	ep, problem := iface.Function("reference_media_to_video")
	fatal(t, problem)
	root := t.TempDir()
	payloadPath := filepath.Join(root, "reference.json")
	must(t, os.WriteFile(payloadPath, []byte(`{"prompt":"A lighthouse beside the sea","references":[{"type":"image"}],"seed":12345,"mute":true}`), 0600))
	input, _, problem := launch.ParsePayload(ep, nil, payloadPath)
	fatal(t, problem)
	if payloadProblem("paul/minimax-h3", ep, input) == nil {
		t.Fatal("missing image reference passed before staging")
	}
	imagePath := filepath.Join(root, "reference.png")
	file, err := os.Create(imagePath)
	must(t, err)
	must(t, png.Encode(file, image.NewRGBA(image.Rect(0, 0, 64, 64))))
	must(t, file.Close())
	input, assets, problem := launch.ParseAssets(ep, input, []string{"references.0.image=" + imagePath}, nil, nil)
	fatal(t, problem)
	fatal(t, payloadProblem("paul/minimax-h3", ep, input))
	if len(assets) != 1 || assets[0].FieldPath != "references.0.image" || assets[0].Length <= 0 ||
		assets[0].MediaType != "image/png" || assets[0].MaxBytes != 64<<20 || !strings.Contains(string(input), assets[0].Digest) {
		t.Fatalf("staging lost the declared asset: %+v", assets)
	}
	for name, reference := range map[string]string{
		"missing tag":   `{"image":"sha256:x"}`,
		"unknown tag":   `{"type":"other","image":"sha256:x"}`,
		"wrong branch":  `{"type":"video","image":"sha256:x"}`,
		"missing asset": `{"type":"image"}`,
		"null asset":    `{"type":"image","image":null}`,
		"not an object": `"image"`,
	} {
		t.Run(name, func(t *testing.T) {
			payload := json.RawMessage(`{"prompt":"test","references":[` + reference + `]}`)
			if payloadProblem("paul/minimax-h3", ep, payload) == nil {
				t.Fatalf("invalid tagged reference passed: %s", reference)
			}
		})
	}
	for _, reference := range []string{`{"type":"image","image":"ref"}`, `{"type":"video","video":"ref"}`, `{"type":"audio","audio":"ref"}`} {
		fatal(t, payloadProblem("paul/minimax-h3", ep, json.RawMessage(`{"prompt":"test","references":[`+reference+`]}`)))
	}
	// A field the selected branch does not declare is dropped and named, never refused.
	for reference, kept := range map[string]string{
		`{"type":"image","image":"ref","other":1}`:       `{"image":"ref","type":"image"}`,
		`{"type":"image","image":"ref","video":"ref2"}`:  `{"image":"ref","type":"image"}`,
		`{"type":"audio","audio":"ref","extra":{"a":1}}`: `{"audio":"ref","type":"audio"}`,
	} {
		payload := `{"prompt":"test","style":"x","references":[{"type":"video","video":"v"},` + reference + `]}`
		cleaned, ignored, problem := launch.ValidatePayload("paul/minimax-h3", ep, json.RawMessage(payload))
		fatal(t, problem)
		if len(ignored) != 2 || !strings.HasPrefix(ignored[0], "references[1].") || ignored[1] != "style" ||
			string(cleaned) != `{"prompt":"test","references":[{"type":"video","video":"v"},`+kept+`]}` {
			t.Fatalf("%s: ignored %v, cleaned %s", reference, ignored, cleaned)
		}
		fatal(t, payloadProblem("paul/minimax-h3", ep, cleaned))
	}
}

func TestTaggedUnionKeepsLiteralAndOrdinaryUnionValidation(t *testing.T) {
	raw := []byte(`{"application":"proof:app","entrypoints":[{"name":"run","request":{"fields":[{"name":"item","type":{"tag_field":"kind","union":[{"tag":1,"fields":[{"name":"value","type":"int"}]},{"tag":2,"fields":[{"name":"value","type":"str"}]}]}},{"name":"optional","type":{"union":["null","int"]}}]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[]}`)
	iface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	ep, problem := iface.Function("run")
	fatal(t, problem)
	for _, payload := range []string{`{"item":{"kind":1,"value":7},"optional":null}`, `{"item":{"kind":2,"value":"seven"},"optional":7}`} {
		fatal(t, payloadProblem("proof", ep, []byte(payload)))
	}
	for _, payload := range []string{`{"item":{"kind":1,"value":"seven"},"optional":null}`, `{"item":{"kind":"1","value":7},"optional":null}`, `{"item":{"kind":3,"value":7},"optional":null}`, `{"item":{"kind":1,"value":7},"optional":"seven"}`} {
		if payloadProblem("proof", ep, []byte(payload)) == nil {
			t.Fatalf("invalid discriminated or ordinary union passed: %s", payload)
		}
	}
}

func TestUntaggedStructDropsAnEmptyFieldName(t *testing.T) {
	raw := []byte(`{"application":"proof:app","entrypoints":[{"name":"run","request":{"fields":[{"name":"item","type":{"fields":[{"name":"value","type":"int"}]}}]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[]}`)
	iface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	ep, problem := iface.Function("run")
	fatal(t, problem)
	fatal(t, payloadProblem("proof", ep, []byte(`{"item":{"value":7}}`)))
	cleaned, ignored, problem := launch.ValidatePayload("proof", ep, []byte(`{"item":{"value":7,"":"undeclared"}}`))
	fatal(t, problem)
	if len(ignored) != 1 || ignored[0] != "item." || string(cleaned) != `{"item":{"value":7}}` {
		t.Fatalf("an untagged struct treated the empty field name as a discriminator: %v %s", ignored, cleaned)
	}
}

// payloadProblem is ValidatePayload's verdict alone.
func payloadProblem(pkg string, ep *launch.Entrypoint, payload json.RawMessage) *exit.Error {
	_, _, problem := launch.ValidatePayload(pkg, ep, payload)
	return problem
}
