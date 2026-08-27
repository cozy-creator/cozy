package main

import (
	"bytes"
	"fmt"
	"os"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/launch"
)

func sectionDescriptor() {
	head("endpoint descriptor/1 — exact bytes and the one consumed schema grammar")
	if path := flag("file", ""); path != "" {
		data, err := os.ReadFile(path)
		must("reading Runtime descriptor", err)
		doc, problem := launch.DecodeDescriptor(data)
		detail := fmt.Sprintf("%v", problem)
		if doc != nil {
			detail = fmt.Sprintf("%s · %d bytes", doc.Digest, len(data))
		}
		check("Creator consumes the exact Runtime-generated descriptor",
			problem == nil, detail)
	}
	raw := []byte(`{"application":"probe:app","entrypoints":[{"hidden":false,"name":"run","request":{"fields":[{"constraints":{"gt":0},"name":"strength","type":"float","wire":"required"},{"name":"mode","type":{"literal":["fast","quality"]},"wire":"required"}]},"result":{"fields":[]}}],"format":"cozy.endpoint.descriptor/1","jobs":[]}`)
	descriptor, problem := launch.DecodeDescriptor(raw)
	want, err := canonical.Spell(canonical.Digest(raw))
	must("spelling descriptor digest", err)
	check("the exact stored bytes are the descriptor's sole identity",
		problem == nil && descriptor.Digest == want, fmt.Sprintf("%s", want))
	check("entrypoint kind is inferred from collection membership",
		problem == nil && len(descriptor.Entrypoints) == 1 && descriptor.Entrypoints[0].Kind == "entrypoint", "")

	if problem == nil {
		ep := &descriptor.Entrypoints[0]
		check("gt is a real Creator validator: 0 refuses",
			launch.ValidatePayload(ep, []byte(`{"strength":0,"mode":"fast"}`)) != nil, "")
		check("gt and enum-as-literal admit the valid payload",
			launch.ValidatePayload(ep, []byte(`{"strength":0.25,"mode":"quality"}`)) == nil, "")
	}

	spaced := bytes.Replace(raw, []byte(`,"entrypoints"`), []byte(", \"entrypoints\""), 1)
	spacedDoc, spacedProblem := launch.DecodeDescriptor(spaced)
	check("whitespace changes exact descriptor identity",
		spacedProblem == nil && spacedDoc.Digest != want, "")

	for name, planted := range map[string][]byte{
		"embedded surface_digest": bytes.Replace(raw, []byte(`{"application"`),
			[]byte(`{"surface_digest":"sha256:`+strings.Repeat("0", 64)+`","application"`), 1),
		"redundant callable kind": bytes.Replace(raw, []byte(`{"hidden"`),
			[]byte(`{"kind":"entrypoint","hidden"`), 1),
		"unsupported constraint": bytes.Replace(raw, []byte(`"gt":0`), []byte(`"lt":1`), 1),
		"retired enum grammar": bytes.Replace(raw, []byte(`{"literal":["fast","quality"]}`),
			[]byte(`{"enum":"Mode","values":["fast","quality"]}`), 1),
	} {
		_, refusal := launch.DecodeDescriptor(planted)
		check(name+" refuses at the descriptor boundary", refusal != nil, "")
	}
}
