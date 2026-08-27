package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/launch"
)

func sectionDescriptor() {
	head("endpoint descriptor/1 — semantic identity and the one consumed schema grammar")
	sectionDescriptorJCS()
	if path := flag("file", ""); path != "" {
		data, err := os.ReadFile(path)
		must("reading Runtime descriptor", err)
		doc, problem := launch.DecodeDescriptor(data)
		detail := fmt.Sprintf("%v", problem)
		if doc != nil {
			detail = fmt.Sprintf("%s · %d bytes", doc.Digest, len(data))
		}
		check("Creator consumes the Runtime-generated descriptor",
			problem == nil, detail)
	}
	raw := []byte(`{"application":"probe:app","entrypoints":[{"hidden":false,"name":"run","request":{"fields":[{"constraints":{"gt":0},"name":"strength","type":"float","wire":"required"},{"name":"mode","type":{"literal":["fast","quality"]},"wire":"required"}]},"result":{"fields":[]}}],"format":"cozy.endpoint.descriptor/1","jobs":[]}`)
	descriptor, problem := launch.DecodeDescriptor(raw)
	want, err := canonical.Spell(canonical.Digest(raw))
	must("spelling descriptor digest", err)
	check("canonical content is the descriptor's sole identity",
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
	check("whitespace does not change descriptor identity",
		spacedProblem == nil && spacedDoc.Digest == want && bytes.Equal(spacedDoc.Raw, raw), "")
	reordered := []byte(`{"jobs":[],"format":"cozy.endpoint.descriptor/1","entrypoints":[{"result":{"fields":[]},"request":{"fields":[{"wire":"required","type":"float","name":"strength","constraints":{"gt":0}},{"wire":"required","type":{"literal":["fast","quality"]},"name":"mode"}]},"name":"run","hidden":false}],"application":"probe:app"}`)
	reorderedDoc, reorderedProblem := launch.DecodeDescriptor(reordered)
	check("key order does not change descriptor identity",
		reorderedProblem == nil && reorderedDoc.Digest == want && bytes.Equal(reorderedDoc.Raw, raw), "")
	changed := bytes.Replace(raw, []byte(`"name":"run"`), []byte(`"name":"other"`), 1)
	changedDoc, changedProblem := launch.DecodeDescriptor(changed)
	check("a meaning change moves descriptor identity",
		changedProblem == nil && changedDoc.Digest != want, "")

	for name, planted := range map[string][]byte{
		"duplicate key": bytes.Replace(raw, []byte(`{"application"`),
			[]byte(`{"application":"other","application"`), 1),
		"non-finite number": bytes.Replace(raw, []byte(`"gt":0`), []byte(`"gt":NaN`), 1),
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

func sectionDescriptorJCS() {
	path := flag("number-vectors", "fixtures/canonical/es6-numbers.txt")
	corpus, err := os.ReadFile(path)
	must("reading the Runtime ES6 number vectors", err)
	wantCorpus := "973abb151673539cb1713991235beb4f9892e55c3743e6bbb0d398c8e5512302"
	gotCorpus := fmt.Sprintf("%x", sha256.Sum256(corpus))
	check("the pinned Runtime number oracle is unchanged", gotCorpus == wantCorpus, gotCorpus)
	scanner := bufio.NewScanner(bytes.NewReader(corpus))
	rows, mismatches := 0, 0
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != 2 {
			continue
		}
		bits, err := hex.DecodeString(fields[0])
		if err != nil || len(bits) != 8 {
			mismatches++
			continue
		}
		value := math.Float64frombits(binary.LittleEndian.Uint64(bits))
		input := strconv.FormatFloat(value, 'g', -1, 64)
		got, err := canonical.NormalizeJCS([]byte(input))
		if err != nil || string(got) != fields[1] {
			mismatches++
		}
		rows++
	}
	must("scanning the Runtime ES6 number vectors", scanner.Err())
	check("Creator matches Runtime over the full ES6 number corpus",
		rows == 4561 && mismatches == 0, fmt.Sprintf("%d rows, %d mismatches", rows, mismatches))

	input := []byte("{\"\ue000\":\"bmp\",\"s\":\"\\b\\t\\n\\f\\r\\u0000\\\"\\\\\u2028\u2029\",\"\U00010000\":\"astral\"}")
	want := []byte("{\"s\":\"\\b\\t\\n\\f\\r\\u0000\\\"\\\\\u2028\u2029\",\"\U00010000\":\"astral\",\"\ue000\":\"bmp\"}")
	got, err := canonical.NormalizeJCS(input)
	check("Creator matches Runtime string escaping and UTF-16 key order",
		err == nil && bytes.Equal(got, want), fmt.Sprintf("%q", got))

	refused := 0
	for _, raw := range []string{
		`{"s":"\uD800"}`,
		`{"s":"\uDC00"}`,
		`{"s":"\uD800x"}`,
	} {
		if _, err := canonical.NormalizeJCS([]byte(raw)); canonical.Code(err) == "unicode_scalar" {
			refused++
		}
	}
	check("unpaired surrogate escapes refuse", refused == 3, fmt.Sprintf("%d/3", refused))
}
