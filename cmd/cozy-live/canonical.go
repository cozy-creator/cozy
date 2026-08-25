package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// The conformance corpus is worker-protocol's FROZEN fixtures. `fixtures/canonical/<n>.bin`
// is the marshaled message and `<n>.json` is its ONE canonical document — sha256 over that
// file IS the wire digest. So the check is exact in both directions: this writer must
// reproduce those bytes from the message, and this reader must accept them and refuse
// each frozen twin with the named code.
func sectionCanonical() {
	fixtures := flag("fixtures", "/home/fidika/cozy_v2/worker-protocol/fixtures")

	manifest := struct {
		Canonical map[string]struct {
			Bytes int    `json:"bytes"`
			ID    string `json:"id"`
			Type  string `json:"type"`
		} `json:"canonical"`
		Red map[string]struct {
			Rule string `json:"rule"`
			Type string `json:"type"`
		} `json:"red"`
	}{}
	data, err := os.ReadFile(filepath.Join(fixtures, "MANIFEST.json"))
	must("reading the fixture manifest", err)
	must("parsing the fixture manifest", json.Unmarshal(data, &manifest))

	head("this Go writer against worker-protocol's frozen canonical documents")
	names := make([]string, 0, len(manifest.Canonical))
	for name := range manifest.Canonical {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		row := manifest.Canonical[name]
		msg := messageFor(row.Type)
		if msg == nil {
			check(name, false, "no binding for "+row.Type)
			continue
		}
		wire, err := os.ReadFile(filepath.Join(fixtures, "canonical", name+".bin"))
		must("reading "+name+".bin", err)
		frozen, err := os.ReadFile(filepath.Join(fixtures, "canonical", name+".json"))
		must("reading "+name+".json", err)
		if err := proto.Unmarshal(wire, msg); err != nil {
			check(name, false, "the frozen message does not decode: "+err.Error())
			continue
		}
		mine, digest, cerr := canonical.Identity(msg)
		if cerr != nil {
			check(name, false, "this writer refused it: "+cerr.Error())
			continue
		}
		spelled, _ := canonical.Spell(digest)
		byteEqual := bytes.Equal(mine, frozen)
		check(name+" bytes", byteEqual, fmt.Sprintf("%d B, byte-for-byte", len(mine)))
		check(name+" id", spelled == row.ID, spelled)
	}

	head("this Go reader accepts the frozen documents (format + closed key set + re-emit law)")
	for _, name := range names {
		row := manifest.Canonical[name]
		frozen, err := os.ReadFile(filepath.Join(fixtures, "canonical", name+".json"))
		must("reading "+name+".json", err)
		doc, rerr := canonical.Read(frozen, messageFor(row.Type))
		check(name+" reads", rerr == nil && doc != nil, detailOf(rerr))
	}

	head("red arms: every frozen SEMANTIC TWIN is refused, by its own code")
	// The twins are the ExecutionSpec document with exactly one rule broken. The
	// expected code is the rule the fixture names.
	expect := map[string]string{
		"twin_duplicate_key":     "duplicate_key",
		"twin_float_number":      "non_integer_number",
		"twin_integer_over_2_53": "number_range",
		"twin_key_order":         "noncanonical_encoding",
		"twin_non_ascii":         "non_ascii_field",
		"twin_trailing_newline":  "noncanonical_encoding",
		"twin_unknown_key":       "unknown_field",
		"twin_whitespace":        "noncanonical_encoding",
	}
	twins := make([]string, 0, len(expect))
	for name := range expect {
		twins = append(twins, name)
	}
	sort.Strings(twins)
	for _, name := range twins {
		body, err := os.ReadFile(filepath.Join(fixtures, "red", name+".json"))
		if err != nil {
			check(name, false, "fixture missing: "+err.Error())
			continue
		}
		_, rerr := canonical.Read(body, &pb.ExecutionSpec{})
		got := canonical.Code(rerr)
		check(name+" refuses", got == expect[name],
			fmt.Sprintf("%d B -> %s (wanted %s)", len(body), orNone(got), expect[name]))
	}

	head("red arms: the document plane's own refusals, live")
	// A document of the WRONG kind. The `format` tag is what domain-separates two
	// documents with equal fields, and it is checked before any field is read.
	specBytes, err := os.ReadFile(filepath.Join(fixtures, "canonical", "exec_spec_serving.json"))
	must("reading exec_spec_serving.json", err)
	_, rerr := canonical.Read(specBytes, &pb.TerminalBody{})
	check("an ExecutionSpec read as a TerminalBody refuses", canonical.Code(rerr) == "unknown_format",
		detailOf(rerr))

	// A digest field that is not 32 bytes has no canonical spelling at all.
	bad := &pb.TerminalBody{RequestId: "req-1", Attempt: 1, ExecSpecDigest: []byte{1, 2, 3}}
	_, _, cerr := canonical.Identity(bad)
	check("a 3-byte *_digest refuses", canonical.Code(cerr) == "malformed_digest", detailOf(cerr))

	// A non-printable-ASCII field: the protocol profile is narrower than JSON, and the
	// writer refuses rather than spelling it.
	nonascii := &pb.TerminalBody{RequestId: "req-1", Attempt: 1, SafeMessage: "café"}
	_, _, cerr = canonical.Identity(nonascii)
	check("a non-ASCII field refuses", canonical.Code(cerr) == "non_ascii_field", detailOf(cerr))

	// Truncated bytes are not a document.
	_, rerr = canonical.Read(specBytes[:len(specBytes)-1], &pb.ExecutionSpec{})
	check("truncated canonical bytes refuse", rerr != nil, detailOf(rerr))
}

func messageFor(name string) proto.Message {
	switch name {
	case "cozy.worker.v1.ExecutionSpec":
		return &pb.ExecutionSpec{}
	case "cozy.worker.v1.TerminalBody":
		return &pb.TerminalBody{}
	case "cozy.worker.v1.PurgeBody":
		return &pb.PurgeBody{}
	}
	return nil
}

func detailOf(err error) string {
	if err == nil {
		return "accepted"
	}
	return err.Error()
}

func orNone(s string) string {
	if s == "" {
		return "ACCEPTED"
	}
	return s
}
