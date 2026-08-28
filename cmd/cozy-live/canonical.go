package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

// The conformance corpus is worker-protocol's FROZEN fixtures at schema rev 3.
// `fixtures/canonical/<n>.bin` is the marshaled message and `<n>.json` is its ONE canonical
// document — sha256 over that file IS the wire digest. So the check is exact in both
// directions: this writer must reproduce those bytes from the message, and this reader must
// accept them and refuse each frozen twin with the named code.
//
// One document has no `.bin`: `wire_schema` is DERIVED from the schema's own descriptor
// rather than marshaled from a message in it. It gets the stronger arm instead — its
// document id must equal the `SchemaDigest` constant this binding was generated with, which
// is the whole of #530-A1 checked from the consumer's side.
func sectionCanonical() {
	fixtures := flag("fixtures", "/home/fidika/cozy_v2/worker-protocol/fixtures")

	manifest := struct {
		Canonical map[string]struct {
			Bytes    int    `json:"bytes"`
			ID       string `json:"id"`
			Type     string `json:"type"`
			Document string `json:"document"`
		} `json:"canonical"`
		Red map[string]struct {
			Rule string `json:"rule"`
			Type string `json:"type"`
		} `json:"red"`
	}{}
	data, err := os.ReadFile(filepath.Join(fixtures, "MANIFEST.json"))
	must("reading the fixture manifest", err)
	must("parsing the fixture manifest", json.Unmarshal(data, &manifest))

	head("the schema fence: this binding's own SchemaDigest IS the corpus's WireSchema document")
	// A version CONSTANT that can agree while the bytes disagree is not a fence (#530-A1).
	// This is the consumer half of that argument: the digest `regen.sh` derived from the
	// descriptor it generated these bindings from, compared against the frozen document.
	// A stale `protocol/` copy fails here without any peer being dialed.
	schema := manifest.Canonical["wire_schema"]
	check("wire_identity.go's SchemaDigest matches the frozen WireSchema/1 id",
		pb.SchemaDigest == schema.ID, pb.SchemaDigest)
	schemaBytes, err := os.ReadFile(filepath.Join(fixtures, "canonical", "wire_schema.json"))
	must("reading wire_schema.json", err)
	spelledSchema, _ := canonical.Spell(canonical.Digest(schemaBytes))
	check("and the digest is over exactly those document bytes",
		spelledSchema == schema.ID, fmt.Sprintf("%d B -> %s", len(schemaBytes), shortID(spelledSchema)))
	check("this binding declares wire schema rev 3 on the linear-train minor 0",
		pb.WireSchemaRev == 3 && pb.WireMinor == 0,
		fmt.Sprintf("rev %d, minor %d", pb.WireSchemaRev, pb.WireMinor))

	head("this Go writer against worker-protocol's frozen canonical documents")
	names := make([]string, 0, len(manifest.Canonical))
	for name := range manifest.Canonical {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		row := manifest.Canonical[name]
		if name == "wire_schema" {
			// DERIVED from the schema's own descriptor rather than marshaled from a message
			// in it, so there is no `.bin` and no binding to unmarshal into. Its arms are
			// above, and they are the stronger ones.
			continue
		}
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
		// THE DOCUMENT VERSION IS PART OF THE IDENTITY (#536e). An AttemptOutcomeBody spelled
		// under `/1` would digest to a document that does not exist, and every outcome this
		// host acked would name the wrong one. The map is read independently per language,
		// so this is the arm that proves the two ends read the same map.
		check(name+" format tag", canonical.Format(msg) == row.Document, canonical.Format(msg))
	}

	head("this Go reader accepts the frozen documents (format + closed key set + re-emit law)")
	for _, name := range names {
		row := manifest.Canonical[name]
		msg := messageFor(row.Type)
		if msg == nil {
			continue
		}
		frozen, err := os.ReadFile(filepath.Join(fixtures, "canonical", name+".json"))
		must("reading "+name+".json", err)
		doc, rerr := canonical.Read(frozen, msg)
		check(name+" reads", rerr == nil && doc != nil, detailOf(rerr))
	}

	head("red arms: every frozen SEMANTIC TWIN is refused, by its own code")
	// The twins are a frozen document with exactly one rule broken. The expected code is
	// the rule the fixture names.
	expect := map[string]struct {
		code string
		msg  proto.Message
	}{
		"twin_duplicate_key":  {"duplicate_key", &pb.InvocationSpec{}},
		"twin_float":          {"non_integer_number", &pb.InvocationSpec{}},
		"twin_unknown_key":    {"unknown_field", &pb.InvocationSpec{}},
		"twin_whitespace":     {"noncanonical_encoding", &pb.InvocationSpec{}},
		"twin_libc_unspelled": {"libc_unspelled", &pb.EndpointEnvironmentSpec{}},
	}
	twins := make([]string, 0, len(expect))
	for name := range expect {
		twins = append(twins, name)
	}
	sort.Strings(twins)
	for _, name := range twins {
		want := expect[name]
		body, err := os.ReadFile(filepath.Join(fixtures, "red", name+".json"))
		if err != nil {
			check(name, false, "fixture missing: "+err.Error())
			continue
		}
		_, rerr := canonical.Read(body, want.msg)
		got := canonical.Code(rerr)
		check(name+" refuses", got == want.code,
			fmt.Sprintf("%d B -> %s (wanted %s)", len(body), orNone(got), want.code))
	}

	head("red arms: the document plane's own refusals, live")
	// A document of the WRONG kind. The `format` tag is what domain-separates two
	// documents with equal fields, and it is checked before any field is read.
	specBytes, err := os.ReadFile(filepath.Join(fixtures, "canonical", "invocation_spec_serving.json"))
	must("reading invocation_spec_serving.json", err)
	_, rerr := canonical.Read(specBytes, &pb.AttemptOutcomeBody{})
	check("an InvocationSpec read as an AttemptOutcomeBody refuses",
		canonical.Code(rerr) == "unknown_format", detailOf(rerr))

	// The VERSION half of the same fence: an AttemptOutcomeBody/2 read against a reader
	// that expected /1 would refuse on the tag, which is why the version is IN the tag.
	outcomeBytes, err := os.ReadFile(filepath.Join(fixtures, "canonical", "attempt_outcome_body_succeeded.json"))
	must("reading attempt_outcome_body_succeeded.json", err)
	_, rerr = canonical.Read(outcomeBytes, &pb.InvocationSpec{})
	check("an AttemptOutcomeBody read as an InvocationSpec refuses",
		canonical.Code(rerr) == "unknown_format", detailOf(rerr))

	// A `bytes` digest field that is not 32 bytes has no canonical spelling at all. The
	// subject is a WorkerSnapshotBody rather than an outcome because the BYTES NAMING LAW
	// applies to `bytes` fields: inside a document the same digest is SPELLED
	// `sha256:<hex>`, and the spelling is what the wire field's type decides.
	bad := &pb.WorkerSnapshotBody{AcceptedPlacementSetDigest: []byte("abc")}
	_, _, cerr := canonical.Identity(bad)
	check("a 3-byte *_digest refuses", canonical.Code(cerr) == "malformed_digest", detailOf(cerr))

	// A non-printable-ASCII field: the protocol profile is narrower than JSON, and the
	// writer refuses rather than spelling it.
	nonascii := &pb.AttemptOutcomeBody{RequestId: "req-1", AttemptOrdinal: 1, SafeMessage: "café"}
	_, _, cerr = canonical.Identity(nonascii)
	check("a non-ASCII field refuses", canonical.Code(cerr) == "non_ascii_field", detailOf(cerr))

	// Truncated bytes are not a document.
	_, rerr = canonical.Read(specBytes[:len(specBytes)-1], &pb.InvocationSpec{})
	check("truncated canonical bytes refuse", rerr != nil, detailOf(rerr))
}

// messageFor is the binding for each DOCUMENT SHAPE the corpus carries. It is the rev-2
// set: the retired TerminalBody and DeploymentSet have no entry, because they have no
// message to have one.
func messageFor(name string) proto.Message {
	switch name {
	case "cozy.worker.v1.InvocationSpec":
		return &pb.InvocationSpec{}
	case "cozy.worker.v1.AttemptOutcomeBody":
		return &pb.AttemptOutcomeBody{}
	case "cozy.worker.v1.PlacementSet":
		return &pb.PlacementSet{}
	case "cozy.worker.v1.PlacementSpec":
		return &pb.PlacementSpec{}
	case "cozy.worker.v1.EndpointEnvironmentSpec":
		return &pb.EndpointEnvironmentSpec{}
	case "cozy.worker.v1.WorkerSnapshotBody":
		return &pb.WorkerSnapshotBody{}
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
