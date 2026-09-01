package producttest

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/modeltransfer"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// TestCanonicalDocuments is the identity fence. Every document that crosses a repo or
// process boundary is named by the sha256 of its canonical bytes, so two independent
// writers in two languages must produce byte-identical documents or nothing downstream —
// digests, plan ids, terminal admission — agrees at all. It is the cheapest test here and
// the one whose failure is least visible any other way.
//
// It runs ALWAYS. The corpus is vendored at testdata/worker-protocol, digest-fenced
// by vendored_test.go, so this needs no token, no sibling checkout and no flag. It
// used to resolve a corpus off the local disk and t.Skipf when it was absent, which
// is how it sat green and inert on every CI run while a whole wire minor of skew
// went unnoticed. A skip is indistinguishable from a pass.
func TestCanonicalDocuments(t *testing.T) {
	fixtureDir := corpusDir
	var manifest struct {
		Canonical map[string]struct{ ID, Type, Document string } `json:"canonical"`
		WireMinor uint32                                         `json:"wire_minor"`
	}
	data, err := os.ReadFile(filepath.Join(fixtureDir, "MANIFEST.json"))
	must(t, err)
	must(t, json.Unmarshal(data, &manifest))

	// The corpus names the wire minor it was frozen at. This repo is the RecordOwner, so a
	// stale vendored binding here advertises a minor the workers have already moved past on
	// every Claim and DesiredWorkerState. The document bytes below do not move on an
	// additive bump, so nothing else in this test would notice.
	if pb.WireMinor != manifest.WireMinor {
		t.Errorf("vendored wire minor %d != the frozen corpus's %d: re-vendor protocol/ from "+
			"worker-protocol", pb.WireMinor, manifest.WireMinor)
	}

	// This Go writer against the frozen documents, and this Go reader back over them.
	names := make([]string, 0, len(manifest.Canonical))
	for name := range manifest.Canonical {
		names = append(names, name)
	}
	sort.Strings(names)
	arms := 0
	for _, name := range names {
		row := manifest.Canonical[name]
		msg := messageFor(row.Type)
		if msg == nil {
			t.Errorf("%s: no binding for %s", name, row.Type)
			continue
		}
		wire, err := os.ReadFile(filepath.Join(fixtureDir, "canonical", name+".bin"))
		must(t, err)
		frozen, err := os.ReadFile(filepath.Join(fixtureDir, "canonical", name+".json"))
		must(t, err)
		must(t, proto.Unmarshal(wire, msg))
		mine, digest, cerr := canonical.Identity(msg)
		if cerr != nil {
			t.Errorf("%s: this writer refused the frozen message: %v", name, cerr)
			continue
		}
		if !bytes.Equal(mine, frozen) {
			t.Errorf("%s: this writer produced %d B, the frozen document is %d B", name, len(mine), len(frozen))
		}
		if spelled, _ := canonical.Spell(digest); spelled != row.ID {
			t.Errorf("%s: id %s != frozen %s", name, spelled, row.ID)
		}
		// THE DOCUMENT VERSION IS PART OF THE IDENTITY (#536e). Every current
		// pre-release document is spelled under the sole `/1` format.
		if canonical.Format(msg) != row.Document {
			t.Errorf("%s: format tag %s != frozen %s", name, canonical.Format(msg), row.Document)
		}
		if _, rerr := canonical.Read(frozen, msg); rerr != nil {
			t.Errorf("%s: this reader refused the frozen document: %v", name, rerr)
		}
		arms++
	}
	// A corpus that shrank to nothing would otherwise pass this test green, which is the
	// same vacuous pass a silent skip gives. Every canonical document must be exercised.
	if arms != len(manifest.Canonical) || arms == 0 {
		t.Fatalf("exercised %d of %d canonical documents", arms, len(manifest.Canonical))
	}

	// RED: every frozen SEMANTIC TWIN — one frozen document with exactly one rule broken —
	// is refused by the code the fixture names.
	for name, want := range map[string]struct {
		code string
		msg  proto.Message
	}{
		"twin_duplicate_key": {"duplicate_key", &pb.InvocationSpec{}},
		"twin_float":         {"non_integer_number", &pb.InvocationSpec{}},
		"twin_unknown_key":   {"unknown_field", &pb.InvocationSpec{}},
		"twin_whitespace":    {"noncanonical_encoding", &pb.InvocationSpec{}},
	} {
		body, err := os.ReadFile(filepath.Join(fixtureDir, "red", name+".json"))
		must(t, err)
		_, rerr := canonical.Read(body, want.msg)
		if got := canonical.Code(rerr); got != want.code {
			t.Errorf("%s: refused as %q, wanted %q", name, got, want.code)
		}
		arms++
	}

	// RED: the document plane's own refusals. The `format` tag domain-separates two
	// documents with equal fields and is checked before any field is read.
	spec, err := os.ReadFile(filepath.Join(fixtureDir, "canonical", "invocation_spec_serving.json"))
	must(t, err)
	if _, rerr := canonical.Read(spec, &pb.AttemptOutcomeBody{}); canonical.Code(rerr) != "unknown_format" {
		t.Errorf("an InvocationSpec read as an AttemptOutcomeBody was not refused: %v", rerr)
	}
	if _, rerr := canonical.Read(spec[:len(spec)-1], &pb.InvocationSpec{}); rerr == nil {
		t.Error("truncated canonical bytes were accepted")
	}
	// A `bytes` digest field that is not 32 bytes has no canonical spelling at all.
	_, _, cerr := canonical.Identity(&pb.WorkerSnapshotBody{AcceptedPlacementSetDigest: []byte("abc")})
	if canonical.Code(cerr) != "malformed_digest" {
		t.Errorf("a 3-byte *_digest was not refused: %v", cerr)
	}
	// The protocol profile is narrower than JSON: a non-printable-ASCII field refuses
	// rather than being spelled.
	_, _, cerr = canonical.Identity(&pb.AttemptOutcomeBody{
		RequestId: "req-1", AttemptOrdinal: 1, SafeMessage: "café"})
	if canonical.Code(cerr) != "non_ascii_field" {
		t.Errorf("a non-ASCII field was not refused: %v", cerr)
	}
	arms += 5 // the wire-minor agreement and the document plane's own refusals, above
	t.Logf("%s: %d arms at wire minor %d (%d canonical documents, 4 semantic twins, "+
		"4 plane refusals, 1 wire-minor agreement)",
		fixtureDir, arms, manifest.WireMinor, len(manifest.Canonical))
}

func messageFor(name string) proto.Message {
	switch name {
	case "cozy.worker.v1.InvocationSpec":
		return &pb.InvocationSpec{}
	case "cozy.worker.v1.AttemptOutcomeBody":
		return &pb.AttemptOutcomeBody{}
	case "cozy.worker.v1.WeightsReceipt":
		return &pb.WeightsReceipt{}
	case "cozy.worker.v1.ClaimProof":
		return &pb.ClaimProof{}
	case "cozy.worker.v1.DownloadDelegation":
		return &pb.DownloadDelegation{}
	case "cozy.worker.v1.PlacementSet":
		return &pb.PlacementSet{}
	case "cozy.worker.v1.PrivatePackageRevision":
		return &pb.PrivatePackageRevision{}
	case "cozy.worker.v1.WorkerSnapshotBody":
		return &pb.WorkerSnapshotBody{}
	}
	return nil
}

// TestNumberProfile is the cross-language float hazard, in isolation. Go and Python must
// spell every double the same way or two canonical documents describing the same thing
// digest differently, and the whole identity plane silently forks at the boundary.
// `testdata/canonical/es6-numbers.txt` is Runtime's own oracle, pinned by digest.
func TestNumberProfile(t *testing.T) {
	corpus, err := os.ReadFile("testdata/canonical/es6-numbers.txt")
	must(t, err)
	// Pinned: a corpus that moved is a different question, not a passing answer.
	if got := hex.EncodeToString(sha256Of(corpus)); got !=
		"973abb151673539cb1713991235beb4f9892e55c3743e6bbb0d398c8e5512302" {
		t.Fatalf("the pinned Runtime number oracle moved: %s", got)
	}
	rows, admitted, refused := 0, 0, 0
	safe := float64((int64(1) << 53) - 1)
	scanner := bufio.NewScanner(bytes.NewReader(corpus))
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != 2 {
			continue
		}
		bits, err := hex.DecodeString(fields[0])
		if err != nil || len(bits) != 8 {
			t.Fatalf("row %d is not a float64 bit pattern: %q", rows, fields[0])
		}
		rows++
		value := math.Float64frombits(binary.LittleEndian.Uint64(bits))
		got, err := canonical.NormalizeJCS([]byte(strconv.FormatFloat(value, 'g', -1, 64)))
		switch {
		case math.Abs(value) > safe:
			// Outside the exactly-representable integer range the profile REFUSES rather
			// than spelling a number the other end would round differently.
			if err == nil {
				t.Errorf("%v is outside the safe range and was admitted", value)
			}
			refused++
		case err != nil || string(got) != fields[1]:
			t.Errorf("%v: Cozy spelled %q, Runtime spelled %q (%v)", value, got, fields[1], err)
		default:
			admitted++
		}
	}
	must(t, scanner.Err())
	if rows != 4561 || admitted == 0 || refused == 0 {
		t.Fatalf("%d rows, %d admitted, %d profile refusals", rows, admitted, refused)
	}
	t.Logf("%d rows, %d admitted, %d profile refusals, 0 mismatches", rows, admitted, refused)

	// Every spelling of one boundary value collapses to ONE closed identity.
	for _, twins := range [][]string{
		{`{"v":9007199254740991}`, `{"v":9007199254740991.0}`, `{"v":9.007199254740991e15}`},
		{`{"v":-9007199254740991}`, `{"v":-9007199254740991.0}`, `{"v":-9.007199254740991e15}`},
	} {
		identities := map[string]bool{}
		for _, twin := range twins {
			normalized, err := canonical.NormalizeJCS([]byte(twin))
			must(t, err)
			again, err := canonical.NormalizeJCS(normalized)
			must(t, err)
			if !bytes.Equal(again, normalized) {
				t.Errorf("%s does not normalize to a closed form", twin)
			}
			identities[string(normalized)] = true
		}
		if len(identities) != 1 {
			t.Errorf("%v produced %d identities", twins, len(identities))
		}
	}
	for _, source := range []string{
		`{"v":9007199254740992}`, `{"v":9007199254740992.0}`, `{"v":9.007199254740992e15}`,
		`{"v":-9007199254740992}`, `{"v":-9007199254740992.0}`, `{"v":-9.007199254740992e15}`,
		`{"v":1000000000000000000000}`, `{"v":1e21}`,
	} {
		if _, err := canonical.NormalizeJCS([]byte(source)); canonical.Code(err) != "number_range" {
			t.Errorf("%s was not refused as number_range: %v", source, err)
		}
	}
	// String escaping and UTF-16 key order, which are the other half of "two writers agree".
	input := []byte("{\"\ue000\":\"bmp\",\"s\":\"\\b\\t\\n\\f\\r\\u0000\\\"\\\\\u2028\u2029\",\"\U00010000\":\"astral\"}")
	want := []byte("{\"s\":\"\\b\\t\\n\\f\\r\\u0000\\\"\\\\\u2028\u2029\",\"\U00010000\":\"astral\",\"\ue000\":\"bmp\"}")
	if got, err := canonical.NormalizeJCS(input); err != nil || !bytes.Equal(got, want) {
		t.Errorf("escaping/key order: %q (%v)", got, err)
	}
	for _, raw := range []string{`{"s":"\uD800"}`, `{"s":"\uDC00"}`, `{"s":"\uD800x"}`} {
		if _, err := canonical.NormalizeJCS([]byte(raw)); canonical.Code(err) != "unicode_scalar" {
			t.Errorf("%s: unpaired surrogate was not refused: %v", raw, err)
		}
	}
}

// TestPackageDescriptor is the OTHER side of the identity plane: the grammar Runtime
// authors and Cozy consumes at install. Identity is the canonical content and nothing
// else, and a descriptor that cannot be read exactly is refused rather than guessed at.
func TestPackageDescriptor(t *testing.T) {
	raw := []byte(`{"application":"probe:app","entrypoints":[{"name":"run","request":{"fields":[{"constraints":{"gt":0},"name":"strength","type":"float"},{"name":"mode","type":{"literal":["fast","quality"]}}]},"result":{"fields":[]}}],"format":"cozy.package.descriptor/1","jobs":[]}`)
	want, err := canonical.Spell(canonical.Digest(raw))
	must(t, err)
	doc, problem := launch.DecodeDescriptor(raw)
	fatal(t, problem)
	if doc.Digest != want {
		t.Fatalf("descriptor identity %s != %s", doc.Digest, want)
	}
	if len(doc.Entrypoints) != 1 || doc.Entrypoints[0].Kind != "entrypoint" {
		t.Fatalf("entrypoint kind is not inferred from collection membership: %+v", doc.Entrypoints)
	}
	// The declared constraints are real Cozy validators, not documentation.
	ep := &doc.Entrypoints[0]
	if ep.Request.Fields[0].Wire != "required" || ep.Request.Fields[1].Wire != "required" {
		t.Fatalf("absent wire did not derive required: %+v", ep.Request.Fields)
	}
	if launch.ValidatePayload(ep, []byte(`{"strength":0,"mode":"fast"}`)) == nil {
		t.Error("gt:0 admitted 0")
	}
	if e := launch.ValidatePayload(ep, []byte(`{"strength":0.25,"mode":"quality"}`)); e != nil {
		t.Errorf("a valid payload was refused: %s", e.Message)
	}
	// Whitespace and key order are NOT identity; a meaning change is.
	for _, same := range [][]byte{
		bytes.Replace(raw, []byte(`,"entrypoints"`), []byte(", \"entrypoints\""), 1),
		[]byte(`{"jobs":[],"format":"cozy.package.descriptor/1","entrypoints":[{"result":{"fields":[]},"request":{"fields":[{"type":"float","name":"strength","constraints":{"gt":0}},{"type":{"literal":["fast","quality"]},"name":"mode"}]},"name":"run"}],"application":"probe:app"}`),
	} {
		got, problem := launch.DecodeDescriptor(same)
		if problem != nil || got.Digest != want || !bytes.Equal(got.Raw, raw) {
			t.Errorf("a spelling change moved descriptor identity: %v", problem)
		}
	}
	changed := bytes.Replace(raw, []byte(`"name":"run"`), []byte(`"name":"other"`), 1)
	if got, problem := launch.DecodeDescriptor(changed); problem != nil || got.Digest == want {
		t.Error("a meaning change did not move descriptor identity")
	}
	for name, planted := range map[string][]byte{
		"duplicate key": bytes.Replace(raw, []byte(`{"application"`),
			[]byte(`{"application":"other","application"`), 1),
		"non-finite number": bytes.Replace(raw, []byte(`"gt":0`), []byte(`"gt":NaN`), 1),
		"embedded surface_digest": bytes.Replace(raw, []byte(`{"application"`),
			[]byte(`{"surface_digest":"sha256:`+strings.Repeat("0", 64)+`","application"`), 1),
		"redundant callable kind": bytes.Replace(raw, []byte(`{"name":"run"`),
			[]byte(`{"kind":"entrypoint","name":"run"`), 1),
		"retired hidden marker": bytes.Replace(raw, []byte(`{"name":"run"`),
			[]byte(`{"hidden":false,"name":"run"`), 1),
		"explicit required wire": bytes.Replace(raw, []byte(`"type":"float"`),
			[]byte(`"type":"float","wire":"required"`), 1),
		"unsupported constraint": bytes.Replace(raw, []byte(`"gt":0`), []byte(`"lt":1`), 1),
		"retired enum grammar": bytes.Replace(raw, []byte(`{"literal":["fast","quality"]}`),
			[]byte(`{"enum":"Mode","values":["fast","quality"]}`), 1),
		"retired model production graph": bytes.Replace(raw, []byte(`"jobs":[]`),
			[]byte(`"jobs":[],"model_productions":[]`), 1),
	} {
		if _, refusal := launch.DecodeDescriptor(planted); refusal == nil {
			t.Errorf("%s was accepted at the descriptor boundary", name)
		}
	}

	// Model inputs are not hardware facts: this job could transform their TensorFS bytes on
	// CPU. The immutable release dependency set selects the machine class instead.
	gpuRaw := []byte(`{"application":"h3:tools","entrypoints":[],"format":"cozy.package.descriptor/1","jobs":[{"models":[{"class":"H3Dits","component_use":{},"path":"four_lane.models.dits","source_profile":"hf/minimax-h3/native-dual-bf16/1","stamps":{}}],"name":"four_lane","publishes":false,"request":{"fields":[]},"result":{"fields":[]}}]}`)
	gpuDescriptor, problem := launch.DecodeDescriptor(gpuRaw)
	fatal(t, problem)
	_, problem = gpuDescriptor.Function("four_lane")
	fatal(t, problem)
	if launch.AcceleratorRequired([]string{
		"cozy-jobs==0.0.14", "cozy-runtime<1.0.0,>=0.0.34",
	}) {
		t.Fatal("the SDXL TensorFS/NumPy release selected accelerator capacity")
	}
	cpuRaw := []byte(`{"application":"probe:app","entrypoints":[],"format":"cozy.package.descriptor/1","jobs":[{"name":"scan","publishes":false,"request":{"fields":[]},"result":{"fields":[]}}]}`)
	cpuDescriptor, problem := launch.DecodeDescriptor(cpuRaw)
	fatal(t, problem)
	_, problem = cpuDescriptor.Function("scan")
	fatal(t, problem)
	if !launch.AcceleratorRequired([]string{
		"cozy-jobs==0.0.13", "cozy-runtime<1.0.0,>=0.0.33", "msgspec>=0.19",
		"numpy>=1.26", "torch<3,>=2.13",
	}) {
		t.Fatal("the H3 release's exact torch requirement did not select accelerator capacity")
	}
	authoredResources := bytes.Replace(cpuRaw, []byte(`"publishes":false`),
		[]byte(`"publishes":false,"resources":{"gpu_count":1,"requires":"sm90+"}`), 1)
	if _, refusal := launch.DecodeDescriptor(authoredResources); refusal == nil {
		t.Fatal("author-supplied resource requirements were accepted")
	}
}

// TestDynamicProducerOutputContracts is the model-upload dry-run boundary. A generic
// source-derived producer cannot name one topology before it sees the selected source,
// while an H3-style producer that does declare a contract must still match it exactly.
func TestDynamicProducerOutputContracts(t *testing.T) {
	raw := []byte(`{"application":"quantize:app","entrypoints":[],"format":"cozy.package.descriptor/1","jobs":[{"models":[{"class":"QuantizationSource","component_use":{},"path":"produce.models.source","source_profile":"civitai/sdxl/single-file/1","stamps":{}}],"name":"produce","publishes":false,"request":{"fields":[]},"result":{"fields":[]},"weights_outputs":[{"max_bytes":17179869184,"mime_type":"application/vnd.cozy.model-manifest","output_id":"bf16"},{"max_bytes":17179869184,"mime_type":"application/vnd.cozy.model-manifest","output_id":"fp8"},{"max_bytes":17179869184,"mime_type":"application/vnd.cozy.model-manifest","output_id":"mxfp8"}]}]}`)
	descriptor, problem := launch.DecodeDescriptor(raw)
	fatal(t, problem)
	job, problem := descriptor.Function("produce")
	fatal(t, problem)
	if problem := modeltransfer.ValidateProducer("paul/quantize@v1/produce", job); problem != nil {
		t.Fatalf("generic model producer was refused before source-derived contracts exist: %s", problem.Message)
	}

	profiles := map[string]string{"source": "civitai/sdxl/single-file/1"}
	intent := &records.ModelTransferIntent{Kind: "model-upload", Destination: "paul/sdxl",
		Source: "civitai://1", SourceSelection: "sha256:" + strings.Repeat("1", 64),
		SourceProfiles: profiles, Outputs: []records.ModelTransferOutput{
			{Name: "bf16"}, {Name: "fp8"}, {Name: "mxfp8"},
		}}
	spec := orchestrator.Submission{Package: "paul/quantize", Entrypoint: "produce",
		ModelTransfer: intent, ProducerProfiles: profiles, WeightsOutputs: []orchestrator.WeightsOutput{
			{OutputID: "bf16"}, {OutputID: "fp8"}, {OutputID: "mxfp8"},
		}}
	if problem := modeltransfer.ValidateSubmission(spec); problem != nil {
		t.Fatalf("matching open output contracts were refused: %s", problem.Message)
	}

	contract := &records.ModelTransferContract{TopologyDigest: "sha256:" + strings.Repeat("2", 64),
		Encodings: []string{"sha256:" + strings.Repeat("3", 64)}}
	spec.ModelTransfer.Outputs[0].RequiredContract = contract
	if problem := modeltransfer.ValidateSubmission(spec); problem == nil {
		t.Fatal("an intent-authored contract absent from the descriptor was accepted")
	}
	spec.WeightsOutputs[0].RequiredContract = &records.ModelTransferContract{
		TopologyDigest: contract.TopologyDigest, Encodings: append([]string(nil), contract.Encodings...)}
	if problem := modeltransfer.ValidateSubmission(spec); problem != nil {
		t.Fatalf("an exact declared contract was refused: %s", problem.Message)
	}
	spec.WeightsOutputs[0].RequiredContract.TopologyDigest = "sha256:" + strings.Repeat("4", 64)
	if problem := modeltransfer.ValidateSubmission(spec); problem == nil {
		t.Fatal("a descriptor/intent topology mismatch was accepted")
	}
	if problem := modeltransfer.ValidateSubmission(orchestrator.Submission{
		WeightsOutputs: []orchestrator.WeightsOutput{{OutputID: "ordinary-job-output"}},
	}); problem != nil {
		t.Fatalf("ordinary job publication was changed by model-transfer validation: %s", problem.Message)
	}
}
