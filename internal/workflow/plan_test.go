package workflow

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy-creator-v2/internal/records"
)

const testDigest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func testPlan(steps int) Plan {
	out := Plan{Format: PlanFormat, CreativePlanDigest: testDigest}
	for index := 0; index < steps; index++ {
		out.Steps = append(out.Steps, Step{
			Endpoint: "org/ep", EndpointReleaseID: "org/ep@1.0.0",
			Entrypoint: "run", EntrypointBindingPlanID: testDigest,
			PayloadBase64: base64.StdEncoding.EncodeToString([]byte(`{}`)),
			Outputs:       []string{"video"},
		})
	}
	return out
}

func TestMaterializeBackwardOutputBinding(t *testing.T) {
	plan := testPlan(2)
	plan.Steps[1].PayloadBase64 = base64.StdEncoding.EncodeToString([]byte(`{"first_frame":""}`))
	plan.Steps[1].Bindings = []OutputBinding{{FieldPath: "first_frame", PriorStep: 1,
		OutputName: "video", ExpectedMediaKind: "video"}}
	materialized, _, _, resolved, problem := MaterializeStep(&plan, 2, nil, []ResolvedOutput{{
		Binding: plan.Steps[1].Bindings[0], RequestID: "req-one", Attempt: 3,
		Output: records.Output{OutputID: "video", Digest: testDigest, Length: 99,
			MimeType: "video/mp4", Path: "/private/output"},
	}})
	if problem != nil || len(materialized.Assets) != 1 || materialized.Assets[0].Order != 0 ||
		len(resolved) != 1 || resolved[0].PriorAttempt != 3 {
		t.Fatalf("materialized=%#v resolved=%#v problem=%v", materialized, resolved, problem)
	}
	payload, _ := base64.StdEncoding.DecodeString(materialized.PayloadBase64)
	if !strings.Contains(string(payload), testDigest) || strings.Contains(string(payload), "/private") {
		t.Fatalf("payload leaks or misses identity: %s", payload)
	}
	wrong := plan.Steps[1].Bindings[0]
	wrong.ExpectedMediaKind = "image"
	plan.Steps[1].Bindings[0] = wrong
	if _, _, _, _, problem := MaterializeStep(&plan, 2, nil, []ResolvedOutput{{
		Binding: wrong, RequestID: "req-one", Attempt: 3,
		Output: records.Output{OutputID: "video", Digest: testDigest, Length: 99,
			MimeType: "video/mp4"},
	}}); problem == nil || problem.ErrName() != "workflow_output_kind" {
		t.Fatalf("kind mismatch: %#v", problem)
	}
}

func encodePlan(t *testing.T, plan Plan) []byte {
	t.Helper()
	data, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestPlanIdentityIgnoresPresentation(t *testing.T) {
	plan := testPlan(2)
	one, canonicalOne, digestOne, problem := DecodePlan(encodePlan(t, plan))
	if problem != nil || one == nil {
		t.Fatalf("first decode: %v", problem)
	}
	compact, _ := json.Marshal(plan)
	_, canonicalTwo, digestTwo, problem := DecodePlan(compact)
	if problem != nil || digestOne != digestTwo || string(canonicalOne) != string(canonicalTwo) {
		t.Fatalf("presentation moved identity: %v %s %s", problem, digestOne, digestTwo)
	}
}

func TestPlanRefusesBoundsAndForwardEdges(t *testing.T) {
	tooMany := testPlan(17)
	if _, _, _, problem := DecodePlan(encodePlan(t, tooMany)); problem == nil ||
		problem.ErrName() != "workflow_step_count" {
		t.Fatalf("17 steps: %#v", problem)
	}
	forward := testPlan(2)
	forward.Steps[0].Bindings = []OutputBinding{{FieldPath: "image", PriorStep: 2,
		OutputName: "video", ExpectedMediaKind: "video"}}
	if _, _, _, problem := DecodePlan(encodePlan(t, forward)); problem == nil ||
		problem.ErrName() != "workflow_step_invalid" {
		t.Fatalf("forward edge: %#v", problem)
	}
	mutable := testPlan(1)
	mutable.Steps[0].EndpointReleaseID = "org/ep@latest"
	if _, _, _, problem := DecodePlan(encodePlan(t, mutable)); problem == nil {
		t.Fatal("mutable release accepted")
	}
}

func TestPlanRefusesUnknownFields(t *testing.T) {
	data := encodePlan(t, testPlan(1))
	data = []byte(strings.Replace(string(data), `"steps":`, `"smuggled":true,"steps":`, 1))
	if _, _, _, problem := DecodePlan(data); problem == nil ||
		problem.ErrName() != "workflow_plan_malformed" {
		t.Fatalf("unknown field: %#v", problem)
	}
}

func TestPlanRefusesUnboundedAndReservedAssetPaths(t *testing.T) {
	for _, path := range []string{"references.1000000000.image", "payload", "tree:dataset"} {
		plan := testPlan(1)
		plan.Steps[0].Assets = []AssetClaim{{FieldPath: path, Digest: testDigest,
			Length: 1, MediaType: "image/png", Kind: "image"}}
		if _, _, _, problem := DecodePlan(encodePlan(t, plan)); problem == nil {
			t.Fatalf("path %q accepted", path)
		}
	}
}
