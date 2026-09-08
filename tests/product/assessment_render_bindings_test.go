package producttest

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/assessment"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func assessmentRecord(t *testing.T, err *exit.Error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
func assessmentJSON(t *testing.T, value any) []byte {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	raw, err = canonical.NormalizeJCS(raw)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
func assessmentDocument(t *testing.T, value proto.Message) []byte {
	t.Helper()
	raw, err := canonical.Bytes(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// This test exercises the real immutable Creator journal and canonical wire
// documents. Its tiny input/output records are association fixtures, not a claim
// that the numerical evaluator or a GPU executed these render requests.
func TestRetainedRenderBindingRejectsChangedWorkloadAndArmEvidence(t *testing.T) {
	st, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	assessmentRecord(t, problem)
	defer st.Close()
	for _, id := range []string{"parent-install", "render-install"} {
		assessmentRecord(t, st.RecordInstall(records.PackageInstall{ID: id, Package: "local/" + id, Version: "1.0.0", SourceKind: "local"}))
	}
	revision := "sha256:" + strings.Repeat("a", 64)
	binding := records.ChildBinding{ParentInstallID: "parent-install", ChildInstallID: "render-install", InterfaceDigest: "sha256:" + strings.Repeat("b", 64), Module: "proof", Export: "render", Entrypoint: "render", LocalRevisionDigest: revision}
	assessmentRecord(t, st.RecordChildBindings([]records.ChildBinding{binding}))
	parent, _, problem := st.Submit(records.Request{ID: "parent", IdemKey: "parent", BodyDigest: assessmentDigest([]byte("parent")), Kind: "job", Package: "local/parent", Entrypoint: "main", InstallID: "parent-install", RetainWork: true, Payload: []byte(`{}`)})
	assessmentRecord(t, problem)
	assessmentRecord(t, st.SpawnWorker(records.WorkerProcess{InstanceID: "instance", Package: "local/render", WorkerID: "worker", Devices: []string{"cpu"}}))
	payload := []byte(`{"prompt":"fixture","seed":8}`)
	workload := assessmentJSON(t, map[string]any{"entrypoint": "proof.render", "payloads": []json.RawMessage{payload}, "prompts": []string{"fixture"}, "seeds": []int{8}, "width": 1, "height": 1, "frames": 1, "steps": 1, "fps": 0, "capture": []string{}, "capture_steps": []int{0}, "checklists": "", "checklist_ids": []string{}})
	workloads := append(append([]byte{'['}, workload...), ']')
	var info assessment.RenderInspection
	info.Subject.Candidate = assessmentDigest([]byte("candidate checkpoint"))
	info.Subject.Reference = assessmentDigest([]byte("reference checkpoint"))
	info.Subject.Workloads = append(info.Subject.Workloads, struct {
		Entrypoint string `json:"entrypoint"`
		Digest     string `json:"digest"`
	}{"proof.render", assessmentDigest(workload)})
	info.Subject.Arms = map[string]assessment.RenderArm{}
	info.Environment = map[string]assessment.Environment{}
	for i, name := range []string{"reference", "repeat", "candidate"} {
		id := "render-" + name
		checkpoint := info.Subject.Reference
		if name == "candidate" {
			checkpoint = info.Subject.Candidate
		}
		intent := assessmentJSON(t, map[string]any{"interface_digest": binding.InterfaceDigest, "module": binding.Module, "export": binding.Export, "request": json.RawMessage(payload)})
		request, _, problem := st.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: assessmentDigest([]byte(id)), Kind: "job", Package: "local/render-install", Entrypoint: "render", InstallID: binding.ChildInstallID, RetainWork: true, ParentRequestID: parent.ID, ParentCallIndex: int64(i), ChildIntentDigest: assessmentDigest(intent), ChildTargetDigest: revision, Payload: payload, Models: []records.ModelRef{{Slot: "model", Manifest: checkpoint}}})
		assessmentRecord(t, problem)
		spec := assessmentDocument(t, &pb.InvocationSpec{PayloadDigest: assessmentDigest(payload), Inputs: []*pb.InputBinding{{InputId: "model:model", Digest: checkpoint, Length: 1}}, Spec: &pb.InvocationSpec_Job{Job: &pb.JobInvocationSpec{BuildId: revision}}})
		ordinal, problem := st.Dispatch(records.Attempt{RequestID: request.ID, InstanceID: "instance", SessionID: "boot", InvocationDigest: assessmentDigest(spec), InvocationCanonical: spec})
		assessmentRecord(t, problem)
		assessmentRecord(t, st.OfferDispatch(id, ordinal, "boot"))
		frame := image.NewNRGBA(image.Rect(0, 0, 1, 1))
		frame.SetNRGBA(0, 0, color.NRGBA{R: uint8(i), A: 255})
		var imageBytes bytes.Buffer
		if err := png.Encode(&imageBytes, frame); err != nil {
			t.Fatal(err)
		}
		media := imageBytes.Bytes()
		mediaDigest := canonical.Digest(media)
		terminal := assessmentDocument(t, &pb.AttemptOutcomeBody{RequestId: id, AttemptOrdinal: uint64(ordinal), InvocationSpecDigest: assessmentDigest(spec), Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, ExecutionStarted: true, OutputManifest: &pb.OutputManifest{Outputs: []*pb.OutputEntry{{OutputId: "image", Digest: mediaDigest, Length: uint64(len(media)), MimeType: "image/png"}}}})
		_, problem = st.AcceptTerminal(records.Terminal{RequestID: id, Attempt: ordinal, SessionID: "boot", InvocationDigest: assessmentDigest(spec), TerminalID: "out-" + id, TerminalDigest: assessmentDigest(terminal), Status: "SUCCEEDED", Body: terminal, RequestState: "succeeded"})
		assessmentRecord(t, problem)
		info.Subject.Arms[name] = assessment.RenderArm{Checkpoint: checkpoint, Requests: []string{id}, Media: []string{assessmentDigest(media)}}
		info.Environment[name] = assessment.Environment{WorkerBootID: "boot"}
	}
	assertRefused := func(label string, altered assessment.RenderInspection, preimages []byte) {
		t.Helper()
		if _, problem := assessment.VerifyRenderBindings(st, parent.ID, altered, preimages); problem == nil {
			t.Fatalf("%s accepted", label)
		}
	}
	bound, problem := assessment.VerifyRenderBindings(st, parent.ID, info, workloads)
	assessmentRecord(t, problem)
	if len(bound) != 3 {
		t.Fatalf("bound %d renders", len(bound))
	}
	clone := func() assessment.RenderInspection {
		raw, _ := json.Marshal(info)
		var value assessment.RenderInspection
		_ = json.Unmarshal(raw, &value)
		return value
	}
	for _, field := range []string{"media", "checkpoint", "boot", "repeat_request", "request_order"} {
		changed := clone()
		arm := changed.Subject.Arms["candidate"]
		switch field {
		case "media":
			arm.Media[0] = assessmentDigest([]byte("another image"))
		case "checkpoint":
			arm.Checkpoint = info.Subject.Reference
		case "boot":
			changed.Environment["candidate"] = assessment.Environment{WorkerBootID: "other-boot"}
		case "repeat_request":
			repeat := changed.Subject.Arms["repeat"]
			repeat.Requests = changed.Subject.Arms["reference"].Requests
			changed.Subject.Arms["repeat"] = repeat
		case "request_order":
			arm.Requests = changed.Subject.Arms["reference"].Requests
		}
		changed.Subject.Arms["candidate"] = arm
		assertRefused(field, changed, workloads)
	}
	changedWorkload := []byte(strings.Replace(string(workload), `"seed":8`, `"seed":9`, 1))
	changed := clone()
	changed.Subject.Workloads[0].Digest = assessmentDigest(changedWorkload)
	assertRefused("changed workload payload despite a self-consistent preimage hash", changed, append(append([]byte{'['}, changedWorkload...), ']'))
	changed = clone()
	changed.Subject.Workloads[0].Digest = assessmentDigest([]byte("another workload"))
	assertRefused("wrong workload digest", changed, workloads)
	changed = clone()
	changed.Subject.Workloads[0].Entrypoint = "other.render"
	assertRefused("changed captured callable", changed, workloads)
	assertRefused("noncanonical workload", info, append([]byte(" "), workloads...))
	// Retaining the association is sufficient for this narrow checker; deliberately
	// no environment/capture completeness assertion or Hub mutation follows it.
	for _, row := range bound {
		if row.OutcomeDigest == "" || row.InvocationDigest == "" {
			t.Fatal(fmt.Sprint(row))
		}
	}
}

func assessmentDigest(raw []byte) string {
	value, _ := canonical.Spell(canonical.Digest(raw))
	return value
}

// The canonical reader remains the numerical/schema authority; Creator consumes
// its exact association projection and binds the original artifact identity.
var assessmentEvaluatorPath = flag.String("assessment-v3-evaluator", "", "installed canonical @3 report reader for association proof")

func TestAssessmentV3InspectorIdentityBoundary(t *testing.T) {
	evaluator := *assessmentEvaluatorPath
	if evaluator == "" {
		t.Skip("requires the installed cozy-eval @3 reader")
	}
	report, err := os.ReadFile("testdata/assessment-v3.json")
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", filepath.Dir(evaluator))
	inspected, problem := assessment.Inspect(t.Context(), report, []string{"PYTHONNOUSERSITE=1"})
	assessmentRecord(t, problem)
	info, problem := assessment.ReadRenderInspection(report, inspected)
	assessmentRecord(t, problem)
	if info.Report.Digest != assessmentDigest(report) || info.Verdict != "pass" || len(info.Subject.Arms) != 3 {
		t.Fatal("reader association changed")
	}
	for _, raw := range [][]byte{nil, append([]byte(" "), report...), report[:len(report)-1]} {
		if _, problem := assessment.ReadRenderInspection(raw, inspected); problem == nil {
			t.Fatal("changed report bytes accepted")
		}
	}
	info.Schema = "cozy-eval/report-inspection@2"
	if _, problem := assessment.ReadRenderInspection(report, assessmentJSON(t, info)); problem == nil {
		t.Fatal("retired inspector accepted")
	}
}
