package producttest

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/assessment"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
)

// These are explicit association fixtures in the real owner database; Runtime's
// native peer tests separately prove where byte receipts and observations originate.
func observedAssessment(t *testing.T, observed bool) (*records.Store, string, assessment.RenderInspection, []byte, []assessment.BoundRender) {
	t.Helper()
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	for _, id := range []string{"assessment-owner", "assessment-render"} {
		fatal(t, store.RecordInstall(records.PackageInstall{ID: id, Package: "local/" + id, Version: "1", SourceKind: "local"}))
	}
	outer := offerChildParent(t, store, recordPrivateTransaction(t, store, "assessment-outer", ""))
	parent, _, problem := store.SubmitChild(records.Request{ID: "observed-parent", IdemKey: "observed-parent", BodyDigest: childDigest("1"), Kind: "job", Package: "local/parent", Entrypoint: "main", InstallID: "assessment-owner", RetainWork: true, Payload: []byte(`{}`), ParentRequestID: outer.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("e"), ChildTargetDigest: childDigest("f"), ChildArtifacts: true, Outputs: "report,workloads"}, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	parent = offerChildParent(t, store, parent)
	binding := records.ChildBinding{ParentInstallID: parent.InstallID, ChildInstallID: "assessment-render", Module: "fixture", Export: "render", Entrypoint: "render"}
	fatal(t, store.RecordChildBindings([]records.ChildBinding{binding}))
	artifacts := map[string]records.ModelArtifact{}
	for index, name := range []string{"reference", "candidate"} {
		manifest := assessmentDigest([]byte(name))
		call := records.NativeCall{ID: "source-" + name, ParentRequestID: parent.ID, CallIndex: int64(index), Kind: "source", Operation: "convert_cozytensors", IntentDigest: assessmentDigest([]byte("source-" + name)), Request: []byte(`{}`)}
		_, _, problem := store.AcceptNativeCall(call, 1, childDigest("1"), "private-boot")
		fatal(t, problem)
		receipt := assessmentJSON(t, map[string]any{"manifest": map[string]any{"length": 100, "sha256": manifest[7:]}, "transaction_id": assessmentDigest([]byte("transaction-" + name))})
		artifact := records.ModelArtifact{ProducerRequestID: call.ID, OutputSlot: "model", Manifest: records.ArtifactObjectRef{Digest: manifest, Length: 100}, TensorFSReceiptDigest: assessmentDigest(receipt)}
		fatal(t, store.CompleteNativeCallAt(call.ID, assessmentJSON(t, artifact), receipt, "private-worker", "private-boot"))
		artifacts[name] = artifact
	}
	payload := []byte(`{"prompt":"fixture","seed":8}`)
	workload := assessmentJSON(t, map[string]any{"entrypoint": "fixture.render", "payloads": []json.RawMessage{payload}, "prompts": []string{"fixture"}, "seeds": []int{8}, "width": 1, "height": 1, "frames": 1, "steps": 1, "fps": 0, "capture": []string{"dit"}, "capture_steps": []int{0}, "checklists": "", "checklist_ids": []string{}})
	workloads := append(append([]byte{'['}, workload...), ']')
	var info assessment.RenderInspection
	info.Subject.Reference = artifacts["reference"].Manifest.Digest
	info.Subject.Candidate = artifacts["candidate"].Manifest.Digest
	info.Subject.Workloads = append(info.Subject.Workloads, struct {
		Entrypoint string `json:"entrypoint"`
		Digest     string `json:"digest"`
	}{"fixture.render", assessmentDigest(workload)})
	info.Subject.Arms = map[string]assessment.RenderArm{}
	info.Environment = map[string]assessment.Environment{}
	for index, name := range []string{"reference", "repeat", "candidate"} {
		model := artifacts["reference"]
		if name == "candidate" {
			model = artifacts[name]
		}
		arguments := assessmentJSON(t, map[string]any{"prompt": "fixture", "seed": 8, "model": model})
		capture := []byte(`{"components":["dit"],"steps":[0]}`)
		intent := assessmentJSON(t, map[string]any{"module": binding.Module, "export": binding.Export, "request": json.RawMessage(arguments), "capture": json.RawMessage(capture)})
		request, _, problem := store.SubmitChild(records.Request{ID: "render-" + name, IdemKey: "render-" + name, BodyDigest: assessmentDigest(arguments), Package: "local/render", Entrypoint: binding.Entrypoint, InstallID: binding.ChildInstallID, Kind: "job", Payload: arguments, Capture: string(capture), Outputs: "image,runtime.capture", ChildArtifacts: true, Models: []records.ModelRef{{Slot: "model", Manifest: model.Manifest.Digest, ManifestLength: 100}}, ParentRequestID: parent.ID, ParentCallIndex: int64(index + 2), ChildIntentDigest: assessmentDigest(intent), ChildTargetDigest: assessmentDigest(assessmentJSON(t, map[string]any{"installation_id": binding.ChildInstallID, "entrypoint": binding.Entrypoint, "module": binding.Module, "export": binding.Export}))}, 1, childDigest("1"), "private-boot", nil)
		fatal(t, problem)
		spec := assessmentDocument(t, map[string]any{"payload_digest": assessmentDigest(arguments), "inputs": []any{map[string]any{"input_id": "model:model", "digest": model.Manifest.Digest, "length": 100, "_format": "cozy.worker.v1.InputBinding/1"}}, "job": map[string]any{"installation_id": binding.ChildInstallID, "_format": "cozy.worker.v1.JobInvocationSpec/1"}, "capture": map[string]any{"components": []string{"dit"}, "steps": []uint32{0}, "_format": "cozy.worker.v1.ActivationCapture/1"}, "_format": "cozy.worker.v1.InvocationSpec/1"})
		ordinal, problem := store.Dispatch(records.Attempt{RequestID: request.ID, InstanceID: "private-worker", SessionID: "private-boot", InvocationDigest: assessmentDigest(spec), InvocationCanonical: spec})
		fatal(t, problem)
		fatal(t, store.OfferDispatch(request.ID, ordinal, "private-boot"))
		mediaDigest := assessmentDigest([]byte("media-" + name))
		captureDigest := assessmentDigest([]byte("capture-content-" + name))
		media := records.ByteOutput{RequestID: request.ID, Attempt: ordinal, OutputID: "image", Digest: mediaDigest, Length: 10, MimeType: "image/png", ProducerRootID: assessmentDigest([]byte("media-root-" + name)), ReceiptDigest: assessmentDigest([]byte("media-receipt-" + name)), ManifestID: assessmentDigest([]byte("media-manifest-" + name)), ManifestLength: 100, ContentBytes: 10}
		captureOutput := records.ByteOutput{RequestID: request.ID, Attempt: ordinal, OutputID: "runtime.capture", Digest: assessmentDigest([]byte("capture-manifest-" + name)), Length: 200, MimeType: "application/vnd.cozy.tree-manifest", ProducerRootID: assessmentDigest([]byte("capture-root-" + name)), ReceiptDigest: assessmentDigest([]byte("capture-receipt-" + name)), ManifestID: assessmentDigest([]byte("capture-manifest-" + name)), ManifestLength: 200, ContentBytes: 30}
		body := map[string]any{"request_id": request.ID, "attempt_ordinal": uint64(ordinal), "invocation_spec_digest": assessmentDigest(spec), "status": 1, "execution_started": true, "output_manifest": map[string]any{"outputs": []any{map[string]any{"output_id": "image", "digest": fixtureDigest(assessmentRawDigest(t, mediaDigest)), "length": 10, "mime_type": "image/png", "native_tree": media.NativeRef(), "_format": "cozy.worker.v1.OutputEntry/1"}, map[string]any{"output_id": "runtime.capture", "digest": fixtureDigest(assessmentRawDigest(t, captureOutput.Digest)), "length": 200, "mime_type": captureOutput.MimeType, "native_tree": captureOutput.NativeRef(), "_format": "cozy.worker.v1.OutputEntry/1"}}, "_format": "cozy.worker.v1.OutputManifest/1"}, "_format": "cozy.worker.v1.AttemptOutcomeBody/1"}
		if observed {
			body["observation"] = map[string]any{"environment": map[string]any{"runtime_version": "fixture-runtime", "accelerator": "CPU", "worker_boot_id": "private-boot", "execution_lane": "eager", "_format": "cozy.worker.v1.ExecutionEnvironment/1"}, "capture": map[string]any{"output_id": "runtime.capture", "content_digest": fixtureDigest(assessmentRawDigest(t, captureDigest)), "_format": "cozy.worker.v1.ActivationCaptureResult/1"}, "_format": "cozy.worker.v1.ExecutionObservation/1"}
		}
		terminal := assessmentDocument(t, body)
		_, problem = store.AcceptTerminal(records.Terminal{RequestID: request.ID, Attempt: ordinal, SessionID: "private-boot", InvocationDigest: assessmentDigest(spec), TerminalID: "out-" + name, TerminalDigest: assessmentDigest(terminal), Status: "SUCCEEDED", Body: terminal, RequestState: "succeeded", ByteOutputs: []records.ByteOutput{media, captureOutput}})
		fatal(t, problem)
		fatal(t, store.Closed(request.ID, ordinal))
		for _, output := range []records.ByteOutput{media, captureOutput} {
			hold, problem := store.ReserveByteResult(request.ID, output)
			fatal(t, problem)
			fatal(t, store.ConfirmNativeArtifact(hold.RetentionID, "private-worker", "private-boot"))
		}
		info.Subject.Arms[name] = assessment.RenderArm{Checkpoint: model.Manifest.Digest, Requests: []string{request.ID}, Media: []string{mediaDigest}, Captures: []string{captureDigest}}
		info.Environment[name] = assessment.Environment{RuntimeVersion: "fixture-runtime", Accelerator: "CPU", WorkerBootID: "private-boot", ExecutionLane: "eager"}
	}
	bound, problem := assessment.VerifyRenderBindings(store, parent.ID, info, workloads)
	fatal(t, problem)
	return store, parent.ID, info, workloads, bound
}
func TestAssessmentRequiresExactObservedEnvironmentCaptureAndHeldOutputs(t *testing.T) {
	store, owner, info, workloads, bound := observedAssessment(t, true)
	holds, problem := assessment.VerifyObservations(store, owner, info, workloads, bound)
	fatal(t, problem)
	if len(holds) != 6 {
		t.Fatal("not every media/capture root was bound")
	}
	for _, field := range []string{"runtime", "image", "accelerator", "driver", "cuda", "lane", "contract", "kernel", "capture"} {
		t.Run(field, func(t *testing.T) {
			raw, _ := json.Marshal(info)
			var changed assessment.RenderInspection
			_ = json.Unmarshal(raw, &changed)
			env := changed.Environment["candidate"]
			switch field {
			case "runtime":
				env.RuntimeVersion = "invented"
			case "image":
				env.WorkerImage = childDigest("a")
			case "accelerator":
				env.Accelerator = "invented"
			case "driver":
				env.Driver = "invented"
			case "cuda":
				env.CUDA = "invented"
			case "lane":
				env.ExecutionLane = "compiled"
			case "contract":
				env.ExecutionContract = childDigest("a")
			case "kernel":
				env.KernelSymbol = "invented"
			case "capture":
				arm := changed.Subject.Arms["candidate"]
				arm.Captures[0] = assessmentDigest([]byte("capture-manifest-candidate"))
				changed.Subject.Arms["candidate"] = arm
			}
			changed.Environment["candidate"] = env
			if _, problem := assessment.VerifyObservations(store, owner, changed, workloads, bound); problem == nil {
				t.Fatal("unobserved claim accepted")
			}
		})
	}
	fatal(t, store.BeginNativeArtifactRelease("render-candidate", false))
	if _, problem := assessment.VerifyObservations(store, owner, info, workloads, bound); problem == nil {
		t.Fatal("released output custody accepted")
	}
}
func TestAssessmentRefusesMissingObservationsInsteadOfManufacturingFacts(t *testing.T) {
	store, owner, info, workloads, bound := observedAssessment(t, false)
	if _, problem := assessment.VerifyObservations(store, owner, info, workloads, bound); problem == nil {
		t.Fatal("missing runtime observation accepted")
	}
}

func assessmentRawDigest(t *testing.T, value string) []byte {
	t.Helper()
	raw, err := canonical.Raw(value)
	must(t, err)
	return raw
}
