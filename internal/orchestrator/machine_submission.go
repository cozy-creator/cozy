package orchestrator

import (
	"encoding/base64"
	"encoding/json"
	"sort"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// MachineJobSubmission reuses the ordinary invocation and job-directive codecs.
// The first ordinal is an intake proposal; only Runtime records its attempts.
func MachineJobSubmission(request records.Request, capture localpackage.ExecutionCapture, plan *JobPlan, byteInputs []*pb.InputAccess) (*pb.MachineExecutionSubmit, *exit.Error) {
	if !request.IsJob() || plan == nil || plan.DescriptorID != request.PlanID {
		return nil, exit.New(exit.Conflict, "machine job no longer names its captured declaration")
	}
	rootCapture, installationID, problem := machineRoot(request, capture, byteInputs)
	if problem != nil {
		return nil, problem
	}
	if request.LocalInstallationID == "" && plan.InstallationID != installationID {
		return nil, exit.New(exit.Conflict, "machine job changed its selected installation")
	}
	weights, problem := decodeWeightsOutputs(request.WeightsOutputs)
	if problem != nil {
		return nil, problem
	}
	limit := uint64(DefaultMaxOutputMiB) << 20
	outputs := invocationOutputBindings(splitList(request.Outputs), weights, limit)
	if len(jobModels(request)) != len(request.OwnModels()) {
		return nil, exit.Named(exit.Structural, "machine_execution.model_identity_missing", "root Model inputs require exact manifest identities and lengths")
	}
	models := jobModels(request)
	modelInputs := modelAccess(request)
	for index, model := range models {
		if model.CatalogRepository == "" {
			continue
		}
		if model.CatalogRepository != model.Model {
			return nil, exit.Named(exit.Structural, "machine_execution.model_source_changed", "catalog Model input differs from its resolved repository")
		}
		modelInputs[index].CatalogModel = &pb.CatalogModelSource{Repository: model.CatalogRepository}
	}
	payloadDigest := spellOf(canonical.Digest(request.Payload))
	publication := &pb.PublicationContract{GrantId: home.ScratchRepo(request.Org, request.ID), Outputs: outputs}
	spec := &pb.InvocationSpec{
		DeadlineUnixMs: request.DeadlineUnixMS, InstallationId: installationID,
		PayloadDigest: payloadDigest, Inputs: inputBindings(request, payloadDigest), Outputs: outputs,
		AttentionKernel: request.AttentionKernel,
		Spec:            &pb.InvocationSpec_Job{Job: &pb.JobInvocationSpec{InstallationId: installationID, JobDescriptorId: request.PlanID, PublicationContract: publication}},
	}
	root := *plan
	root.InstallationID, root.OrchestrationParent, root.FrozenDirective = installationID, nil, nil
	// The frozen capture also contains self-serving exports which need not have
	// separate install binding rows. Apply the same composition-parent rule to
	// those actual capabilities, rather than inferring a GPU from package imports.
	hasCallees, isCallee := false, false
	for _, binding := range rootCapture.List("bindings") {
		if binding.Str("caller_installation_id") == installationID {
			hasCallees = true
		}
		if binding.Str("callee_installation_id") == installationID && binding.Str("entrypoint") == request.Entrypoint {
			isCallee = true
		}
	}
	if !root.AcceleratorDeclared && hasCallees && !isCallee && len(request.OwnModels()) == 0 && len(root.WeightsOutputs) == 0 {
		root.NeedsAccelerator = false
	}
	// A device-less root takes the machine's CPU slot, so the managed model children it
	// calls may hold devices; its own Models are derive-only views. A Runtime that does not
	// report `cpu_slot_model_inputs` puts a root holding Models or weights on its device
	// slot and refuses it as orchestration, so that root keeps the device lane there.
	root.Orchestration = !root.NeedsAccelerator &&
		(root.CPUSlotModelInputs || len(request.OwnModels()) == 0 && len(root.WeightsOutputs) == 0)
	if root.Orchestration && root.RSSCap == DefaultJobRSSCap {
		// The legacy local launch ceiling is not an authored memory demand.
		// Let Runtime admit a CPU caller using its own measured host policy.
		root.RSSCap = 0
	}
	prepared := &pb.DesiredWorkerState{Mode: &pb.DesiredWorkerState_Job{Job: jobDirectiveWithLimit(&root, limit)}}
	return machineSubmission(request, capture, spec, append(modelInputs, byteInputs...), prepared, "")
}

// MachineServingSubmission submits one inference root. The desired state is the placement
// set the machine prepared, and the offer names the placement holding the root. No device
// pin is sent: Runtime chooses the devices, and the group width unless the request's
// models name an exact GPU count.
func MachineServingSubmission(request records.Request, capture localpackage.ExecutionCapture, prepared *pb.DesiredPlacementSet, byteInputs []*pb.InputAccess) (*pb.MachineExecutionSubmit, *exit.Error) {
	if request.IsJob() || prepared == nil {
		return nil, exit.New(exit.Conflict, "machine inference needs its prepared placement")
	}
	_, installationID, problem := machineRoot(request, capture, byteInputs)
	if problem != nil {
		return nil, problem
	}
	placement, planID, problem := ServingPlacement(prepared.PlacementSetCanonicalBytes, installationID, request)
	if problem != nil {
		return nil, problem
	}
	if planID != request.PlanID {
		return nil, exit.New(exit.Conflict, "machine inference no longer names its prepared binding")
	}
	payloadDigest := spellOf(canonical.Digest(request.Payload))
	spec := &pb.InvocationSpec{
		DeadlineUnixMs: request.DeadlineUnixMS, InstallationId: installationID,
		PayloadDigest: payloadDigest, Inputs: inputBindings(request, payloadDigest),
		Outputs:         invocationOutputBindings(splitList(request.Outputs), nil, uint64(DefaultMaxOutputMiB)<<20),
		AttentionKernel: request.AttentionKernel,
		Spec: &pb.InvocationSpec_Serving{Serving: &pb.ServingInvocationSpec{
			EntrypointBindingDigest: planID, BindingsDigest: placement.Str("bindings_digest"), AttemptBindingId: planID,
		}},
	}
	if request.Capture != "" {
		spec.Capture = &pb.ActivationCapture{}
		if err := json.Unmarshal([]byte(request.Capture), spec.Capture); err != nil {
			return nil, exit.New(exit.Validation, "recorded capture options are invalid")
		}
	}
	state := &pb.DesiredWorkerState{Mode: &pb.DesiredWorkerState_PlacementSet{PlacementSet: &pb.DesiredPlacementSet{
		PlacementSetDigest: prepared.PlacementSetDigest, PlacementSetCanonicalBytes: prepared.PlacementSetCanonicalBytes,
		ExecutionGpus: uint32(records.Width(request.Models, 0)),
	}}}
	return machineSubmission(request, capture, spec, byteInputs, state, placement.Str("placement_id"))
}

// ServingPlacement is the prepared placement installing `installationID` and the binding it
// authored for the request's callable, which must serve the request's exact selection.
func ServingPlacement(setBytes []byte, installationID string, request records.Request) (canonical.Doc, string, *exit.Error) {
	doc, err := canonical.Read(setBytes, &pb.PlacementSet{})
	if err != nil {
		return nil, "", exit.New(exit.Conflict, "machine preparation returned an invalid placement document")
	}
	for _, row := range doc.List("placements") {
		if row.Str("installation_id") != installationID {
			continue
		}
		if problem := requireAdapterEcho(request.Models, placementModels(request.Package, row)); problem != nil {
			return nil, "", problem
		}
		if planID, ok := entrypointServes(row, logicalOf(request)); ok && validDigest(planID) {
			return row, planID, nil
		}
	}
	return nil, "", exit.Named(exit.Conflict, "machine_execution.placement_absent",
		"the prepared placement does not bind %s to its selected models", request.Entrypoint)
}

// machineRoot checks the parts every root submission shares: the captured root
// installation, and one native receipt per staged root byte input.
func machineRoot(request records.Request, capture localpackage.ExecutionCapture, byteInputs []*pb.InputAccess) (canonical.Doc, string, *exit.Error) {
	rootCapture, err := canonical.Read(capture.Canonical, &pb.MachineExecutionCapture{})
	if err != nil {
		return nil, "", exit.New(exit.Conflict, "machine execution has no canonical capture")
	}
	installationID := rootCapture.Str("root_installation_id")
	if installationID == "" || request.LocalInstallationID != "" && request.LocalInstallationID != installationID {
		return nil, "", exit.New(exit.Conflict, "machine execution changed its selected installation")
	}
	if request.Trees != "" || request.ModelTransfer != nil {
		return nil, "", exit.Named(exit.Structural, "machine_execution.inputs_not_staged", "this input shape has no machine-side staging path yet; execution was not submitted")
	}
	if len(byteInputs) != len(request.Assets) {
		return nil, "", exit.Named(exit.Structural, "machine_execution.inputs_not_staged", "root bytes have no exact native input receipt")
	}
	for index, asset := range request.Assets {
		input := byteInputs[index]
		if input == nil || input.InputId != asset.FieldPath || input.Url != "" || input.NativeTree == nil || records.ValidateByteRef(input.NativeTree.Source) != nil || input.NativeTree.RetentionId == "" {
			return nil, "", exit.New(exit.Conflict, "root byte grant changed its staged field or native receipt")
		}
	}
	return rootCapture, installationID, nil
}

func machineSubmission(request records.Request, capture localpackage.ExecutionCapture, spec *pb.InvocationSpec,
	inputs []*pb.InputAccess, prepared *pb.DesiredWorkerState, placementID string) (*pb.MachineExecutionSubmit, *exit.Error) {
	raw, digest, err := canonical.Identity(spec)
	if err != nil {
		return nil, exit.Internalf("cannot encode machine invocation: %s", err)
	}
	access := make([]*pb.OutputAccess, 0, len(spec.Outputs))
	for _, output := range spec.Outputs {
		access = append(access, &pb.OutputAccess{OutputId: output.OutputId})
	}
	inputs = append([]*pb.InputAccess{{InputId: "payload", Url: "data:application/json;base64," + base64.StdEncoding.EncodeToString(request.Payload)}}, inputs...)
	sort.Slice(inputs, func(i, j int) bool { return inputs[i].InputId < inputs[j].InputId })
	prepared.Revision, prepared.WireMinor, prepared.Posture = 1, pb.WireMinor, pb.Posture_POSTURE_ACCEPTING
	return &pb.MachineExecutionSubmit{
		SubmissionId: records.MachineSubmissionID(request.IdemKey), CaptureCanonicalBytes: capture.Canonical, CaptureDigest: capture.Digest,
		PayloadCanonicalBytes: request.Payload, MaxAttempts: uint32(MaxRequeues + 1),
		Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, PlacementId: placementID,
			InvocationSpecCanonicalBytes: raw, InvocationSpecDigest: digest,
			Grant: &pb.DeliveryGrant{InvocationSpecDigest: digest, Inputs: inputs, Outputs: access}},
		PreparedState: prepared,
	}, nil
}
