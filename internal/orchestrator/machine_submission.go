package orchestrator

import (
	"encoding/base64"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// MachineJobSubmission reuses the ordinary invocation and job-directive codecs.
// The first ordinal is an intake proposal; only Runtime records its attempts.
func MachineJobSubmission(request records.Request, capture localpackage.ExecutionCapture, plan *JobPlan) (*pb.MachineExecutionSubmit, *exit.Error) {
	if !request.IsJob() || plan == nil || plan.DescriptorID != request.PlanID || request.LocalPackageDigest == "" {
		return nil, exit.New(exit.Conflict, "machine job no longer names its captured declaration")
	}
	if len(request.Assets) > 0 || len(request.Models) > 0 || request.Trees != "" || request.ModelTransfer != nil {
		return nil, exit.Named(exit.Structural, "machine_execution.inputs_not_staged", "this input shape has no machine-side staging path yet; execution was not submitted")
	}
	weights, problem := decodeWeightsOutputs(request.WeightsOutputs)
	if problem != nil {
		return nil, problem
	}
	limit := uint64(DefaultMaxOutputMiB) << 20
	outputs := invocationOutputBindings(splitList(request.Outputs), weights, limit)
	access := make([]*pb.OutputAccess, 0, len(outputs))
	for _, output := range outputs {
		access = append(access, &pb.OutputAccess{OutputId: output.OutputId})
	}
	payloadDigest := spellOf(canonical.Digest(request.Payload))
	publication := &pb.PublicationContract{GrantId: home.ScratchRepo(request.Org, request.ID), Outputs: outputs}
	spec := &pb.InvocationSpec{
		DeadlineUnixMs: request.DeadlineUnixMS,
		PayloadDigest:  payloadDigest, Inputs: inputBindings(request, payloadDigest), Outputs: outputs,
		AttentionKernel: request.AttentionKernel,
		Spec:            &pb.InvocationSpec_Job{Job: &pb.JobInvocationSpec{BuildId: request.LocalPackageDigest, JobDescriptorId: request.PlanID, PublicationContract: publication}},
	}
	raw, digest, err := canonical.Identity(spec)
	if err != nil {
		return nil, exit.Internalf("cannot encode machine invocation: %s", err)
	}
	root := *plan
	root.BuildID, root.OrchestrationParent, root.FrozenDirective = request.LocalPackageDigest, nil, nil
	// A captured CPU caller must not occupy the execution lane its managed
	// model children need. Device-bearing roots keep their declared lane.
	root.Orchestration = !root.NeedsAccelerator && len(request.Models) == 0 && len(root.WeightsOutputs) == 0
	if root.Orchestration && root.RSSCap == DefaultJobRSSCap {
		// The legacy local launch ceiling is not an authored memory demand.
		// Let Runtime admit a CPU caller using its own measured host policy.
		root.RSSCap = 0
	}
	directive := jobDirectiveWithLimit(&root, limit)
	return &pb.MachineExecutionSubmit{
		SubmissionId: request.IdemKey, CaptureCanonicalBytes: capture.Canonical, CaptureDigest: capture.Digest,
		PayloadCanonicalBytes: request.Payload, MaxAttempts: uint32(MaxRequeues + 1),
		Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1,
			InvocationSpecCanonicalBytes: raw, InvocationSpecDigest: digest,
			Grant: &pb.DeliveryGrant{InvocationSpecDigest: digest, Inputs: []*pb.InputAccess{{InputId: "payload", Url: "data:application/json;base64," + base64.StdEncoding.EncodeToString(request.Payload)}}, Outputs: access}},
		PreparedState: &pb.DesiredWorkerState{Revision: 1, WireMinor: pb.WireMinor, Posture: pb.Posture_POSTURE_ACCEPTING, Mode: &pb.DesiredWorkerState_Job{Job: directive}},
	}, nil
}
