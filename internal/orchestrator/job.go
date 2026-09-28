package orchestrator

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/home"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// THE JOB BRANCH (cl-004). A job is an ATTEMPT CLASS on this one orchestrator, not a
// second scheduler: it reuses the request row, the ordinal law, the dispatch queue, the
// terminal transaction, the requeue projection and the event plane unchanged. What
// differs is exactly what cr-009 says differs — a `JobDirective` instead of a
// `ServingDirective`, a `JobExecutionSpec` instead of a `ServingExecutionSpec`, job
// capacity instead of serving capacity, and a grant that writes into a DURABLE
// PUBLICATION ROOT instead of the package's result store.
//
// The two things this file owns that the serving lane has no version of:
//
//  1. THE PUBLICATION ROOT. A bounded job is reclaimed at its terminal; its worker root
//     and its attempt spool go with it. So the destinations a job is granted are not
//     under either — they are under `<COZY_HOME>/publications/<org>/_job-<request-id>`,
//     which nothing in the reclaim path touches. `publicationDest` is the fence: a
//     destination that resolves outside that root is refused BEFORE the attempt is
//     dispatched, so the escape is unrepresentable rather than defended against.
//  2. THE DURABLE CHECKPOINT EXCHANGE's orchestrator half. The worker has already made
//     the save durable in its own journal; this side records the identity and answers
//     with a receipt. It is a SECOND observation, never a second authority (law 9).

// JobPlan is cozy-creator's LOCAL JOB PLAN RECORD: the resolution of one
// `job_descriptor_id` against this machine. Same seam as `Binding`, one lane over — the
// orchestrator names the job by DIGEST and never ships a path, and the worker resolves
// that digest against a small local file under `<worker home>/job-plans/`.
type JobPlan struct {
	Function     string
	DescriptorID string
	// InstallationID names a retained installed resource, never package bytes.
	InstallationID string
	// Outputs are the job's declared asset result field paths — the output ids the
	// publication grant names, one destination each. Grants mint off the DECLARATION.
	Outputs []string
	// WeightsOutputs is the explicit WeightsSink subset. Empty keeps an ordinary asset
	// job on the existing publication path; non-empty is the M0 weights-only contract.
	WeightsOutputs []WeightsOutput
	// Record is the closed key set `plan.py::JobBinding.read` accepts. An unknown key is
	// a refusal at the worker, which is what makes "closed at both ends" a fact.
	Record           map[string]any
	RSSCap           int64
	NeedsAccelerator bool
	// AcceleratorDeclared: the job's interface names its device; capture does not reclassify it.
	AcceleratorDeclared bool
	// CPUSlotModelInputs: the machine's Runtime puts a device-less root holding Model inputs
	// or weights outputs on its CPU slot (MachineExecutionWorkspace.cpu_slot_model_inputs).
	CPUSlotModelInputs  bool
	Orchestration       bool
	OrchestrationParent *JobPlan
	FrozenDirective     *pb.JobDirective
}

const DefaultJobRSSCap int64 = 8 << 30

func jobDirectiveWithLimit(plan *JobPlan, outputLimit uint64) *pb.JobDirective {
	if plan.FrozenDirective != nil {
		return proto.Clone(plan.FrozenDirective).(*pb.JobDirective)
	}
	directive := &pb.JobDirective{
		InstallationId:  plan.InstallationID,
		JobDescriptorId: plan.DescriptorID,
		ResourceCaps: &pb.ResourceCaps{
			DeviceRequired: gpuCountOf(plan) > 0,
			MaxRssBytes:    uint64(plan.RSSCap),
		},
		// The DIRECTIVE's publication contract is the worker-level authorization;
		// the per-attempt one rides the InvocationSpec, because a worker that
		// drains a queue publishes into a different scratch repo per request.
		PublicationContract: &pb.PublicationContract{
			GrantId: home.ScratchRepo("local", "queue"),
			Outputs: invocationOutputBindings(plan.Outputs, plan.WeightsOutputs, outputLimit),
		},
		// TERMINAL AND RECLAIM, everywhere. A job worker is one immutable build
		// running one bounded attempt; deep queueing is the orchestrator's dispatch
		// queue, not a warm worker (audit-adopted, 2026-08-26).
		ReclaimOnTerminal: true,
		DeviceCount:       uint32(gpuCountOf(plan)),
		Orchestration:     plan.Orchestration,
	}
	if plan.OrchestrationParent != nil {
		directive.OrchestrationParent = jobDirectiveWithLimit(plan.OrchestrationParent, outputLimit)
	}
	return directive
}

// gpuCountOf is the job's device FLOOR: how many accelerators the attempt may not start
// without. It is 0 or 1 and stays 0 or 1 on a wide pod, deliberately (cl-179).
//
// A width is not a floor. `JobDirective.device_count` is, in the runtime's own words, a
// floor and never concurrency; a job is one bounded attempt over a derive-only view of its model
// inputs — it shards nothing, so a job on a four-card pod needs one card and would be
// refused by a floor of four. Sequence parallelism is Runtime's grant to a serving call,
// not a job resource cap. What keeps a job off a wide pod is the selection
// side, where a wide product is excluded for a job outright rather than bought and idled.
func gpuCountOf(p *JobPlan) int64 {
	if p.NeedsAccelerator {
		return 1
	}
	return 0
}

// ------------------------------------------------------------------ the publication root

func splitList(joined string) []string {
	out := []string{}
	for _, v := range strings.Split(joined, ",") {
		if v = strings.TrimSpace(v); v != "" {
			out = append(out, v)
		}
	}
	return out
}

// ------------------------------------------------------------------ the publication

// THE CHECKPOINT HALF IS A RUNTIME-BORDER SEAM, NOT A CREATOR PROTOCOL.
//
// A job that produces canonical bytes writes them through the ONE TensorFS border and the
// PUBLICATION TRANSACTION is the runtime's (cr-005/cr-009, jobs.md). What this orchestrator
// owes is the half it owns: validate ONE typed durable publication receipt the runtime
// produced, root what that receipt names, and record the catalog projection.
//
// That receipt does not exist yet at the runtime border, so NOTHING here interprets a job's
// result to guess at one. An earlier version of this file read an author result member
// named `published` and manufactured TensorFS roots from it: that is a second, untyped
// publication protocol invented on the wrong side of the boundary, and it was wrong in
// three separate ways at once — it read an author-chosen field name as if it were a
// contract, it registered several snapshots under ONE root name (registration REPLACES the
// prior mapping — `meta.rs::register_root` — so all but the last were silently lost), and
// it made cozy-creator the producer of a fact TensorFS and cozy-runtime own. Deleted whole
// (audit-adopted, 2026-08-26).
//
// Until the typed receipt lands, a job's publication is its ASSET plane: the landed writes
// under the durable publication root, recorded in the one lifecycle authority.

// ------------------------------------------------------------------ the checkpoint lane
