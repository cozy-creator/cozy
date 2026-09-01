package orchestrator

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// THE JOB BRANCH (cl-004). A job is an ATTEMPT CLASS on this one orchestrator, not a
// second scheduler: it reuses the request row, the ordinal law, the dispatch queue, the
// terminal transaction, the requeue projection and the event plane unchanged. What
// differs is exactly what cr-009 says differs — a `JobDirective` instead of a
// `ServingDirective`, a `JobExecutionSpec` instead of a `ServingExecutionSpec`, job
// capacity instead of serving capacity, and a grant that writes into a DURABLE
// PUBLICATION ROOT instead of a disposable attempt directory.
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
	// Outputs are the job's declared asset result field paths — the output ids the
	// publication grant names, one destination each. Grants mint off the DECLARATION.
	Outputs []string
	// WeightsOutputs is the explicit WeightsSink subset. Empty keeps an ordinary asset
	// job on the existing publication path; non-empty is the M0 weights-only contract.
	WeightsOutputs []WeightsOutput
	// Record is the closed key set `plan.py::JobBinding.read` accepts. An unknown key is
	// a refusal at the worker, which is what makes "closed at both ends" a fact.
	Record   map[string]any
	RSSCap   int64
	GPUCount int64
}

const DefaultJobRSSCap int64 = 8 << 30

// stageJobPlans writes one job plan record per declared job into the worker's own home.
// The file name is the descriptor id's hex, which is how the supervisor finds it.
func stageJobPlans(workerHome string, plans []*JobPlan) *exit.Error {
	dir := filepath.Join(workerHome, "job-plans")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return exit.Internalf("cannot create the job plan directory %s: %s", dir, err)
	}
	for _, p := range plans {
		data, err := json.MarshalIndent(p.Record, "", "  ")
		if err != nil {
			return exit.Internalf("cannot render the job plan record: %s", err)
		}
		name := strings.TrimPrefix(p.DescriptorID, "sha256:") + ".json"
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o644); err != nil {
			return exit.Internalf("cannot stage the job plan record: %s", err)
		}
	}
	return nil
}

// sendJobDirective issues the JOB-mode full-replace desired state. A DesiredWorkerState is
// a discriminated COMPLETE replacement carrying its own `mode` oneof, so a worker is in
// exactly ONE mode until the next revision — the branch is stated here rather than
// inferred from which field happens to be populated (cr-009's `fabd6fc` lesson). Job mode
// hosts no PLACEMENT at all: there is no set, no serving axis, and no placement_id on its
// attempts.
func (c *Orchestrator) sendJobDirective(s *session, w *worker) *exit.Error {
	plan := w.spec.Placement.Jobs[0]
	rev := c.nextRevision()
	c.mu.Lock()
	w.revision = rev
	c.mu.Unlock()
	d := &pb.DesiredWorkerState{
		Revision: rev, WireMinor: pb.WireMinor,
		Posture: pb.Posture_POSTURE_ACCEPTING,
		Mode: &pb.DesiredWorkerState_Job{Job: &pb.JobDirective{
			BuildId:         w.spec.Placement.PackageRevisionDigest,
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
				Outputs: invocationOutputBindings(plan.Outputs, plan.WeightsOutputs, c.maxOutputBytes()),
			},
			// TERMINAL AND RECLAIM, everywhere. A job worker is one immutable build
			// running one bounded attempt; deep queueing is the orchestrator's dispatch
			// queue, not a warm worker (audit-adopted, 2026-08-26).
			ReclaimOnTerminal: true,
			DeviceCount:       uint32(gpuCountOf(plan)),
		}},
	}
	d.RecordOwnerEpoch, d.ControlStreamGeneration, d.WorkerBootId = recordOwnerEpoch, s.generation, s.bootID
	if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_DesiredState{DesiredState: d}}) {
		return exit.Unavailablef("worker %s control stream closed before job directive send", w.instanceID)
	}
	c.logf("DesiredWorkerState revision=%d posture=accepting JOB %s (%s) -> %s",
		rev, plan.Function, shortDigest(plan.DescriptorID), s.bootID)
	return nil
}

func (c *Orchestrator) ConvergeRemoteJob(instanceID string, spec WorkerLaunchSpec) *exit.Error {
	if spec.Connection == nil || !spec.IsJob() || len(spec.Placement.Jobs) != 1 {
		return exit.Internalf("remote job convergence requires one attached job spec")
	}
	c.mu.Lock()
	w := c.workers[instanceID]
	var s *session
	if w != nil {
		s = c.sessions[w.bootID]
		w.spec = spec
		w.planIDs = []string{spec.Placement.Jobs[0].DescriptorID}
		w.desiredRefusal = nil
	}
	c.mu.Unlock()
	if w == nil || s == nil {
		return exit.Unavailablef("worker %s holds no claimed control stream", instanceID)
	}
	return c.sendJobDirective(s, w)
}

func gpuCountOf(p *JobPlan) int64 {
	if p.GPUCount > 0 {
		return p.GPUCount
	}
	if n, ok := p.Record["gpu_count"].(int64); ok {
		return n
	}
	return 0
}

// ------------------------------------------------------------------ the publication root

// publicationDest resolves ONE granted destination under this request's publication root,
// and REFUSES anything that leaves it.
//
// The fence is the resolution itself, not a scan for suspicious characters: the path is
// joined and cleaned, then the result must still be inside the root. A destination that
// escapes cannot be granted, so there is no state in which a job holds a capability to
// write outside its own publication.
func publicationDest(root, outputID string) (string, *exit.Error) {
	clean := filepath.Clean(filepath.Join(root, outputID))
	rel, err := filepath.Rel(root, clean)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) ||
		filepath.IsAbs(outputID) {
		return "", exit.Named(exit.Validation, "publication_escape",
			"output %q resolves to %s, outside this job's publication root %s",
			outputID, clean, root).
			WithRemedy("a job may be granted destinations under its own publication root and nowhere else")
	}
	return clean, nil
}

// FenceOutputID admits a serving output id only as ONE path element: no separators, no
// "..", never empty. It is the serving counterpart of publicationDest — the grant, the
// mirror and `--out` all resolve `dir/id`, so the id itself must be unable to leave dir.
func FenceOutputID(id string) *exit.Error {
	if id == "" || id == "." || id == ".." || strings.ContainsAny(id, `/\`) {
		return exit.Named(exit.Validation, "output_id_escape",
			"output id %q is not a single path element", id).
			WithRemedy("an output is granted one file under its attempt directory and nowhere else")
	}
	return nil
}

// jobGrant builds the LOCAL delivery grant for one job attempt: the payload as the input
// `payload`, one input per MATERIALIZED input tree (`tree:<ref>`), and one destination per
// granted result field path — every one of them under the publication root.
//
// There is no credential here either. A local grant is a CAS root plus a directory, and a
// job's directory is the durable one.
func (c *Orchestrator) jobGrant(req records.Request, attempt uint64) (*pb.DeliveryGrant, string, *exit.Error) {
	// THE JOB WRITES INTO A STAGE, never into the addressable publication root. The root
	// is where a COMMITTED bundle lives; the stage is where an attempt in flight puts its
	// bytes, and `promote` moves them across after the terminal is verified.
	root := c.opt.Layout.PublicationStage(req.Org, req.ID, attempt)
	inDir := filepath.Join(c.opt.Layout.AttemptDir(req.ID, attempt), "in")
	if err := os.MkdirAll(root, 0o755); err != nil {
		return nil, "", exit.Internalf("cannot create the publication stage %s: %s", root, err)
	}
	if err := os.MkdirAll(inDir, 0o755); err != nil {
		return nil, "", exit.Internalf("cannot create the attempt input directory %s: %s", inDir, err)
	}
	payloadPath := filepath.Join(inDir, "payload")
	if err := os.WriteFile(payloadPath, req.Payload, 0o644); err != nil {
		return nil, "", exit.Internalf("cannot stage the job payload: %s", err)
	}
	g := &pb.DeliveryGrant{
		FileBaseUrl: "file://" + root,
		// NO EXPIRY — the same reading as the serving grant, and a job is where the old
		// constant did real damage: 6 hours is a ceiling on what a run-to-completion
		// conversion may be, and job-001 is the incident where one of these killed a
		// 99 GB weights that had already landed. The stage's lifetime is the ATTEMPT's:
		// `promote` moves the bytes across once the terminal is verified, and that is the
		// event that ends this grant's usefulness. A clock never knew about it.
		ExpiresAtUnix: 0,
		// ACCESS ONLY (#439): identities live in the InvocationSpec's bindings.
		Inputs: []*pb.InputAccess{{InputId: "payload", Url: "file://" + payloadPath}},
	}
	for _, model := range req.Models {
		if model.ManifestLength <= 0 {
			continue
		}
		g.Inputs = append(g.Inputs, &pb.InputAccess{
			InputId: "model:" + model.Slot, Url: "model://" + model.Manifest,
		})
	}
	// THE INPUT TREES. A tree's bytes are not re-hashed at the grant: a tree is a
	// materialized TensorFS snapshot and its verification is the store's own verified
	// read path (law 18, tfs-007). What this side owns is that the grant NAMES it — a
	// field naming a ref the grant does not cover never reaches a filesystem.
	for _, pair := range splitList(req.Trees) {
		ref, dir, ok := strings.Cut(pair, "=")
		if !ok {
			continue
		}
		abs, err := filepath.Abs(dir)
		if err != nil {
			return nil, "", exit.Internalf("cannot resolve the input tree %s: %s", dir, err)
		}
		g.Inputs = append(g.Inputs, &pb.InputAccess{Url: "file://" + abs, InputId: "tree:" + ref})
	}
	for _, id := range splitList(req.Outputs) {
		dest, e := publicationDest(root, id)
		if e != nil {
			return nil, "", e
		}
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			return nil, "", exit.Internalf("cannot create %s: %s", filepath.Dir(dest), err)
		}
		g.Outputs = append(g.Outputs, &pb.OutputAccess{OutputId: id, Url: "file://" + dest})
	}
	return g, root, nil
}

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

// publicationOf builds the row that becomes durable INSIDE the terminal transaction.
func (c *Orchestrator) publicationOf(req records.Request, attempt uint64, status, cause string,
	outputs []records.Output) *records.Publication {
	var bytes int64
	for _, o := range outputs {
		bytes += o.Length
	}
	return &records.Publication{
		RequestID: req.ID, Attempt: int64(attempt),
		Repo: home.ScratchRepo(req.Org, req.ID),
		Root: c.opt.Layout.PublicationRoot(req.Org, req.ID),
		// The VERDICT is stamped as metadata and gates nothing: a failed run's writes
		// still land (jobs.md), and a publication that only existed on success would throw
		// away exactly the evidence a failed conversion is worth keeping for.
		Status: status, Cause: cause,
		Entries: int64(len(outputs)), Bytes: bytes,
	}
}

// promote moves one attempt's staged writes into the addressable publication root. It runs
// AFTER the terminal document has been verified and immediately BEFORE the transaction
// that makes the publication exist, so the addressable path only ever holds bytes some
// terminal vouched for: a job never writes there itself, and a crash before the commit
// leaves the root without them.
func (c *Orchestrator) promote(req records.Request, attempt uint64, outputs []records.Output) *exit.Error {
	root := c.opt.Layout.PublicationRoot(req.Org, req.ID)
	for i, o := range outputs {
		dest, e := publicationDest(root, o.OutputID)
		if e != nil {
			return e
		}
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			return exit.Internalf("cannot create %s: %s", filepath.Dir(dest), err)
		}
		// RENAME, within one directory tree: atomic per entry, so a reader sees each
		// published file whole or not at all.
		//
		// AND IDEMPOTENT, because a terminal is replayed byte-identically until it is
		// acked (law 9) and every step it triggers is therefore run again. A crash between
		// this promotion and the commit leaves the bytes at the destination and nothing at
		// the stage; the replay must read that as "already promoted", not as a broken
		// filesystem. Found live by cl-004's crash arm, where the replayed terminal was
		// REFUSED forever on a rename whose work was already done.
		if err := os.Rename(o.Path, dest); err != nil {
			if _, absent := os.Stat(o.Path); absent != nil {
				if _, there := os.Stat(dest); there == nil {
					outputs[i].Path = dest
					c.logf("publication %s: %s was already promoted; the replayed terminal "+
						"finds it in place", home.ScratchRepo(req.Org, req.ID), o.OutputID)
					continue
				}
			}
			return exit.Internalf("cannot promote %s into the publication root: %s", o.OutputID, err)
		}
		outputs[i].Path = dest
	}
	return nil
}

// ------------------------------------------------------------------ the checkpoint lane

// onCheckpoint answers the DURABLE checkpoint exchange. The worker blocked on this frame
// over the control lane — deliberately not the lossy one — and the identity it presents
// is closed: repeating it replays the receipt, and the same operation/logical key with
// different bytes is a CONFLICT, never a replacement.
func (c *Orchestrator) onCheckpoint(s *session, r *pb.JobCheckpointRequest) {
	digest, _ := canonical.Spell(r.ContentDigest)
	// The attempt is part of the identity AND the authority: the row is admitted only if
	// `RequestId#AttemptOrdinal` is an attempt still open under THIS session (#553a). A
	// worker cannot declare a checkpoint against another worker's run, and no worker can
	// declare one against a run that has already ended.
	row, outcome, e := c.opt.Store.RecordCheckpoint(records.Checkpoint{
		RequestID: r.RequestId, Attempt: int64(r.AttemptOrdinal),
		OperationKey: r.OperationKey, LogicalKey: r.LogicalKey, ContentDigest: digest,
		SessionID: s.bootID,
	})
	if e != nil {
		c.logf("checkpoint %s/%s of %s#%d NOT journaled: %s",
			r.OperationKey, r.LogicalKey, r.RequestId, r.AttemptOrdinal, e.Message)
		s.send(checkpointReceipt(s, r, "", pb.CheckpointOutcome_CHECKPOINT_OUTCOME_REFUSED,
			pb.CheckpointFaultCode_CHECKPOINT_FAULT_CODE_UNKNOWN_ATTEMPT, e.Message))
		return
	}
	c.logf("checkpoint %s/%s of %s#%d %s (%s)", r.OperationKey, r.LogicalKey,
		r.RequestId, r.AttemptOrdinal, outcome, shortDigest(digest))
	switch outcome {
	case "CONFLICT":
		s.send(checkpointReceipt(s, r, row.ReceiptID, pb.CheckpointOutcome_CHECKPOINT_OUTCOME_CONFLICT,
			pb.CheckpointFaultCode_CHECKPOINT_FAULT_CODE_IDENTITY_CONFLICT,
			"this identity is already journaled at "+shortDigest(row.ContentDigest)))
	default:
		s.send(checkpointReceipt(s, r, row.ReceiptID,
			pb.CheckpointOutcome_CHECKPOINT_OUTCOME_RECORDED, 0, ""))
	}
}

func checkpointReceipt(s *session, r *pb.JobCheckpointRequest, receiptID string,
	outcome pb.CheckpointOutcome, code pb.CheckpointFaultCode, detail string) *pb.RecordOwnerFrame {
	receipt := &pb.JobCheckpointReceipt{
		RequestId: r.RequestId, AttemptOrdinal: r.AttemptOrdinal, OperationKey: r.OperationKey,
		LogicalKey: r.LogicalKey, ContentDigest: r.ContentDigest,
		ReceiptId: receiptID, Outcome: outcome,
	}
	receipt.RecordOwnerEpoch, receipt.ControlStreamGeneration, receipt.WorkerBootId =
		recordOwnerEpoch, s.generation, s.bootID
	if detail != "" {
		receipt.Fault = &pb.CheckpointFault{Code: code, Detail: detail}
	}
	return &pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_CheckpointReceipt{CheckpointReceipt: receipt}}
}
