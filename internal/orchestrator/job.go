package orchestrator

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
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
	// BuildID is `build_id`: the BUILD this job plan record is written under, and the
	// value `JobDirective.build_id` must name. See `JobBuildID` for why it is the
	// preparing placement's own identity and never the PlacementSet digest.
	BuildID string
	// Outputs are the job's declared asset result field paths — the output ids the
	// publication grant names, one destination each. Grants mint off the DECLARATION.
	Outputs []string
	// WeightsOutputs is the explicit WeightsSink subset. Empty keeps an ordinary asset
	// job on the existing publication path; non-empty is the M0 weights-only contract.
	WeightsOutputs []WeightsOutput
	// Record is the closed key set `plan.py::JobBinding.read` accepts. An unknown key is
	// a refusal at the worker, which is what makes "closed at both ends" a fact.
	Record              map[string]any
	RSSCap              int64
	NeedsAccelerator    bool
	Orchestration       bool
	OrchestrationParent *JobPlan
	FrozenDirective     *pb.JobDirective
}

const DefaultJobRSSCap int64 = 8 << 30

// JobBuildID reads the BUILD a job plan record is staged under out of the prepared
// PlacementSet document itself.
//
// IT IS NOT THE PlacementSet DIGEST. This owner UNITES the sets several packages
// prepared into one document (`unitePreparedPlacementSets`), so the set digest a worker
// would have to compare against is a value that worker cannot know while it is staging
// its records — and a rented worker stages them itself, from
// `package_prepare.py::_prepare_published`, before this owner has united anything. The
// identity a preparation CAN name is its own placement's, and that is what the runtime
// writes: `environment_digest` for a published preparation, and the complete
// `local_revision_digest` for a transported private preparation. A source-local
// development install instead has its project wheel or captured source digest.
// This reads those existing identities from the document, so the local lane — which stages the record
// here, in Go — writes the same value the remote lane's worker wrote for itself.
func JobBuildID(setBytes []byte, pkg string) (string, *exit.Error) {
	doc, err := canonical.Read(setBytes, &pb.PlacementSet{})
	if err != nil {
		return "", exit.Named(exit.Structural, "job_build_identity_unreadable",
			"the prepared PlacementSet for %s is not its canonical document: %s", pkg, err)
	}
	for _, row := range doc.List("placements") {
		if development := row.Sub("development"); development.Str("package") == pkg {
			id := development.Str("local_revision_digest")
			if id == "" {
				id = development.Sub("project_wheel").Sub("ref").Str("digest")
			}
			if id == "" {
				// A local immutable source install has no transported project wheel.
				// Its captured source closure is the build identity; the environment
				// remains separately bound in the invocation and writer fingerprint.
				id = development.Str("source_digest")
			}
			if id == "" {
				return "", exit.Named(exit.Structural, "job_build_identity_missing",
					"the development placement for %s names no project wheel", pkg)
			}
			return id, nil
		}
		if row.Sub("package").Str("package") != pkg {
			continue
		}
		id := row.Str("environment_digest")
		if id == "" {
			return "", exit.Named(exit.Structural, "job_build_identity_missing",
				"the prepared placement for %s names no environment identity", pkg)
		}
		return id, nil
	}
	return "", exit.Named(exit.Structural, "job_build_identity_missing",
		"the prepared PlacementSet carries no placement for %s", pkg)
}

// stageJobPlans writes one job plan record per declared job into the worker's own home.
// The file name is the descriptor id's hex, which is how the supervisor finds it.
func stageJobPlans(workerHome string, plans []*JobPlan) *exit.Error {
	for _, p := range plans {
		for _, id := range []string{p.BuildID, p.DescriptorID} {
			raw, err := canonical.Raw(id)
			spelled, _ := canonical.Spell(raw)
			if err != nil || spelled != id {
				return exit.New(exit.Validation, "job plan path requires exact canonical build and descriptor digests")
			}
		}
		dir := filepath.Join(workerHome, "job-plans", strings.TrimPrefix(p.BuildID, "sha256:"))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return exit.Internalf("cannot create the job plan directory: %s", err)
		}
		data, err := json.MarshalIndent(p.Record, "", "  ")
		if err != nil {
			return exit.Internalf("cannot render the job plan record: %s", err)
		}
		name := strings.TrimPrefix(p.DescriptorID, "sha256:") + ".json"
		path := filepath.Join(dir, name)
		out, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o444)
		if os.IsExist(err) {
			held, readError := os.ReadFile(path)
			if readError != nil || !bytes.Equal(held, data) {
				return exit.Named(exit.Conflict, "job.plan_changed", "the exact job build and descriptor already name different plan bytes")
			}
			continue
		}
		if err != nil {
			return exit.Internalf("cannot stage exact job plan: %s", err)
		}
		_, writeError := out.Write(data)
		syncError := out.Sync()
		closeError := out.Close()
		if writeError != nil || syncError != nil || closeError != nil {
			return exit.Internalf("cannot durably stage exact job plan")
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
func (c *Orchestrator) sendJobDirective(s *session, w *worker, replacement *WorkerLaunchSpec) *exit.Error {
	c.mu.Lock()
	spec := w.spec
	if replacement != nil {
		spec = *replacement
	}
	plan := spec.Placement.Jobs[0]
	if plan.BuildID == "" {
		c.mu.Unlock()
		return exit.Named(exit.Structural, "job_build_identity_missing",
			"job %s carries no build identity to name in its directive", plan.Function)
	}
	rev := c.nextRevisionLocked()
	// Publish the selection and its new readiness fence together. A concurrent
	// old capacity report must never make the replacement spec dispatchable.
	if replacement != nil {
		w.spec = spec
		w.planIDs = []string{plan.DescriptorID}
		if parent := plan.OrchestrationParent; parent != nil {
			w.planIDs = append(w.planIDs, parent.DescriptorID)
		}
		w.desiredRefusal = nil
	}
	w.revision = rev
	c.mu.Unlock()
	d := &pb.DesiredWorkerState{
		Revision: rev, WireMinor: pb.WireMinor,
		Posture: pb.Posture_POSTURE_ACCEPTING,
		Mode:    &pb.DesiredWorkerState_Job{Job: c.jobDirective(plan)},
	}
	d.RecordOwnerEpoch, d.ControlStreamEpoch, d.WorkerBootId = recordOwnerEpoch, s.epoch, s.bootID
	if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_DesiredState{DesiredState: d}}) {
		return exit.Unavailablef("worker %s control stream closed before job directive send", w.instanceID)
	}
	c.logf("DesiredWorkerState revision=%d posture=accepting JOB %s (%s) -> %s",
		rev, plan.Function, shortDigest(plan.DescriptorID), s.bootID)
	return nil
}

func (c *Orchestrator) jobDirective(plan *JobPlan) *pb.JobDirective {
	if plan.FrozenDirective != nil {
		return proto.Clone(plan.FrozenDirective).(*pb.JobDirective)
	}
	directive := &pb.JobDirective{
		BuildId:         plan.BuildID,
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
		Orchestration:     plan.Orchestration,
	}
	if plan.OrchestrationParent != nil {
		directive.OrchestrationParent = c.jobDirective(plan.OrchestrationParent)
	}
	return directive
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
	}
	c.mu.Unlock()
	if w == nil || s == nil {
		return exit.Unavailablef("worker %s holds no claimed control stream", instanceID)
	}
	return c.sendJobDirective(s, w, &spec)
}

// gpuCountOf is the job's device FLOOR: how many accelerators the attempt may not start
// without. It is 0 or 1 and stays 0 or 1 on a wide pod, deliberately (cl-179).
//
// A width is not a floor. `JobDirective.device_count` is, in the runtime's own words, a
// floor and never concurrency; a job is one bounded attempt over a derive-only view of its model
// inputs — it shards nothing, so a job on a four-card pod needs one card and would be
// refused by a floor of four. Sequence parallelism is a SERVING lane property expressed as
// a device pin, not a job resource cap. What keeps a job off a wide pod is the selection
// side, where a wide product is excluded for a job outright rather than bought and idled.
func gpuCountOf(p *JobPlan) int64 {
	if p.NeedsAccelerator {
		return 1
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
			WithRemedy("an output is granted one digest-named file in its store directory and nowhere else")
	}
	return nil
}

// jobGrant builds the LOCAL delivery grant for one job attempt: the payload INLINE as the
// input `payload`, one input per MATERIALIZED input tree (`tree:<ref>`), and one
// destination per granted result field path — every one of them under the publication
// root.
//
// There is no credential here either. A local grant is a CAS root plus a directory, and a
// job's directory is the durable one.
func (c *Orchestrator) jobGrant(req records.Request, attempt uint64) (*pb.DeliveryGrant, string, *exit.Error) {
	// THE JOB WRITES INTO A STAGE, never into the addressable publication root. The root
	// is where a COMMITTED bundle lives; the stage is where an attempt in flight puts its
	// bytes, and `promote` moves them across after the terminal is verified.
	root := c.opt.Layout.PublicationStage(req.Org, req.ID, attempt)
	if err := os.MkdirAll(root, 0o755); err != nil {
		return nil, "", exit.Internalf("cannot create the publication stage %s: %s", root, err)
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
		Inputs: []*pb.InputAccess{{InputId: "payload", Url: payloadURL(req.Payload)}},
	}
	g.Inputs = append(g.Inputs, modelAccess(req)...)
	assets, problem := localAssetAccess(req)
	if problem != nil {
		return nil, "", problem
	}
	g.Inputs = append(g.Inputs, assets...)
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

// cleanupPublication settles the publication plane for one SETTLED job request: the
// staging tree is over either way, and a root no committed publication row names holds
// nothing a reader can reach — the audit's 51 empty directories were exactly these,
// left by failed and canceled jobs (cl-116). Empty parents are pruned so the plane
// itself disappears when no job holds it. Idempotent, like everything after a terminal.
func (c *Orchestrator) cleanupPublication(req records.Request) {
	if !req.IsJob() {
		return
	}
	root := c.opt.Layout.PublicationRoot(req.Org, req.ID)
	publication, problem := c.opt.Store.PublicationOf(req.ID)
	if problem != nil {
		c.logf("request %s publication cleanup deferred: %s", req.ID, problem.Message)
		return
	}
	target := filepath.Join(root, ".staging")
	if publication == nil {
		target = root
	}
	if err := os.RemoveAll(target); err != nil {
		c.logf("request %s publication cleanup deferred: %s", req.ID, err)
		return
	}
	prunePublicationParents(c.opt.Layout, root)
}

// prunePublicationParents removes the empty directories a settled job leaves between
// its root and the plane, the plane included. Each remove refuses on a non-empty
// directory, so a committed sibling publication is structurally safe.
func prunePublicationParents(l home.Layout, root string) {
	for dir := root; strings.HasPrefix(dir, l.Publications); dir = filepath.Dir(dir) {
		if os.Remove(dir) != nil {
			return
		}
	}
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
	receipt.RecordOwnerEpoch, receipt.ControlStreamEpoch, receipt.WorkerBootId =
		recordOwnerEpoch, s.epoch, s.bootID
	if detail != "" {
		receipt.Fault = &pb.CheckpointFault{Code: code, Detail: detail}
	}
	return &pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_CheckpointReceipt{CheckpointReceipt: receipt}}
}
