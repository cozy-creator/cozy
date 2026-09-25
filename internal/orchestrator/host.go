package orchestrator

// THE HOST LANE (proto-025). A rented pod's supervisor is the pod-resident half of the host
// role this daemon performs in-process for a local worker: it downloads, verifies, and obtains
// PlacementSet bytes from the Runtime's loopback preparation. This owner asks it to PREPARE
// over PodHost, off the control stream, and then sends the prepared placement_set bytes
// itself as DesiredWorkerState on WorkerControl -- the identical three steps `converge` runs
// for a local install (prepare -> placement_set bytes -> desired_state). Before minor 21 the
// supervisor did all of this inline on the control stream's read loop, and every offer, ack
// and cancel for the package already serving waited behind package B's download.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type prepareOpener func(context.Context) (grpc.ServerStreamingClient[pb.PrepareEvent], error)

// issueThroughHost starts one preparation and returns. The caller has already journaled the
// logical desire on the worker (so a reconnect re-issues it); the prepare runs on its own
// goroutine because its callers include the control stream's own receive loop, and a
// multi-GiB materialization must never sit on that loop on either end of the wire.
//
// hostPrepareSeq numbers the desire. A prepare that finishes after a newer desire was issued
// sends nothing: its bytes describe a set this owner no longer wants, and a full-replace
// desired state carrying them would unload the newer one. The desired revision is minted
// HERE, not at the send, so every wait on `acceptedRevision >= revision` stays closed for the
// whole preparation rather than passing on the previous set's acceptance.
func (c *Orchestrator) issueThroughHost(s *session, w *worker, label string, open prepareOpener) *exit.Error {
	if s.host == nil {
		return exit.Internalf("worker %s has no PodHost lane to prepare %s on", w.instanceID, label)
	}
	rev := c.nextRevision()
	c.mu.Lock()
	w.hostPrepareSeq++
	seq := w.hostPrepareSeq
	w.revision, w.desiredRefusal = rev, nil
	c.mu.Unlock()
	c.logf("PodHost prepare %s#%d for revision %d -> %s", label, seq, rev, s.bootID)
	go c.prepareThroughHost(s, w, seq, rev, label, open)
	return nil
}

// packagePrepare is one package's own PreparePackageSet call inside a package_set desire:
// the exact download set naming that one package and its models.
type packagePrepare struct {
	label string
	// pkg names the package this preparation carries.
	pkg string
	// ref is the exact release this preparation carries — what the facts fetch
	// names against the rental-scoped prepare-facts route.
	ref         *pb.DownloadPackageRef
	downloadSet []byte
}

// issuePackagePrepares starts one package_set desire and returns. The desire holds ONE
// prepare per package — the Runtime prepares exactly one package per call — issued
// SEQUENTIALLY on one goroutine, and the united placement set is sent as one full-replace
// desired state only after every package's preparation returned its exact bytes.
// issueThroughHost's seq/revision law applies to the sequence as a whole: it is one
// logical desire, superseded between prepares exactly as a single prepare is superseded.
func (c *Orchestrator) issuePackagePrepares(s *session, w *worker, label string, prepares []packagePrepare) *exit.Error {
	if s.host == nil {
		return exit.Internalf("worker %s has no PodHost lane to prepare %s on", w.instanceID, label)
	}
	rev := c.nextRevision()
	c.mu.Lock()
	w.hostPrepareSeq++
	seq := w.hostPrepareSeq
	w.revision, w.desiredRefusal = rev, nil
	c.mu.Unlock()
	c.logf("PodHost prepare %s#%d for revision %d -> %s", label, seq, rev, s.bootID)
	go c.preparePackagesThroughHost(s, w, seq, rev, label, prepares)
	return nil
}

func (c *Orchestrator) prepareThroughHost(s *session, w *worker, seq, rev uint64, label string, open prepareOpener) {
	result := c.runHostPrepare(s, w, seq, label, open)
	if !c.settleHostPrepare(s, w, seq, label, result) {
		return
	}
	c.convergePrepared(s, w, seq, rev, label, result.set)
}

func (c *Orchestrator) preparePackagesThroughHost(s *session, w *worker, seq, rev uint64,
	label string, prepares []packagePrepare) {
	sets := make([]*pb.DesiredPlacementSet, 0, len(prepares))
	for _, prep := range prepares {
		c.mu.Lock()
		superseded := w.hostPrepareSeq
		c.mu.Unlock()
		if superseded != seq {
			c.logf("PodHost prepare %s#%d superseded by #%d between packages", prep.label, seq, superseded)
			return
		}
		// MINOR 31 (xs-019): the call carries the hub-known release facts on
		// fields 3-6; the pod host refuses one without them. They are fetched
		// here, on the prepare's own goroutine, because the source is a network
		// call and issuePackageSet's callers include the control stream's
		// receive loop.
		facts, problem := c.opt.RentalPrepareFacts(s.ctx, w.spec.Connection, prep.ref)
		if problem != nil {
			if problem.Code == exit.Unavailable || problem.Code == exit.Deadline {
				c.setDesiredUnavailable(w, seq, exit.Unavailablef(
					"PodHost prepare %s has no release facts yet: %s", prep.label, problem.Message))
			} else {
				c.setDesiredRefusal(w, seq, exit.Named(exit.Structural, "worker.prepare_facts_refused",
					"the hub refused the release facts for %s: %s", prep.label, problem.Message))
			}
			return
		}
		if problem := requireAdapterDownloadPeer(s, prep.downloadSet); problem != nil {
			c.setDesiredRefusal(w, seq, problem)
			return
		}
		call := &pb.PreparePackageSetCall{SupportsModelMaterializationRecovery: true, Claim: s.claim, PackageSet: &pb.DesiredPackageSet{
			DownloadDelegation: append([]byte(nil), prep.downloadSet...),
		},
			Application:    facts.Application,
			ModelSlotPaths: append([]string(nil), facts.ModelSlotPaths...),
			ImageInventory: facts.ImageInventory,
			PythonRequires: facts.PythonRequires, PythonVersion: facts.PythonVersion,
			LockedRequirements: append([]byte(nil), facts.LockedRequirements...),
		}
		result := c.runHostPrepare(s, w, seq, prep.label,
			func(ctx context.Context) (grpc.ServerStreamingClient[pb.PrepareEvent], error) {
				return s.host.PreparePackageSet(ctx, call)
			})
		if !c.settleHostPrepare(s, w, seq, prep.label, result) {
			return
		}
		sets = append(sets, result.set)
	}
	united, err := unitePreparedPlacementSets(sets)
	if err != nil {
		c.setDesiredRefusal(w, seq, exit.Named(exit.Structural, "worker.prepare_document_invalid",
			"the pod host's prepared PlacementSets for %s cannot be united: %s", label, err))
		return
	}
	c.convergePrepared(s, w, seq, rev, label, united)
}

// hostPrepareResult is one preparation's end: the exact prepared set, the host's typed
// refusal text, an inadmissible-document fault, or a transport end without a verdict.
type hostPrepareResult struct {
	set     *pb.DesiredPlacementSet
	refusal string
	fault   *exit.Error
	err     error
}

// runHostPrepare opens one prepare stream and consumes it to its end, returning the
// verified prepared PlacementSet or the verdict that stopped it.
//
// EVERY EVENT IS AN OBSERVATION, not only the ones that change the stage (cl-121). This
// loop used to log on a stage change and discard the rest, so a multi-hour materialization
// produced four lines in a daemon log and nothing anywhere a person looks. The pod's
// counters are the only measurement of that work anyone has; they belong in the phase lane
// the moment they arrive.
func (c *Orchestrator) runHostPrepare(s *session, w *worker, seq uint64, label string, open prepareOpener) hostPrepareResult {
	stream, err := open(s.ctx)
	if err != nil {
		return classifyPrepareEnd(err)
	}
	c.mu.Lock()
	machine := c.machineWord(w.instanceID)
	c.mu.Unlock()
	var stage pb.PrepareStage
	for {
		event, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				err = status.Error(codes.FailedPrecondition, "the host closed the prepare stream without a terminal event")
			}
			if problem := RuntimeRequirementTrailer(stream.Trailer()); problem != nil {
				return hostPrepareResult{fault: problem}
			}
			return classifyPrepareEnd(err)
		}
		c.observePrepareEvent(w.instanceID, machine, label, event)
		if event.Stage != stage {
			stage = event.Stage
			c.logf("PodHost prepare %s#%d %s (%d/%d B)", label, seq,
				trimEnum(pb.PrepareStage_name[int32(stage)], "PREPARE_STAGE_"),
				event.TransferredBytes, event.TotalBytes)
		}
		switch event.Stage {
		case pb.PrepareStage_PREPARE_STAGE_REFUSED:
			if problem := RuntimeRequirementEvent(event); problem != nil {
				return hostPrepareResult{fault: problem}
			}
			if event.SafeCode == "package_environment_dependency_base_conflict" {
				return hostPrepareResult{fault: exit.Named(exit.Structural, "machine_execution.runtime_requirement", "%s", event.SafeDetail)}
			}
			return hostPrepareResult{refusal: event.SafeCode + ": " + event.SafeDetail}
		case pb.PrepareStage_PREPARE_STAGE_PREPARED:
			prepared := event.PlacementSet
			if prepared == nil || !bytes.Equal(canonical.Digest(prepared.PlacementSetCanonicalBytes),
				prepared.PlacementSetDigest) {
				return hostPrepareResult{fault: exit.Named(exit.Structural, "worker.prepare_identity_mismatch",
					"the pod host's prepared PlacementSet for %s does not hash to its digest", label)}
			}
			if _, err := canonical.Read(prepared.PlacementSetCanonicalBytes, &pb.PlacementSet{}); err != nil {
				return hostPrepareResult{fault: exit.Named(exit.Structural, "worker.prepare_document_invalid",
					"the pod host's prepared PlacementSet for %s is inadmissible: %s", label, err)}
			}
			return hostPrepareResult{set: prepared}
		}
	}
}

// settleHostPrepare records a preparation's verdict, answering whether the caller holds a
// prepared set to carry forward. A typed refusal or an inadmissible document is the
// worker's word on the whole desire (a full-replace set missing one member must not be
// sent); a transport end without a verdict is not. It releases the waiting launch as
// Unavailable, leaving the request queued; the fleet observer re-issues the journaled desire.
func (c *Orchestrator) settleHostPrepare(s *session, w *worker, seq uint64, label string, result hostPrepareResult) bool {
	switch {
	case result.refusal != "":
		c.hostPrepareRefused(s, w, seq, label, result.refusal)
	case result.fault != nil:
		c.setDesiredRefusal(w, seq, result.fault)
	case result.err != nil:
		c.setDesiredUnavailable(w, seq, exit.Unavailablef(
			"PodHost prepare %s ended without a verdict: %v", label, result.err))
	default:
		return result.set != nil
	}
	return false
}

// classifyPrepareEnd sorts a prepare stream's end into "no verdict, the reconnect re-issues"
// and the host's typed verdict on this desire.
//
// THE POLARITY IS THE POINT. This listed the terminal codes and let everything else fall
// through to a redial, which means every status nobody thought of became an unbounded retry
// on a rented pod. codes.NotFound was one of them: the desire names an immutable identity,
// so a host that cannot find it now cannot find it on the next dial either, and the owner
// re-issued the same revision forever. That is proto-035's rule — a refusal is resumable
// only if the refusing party could answer differently to the IDENTICAL request later — and
// a deny-list cannot enforce it, because the codes that violate it are exactly the ones not
// yet enumerated. It is also how the Unimplemented loop happened (cozy #298), eleven lines
// from `RentalPrepareFacts`, which has had the right polarity all along.
//
// So: an allow-list of the codes that describe a condition of the MOMENT rather than of the
// request. Anything else — NotFound, OutOfRange, DataLoss, AlreadyExists, and whatever is
// added next — is the host's word on this desire, and redialing cannot change it.
func classifyPrepareEnd(err error) hostPrepareResult {
	switch status.Code(err) {
	case codes.OK:
		// Not reached: this is only called with a non-nil end. Never a refusal.
		return hostPrepareResult{err: err}
	case codes.Unavailable, codes.DeadlineExceeded, codes.Canceled,
		codes.ResourceExhausted, codes.Aborted, codes.Internal, codes.Unknown:
		// The pod is down, busy, restarting, contended, or the transport broke under a
		// raw error. Internal and Unknown stay here on purpose: a connection reset mid
		// stream surfaces as one of them, and calling that a permanent verdict would
		// strand a paid pod on a network blip.
		return hostPrepareResult{err: err}
	default:
		return hostPrepareResult{refusal: status.Convert(err).Message()}
	}
}

// hostPrepareRefused records the host lane's refusal. It is the worker's final word on
// this desire: redialing cannot make that exact revision acceptable.
//
// It used to carry ONE exception (xs-007 row 5): a refusal naming this owner's lapsed
// download delegation was answered by re-issuing under a freshly signed one, because the
// credential aged out mid-transfer. The credential is deleted (owner ruling 2026-09-03),
// so a download can no longer become unauthorized by running long, and the exception has
// nothing left to except.
func (c *Orchestrator) hostPrepareRefused(_ *session, w *worker, seq uint64, _, detail string) {
	if len(detail) > 1024 {
		detail = detail[:1024] + "…"
	}
	c.mu.Lock()
	if w.hostPrepareSeq != seq {
		live := w.hostPrepareSeq
		c.mu.Unlock()
		c.logf("worker %s: PodHost prepare #%d was superseded by #%d before its refusal "+
			"landed; the newer prepare carries its own verdict: %s", w.instanceID, seq, live, detail)
		return
	}
	revision := w.revision
	w.desiredRefusal = exit.Named(exit.Structural, "worker.desired_state_refused",
		"worker rejected desired revision %d before applying it: %s", revision, detail)
	c.mu.Unlock()
	c.logf("worker %s: desired revision %d REFUSED before it was applied: %s",
		w.instanceID, revision, detail)
}

func (c *Orchestrator) setDesiredRefusal(w *worker, seq uint64, e *exit.Error) {
	c.mu.Lock()
	if w.hostPrepareSeq != seq {
		live := w.hostPrepareSeq
		c.mu.Unlock()
		c.logf("worker %s: PodHost prepare #%d was superseded by #%d before its refusal "+
			"landed; the newer prepare carries its own verdict: %s", w.instanceID, seq, live, e.Message)
		return
	}
	w.desiredRefusal = e
	instance, revision := w.instanceID, w.revision
	c.mu.Unlock()
	c.logf("worker %s: desired revision %d REFUSED before it was applied: %s",
		instance, revision, e.Message)
}

func (c *Orchestrator) setDesiredUnavailable(w *worker, seq uint64, e *exit.Error) {
	c.mu.Lock()
	if w.hostPrepareSeq != seq {
		live := w.hostPrepareSeq
		c.mu.Unlock()
		c.logf("worker %s: PodHost prepare #%d was superseded by #%d before its deferral "+
			"landed: %s", w.instanceID, seq, live, e.Message)
		return
	}
	w.desiredRefusal = e
	instance, revision := w.instanceID, w.revision
	c.mu.Unlock()
	c.logf("worker %s: desired revision %d DEFERRED without a verdict: %s",
		instance, revision, e.Message)
}

// convergePrepared is step three: the exact bytes this owner has verified become the desired
// state, over the same frame a local install sends.
func (c *Orchestrator) convergePrepared(s *session, w *worker, seq, rev uint64, label string, prepared *pb.DesiredPlacementSet) {
	setBytes := append([]byte(nil), prepared.PlacementSetCanonicalBytes...)
	digest := append([]byte(nil), prepared.PlacementSetDigest...)
	// THE WIDTH IS AUTHORED HERE TOO. This is the rental's own convergence path — the pod
	// prepared its own bytes and this owner relays them — so the device pin has to be
	// authored over the SAME rule as a locally prepared set, or a wide pod would take a
	// placement with no pin and refuse `device_group_unsupported` after the hour started.
	pins, problem := devicePins(setBytes, w.spec.Devices)
	if problem != nil {
		c.setDesiredRefusal(w, seq, problem)
		return
	}
	c.mu.Lock()
	if w.hostPrepareSeq != seq {
		c.mu.Unlock()
		c.logf("PodHost prepare %s#%d superseded by #%d before its bytes were sent", label, seq, w.hostPrepareSeq)
		return
	}
	// Model preparation can outlive cancellation, including when the previous
	// child was already serving. Revalidate before ANY final desired-state send.
	if w.preparingRequest != "" {
		request, problem := c.opt.Store.RequestRow(w.preparingRequest)
		if problem == nil && request != nil {
			problem = c.rentalPreparationAllowedLocked(w, *request)
		}
		if c.sessions[s.bootID] != s || c.workers[w.instanceID] != w || request == nil || problem != nil {
			c.mu.Unlock()
			c.setDesiredUnavailable(w, seq, exit.Unavailablef("private preparation lost its current execution authority"))
			return
		}
		parent, problem := c.activeParentFor(*request)
		if problem != nil || !proto.Equal(parent, w.orchestrationParent) {
			c.mu.Unlock()
			c.setDesiredUnavailable(w, seq, exit.Unavailablef("private preparation changed its exact CPU parent"))
			return
		}
	}
	if w.spec.IsJob() {
		ownedChild, actingJob, actingParent := false, false, ""
		if w.preparingRequest != "" {
			request, problem := c.opt.Store.RequestRow(w.preparingRequest)
			if problem == nil && request != nil {
				actingJob, actingParent = request.IsJob(), request.ParentRequestID
			}
			ownedChild = w.orchestrationParent != nil && problem == nil && request != nil &&
				request.ParentRequestID != "" && c.rentalPreparationAllowedLocked(w, *request) == nil
		}
		retained, problem := c.opt.Store.RentalHasRetainedJob(w.spec.Connection.RentalID)
		if !ownedChild && (problem != nil || retained || !c.idleRentalWorkerLocked(w) ||
			c.modeClaimedLocked(w, actingJob, w.preparingRequest, actingParent)) {
			c.mu.Unlock()
			c.setDesiredUnavailable(w, seq, exit.Unavailablef("rented worker is finishing its current job before changing mode"))
			return
		}
		// The wire carries a full replacement. Match it locally before its first
		// report arrives, otherwise serving capacity is interpreted as job capacity.
		// Native checkpoint/workspace custody stays with the supervisor and TensorFS.
		w.spec.Placement = DesiredPlacement{}
		w.planIDs = nil
		w.remotePlacements = map[string]DesiredPlacement{}
		w.observedRemote = map[string]remotePlacementObservation{}
		w.dispatchable, w.materializable = map[string]bool{}, map[string]bool{}
		w.jobReady = nil
		w.jobsAvail, w.reportedJobs = 0, 0
	}
	w.setDigest, w.setBytes = digest, setBytes
	c.mu.Unlock()
	d := &pb.DesiredWorkerState{
		RecordOwnerEpoch: recordOwnerEpoch, ControlStreamEpoch: s.epoch, WorkerBootId: s.bootID,
		Revision: rev, WireMinor: pb.WireMinor, Posture: pb.Posture_POSTURE_ACCEPTING,
		Mode: &pb.DesiredWorkerState_PlacementSet{PlacementSet: &pb.DesiredPlacementSet{
			PlacementSetDigest: digest, PlacementSetCanonicalBytes: setBytes, DevicePins: pins, OrchestrationParent: w.orchestrationParent}},
	}
	if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_DesiredState{DesiredState: d}}) {
		c.logf("PodHost prepare %s#%d: control stream closed before the placement_set send", label, seq)
		return
	}
	c.logf("DesiredWorkerState revision=%d placement_set=%s (%d canonical bytes, prepared by the host as %s) "+
		"envelope=[%s] pins=%d -> %s",
		rev, shortDigest(shortNone(digest)), len(setBytes), label,
		strings.Join(w.spec.Devices, ","), len(pins), s.bootID)
}

func hostLabel(kind, id string) string { return fmt.Sprintf("%s(%s)", kind, id) }

// observePrepareEvent folds one PrepareEvent into the phase lane. The subject is the
// WORKER, not a request: one preparation serves every request queued behind it, and
// recording it per request would make the same bytes look like several transfers.
//
// The mapping is the pod's own stage vocabulary and nothing else. A stage this owner does
// not recognise records no phase rather than a guessed one, and the terminal stages record
// none because a request that has left preparation is no longer IN it.
//
// TotalBytes is 0 until the plan is bounded (worker.proto), so it is forwarded as declared:
// zero means "no denominator yet", which the renderer shows as bytes and a rate rather
// than as a fraction of nothing.
func (c *Orchestrator) observePrepareEvent(instanceID, machine, label string, event *pb.PrepareEvent) {
	name := ""
	switch event.GetStage() {
	case pb.PrepareStage_PREPARE_STAGE_RESOLVED:
		name = PhaseResolving
	case pb.PrepareStage_PREPARE_STAGE_DOWNLOADING:
		name = PhaseDownloading
	case pb.PrepareStage_PREPARE_STAGE_PREPARING:
		name = PhasePreparing
	case pb.PrepareStage_PREPARE_STAGE_PREPARED:
		// The bytes are in and the placement is about to be sent: what remains is the
		// worker making it resident, which is the phase the routing already names.
		name = PhaseWarming
	default:
		return
	}
	sample := PhaseSample{Name: name, Machine: machine, Detail: label}
	// PrepareEvent counters are cumulative across the entire call. After download,
	// they still describe the completed transfer, not package setup or GPU loading.
	if name != PhaseDownloading {
		c.ObservePhase(instanceID, sample)
		return
	}
	models := make([]ModelDownloadProgress, 0, len(event.GetModelProgress()))
	for _, item := range event.GetModelProgress() {
		ref := item.GetModel()
		if ref == nil {
			continue
		}
		models = append(models, ModelDownloadProgress{
			Model: ref.GetModel(), Release: ref.GetRelease(), Lane: ref.GetLane(),
			Slot: ref.GetSlot(), Manifest: ref.GetManifest(),
			Moved: item.GetTransferredBytes(), Total: item.GetTotalBytes(),
			OriginBytes: item.GetOriginBytes(), CachedBytes: item.GetCachedBytes(),
		})
	}
	sample.Models = models
	sample.HasBytes = event.GetTotalBytes() > 0 || event.GetTransferredBytes() > 0
	sample.Moved, sample.Total = event.GetTransferredBytes(), event.GetTotalBytes()
	c.ObservePhase(instanceID, sample)
}

// RuntimeRequirementTrailer preserves authenticated worker dependency facts for both preparation paths.
func RuntimeRequirementTrailer(trailer metadata.MD) *exit.Error {
	first := func(key string) string {
		values := trailer.Get(key)
		if len(values) == 1 && len(values[0]) <= 2048 {
			return values[0]
		}
		return ""
	}
	packageName, distribution := first("cozy-requirement-package"), first("cozy-requirement-distribution")
	required, installed := first("cozy-requirement-required"), first("cozy-requirement-installed")
	if packageName == "" || distribution == "" || required == "" || installed == "" {
		return nil
	}
	name := "machine_execution.package_requirement"
	if distribution == "cozy-runtime" || distribution == "tensorfs" { //cozy:allow distribution metadata, not a binary invocation
		name = "machine_execution.runtime_requirement"
	}
	return exit.Named(exit.Structural, name, "%s requires %s; this worker has %s %s", packageName, required, distribution, installed)
}

// RuntimeRequirementEvent decodes the Host's bounded dependency verdict. It is
// shared by ordinary serving and Runtime-owned root preparation.
func RuntimeRequirementEvent(event *pb.PrepareEvent) *exit.Error {
	if event.SafeCode != "package_runtime_incompatible" && event.SafeCode != "package_sdk_incompatible" {
		return nil
	}
	if len(event.SafeDetail) > 8192 {
		return nil
	}
	var detail struct {
		Package      string `json:"package"`
		Distribution string `json:"distribution"`
		Required     string `json:"required"`
		Installed    string `json:"installed"`
	}
	if json.Unmarshal([]byte(event.SafeDetail), &detail) != nil {
		return nil
	}
	return RuntimeRequirementTrailer(metadata.Pairs("cozy-requirement-package", detail.Package, "cozy-requirement-distribution", detail.Distribution, "cozy-requirement-required", detail.Required, "cozy-requirement-installed", detail.Installed))
}
