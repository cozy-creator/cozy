package orchestrator

import (
	"bytes"
	"context"
	"os"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type localTransfer struct {
	revision           string
	instanceID, bootID string
	epoch              uint64
	canceled           bool
	uploadCancel       context.CancelFunc
	uploadGeneration   uint64
	files              map[string]localTransferFile
	status             map[string]localTransferStatus
	abortC             chan localAbortStatus
}

type localAbortStatus struct {
	outcome pb.LocalPackageAbortOutcome
	code    string
	detail  string
}

type localTransferFile struct {
	digest         []byte
	filename, path string
	// project marks the row whose wheel identity names the package itself. A Creator-side
	// staging fact only: no kind row travels at wire 30.
	project bool
	length  uint64
}

// Bounds match the host ledger. Captured wheel bytes travel only over PodHost.
const (
	maxLocalWheelBytes    = int64(512 << 20)
	maxLocalWheelSetBytes = int64(1 << 30)
)

type localTransferStatus struct {
	state      pb.LocalPackageFileState
	received   uint64
	safeCode   string
	safeDetail string
}

// ConvergeLocalPackage streams one sealed editable revision and then selects it. The
// request row is the recovery authority: once every file acknowledgement is durable there
// (`uploaded` records the boot that verified them), reconnect sends only the exact
// DesiredLocalPackageSet and the pod replays its ledger. The editable refresh converges
// with no request and no durable marker: its next run proves the bytes again if it must.
func (c *Orchestrator) ConvergeLocalPackage(instanceID, operationID string,
	revision localpackage.Installation, uploadedBootID string, uploaded func(bootID string) *exit.Error,
) *exit.Error {
	selected, transfer, problem := localSelection(operationID, revision)
	if problem != nil {
		return problem
	}
	w, s, problem := c.localControl(instanceID)
	if problem != nil {
		return problem
	}
	// One local preparation at a time per pod: a run and the editable refresh converging
	// the same revision must not each ask the pod to prepare over the other's placement.
	w.localMu.Lock()
	defer w.localMu.Unlock()
	if c.localIssued(w, s, revision.ID) {
		c.logf("worker %s already has local revision %s issued on this session; waiting on it",
			instanceID, shortDigest(revision.ID))
		return nil
	}
	if uploadedBootID != "" && uploadedBootID != s.bootID {
		// The exact revision remains durable, but carrier verification is boot-scoped.
		// Re-prove every byte to the replacement pod before moving the durable boot marker.
		uploadedBootID = ""
	}
	if uploadedBootID == "" {
		for uploadedBootID == "" {
			if problem := c.transferLocalPackage(instanceID, operationID, transfer); problem != nil {
				return problem
			}
			w, s, problem = c.localControl(instanceID)
			if problem != nil {
				return problem
			}
			if !c.localTransferVerified(operationID, transfer.revision, s) {
				continue
			}
			if uploaded != nil {
				if problem := uploaded(s.bootID); problem != nil {
					return problem
				}
			}
			uploadedBootID = s.bootID
		}
	}
	if problem := c.hostNothing(instanceID, revision.ID); problem != nil {
		return problem
	}
	for {
		if problem := c.issueLocalPackageSet(s, w, selected); problem == nil {
			c.mu.Lock()
			delete(c.localTransfers, operationID)
			c.mu.Unlock()
			return nil
		} else if problem.ErrName() == "local_package_worker_capacity_unsupported" {
			return problem
		}
		w, s, problem = c.localControl(instanceID)
		if problem != nil {
			return problem
		}
	}
}

// hostNothing empties a pod that holds placements of other revisions and waits for the
// empty set to converge. The pod's Runtime prepares a package only into an empty worker —
// `package_prepare_worker_busy: replace the active package through desired state` — so the
// second revision of an editable package, the edit-and-run loop itself, must unload the
// first before it can be prepared. A pod already holding this revision is left alone.
func (c *Orchestrator) hostNothing(instanceID, revision string) *exit.Error {
	c.mu.Lock()
	w := c.workers[instanceID]
	var s *session
	held := 0
	holdsRevision := false
	if w != nil {
		s = c.sessions[w.bootID]
		held = len(w.observedRemote)
		holdsRevision = w.holdsLocalInstallation(revision)
	}
	c.mu.Unlock()
	if w == nil || s == nil || held == 0 || holdsRevision {
		return nil
	}
	c.logf("worker %s holds %d placement(s) of other revisions; asking it to host nothing "+
		"before %s is prepared", instanceID, held, shortDigest(revision))
	if problem := c.converge(s, w, nil); problem != nil {
		return problem
	}
	c.mu.Lock()
	rev := w.revision
	c.mu.Unlock()
	for {
		c.mu.Lock()
		current := c.workers[instanceID] == w && !w.exited
		// The Runtime never reports an empty set converged (nothing is dispatchable in
		// it); its acceptance plus a report naming no placement is the fact.
		emptied := w.acceptedRevision >= rev && len(w.observedRemote) == 0
		refused, desiredRefusal := w.refusal, w.desiredRefusal
		c.mu.Unlock()
		switch {
		case emptied:
			return nil
		case refused != nil:
			return refused
		case desiredRefusal != nil:
			return desiredRefusal
		case !current:
			return exit.New(exit.Failed, "the rented worker exited while unloading its placements")
		}
		select {
		case <-c.done:
			return exit.Unavailablef("the daemon stopped while the rented worker was unloading")
		case <-time.After(20 * time.Millisecond):
		}
	}
}

// holdsLocalRevision joins the verified set's code identity to the worker's placement
// observation. Placement IDs and local revision digests are different namespaces.
// STAGED is sufficient for model preparation; empty code cannot be dispatchable.
// Called under c.mu, like all reads of the desired set and observed placements.
func (w *worker) holdsLocalInstallation(revision string) bool {
	if revision == "" || len(w.setBytes) == 0 || len(w.setDigest) == 0 {
		return false
	}
	doc, err := canonical.Read(w.setBytes, &pb.PlacementSet{})
	if err != nil {
		return false
	}
	for _, row := range doc.List("placements") {
		if row.Str("installation_id") != revision {
			continue
		}
		observed, ok := w.observedRemote[row.Str("placement_id")]
		if ok && observed.placementSetDigest == spellOf(w.setDigest) &&
			observed.materialization == pb.MaterializationState_MATERIALIZATION_STATE_STAGED {
			return true
		}
	}
	return false
}

// awaitLocalRevision waits until the rented worker REPORTS the local revision among its
// placements — the pod's prepare has landed — or the desire it rides is refused, the
// worker goes, or the daemon stops. Observation only: there is no clock in it.
func (c *Orchestrator) awaitLocalInstallation(instanceID, revision string) *exit.Error {
	for {
		c.mu.Lock()
		w := c.workers[instanceID]
		var held, current bool
		var refused, desiredRefusal *exit.Error
		if w != nil {
			held = w.holdsLocalInstallation(revision)
			current = !w.exited
			refused, desiredRefusal = w.refusal, w.desiredRefusal
		}
		c.mu.Unlock()
		switch {
		case w == nil || !current:
			return exit.New(exit.Failed, "the rented worker exited before it held local revision %s",
				shortDigest(revision))
		case held:
			return nil
		case refused != nil:
			return refused
		case desiredRefusal != nil:
			return desiredRefusal
		}
		select {
		case <-c.done:
			return exit.Unavailablef("the daemon stopped while the rented worker was preparing %s",
				shortDigest(revision))
		case <-time.After(20 * time.Millisecond):
		}
	}
}

func localSelection(operationID string, revision localpackage.Installation) (
	*pb.DesiredLocalPackageSet, *localTransfer, *exit.Error,
) {
	if operationID == "" || revision.ID == "" || revision.Package == "" || revision.Release == "" || len(revision.Files) == 0 || len(revision.Files) > pb.MaxLocalPackageFiles {
		return nil, nil, exit.New(exit.Validation, "private installation inputs are incomplete")
	}
	selected := &pb.DesiredLocalPackageSet{PythonRequires: revision.PythonRequires, PythonVersion: revision.PythonVersion, OperationId: operationID,
		SourceArchive: revision.SourceArchive, DependencyRequirements: revision.DependencyRequirements,
		Package: &pb.DevelopmentPackage{Package: revision.Package, Release: revision.Release, InstallationId: revision.ID}}
	transfer := &localTransfer{revision: revision.ID, files: map[string]localTransferFile{}, status: map[string]localTransferStatus{}, abortC: make(chan localAbortStatus, 1)}
	var total int64
	prior := ""
	for _, file := range revision.Files {
		var digest []byte
		if file.Kind != "source" {
			var err error
			digest, err = canonical.Raw(file.Digest)
			if err != nil {
				return nil, nil, exit.New(exit.Validation, "wheel integrity hash is invalid")
			}
		}
		if file.Length <= 0 || file.Length > maxLocalWheelSetBytes-total || file.Filename <= prior {
			return nil, nil, exit.New(exit.Validation, "source transfer exceeds its bound or repeats a filename")
		}
		total += file.Length
		prior = file.Filename
		transfer.files[file.Filename] = localTransferFile{digest: digest, filename: file.Filename, path: file.Path, project: file.Kind == "project", length: uint64(file.Length)}
		selected.Files = append(selected.Files, &pb.LocalPackageFileRef{Digest: digest, Filename: file.Filename, Length: uint64(file.Length)})
	}
	return selected, transfer, nil
}

// LocalPackageSelection is the passive code-preparation document. Calling it
// neither converges worker state nor creates an execution attempt.
func LocalPackageSelection(operationID string, revision localpackage.Installation) (*pb.DesiredLocalPackageSet, *exit.Error) {
	selected, _, problem := localSelection(operationID, revision)
	return selected, problem
}

// transferLocalPackage sends captured source wheels directly to the claimed host.
// The host owns durable offsets; reconnect resumes verified prefixes without a
// package repository, publication or intermediate object-storage grant.
func (c *Orchestrator) transferLocalPackage(instanceID, operationID string,
	transfer *localTransfer,
) *exit.Error {
	_, current, problem := c.localControl(instanceID)
	if problem != nil {
		return problem
	}
	held, problem := c.bindLocalTransfer(instanceID, operationID, transfer, current)
	if problem != nil {
		return problem
	}
	ordered := transferFilesInOrder(held)
	if problem := c.proveLocalWheelsUnchanged(ordered); problem != nil {
		return problem
	}
	for _, selected := range ordered {
		if problem := c.uploadLocalWheel(current, operationID, held, selected); problem != nil {
			return problem
		}
	}
	return nil
}

func (c *Orchestrator) uploadLocalWheel(current *session, operationID string,
	transfer *localTransfer, selected localTransferFile,
) *exit.Error {
	if current.host == nil {
		return exit.Named(exit.Unavailable, "local_package_direct_transfer_unavailable",
			"worker has no authenticated unpublished package upload service")
	}
	ctx, cancel := context.WithCancel(current.ctx)
	c.mu.Lock()
	if transfer.canceled {
		c.mu.Unlock()
		cancel()
		return exit.New(exit.Canceled, "unpublished package upload was cancelled")
	}
	transfer.uploadGeneration++
	generation := transfer.uploadGeneration
	transfer.uploadCancel = cancel
	c.mu.Unlock()
	defer func() {
		cancel()
		c.mu.Lock()
		if transfer.uploadGeneration == generation {
			transfer.uploadCancel = nil
		}
		c.mu.Unlock()
	}()
	problem := localpackage.UploadFile(ctx, current.host,
		&pb.LocalPackageUploadHeader{Claim: current.claim, OperationId: operationID,
			File: &pb.LocalPackageFileRef{Digest: selected.digest,
				Filename: selected.filename, Length: selected.length}}, selected.path,
		func(status *pb.LocalPackageFileStatus) { c.onLocalPackageFileStatus(current, status) })
	c.mu.Lock()
	canceled := transfer.canceled
	c.mu.Unlock()
	if problem != nil && problem.Code == exit.Canceled && current.ctx.Err() != nil && !canceled {
		// The upload inherits the control stream's lifetime. Losing that stream is
		// not user cancellation: reconnect can resume the Host's durable prefix.
		// Explicit request cancellation sets transfer.canceled before canceling it.
		return exit.Named(exit.Unavailable, "local_package_upload_interrupted",
			"worker control stream ended during unpublished upload; acknowledged bytes can be resumed")
	}
	return problem
}

// proveLocalWheelsUnchanged reads the local files the revision named. The digest is the
// revision's identity, so a wheel edited between staging and transfer is a different revision
// wearing this one's name, and the store would refuse it anyway under the signed checksum.
func (c *Orchestrator) proveLocalWheelsUnchanged(ordered []localTransferFile) *exit.Error {
	for _, selected := range ordered {
		info, err := os.Stat(selected.path)
		if err != nil {
			return exit.Named(exit.Structural, "local_package_wheel_unreadable", "%s", err)
		}
		if !info.Mode().IsRegular() || info.Size() != int64(selected.length) {
			return exit.Named(exit.Structural, "local_package_wheel_changed",
				"local package wheel %s changed before transfer", selected.filename)
		}
	}
	return nil
}

func (c *Orchestrator) bindLocalTransfer(instanceID, operationID string,
	selected *localTransfer, current *session,
) (*localTransfer, *exit.Error) {
	if current == nil || current.instanceID != instanceID {
		return nil, exit.Named(exit.Conflict, "local_package_worker_changed",
			"local package transfer changed worker session")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	held := c.localTransfers[operationID]
	if held != nil && held.revision != selected.revision {
		return nil, exit.Named(exit.Conflict, "local_package_revision_changed",
			"operation %s already transfers another local package revision", operationID)
	}
	if held != nil && held.canceled {
		return nil, exit.New(exit.Canceled,
			"local package transfer %s was canceled", operationID)
	}
	if held == nil {
		held = selected
		c.localTransfers[operationID] = held
	}
	if held.instanceID != instanceID || held.bootID != current.bootID ||
		held.epoch != current.epoch {
		held.instanceID, held.bootID, held.epoch = instanceID, current.bootID,
			current.epoch
		held.status = make(map[string]localTransferStatus, len(held.files))
	}
	return held, nil
}

func (c *Orchestrator) cancelLocalTransfer(operationID string) *exit.Error {
	c.mu.Lock()
	transfer := c.localTransfers[operationID]
	if transfer == nil {
		c.mu.Unlock()
		return nil
	}
	transfer.canceled = true
	transfer.status = nil
	cancelUpload := transfer.uploadCancel
	s := c.sessions[transfer.bootID]
	c.mu.Unlock()
	if cancelUpload != nil {
		cancelUpload()
	}
	if s == nil || s.instanceID != transfer.instanceID || s.epoch != transfer.epoch {
		return exit.Unavailablef("local package transfer has no current worker session to abort")
	}
	abort := &pb.LocalPackageAbort{RecordOwnerEpoch: recordOwnerEpoch,
		ControlStreamEpoch: s.epoch, WorkerBootId: s.bootID, OperationId: operationID}
	if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_LocalPackageAbort{
		LocalPackageAbort: abort}}) {
		return exit.Unavailablef("worker control stream closed before local package abort")
	}
	select {
	case result := <-transfer.abortC:
		if result.outcome == pb.LocalPackageAbortOutcome_LOCAL_PACKAGE_ABORT_OUTCOME_REFUSED {
			return exit.Named(exit.Conflict, result.code,
				"worker refused local package abort: %s", result.detail)
		}
		return nil
	case <-s.ctx.Done():
		return exit.Unavailablef("worker control stream closed during local package abort")
	}
}

func (c *Orchestrator) localTransferVerified(operationID, revision string,
	current *session,
) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	transfer := c.localTransfers[operationID]
	if transfer == nil || transfer.revision != revision || current == nil ||
		transfer.instanceID != current.instanceID || transfer.bootID != current.bootID ||
		transfer.epoch != current.epoch {
		return false
	}
	for digest, file := range transfer.files {
		status := transfer.status[digest]
		if status.state != pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED ||
			status.received != file.length {
			return false
		}
	}
	return true
}

func transferFilesInOrder(transfer *localTransfer) []localTransferFile {
	rows := make([]localTransferFile, 0, len(transfer.files))
	for _, row := range transfer.files {
		rows = append(rows, row)
	}
	// The revision already proved digest order. Rebuild that order without trusting a map.
	for i := 0; i < len(rows); i++ {
		for j := i + 1; j < len(rows); j++ {
			if rows[j].filename < rows[i].filename {
				rows[i], rows[j] = rows[j], rows[i]
			}
		}
	}
	return rows
}

func (c *Orchestrator) localControl(instanceID string) (*worker, *session, *exit.Error) {
	return c.localControlContext(context.Background(), instanceID)
}

func (c *Orchestrator) localControlContext(ctx context.Context, instanceID string) (*worker, *session, *exit.Error) {
	for {
		if ctx.Err() != nil {
			return nil, nil, exit.Unavailablef("worker control wait canceled: %s", ctx.Err())
		}
		c.mu.Lock()
		w := c.workers[instanceID]
		var s *session
		if w != nil && w.snapshotAcknowledged {
			s = c.sessions[w.bootID]
		}
		gone := w == nil || w.exited
		var refused *exit.Error
		if w != nil {
			refused = w.refusal
		}
		c.mu.Unlock()
		switch {
		case s != nil:
			return w, s, nil
		case refused != nil:
			return nil, nil, refused
		case gone:
			return nil, nil, exit.New(exit.Failed,
				"the rented worker exited during local package transfer")
		}
		select {
		case <-ctx.Done():
			return nil, nil, exit.Unavailablef("worker control wait canceled: %s", ctx.Err())
		case <-c.done:
			return nil, nil, exit.Unavailablef("the daemon stopped during local package transfer")
		case <-time.After(20 * time.Millisecond):
		}
	}
}

func (c *Orchestrator) localStatus(operationID, digest string) localTransferStatus {
	c.mu.Lock()
	defer c.mu.Unlock()
	if transfer := c.localTransfers[operationID]; transfer != nil {
		return transfer.status[digest]
	}
	return localTransferStatus{}
}

func (c *Orchestrator) onLocalPackageFileStatus(current *session,
	frame *pb.LocalPackageFileStatus,
) {
	spelled := frame.Filename
	c.mu.Lock()
	defer c.mu.Unlock()
	transfer := c.localTransfers[frame.OperationId]
	if transfer == nil || transfer.canceled || current == nil ||
		transfer.instanceID != current.instanceID || transfer.bootID != current.bootID ||
		transfer.epoch != current.epoch {
		return
	}
	expected, ok := transfer.files[spelled]
	if !ok || !bytes.Equal(frame.Digest, expected.digest) || frame.Filename != expected.filename ||
		frame.Length != expected.length ||
		frame.ReceivedBytes > frame.Length {
		return
	}
	if frame.State != pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING &&
		frame.State != pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED &&
		frame.State != pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_REFUSED {
		return
	}
	prior := transfer.status[spelled]
	if prior.state == pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED &&
		frame.State != prior.state {
		return
	}
	received := frame.ReceivedBytes
	if received < prior.received {
		// A VERDICT IS ABOUT THE FILE, NOT ABOUT THE BYTE COUNT THAT CARRIED IT. The pod's
		// whole-request refusal (`localPackageRefusal`) quotes the grant's identity and
		// carries NO received bytes, so a refusal answering a file some earlier frame had
		// already reported progress on was dropped entire — code, words and all — and the
		// fetch then died on the generic `local_package_transfer_stuck` bound instead of
		// the pod's own reason. The count is what goes backwards; the verdict does not.
		if frame.State != pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_REFUSED {
			return
		}
		received = prior.received
	}
	transfer.status[spelled] = localTransferStatus{state: frame.State,
		received: received, safeCode: frame.SafeCode, safeDetail: frame.SafeDetail}
}

func (c *Orchestrator) onLocalPackageAbortStatus(current *session,
	frame *pb.LocalPackageAbortStatus,
) {
	c.mu.Lock()
	transfer := c.localTransfers[frame.OperationId]
	if transfer == nil || current == nil || transfer.instanceID != current.instanceID ||
		transfer.bootID != current.bootID || transfer.epoch != current.epoch {
		c.mu.Unlock()
		return
	}
	channel := transfer.abortC
	c.mu.Unlock()
	select {
	case channel <- localAbortStatus{outcome: frame.Outcome, code: frame.SafeCode,
		detail: frame.SafeDetail}:
	default:
	}
}

func (c *Orchestrator) replayLocalAborts(current *session, workerID string) {
	rows, problem := c.opt.Store.CanceledLocalPackages(workerID)
	if problem != nil {
		c.logf("cannot replay local package aborts for %s: %s", workerID, problem.Message)
		return
	}
	for _, row := range rows {
		_ = current.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_LocalPackageAbort{
			LocalPackageAbort: &pb.LocalPackageAbort{RecordOwnerEpoch: recordOwnerEpoch,
				ControlStreamEpoch: current.epoch, WorkerBootId: current.bootID,
				OperationId: row.ID}}})
	}
}

// localIssued answers whether this owner already issued revision on the live session
// and the pod has not refused it: prepared, or preparing and not yet reported. A second
// convergence of the same revision then waits on the pod's report instead of asking again.
func (c *Orchestrator) localIssued(w *worker, s *session, revision string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if w.desiredRefusal != nil || w.desiredEpoch != s.epoch {
		return false
	}
	if w.desiredLocal != nil && w.desiredLocal.Package.GetInstallationId() == revision {
		return true
	}
	return w.desiredUnpublishedPlacement != nil &&
		w.desiredUnpublishedPlacement.InstallationId == revision
}

// localOperation names the operation the pod prepared revision under, when this owner
// issued one: a model placement must bind over that operation, whoever converges it.
func (c *Orchestrator) localOperation(w *worker, revision string, fallback string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if w.desiredLocal != nil && w.desiredLocal.Package.GetInstallationId() == revision {
		return w.desiredLocal.OperationId
	}
	if w.desiredUnpublishedPlacement != nil && w.desiredUnpublishedPlacement.InstallationId == revision {
		return w.desiredUnpublishedPlacement.OperationId
	}
	return fallback
}

func (c *Orchestrator) issueLocalPackageSet(s *session, w *worker,
	selected *pb.DesiredLocalPackageSet,
) *exit.Error {
	if selected == nil {
		return exit.Internalf("cannot issue an empty local package set")
	}
	revision := selected.Package.GetInstallationId()
	c.mu.Lock()
	w.desiredLocal = cloneLocalPackageSet(selected)
	w.desiredUnpublishedPlacement = nil
	w.desiredPackages, w.desiredModels, w.desiredDownloadSets = nil, nil, nil
	w.desiredEpoch = s.epoch
	held := w.holdsLocalInstallation(revision)
	c.mu.Unlock()
	if held {
		// The pod reports this exact revision already: nothing to prepare, and asking would
		// be refused as busy. A reconnect re-issue lands here for the set the pod kept.
		c.logf("worker %s already holds local revision %s; nothing to prepare",
			w.instanceID, shortDigest(revision))
		return nil
	}
	call := &pb.PrepareLocalPackageCall{Claim: s.claim, LocalPackageSet: cloneLocalPackageSet(selected)}
	return c.issueThroughHost(s, w, hostLabel("local_package_set", selected.OperationId),
		func(ctx context.Context) (grpc.ServerStreamingClient[pb.PrepareEvent], error) {
			return s.host.PrepareLocalPackage(ctx, call)
		})
}

// ConvergeUnpublishedPlacement binds exact downloaded models to code already admitted under one
// local revision. Creator names logical refs only; Runtime authors the joined PlacementSet.
func (c *Orchestrator) ConvergeUnpublishedPlacement(instanceID, operationID,
	localRevisionDigest string, models []*pb.DownloadModelRef,
) *exit.Error {
	return c.convergeUnpublishedModels(instanceID, operationID, localRevisionDigest, models, nil)
}

func (c *Orchestrator) convergeUnpublishedModels(instanceID, operationID, localRevisionDigest string, models []*pb.DownloadModelRef, native []*pb.NativeModelBinding) *exit.Error {
	if operationID == "" || localRevisionDigest == "" || len(models)+len(native) == 0 {
		return exit.Named(exit.Validation, "private_placement_incomplete",
			"local package placement requires operation, exact revision, and models")
	}
	if len(models) > 0 && c.opt.RentalPackageSet == nil {
		return exit.Named(exit.Unavailable, "rental.package_set_signer_missing",
			"this Cozy daemon has no package_set signer")
	}
	c.mu.Lock()
	w := c.workers[instanceID]
	var s *session
	if w != nil {
		s = c.sessions[w.bootID]
	}
	c.mu.Unlock()
	if w == nil || w.spec.Connection == nil {
		return exit.New(exit.NotFound, "no attached rental worker %s on this host", instanceID)
	}
	if s == nil {
		return exit.Unavailablef("worker %s holds no claimed control stream", instanceID)
	}
	if problem := requireMixedModelInputs(s, len(models) > 0 && len(native) > 0); problem != nil {
		return problem
	}
	// The models bind OVER the local revision, so the pod must hold that revision first:
	// `ConvergeLocalPackage` only ISSUES the prepare, and an unpublished package placement sent on its
	// heels is refused `private_placement_invalid: unpublished package placement revision is not
	// prepared` (found live, L4 `shiranui`, cl-101). Wait on the pod's own report of it.
	if problem := c.awaitLocalInstallation(instanceID, localRevisionDigest); problem != nil {
		return problem
	}
	var downloadSet []byte
	if len(models) > 0 {
		var problem *exit.Error
		downloadSet, problem = c.opt.RentalPackageSet(nil, models)
		if problem != nil {
			return problem
		}
	}
	revision := localRevisionDigest
	if len(downloadSet) == 0 && len(native) == 0 {
		return exit.Named(exit.Validation, "private_placement_download_set_incomplete",
			"local package placement download set is incomplete")
	}
	operationID = c.localOperation(w, revision, operationID)
	selected := &pb.DesiredPrivatePlacementSet{OperationId: operationID,
		InstallationId: revision, DownloadDelegation: downloadSet, NativeModels: native}
	return c.issueUnpublishedPlacementSet(s, w, selected)
}

func (c *Orchestrator) issueUnpublishedPlacementSet(s *session, w *worker,
	selected *pb.DesiredPrivatePlacementSet,
) *exit.Error {
	if selected == nil {
		return exit.Internalf("cannot issue an empty unpublished package placement set")
	}
	if problem := requireAdapterDownloadPeer(s, selected.DownloadDelegation); problem != nil {
		return problem
	}
	for _, native := range selected.NativeModels {
		if native != nil && len(native.Adapters) > 0 {
			if problem := requireAdapterPeer(s, true); problem != nil {
				return problem
			}
			break
		}
	}
	c.mu.Lock()
	w.desiredLocal, w.desiredPackages, w.desiredModels, w.desiredDownloadSets = nil, nil, nil, nil
	w.desiredUnpublishedPlacement = cloneUnpublishedPlacementSet(selected)
	w.desiredEpoch = s.epoch
	c.mu.Unlock()
	call := &pb.PreparePrivatePlacementCall{SupportsModelMaterializationRecovery: true, Claim: s.claim, PrivatePlacementSet: cloneUnpublishedPlacementSet(selected)}
	return c.issueThroughHost(s, w, hostLabel("private_placement_set", selected.OperationId),
		func(ctx context.Context) (grpc.ServerStreamingClient[pb.PrepareEvent], error) {
			return s.host.PreparePrivatePlacement(ctx, call)
		})
}

func cloneLocalPackageSet(in *pb.DesiredLocalPackageSet) *pb.DesiredLocalPackageSet {
	if in == nil {
		return nil
	}
	return proto.Clone(in).(*pb.DesiredLocalPackageSet)
}

func cloneUnpublishedPlacementSet(in *pb.DesiredPrivatePlacementSet) *pb.DesiredPrivatePlacementSet {
	if in == nil {
		return nil
	}
	return proto.Clone(in).(*pb.DesiredPrivatePlacementSet)
}
