package orchestrator

import (
	"bytes"
	"context"
	"errors"
	"io"
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
	source             []byte
	instanceID, bootID string
	epoch              uint64
	canceled           bool
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
	revision localpackage.Revision, uploadedBootID string, uploaded func(bootID string) *exit.Error,
) *exit.Error {
	selected, transfer, problem := localSelection(operationID, revision)
	if problem != nil {
		return problem
	}
	w, s, problem := c.localControl(instanceID)
	if problem != nil {
		return problem
	}
	if problem := requireLocalPackageCapacity(s, len(revision.Files)); problem != nil {
		return problem
	}
	// One local preparation at a time per pod: a run and the editable refresh converging
	// the same revision must not each ask the pod to prepare over the other's placement.
	w.localMu.Lock()
	defer w.localMu.Unlock()
	if c.localIssued(w, s, revision.Digest) {
		c.logf("worker %s already has local revision %s issued on this session; waiting on it",
			instanceID, shortDigest(revision.Digest))
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
	if problem := c.hostNothing(instanceID, revision.Digest); problem != nil {
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
		holdsRevision = w.holdsLocalRevision(revision)
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
func (w *worker) holdsLocalRevision(revision string) bool {
	if !validDigest(revision) || len(w.setBytes) == 0 || len(w.setDigest) == 0 {
		return false
	}
	doc, err := canonical.Read(w.setBytes, &pb.PlacementSet{})
	if err != nil {
		return false
	}
	for _, row := range doc.List("placements") {
		if row.Sub("development").Str("local_revision_digest") != revision {
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
func (c *Orchestrator) awaitLocalRevision(instanceID, revision string) *exit.Error {
	for {
		c.mu.Lock()
		w := c.workers[instanceID]
		var held, current bool
		var refused, desiredRefusal *exit.Error
		if w != nil {
			held = w.holdsLocalRevision(revision)
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

func localSelection(operationID string, revision localpackage.Revision) (
	*pb.DesiredLocalPackageSet, *localTransfer, *exit.Error,
) {
	source, err := canonical.Raw(revision.SourceDigest)
	if err != nil || operationID == "" || revision.Package == "" || revision.Release == "" ||
		!validDigest(revision.Digest) || len(revision.Files) == 0 ||
		len(revision.Files) > pb.MaxLocalPackageFiles {
		return nil, nil, exit.Named(exit.Structural, "local_package_revision_invalid",
			"local package revision is incomplete")
	}
	selected := &pb.DesiredLocalPackageSet{OperationId: operationID,
		Package: &pb.DevelopmentPackage{Package: revision.Package, Release: revision.Release,
			SourceDigest: source}}
	localDigest, _ := canonical.Raw(revision.Digest)
	selected.Package.LocalRevisionDigest = localDigest
	transfer := &localTransfer{revision: revision.Digest, source: source,
		files:  make(map[string]localTransferFile, len(revision.Files)),
		status: make(map[string]localTransferStatus, len(revision.Files)),
		abortC: make(chan localAbortStatus, 1)}
	var prior []byte
	var total uint64
	projects := 0
	for _, file := range revision.Files {
		digest, err := canonical.Raw(file.Digest)
		if file.Kind == "project" {
			projects++
		} else if file.Kind != "dependency" {
			return nil, nil, exit.Named(exit.Structural, "local_package_file_kind_invalid",
				"local package file %s has kind %q", file.Filename, file.Kind)
		}
		if err != nil || file.Length <= 0 || file.Length > maxLocalWheelBytes ||
			(prior != nil && bytes.Compare(prior, digest) >= 0) ||
			file.Length > maxLocalWheelSetBytes-int64(total) {
			return nil, nil, exit.Named(exit.Structural, "local_package_file_invalid",
				"local package file %s has invalid identity or bounds", file.Filename)
		}
		total += uint64(file.Length)
		spelled := file.Digest
		transfer.files[spelled] = localTransferFile{digest: digest, filename: file.Filename,
			path: file.Path, project: file.Kind == "project", length: uint64(file.Length)}
		selected.Files = append(selected.Files, &pb.LocalPackageFileRef{Digest: digest,
			Filename: file.Filename, Length: uint64(file.Length)})
		prior = digest
	}
	if projects != 1 || len(transfer.files) != len(revision.Files) {
		return nil, nil, exit.Named(exit.Structural, "local_package_project_wheel_count",
			"local package revision must carry one project wheel and unique files")
	}
	return selected, transfer, nil
}

// LocalPackageSelection is the passive code-preparation document. Calling it
// neither converges worker state nor creates an execution attempt.
func LocalPackageSelection(operationID string, revision localpackage.Revision) (*pb.DesiredLocalPackageSet, *exit.Error) {
	selected, _, problem := localSelection(operationID, revision)
	return selected, problem
}

// localPackageUploadProblem shares preparation's terminal/refusable distinction.
// A host verdict on an exact header cannot improve by reconnecting.
func localPackageUploadProblem(err error) *exit.Error {
	if result := classifyPrepareEnd(err); result.err == nil {
		return exit.Named(exit.Structural, "local_package_upload_refused", "worker refused unpublished wheel upload: %s", result.refusal)
	}
	return exit.Named(exit.Unavailable, "local_package_upload_interrupted", "unpublished upload interrupted; acknowledged bytes can be resumed: %s", err)
}

// gRPC Send can report EOF before exposing the server's actual terminal status.
func localPackageUploadSendProblem(stream pb.PodHost_LocalPackageUploadClient, err error) *exit.Error {
	if errors.Is(err, io.EOF) {
		if _, terminal := stream.Recv(); terminal != nil {
			err = terminal
		}
	}
	return localPackageUploadProblem(err)
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
	if problem := requireLocalPackageCapacity(current, len(transfer.files)); problem != nil {
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
	defer cancel()
	stream, err := current.host.LocalPackageUpload(ctx)
	if err != nil {
		return localPackageUploadProblem(err)
	}
	defer stream.CloseSend()
	if err := stream.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Header{
		Header: &pb.LocalPackageUploadHeader{Claim: current.claim, OperationId: operationID,
			SourceDigest: transfer.source, File: &pb.LocalPackageFileRef{Digest: selected.digest,
				Filename: selected.filename, Length: selected.length}},
	}}); err != nil {
		return localPackageUploadSendProblem(stream, err)
	}
	file, err := os.Open(selected.path)
	if err != nil {
		return exit.Named(exit.Structural, "local_package_wheel_unreadable", "cannot read captured wheel: %s", err)
	}
	defer file.Close()
	buffer := make([]byte, 1<<20)
	sent := uint64(0)
	first := true
	for {
		status, err := stream.Recv()
		if err != nil {
			return localPackageUploadProblem(err)
		}
		if status.OperationId != operationID || !bytes.Equal(status.SourceDigest, transfer.source) ||
			!bytes.Equal(status.Digest, selected.digest) || status.Filename != selected.filename ||
			status.Length != selected.length || status.ReceivedBytes > selected.length ||
			(!first && status.ReceivedBytes != sent) {
			return exit.Named(exit.Conflict, "local_package_upload_identity_changed", "worker returned another captured file or unexpected upload offset")
		}
		c.onLocalPackageFileStatus(current, status)
		if status.State == pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_REFUSED {
			return exit.Named(exit.Failed, status.SafeCode, "worker refused captured wheel %s: %s", selected.filename, status.SafeDetail)
		}
		if status.State == pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED {
			if status.ReceivedBytes != selected.length {
				return exit.Named(exit.Conflict, "local_package_upload_incomplete", "worker verified an incomplete captured wheel")
			}
			return nil
		}
		if status.State != pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING || status.SafeCode != "" {
			return exit.Named(exit.Unavailable, "local_package_upload_stopped", "worker stopped captured wheel upload: %s", status.SafeDetail)
		}
		if first {
			if _, err := file.Seek(int64(status.ReceivedBytes), io.SeekStart); err != nil {
				return exit.Internalf("cannot resume captured wheel: %s", err)
			}
			sent, first = status.ReceivedBytes, false
		}
		c.mu.Lock()
		canceled := transfer.canceled
		c.mu.Unlock()
		if canceled {
			return exit.New(exit.Canceled, "unpublished package upload was cancelled")
		}
		remaining := selected.length - sent
		if remaining == 0 {
			return exit.Named(exit.Conflict, "local_package_upload_unverified", "worker has all bytes but did not verify the captured wheel")
		}
		chunk := buffer
		if remaining < uint64(len(chunk)) {
			chunk = chunk[:int(remaining)]
		}
		n, err := io.ReadFull(file, chunk)
		if err != nil {
			return exit.Named(exit.Conflict, "local_package_wheel_changed", "captured wheel changed during upload: %s", err)
		}
		if err := stream.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Chunk{
			Chunk: &pb.LocalPackageUploadChunk{Offset: sent, Data: chunk[:n]},
		}}); err != nil {
			return localPackageUploadSendProblem(stream, err)
		}
		sent += uint64(n)
	}
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
	s := c.sessions[transfer.bootID]
	c.mu.Unlock()
	if s == nil || s.instanceID != transfer.instanceID || s.epoch != transfer.epoch {
		return exit.Unavailablef("local package transfer has no current worker session to abort")
	}
	localDigest, localErr := canonical.Raw(transfer.revision)
	if localErr != nil {
		return exit.Internalf("cannot decode local package abort identity: %s", localErr)
	}
	abort := &pb.LocalPackageAbort{RecordOwnerEpoch: recordOwnerEpoch,
		ControlStreamEpoch: s.epoch, WorkerBootId: s.bootID,
		OperationId: operationID, SourceDigest: append([]byte(nil), transfer.source...),
		LocalRevisionDigest: localDigest}
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
			if bytes.Compare(rows[j].digest, rows[i].digest) < 0 {
				rows[i], rows[j] = rows[j], rows[i]
			}
		}
	}
	return rows
}

func (c *Orchestrator) localControl(instanceID string) (*worker, *session, *exit.Error) {
	for {
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
	spelled, err := canonical.Spell(frame.Digest)
	c.mu.Lock()
	defer c.mu.Unlock()
	transfer := c.localTransfers[frame.OperationId]
	if err != nil || transfer == nil || transfer.canceled || current == nil ||
		transfer.instanceID != current.instanceID || transfer.bootID != current.bootID ||
		transfer.epoch != current.epoch ||
		!bytes.Equal(frame.SourceDigest, transfer.source) {
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
		transfer.bootID != current.bootID || transfer.epoch != current.epoch ||
		!bytes.Equal(frame.SourceDigest, transfer.source) {
		c.mu.Unlock()
		return
	}
	digest, err := canonical.Spell(frame.LocalRevisionDigest)
	if err != nil || digest != transfer.revision {
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
		install, installProblem := c.opt.Store.Install(row.InstallID)
		if installProblem != nil || install == nil {
			c.logf("canceled request %s has no source install for local package abort", row.ID)
			continue
		}
		source, sourceErr := canonical.Raw(install.SourceDigest)
		revision, revisionErr := canonical.Raw(row.LocalPackageDigest)
		if sourceErr != nil || revisionErr != nil {
			c.logf("canceled request %s has invalid local package abort identity", row.ID)
			continue
		}
		_ = current.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_LocalPackageAbort{
			LocalPackageAbort: &pb.LocalPackageAbort{RecordOwnerEpoch: recordOwnerEpoch,
				ControlStreamEpoch: current.epoch, WorkerBootId: current.bootID,
				OperationId: row.ID, SourceDigest: source, LocalRevisionDigest: revision}}})
	}
}

// localIssued answers whether this owner already issued revision on the live session
// and the pod has not refused it: prepared, or preparing and not yet reported. A second
// convergence of the same revision then waits on the pod's report instead of asking again.
func (c *Orchestrator) localIssued(w *worker, s *session, revision string) bool {
	raw, err := canonical.Raw(revision)
	if err != nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if w.desiredRefusal != nil || w.desiredEpoch != s.epoch {
		return false
	}
	if w.desiredLocal != nil && bytes.Equal(w.desiredLocal.Package.GetLocalRevisionDigest(), raw) {
		return true
	}
	return w.desiredUnpublishedPlacement != nil &&
		bytes.Equal(w.desiredUnpublishedPlacement.LocalRevisionDigest, raw)
}

// localOperation names the operation the pod prepared revision under, when this owner
// issued one: a model placement must bind over that operation, whoever converges it.
func (c *Orchestrator) localOperation(w *worker, revision []byte, fallback string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if w.desiredLocal != nil && bytes.Equal(w.desiredLocal.Package.GetLocalRevisionDigest(), revision) {
		return w.desiredLocal.OperationId
	}
	if w.desiredUnpublishedPlacement != nil && bytes.Equal(w.desiredUnpublishedPlacement.LocalRevisionDigest, revision) {
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
	if problem := requireLocalPackageCapacity(s, len(selected.Files)); problem != nil {
		return problem
	}
	revision, err := canonical.Spell(selected.Package.GetLocalRevisionDigest())
	if err != nil {
		return exit.Internalf("cannot spell the local revision digest: %s", err)
	}
	c.mu.Lock()
	w.desiredLocal = cloneLocalPackageSet(selected)
	w.desiredUnpublishedPlacement = nil
	w.desiredPackages, w.desiredModels, w.desiredDownloadSets = nil, nil, nil
	w.desiredEpoch = s.epoch
	held := w.holdsLocalRevision(revision)
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
	if operationID == "" || !validDigest(localRevisionDigest) || len(models)+len(native) == 0 {
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
	if problem := c.awaitLocalRevision(instanceID, localRevisionDigest); problem != nil {
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
	revision, err := canonical.Raw(localRevisionDigest)
	if err != nil || (len(downloadSet) == 0 && len(native) == 0) {
		return exit.Named(exit.Validation, "private_placement_download_set_incomplete",
			"local package placement download set is incomplete")
	}
	operationID = c.localOperation(w, revision, operationID)
	selected := &pb.DesiredPrivatePlacementSet{OperationId: operationID,
		LocalRevisionDigest: revision, DownloadDelegation: downloadSet, NativeModels: native}
	return c.issueUnpublishedPlacementSet(s, w, selected)
}

func (c *Orchestrator) issueUnpublishedPlacementSet(s *session, w *worker,
	selected *pb.DesiredPrivatePlacementSet,
) *exit.Error {
	if selected == nil {
		return exit.Internalf("cannot issue an empty unpublished package placement set")
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
	out := &pb.DesiredLocalPackageSet{OperationId: in.OperationId}
	if in.Package != nil {
		out.Package = &pb.DevelopmentPackage{Package: in.Package.Package,
			Release: in.Package.Release, SourceDigest: append([]byte(nil), in.Package.SourceDigest...),
			LocalRevisionDigest: append([]byte(nil), in.Package.LocalRevisionDigest...)}
	}
	for _, file := range in.Files {
		if file != nil {
			out.Files = append(out.Files, &pb.LocalPackageFileRef{Digest: append([]byte(nil), file.Digest...),
				Filename: file.Filename, Length: file.Length})
		}
	}
	return out
}

func cloneUnpublishedPlacementSet(in *pb.DesiredPrivatePlacementSet) *pb.DesiredPrivatePlacementSet {
	if in == nil {
		return nil
	}
	return proto.Clone(in).(*pb.DesiredPrivatePlacementSet)
}

// The Host advertises the intersection with its installed Runtime. ClaimAck alone
// can describe a newer Runtime behind an older Host, so probe before sending bytes.
func requireLocalPackageCapacity(s *session, files int) *exit.Error {
	if files <= pb.LegacyMaxLocalPackageFiles {
		return nil
	}
	if s == nil || s.host == nil {
		return exit.Unavailablef("unpublished package capacity awaits the claimed Host")
	}
	ctx, cancel := context.WithTimeout(s.ctx, 10*time.Second)
	defer cancel()
	info, err := s.host.ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
	if err != nil {
		return exit.Unavailablef("unpublished package capacity probe is unavailable")
	}
	if info == nil || info.WireMinor < pb.ExpandedLocalPackageFilesWireMinor {
		return exit.Named(exit.Conflict, "local_package_worker_capacity_unsupported",
			"unpublished package revision has %d wheels; this worker supports at most %d", files, pb.LegacyMaxLocalPackageFiles).
			WithRemedy("select a worker with protocol minor %d or newer", pb.ExpandedLocalPackageFilesWireMinor)
	}
	return nil
}
