package orchestrator

import (
	"bytes"
	"context"
	"os"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/privatepackage"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type privateTransfer struct {
	revision           string
	source             []byte
	instanceID, bootID string
	epoch              uint64
	canceled           bool
	files              map[string]privateTransferFile
	status             map[string]privateTransferStatus
	abortC             chan privateAbortStatus
}

type privateAbortStatus struct {
	outcome pb.PrivatePackageAbortOutcome
	code    string
	detail  string
}

type privateTransferFile struct {
	digest         []byte
	filename, path string
	kind           pb.LocalDownloadKind
	length         uint64
}

// PrivateWheel is one wheel of an unpublished revision as this daemon holds it on disk.
type PrivateWheel struct {
	Digest, Filename, Kind, Path string
	Length                       int64
}

// PrivateWheelGrantSource is th-094's replacement for the byte relay. It PUTs every wheel the
// store does not already hold and answers with one short-lived read capability per wheel, in
// the order asked. It is a callback for the same reason RentalPackageSet is: the orchestrator
// holds no Tensorhub client, and the account, the token, and the byte plane belong to the
// entrypoint. Calling it again after a stall re-mints the capabilities without re-uploading.
type PrivateWheelGrantSource func(context.Context, []PrivateWheel) ([]string, *exit.Error)

// Object bounds, not control-stream bounds. They are what one revision may cost the store and
// the pod's disk, and they match the pod ledger's own quota. Nothing on the wire carries them:
// the frame that names these wheels carries 33 URLs at most.
const (
	maxPrivateWheelBytes    = int64(512 << 20)
	maxPrivateWheelSetBytes = int64(1 << 30)
)

type privateTransferStatus struct {
	state      pb.PrivatePackageFileState
	received   uint64
	safeCode   string
	safeDetail string
}

// ConvergePrivatePackage streams one sealed editable revision and then selects it. The
// request row is the recovery authority: once every file acknowledgement is durable there,
// reconnect sends only the exact DesiredPrivatePackageSet and the pod replays its ledger.
func (c *Orchestrator) ConvergePrivatePackage(instanceID, requestID string,
	revision privatepackage.Revision, uploadedBootID string,
) *exit.Error {
	selected, transfer, problem := privateSelection(requestID, revision)
	if problem != nil {
		return problem
	}
	w, s, problem := c.privateControl(instanceID)
	if problem != nil {
		return problem
	}
	if uploadedBootID != "" && uploadedBootID != s.bootID {
		// The exact revision remains durable, but carrier verification is boot-scoped.
		// Re-prove every byte to the replacement pod before moving the durable boot marker.
		uploadedBootID = ""
	}
	if uploadedBootID == "" {
		for uploadedBootID == "" {
			if problem := c.transferPrivatePackage(instanceID, requestID, transfer); problem != nil {
				return problem
			}
			w, s, problem = c.privateControl(instanceID)
			if problem != nil {
				return problem
			}
			if !c.privateTransferVerified(requestID, transfer.revision, s) {
				continue
			}
			if problem := c.opt.Store.MarkPrivatePackageUploaded(requestID, revision.Digest,
				s.bootID); problem != nil {
				return problem
			}
			uploadedBootID = s.bootID
		}
	}
	for {
		if problem := c.issuePrivatePackageSet(s, w, selected); problem == nil {
			c.mu.Lock()
			delete(c.privateTransfers, requestID)
			c.mu.Unlock()
			return nil
		}
		w, s, problem = c.privateControl(instanceID)
		if problem != nil {
			return problem
		}
	}
}

func privateSelection(operationID string, revision privatepackage.Revision) (
	*pb.DesiredPrivatePackageSet, *privateTransfer, *exit.Error,
) {
	source, err := canonical.Raw(revision.SourceDigest)
	if err != nil || operationID == "" || revision.Package == "" || revision.Release == "" ||
		!validDigest(revision.Digest) || len(revision.Files) == 0 ||
		len(revision.Files) > pb.MaxPrivatePackageFiles {
		return nil, nil, exit.Named(exit.Structural, "private_package_revision_invalid",
			"private package revision is incomplete")
	}
	selected := &pb.DesiredPrivatePackageSet{OperationId: operationID,
		Package: &pb.DevelopmentPackage{Package: revision.Package, Release: revision.Release,
			SourceDigest: source}}
	privateDigest, _ := canonical.Raw(revision.Digest)
	selected.Package.PrivateRevisionDigest = privateDigest
	transfer := &privateTransfer{revision: revision.Digest, source: source,
		files:  make(map[string]privateTransferFile, len(revision.Files)),
		status: make(map[string]privateTransferStatus, len(revision.Files)),
		abortC: make(chan privateAbortStatus, 1)}
	var prior []byte
	var total uint64
	projects := 0
	for _, file := range revision.Files {
		digest, err := canonical.Raw(file.Digest)
		kind := pb.LocalDownloadKind_LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL
		if file.Kind == "project" {
			kind = pb.LocalDownloadKind_LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL
			projects++
		} else if file.Kind != "dependency" {
			return nil, nil, exit.Named(exit.Structural, "private_package_file_kind_invalid",
				"private package file %s has kind %q", file.Filename, file.Kind)
		}
		if err != nil || file.Length <= 0 || file.Length > maxPrivateWheelBytes ||
			(prior != nil && bytes.Compare(prior, digest) >= 0) ||
			file.Length > maxPrivateWheelSetBytes-int64(total) {
			return nil, nil, exit.Named(exit.Structural, "private_package_file_invalid",
				"private package file %s has invalid identity or bounds", file.Filename)
		}
		total += uint64(file.Length)
		spelled := file.Digest
		transfer.files[spelled] = privateTransferFile{digest: digest, filename: file.Filename,
			path: file.Path, kind: kind, length: uint64(file.Length)}
		selected.Files = append(selected.Files, &pb.PrivatePackageFileRef{Digest: digest,
			Filename: file.Filename, Kind: kind, Length: uint64(file.Length)})
		prior = digest
	}
	if projects != 1 || len(transfer.files) != len(revision.Files) {
		return nil, nil, exit.Named(exit.Structural, "private_package_project_wheel_count",
			"private package revision must carry one project wheel and unique files")
	}
	return selected, transfer, nil
}

// transferPrivatePackage is th-094's whole owner half. It no longer sends a byte: the wheels go
// to the store under capabilities scoped to their own digests, and the pod is handed one read
// capability per wheel to spend against its own download edge.
//
// The loop exists because a capability is short-lived by design. A stall -- the pod saying it
// has a partial prefix and needs a fresh URL -- is answered by minting new grants over the same
// objects, which costs one HTTP round and re-uploads nothing, and the pod resumes from the
// prefix it already holds.
func (c *Orchestrator) transferPrivatePackage(instanceID, operationID string,
	transfer *privateTransfer,
) *exit.Error {
	if c.opt.PrivateWheels == nil {
		return exit.Named(exit.Structural, "private_package_grants_unwired",
			"this daemon cannot upload a private package revision")
	}
	stale := 0
	for {
		_, selectedSession, problem := c.privateControl(instanceID)
		if problem != nil {
			return problem
		}
		held, problem := c.bindPrivateTransfer(instanceID, operationID, transfer, selectedSession)
		if problem != nil {
			return problem
		}
		ordered := transferFilesInOrder(held)
		if problem := c.provePrivateWheelsUnchanged(ordered); problem != nil {
			return problem
		}
		granted, problem := c.grantPrivateWheels(selectedSession, ordered)
		if problem != nil {
			return problem
		}
		// Clear the previous attempt's unverified statuses. A stall left behind by the round
		// these grants were minted to answer would be read as this round's stall, and the loop
		// would re-grant forever without ever waiting for an answer.
		before := c.clearPrivateStalls(operationID)
		frame := &pb.PrivatePackageFetchRequest{RecordOwnerEpoch: recordOwnerEpoch,
			ControlStreamEpoch: selectedSession.epoch,
			WorkerBootId:       selectedSession.bootID, OperationId: operationID,
			SourceDigest: held.source, Files: granted}
		if !selectedSession.send(&pb.RecordOwnerFrame{
			Msg: &pb.RecordOwnerFrame_PrivatePackageFetchRequest{
				PrivatePackageFetchRequest: frame}}) {
			continue
		}
		done, problem := c.awaitPrivateTransfer(instanceID, operationID, ordered, selectedSession)
		if problem != nil {
			return problem
		}
		if done {
			return nil
		}
		// A stall is only worth answering while it is buying ground. A pod that keeps landing
		// bytes gets as many capabilities as it needs; one that lands none twice running is not
		// stalling, it is failing, and saying so beats an unbounded loop.
		if after := c.privateProgress(operationID); after > before {
			stale = 0
		} else if stale++; stale > maxPrivateStalls {
			return exit.Named(exit.Failed, "private_package_transfer_stuck",
				"the worker landed no private package bytes across %d re-granted attempts",
				stale)
		}
	}
}

// maxPrivateStalls is how many consecutive no-progress rounds a transfer may spend. Each one
// costs a grant round trip and nothing else, so the bound is about ending, not about cost.
const maxPrivateStalls = 3

// clearPrivateStalls drops every unverified status and reports the durable byte total the pod
// has acknowledged. A verified file is never cleared: it is settled.
func (c *Orchestrator) clearPrivateStalls(operationID string) uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	transfer := c.privateTransfers[operationID]
	if transfer == nil {
		return 0
	}
	var total uint64
	for digest, status := range transfer.status {
		if status.state == pb.PrivatePackageFileState_PRIVATE_PACKAGE_FILE_STATE_VERIFIED {
			total += status.received
			continue
		}
		total += status.received
		delete(transfer.status, digest)
	}
	return total
}

func (c *Orchestrator) privateProgress(operationID string) uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	transfer := c.privateTransfers[operationID]
	if transfer == nil {
		return 0
	}
	var total uint64
	for _, status := range transfer.status {
		total += status.received
	}
	return total
}

// provePrivateWheelsUnchanged reads the local files the revision named. The digest is the
// revision's identity, so a wheel edited between staging and transfer is a different revision
// wearing this one's name, and the store would refuse it anyway under the signed checksum.
func (c *Orchestrator) provePrivateWheelsUnchanged(ordered []privateTransferFile) *exit.Error {
	for _, selected := range ordered {
		info, err := os.Stat(selected.path)
		if err != nil {
			return exit.Named(exit.Structural, "private_package_wheel_unreadable", "%s", err)
		}
		if !info.Mode().IsRegular() || info.Size() != int64(selected.length) {
			return exit.Named(exit.Structural, "private_package_wheel_changed",
				"private package wheel %s changed before transfer", selected.filename)
		}
	}
	return nil
}

// grantPrivateWheels turns one ordered wheel set into the frame's grants. The URLs are
// memory-only: they are never journaled, never logged, and never survive this frame.
func (c *Orchestrator) grantPrivateWheels(current *session, ordered []privateTransferFile) (
	[]*pb.PrivatePackageFileGrant, *exit.Error,
) {
	wheels := make([]PrivateWheel, 0, len(ordered))
	for _, selected := range ordered {
		spelled, err := canonical.Spell(selected.digest)
		if err != nil {
			return nil, exit.Internalf("cannot spell a private package wheel digest: %s", err)
		}
		kind := "dependency_wheel"
		if selected.kind == pb.LocalDownloadKind_LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL {
			kind = "project_wheel"
		}
		wheels = append(wheels, PrivateWheel{Digest: spelled, Filename: selected.filename,
			Kind: kind, Path: selected.path, Length: int64(selected.length)})
	}
	urls, problem := c.opt.PrivateWheels(current.ctx, wheels)
	if problem != nil {
		return nil, problem
	}
	if len(urls) != len(ordered) {
		return nil, exit.Internalf("private package grants answered for %d of %d wheels",
			len(urls), len(ordered))
	}
	grants := make([]*pb.PrivatePackageFileGrant, 0, len(ordered))
	for i, selected := range ordered {
		if urls[i] == "" || len(urls[i]) > pb.MaxPrivatePackageGrantURLBytes {
			return nil, exit.Named(exit.Internal, "private_package_grant_url_invalid",
				"the read capability for %s is empty or over its bound", selected.filename)
		}
		grants = append(grants, &pb.PrivatePackageFileGrant{Digest: selected.digest,
			Filename: selected.filename, Kind: selected.kind, Length: selected.length,
			Url: urls[i]})
	}
	return grants, nil
}

func (c *Orchestrator) bindPrivateTransfer(instanceID, operationID string,
	selected *privateTransfer, current *session,
) (*privateTransfer, *exit.Error) {
	if current == nil || current.instanceID != instanceID {
		return nil, exit.Named(exit.Conflict, "private_package_worker_changed",
			"private package transfer changed worker session")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	held := c.privateTransfers[operationID]
	if held != nil && held.revision != selected.revision {
		return nil, exit.Named(exit.Conflict, "private_package_revision_changed",
			"operation %s already transfers another private package revision", operationID)
	}
	if held != nil && held.canceled {
		return nil, exit.New(exit.Canceled,
			"private package transfer %s was canceled", operationID)
	}
	if held == nil {
		held = selected
		c.privateTransfers[operationID] = held
	}
	if held.instanceID != instanceID || held.bootID != current.bootID ||
		held.epoch != current.epoch {
		held.instanceID, held.bootID, held.epoch = instanceID, current.bootID,
			current.epoch
		held.status = make(map[string]privateTransferStatus, len(held.files))
	}
	return held, nil
}

func (c *Orchestrator) cancelPrivateTransfer(operationID string) *exit.Error {
	c.mu.Lock()
	transfer := c.privateTransfers[operationID]
	if transfer == nil {
		c.mu.Unlock()
		return nil
	}
	transfer.canceled = true
	transfer.status = nil
	s := c.sessions[transfer.bootID]
	c.mu.Unlock()
	if s == nil || s.instanceID != transfer.instanceID || s.epoch != transfer.epoch {
		return exit.Unavailablef("private package transfer has no current worker session to abort")
	}
	privateDigest, privateErr := canonical.Raw(transfer.revision)
	if privateErr != nil {
		return exit.Internalf("cannot decode private package abort identity: %s", privateErr)
	}
	abort := &pb.PrivatePackageAbort{RecordOwnerEpoch: recordOwnerEpoch,
		ControlStreamEpoch: s.epoch, WorkerBootId: s.bootID,
		OperationId: operationID, SourceDigest: append([]byte(nil), transfer.source...),
		PrivateRevisionDigest: privateDigest}
	if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_PrivatePackageAbort{
		PrivatePackageAbort: abort}}) {
		return exit.Unavailablef("worker control stream closed before private package abort")
	}
	select {
	case result := <-transfer.abortC:
		if result.outcome == pb.PrivatePackageAbortOutcome_PRIVATE_PACKAGE_ABORT_OUTCOME_REFUSED {
			return exit.Named(exit.Conflict, result.code,
				"worker refused private package abort: %s", result.detail)
		}
		return nil
	case <-s.ctx.Done():
		return exit.Unavailablef("worker control stream closed during private package abort")
	}
}

func (c *Orchestrator) privateTransferVerified(operationID, revision string,
	current *session,
) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	transfer := c.privateTransfers[operationID]
	if transfer == nil || transfer.revision != revision || current == nil ||
		transfer.instanceID != current.instanceID || transfer.bootID != current.bootID ||
		transfer.epoch != current.epoch {
		return false
	}
	for digest, file := range transfer.files {
		status := transfer.status[digest]
		if status.state != pb.PrivatePackageFileState_PRIVATE_PACKAGE_FILE_STATE_VERIFIED ||
			status.received != file.length {
			return false
		}
	}
	return true
}

func transferFilesInOrder(transfer *privateTransfer) []privateTransferFile {
	rows := make([]privateTransferFile, 0, len(transfer.files))
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

func (c *Orchestrator) privateControl(instanceID string) (*worker, *session, *exit.Error) {
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
				"the rented worker exited during private package transfer")
		}
		select {
		case <-c.done:
			return nil, nil, exit.Unavailablef("the daemon stopped during private package transfer")
		case <-time.After(20 * time.Millisecond):
		}
	}
}

// awaitPrivateTransfer resolves ONE fetch request. It answers true when every wheel the pod
// reported is verified, false when the set stalled and the caller should re-grant, and a
// problem when the pod refused a wheel outright -- which the schema reserves for bytes that are
// not the named object, and which a fresh URL would never fix.
func (c *Orchestrator) awaitPrivateTransfer(instanceID, operationID string,
	ordered []privateTransferFile, sent *session,
) (bool, *exit.Error) {
	for {
		verified, stalled := 0, false
		for _, selected := range ordered {
			spelled, err := canonical.Spell(selected.digest)
			if err != nil {
				return false, exit.Internalf("cannot spell a private package wheel digest: %s", err)
			}
			status := c.privateStatus(operationID, spelled)
			switch status.state {
			case pb.PrivatePackageFileState_PRIVATE_PACKAGE_FILE_STATE_VERIFIED:
				if status.received != selected.length {
					return false, exit.Named(exit.Structural, "private_package_status_invalid",
						"worker verified %s at %d of %d bytes", selected.filename,
						status.received, selected.length)
				}
				verified++
			case pb.PrivatePackageFileState_PRIVATE_PACKAGE_FILE_STATE_REFUSED:
				return false, exit.Named(exit.Failed, status.safeCode,
					"worker refused private package file %s: %s",
					selected.filename, status.safeDetail)
			case pb.PrivatePackageFileState_PRIVATE_PACKAGE_FILE_STATE_RECEIVING:
				// A RECEIVING status carrying a safe code is the schema's resumable stall.
				if status.safeCode != "" {
					stalled = true
				}
			}
		}
		if verified == len(ordered) {
			return true, nil
		}
		if stalled {
			return false, nil
		}
		c.mu.Lock()
		w := c.workers[instanceID]
		current := w != nil && c.sessions[w.bootID] == sent
		gone := w == nil || w.exited
		var refused *exit.Error
		if w != nil {
			refused = w.refusal
		}
		c.mu.Unlock()
		switch {
		case !current:
			return false, nil // the caller re-grants against the replacement session
		case refused != nil:
			return false, refused
		case gone:
			return false, exit.New(exit.Failed,
				"the rented worker exited during private package transfer")
		}
		select {
		case <-c.done:
			return false, exit.Unavailablef("the daemon stopped during private package transfer")
		case <-time.After(20 * time.Millisecond):
		}
	}
}

func (c *Orchestrator) privateStatus(operationID, digest string) privateTransferStatus {
	c.mu.Lock()
	defer c.mu.Unlock()
	if transfer := c.privateTransfers[operationID]; transfer != nil {
		return transfer.status[digest]
	}
	return privateTransferStatus{}
}

func (c *Orchestrator) onPrivatePackageFileStatus(current *session,
	frame *pb.PrivatePackageFileStatus,
) {
	spelled, err := canonical.Spell(frame.Digest)
	c.mu.Lock()
	defer c.mu.Unlock()
	transfer := c.privateTransfers[frame.OperationId]
	if err != nil || transfer == nil || transfer.canceled || current == nil ||
		transfer.instanceID != current.instanceID || transfer.bootID != current.bootID ||
		transfer.epoch != current.epoch ||
		!bytes.Equal(frame.SourceDigest, transfer.source) {
		return
	}
	expected, ok := transfer.files[spelled]
	if !ok || !bytes.Equal(frame.Digest, expected.digest) || frame.Filename != expected.filename ||
		frame.Kind != expected.kind || frame.Length != expected.length ||
		frame.ReceivedBytes > frame.Length {
		return
	}
	if frame.State != pb.PrivatePackageFileState_PRIVATE_PACKAGE_FILE_STATE_RECEIVING &&
		frame.State != pb.PrivatePackageFileState_PRIVATE_PACKAGE_FILE_STATE_VERIFIED &&
		frame.State != pb.PrivatePackageFileState_PRIVATE_PACKAGE_FILE_STATE_REFUSED {
		return
	}
	prior := transfer.status[spelled]
	if frame.ReceivedBytes < prior.received ||
		prior.state == pb.PrivatePackageFileState_PRIVATE_PACKAGE_FILE_STATE_VERIFIED &&
			frame.State != prior.state {
		return
	}
	transfer.status[spelled] = privateTransferStatus{state: frame.State,
		received: frame.ReceivedBytes, safeCode: frame.SafeCode, safeDetail: frame.SafeDetail}
}

func (c *Orchestrator) onPrivatePackageAbortStatus(current *session,
	frame *pb.PrivatePackageAbortStatus,
) {
	c.mu.Lock()
	transfer := c.privateTransfers[frame.OperationId]
	if transfer == nil || current == nil || transfer.instanceID != current.instanceID ||
		transfer.bootID != current.bootID || transfer.epoch != current.epoch ||
		!bytes.Equal(frame.SourceDigest, transfer.source) {
		c.mu.Unlock()
		return
	}
	digest, err := canonical.Spell(frame.PrivateRevisionDigest)
	if err != nil || digest != transfer.revision {
		c.mu.Unlock()
		return
	}
	channel := transfer.abortC
	c.mu.Unlock()
	select {
	case channel <- privateAbortStatus{outcome: frame.Outcome, code: frame.SafeCode,
		detail: frame.SafeDetail}:
	default:
	}
}

func (c *Orchestrator) replayPrivateAborts(current *session, workerID string) {
	rows, problem := c.opt.Store.CanceledPrivatePackages(workerID)
	if problem != nil {
		c.logf("cannot replay private package aborts for %s: %s", workerID, problem.Message)
		return
	}
	for _, row := range rows {
		source, sourceErr := canonical.Raw(row.PackageRevisionDigest)
		revision, revisionErr := canonical.Raw(row.PrivatePackageDigest)
		if sourceErr != nil || revisionErr != nil {
			c.logf("canceled request %s has invalid private abort identity", row.ID)
			continue
		}
		_ = current.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_PrivatePackageAbort{
			PrivatePackageAbort: &pb.PrivatePackageAbort{RecordOwnerEpoch: recordOwnerEpoch,
				ControlStreamEpoch: current.epoch, WorkerBootId: current.bootID,
				OperationId: row.ID, SourceDigest: source, PrivateRevisionDigest: revision}}})
	}
}

func (c *Orchestrator) issuePrivatePackageSet(s *session, w *worker,
	selected *pb.DesiredPrivatePackageSet,
) *exit.Error {
	if selected == nil {
		return exit.Internalf("cannot issue an empty private package set")
	}
	revision := c.nextRevision()
	c.mu.Lock()
	w.revision, w.desiredRefusal = revision, nil
	w.delegationExpiry = time.Time{} // private wheels travel as minted capabilities, not a delegation
	w.desiredPrivate = clonePrivatePackageSet(selected)
	w.desiredPrivatePlacement = nil
	w.desiredPackages, w.desiredModels = nil, nil
	c.mu.Unlock()
	desired := &pb.DesiredWorkerState{RecordOwnerEpoch: recordOwnerEpoch,
		ControlStreamEpoch: s.epoch, WorkerBootId: s.bootID, Revision: revision,
		Posture: pb.Posture_POSTURE_ACCEPTING, WireMinor: pb.WireMinor,
		Mode: &pb.DesiredWorkerState_PrivatePackageSet{PrivatePackageSet: selected}}
	if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_DesiredState{DesiredState: desired}}) {
		return exit.Unavailablef("worker %s control stream closed before private package send", w.instanceID)
	}
	c.logf("DesiredWorkerState revision=%d private_package_set operation=%s files=%d -> %s",
		revision, selected.OperationId, len(selected.Files), s.bootID)
	return nil
}

// ConvergePrivatePlacement binds exact downloaded models to code already admitted under one
// private revision. Creator signs logical refs only; Runtime authors the joined PlacementSet.
func (c *Orchestrator) ConvergePrivatePlacement(instanceID, operationID,
	privateRevisionDigest string, models []*pb.DownloadModelRef,
) *exit.Error {
	if operationID == "" || !validDigest(privateRevisionDigest) || len(models) == 0 {
		return exit.Named(exit.Validation, "private_placement_incomplete",
			"private modeled placement requires operation, exact revision, and models")
	}
	if c.opt.RentalPackageSet == nil {
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
	delegation, signature, problem := c.opt.RentalPackageSet(w.spec.Connection, nil, models)
	if problem != nil {
		return problem
	}
	revision, err := canonical.Raw(privateRevisionDigest)
	if err != nil || len(delegation) == 0 || len(signature) != 64 {
		return exit.Named(exit.Validation, "private_placement_delegation_incomplete",
			"private modeled placement delegation is incomplete")
	}
	selected := &pb.DesiredPrivatePlacementSet{OperationId: operationID,
		PrivateRevisionDigest: revision, DownloadDelegation: delegation,
		DownloadDelegationSignature: signature}
	return c.issuePrivatePlacementSet(s, w, selected)
}

func (c *Orchestrator) issuePrivatePlacementSet(s *session, w *worker,
	selected *pb.DesiredPrivatePlacementSet,
) *exit.Error {
	if selected == nil {
		return exit.Internalf("cannot issue an empty private placement set")
	}
	revision := c.nextRevision()
	c.mu.Lock()
	w.revision, w.desiredRefusal = revision, nil
	w.delegationExpiry = delegationExpiryOf(selected.DownloadDelegation)
	w.desiredPrivate, w.desiredPackages, w.desiredModels = nil, nil, nil
	w.desiredPrivatePlacement = clonePrivatePlacementSet(selected)
	c.mu.Unlock()
	desired := &pb.DesiredWorkerState{RecordOwnerEpoch: recordOwnerEpoch,
		ControlStreamEpoch: s.epoch, WorkerBootId: s.bootID, Revision: revision,
		Posture: pb.Posture_POSTURE_ACCEPTING, WireMinor: pb.WireMinor,
		Mode: &pb.DesiredWorkerState_PrivatePlacementSet{PrivatePlacementSet: selected}}
	if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_DesiredState{DesiredState: desired}}) {
		return exit.Unavailablef("worker %s control stream closed before private placement send", w.instanceID)
	}
	c.logf("DesiredWorkerState revision=%d private_placement_set operation=%s -> %s",
		revision, selected.OperationId, s.bootID)
	return nil
}

func clonePrivatePackageSet(in *pb.DesiredPrivatePackageSet) *pb.DesiredPrivatePackageSet {
	if in == nil {
		return nil
	}
	out := &pb.DesiredPrivatePackageSet{OperationId: in.OperationId}
	if in.Package != nil {
		out.Package = &pb.DevelopmentPackage{Package: in.Package.Package,
			Release: in.Package.Release, SourceDigest: append([]byte(nil), in.Package.SourceDigest...),
			PrivateRevisionDigest: append([]byte(nil), in.Package.PrivateRevisionDigest...)}
	}
	for _, file := range in.Files {
		if file != nil {
			out.Files = append(out.Files, &pb.PrivatePackageFileRef{Digest: append([]byte(nil), file.Digest...),
				Filename: file.Filename, Kind: file.Kind, Length: file.Length})
		}
	}
	return out
}

func clonePrivatePlacementSet(in *pb.DesiredPrivatePlacementSet) *pb.DesiredPrivatePlacementSet {
	if in == nil {
		return nil
	}
	return &pb.DesiredPrivatePlacementSet{OperationId: in.OperationId,
		PrivateRevisionDigest:       append([]byte(nil), in.PrivateRevisionDigest...),
		DownloadDelegation:          append([]byte(nil), in.DownloadDelegation...),
		DownloadDelegationSignature: append([]byte(nil), in.DownloadDelegationSignature...)}
}
