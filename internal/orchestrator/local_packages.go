package orchestrator

import (
	"bytes"
	"context"
	"os"
	"time"

	"google.golang.org/grpc"

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

// LocalWheel is one wheel of an unpublished revision as this daemon holds it on disk.
type LocalWheel struct {
	Digest, Filename, Kind, Path string
	Length                       int64
}

// LocalWheelGrantSource is th-094's replacement for the byte relay. It PUTs every wheel the
// store does not already hold and answers with one short-lived read capability per wheel, in
// the order asked. It is a callback for the same reason RentalPackageSet is: the orchestrator
// holds no Tensorhub client, and the account, the token, and the byte plane belong to the
// entrypoint. Calling it again after a stall re-mints the capabilities without re-uploading.
type LocalWheelGrantSource func(context.Context, []LocalWheel) ([]string, *exit.Error)

// Object bounds, not control-stream bounds. They are what one revision may cost the store and
// the pod's disk, and they match the pod ledger's own quota. Nothing on the wire carries them:
// the frame that names these wheels carries 33 URLs at most.
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
		_, holdsRevision = w.observedRemote[revision]
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
			_, held = w.observedRemote[revision]
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

// transferLocalPackage is th-094's whole owner half. It no longer sends a byte: the wheels go
// to the store under capabilities scoped to their own digests, and the pod is handed one read
// capability per wheel to spend against its own download edge.
//
// The loop exists because a capability is short-lived by design. A stall -- the pod saying it
// has a partial prefix and needs a fresh URL -- is answered by minting new grants over the same
// objects, which costs one HTTP round and re-uploads nothing, and the pod resumes from the
// prefix it already holds.
func (c *Orchestrator) transferLocalPackage(instanceID, operationID string,
	transfer *localTransfer,
) *exit.Error {
	if c.opt.LocalWheels == nil {
		return exit.Named(exit.Structural, "local_package_grants_unwired",
			"this daemon cannot upload a local package revision")
	}
	stale := 0
	for {
		_, selectedSession, problem := c.localControl(instanceID)
		if problem != nil {
			return problem
		}
		held, problem := c.bindLocalTransfer(instanceID, operationID, transfer, selectedSession)
		if problem != nil {
			return problem
		}
		ordered := transferFilesInOrder(held)
		if problem := c.proveLocalWheelsUnchanged(ordered); problem != nil {
			return problem
		}
		granted, problem := c.grantLocalWheels(selectedSession, ordered)
		if problem != nil {
			return problem
		}
		// Clear the previous attempt's unverified statuses. A stall left behind by the round
		// these grants were minted to answer would be read as this round's stall, and the loop
		// would re-grant forever without ever waiting for an answer.
		before := c.clearLocalStalls(operationID)
		frame := &pb.LocalPackageFetchRequest{RecordOwnerEpoch: recordOwnerEpoch,
			ControlStreamEpoch: selectedSession.epoch,
			WorkerBootId:       selectedSession.bootID, OperationId: operationID,
			SourceDigest: held.source, Files: granted}
		if !selectedSession.send(&pb.RecordOwnerFrame{
			Msg: &pb.RecordOwnerFrame_LocalPackageFetchRequest{
				LocalPackageFetchRequest: frame}}) {
			continue
		}
		done, problem := c.awaitLocalTransfer(instanceID, operationID, ordered, selectedSession)
		if problem != nil {
			return problem
		}
		if done {
			return nil
		}
		// A stall is only worth answering while it is buying ground. A pod that keeps landing
		// bytes gets as many capabilities as it needs; one that lands none twice running is not
		// stalling, it is failing, and saying so beats an unbounded loop.
		if after := c.localProgress(operationID); after > before {
			stale = 0
		} else if stale++; stale > maxLocalStalls {
			return exit.Named(exit.Failed, "local_package_transfer_stuck",
				"the worker landed no local package bytes across %d re-granted attempts",
				stale)
		}
	}
}

// maxLocalStalls is how many consecutive no-progress rounds a transfer may spend. Each one
// costs a grant round trip and nothing else, so the bound is about ending, not about cost.
const maxLocalStalls = 3

// clearLocalStalls drops every unverified status and reports the durable byte total the pod
// has acknowledged. A verified file is never cleared: it is settled.
func (c *Orchestrator) clearLocalStalls(operationID string) uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	transfer := c.localTransfers[operationID]
	if transfer == nil {
		return 0
	}
	var total uint64
	for digest, status := range transfer.status {
		if status.state == pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED {
			total += status.received
			continue
		}
		total += status.received
		delete(transfer.status, digest)
	}
	return total
}

func (c *Orchestrator) localProgress(operationID string) uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	transfer := c.localTransfers[operationID]
	if transfer == nil {
		return 0
	}
	var total uint64
	for _, status := range transfer.status {
		total += status.received
	}
	return total
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

// grantLocalWheels turns one ordered wheel set into the frame's grants. The URLs are
// memory-only: they are never journaled, never logged, and never survive this frame.
func (c *Orchestrator) grantLocalWheels(current *session, ordered []localTransferFile) (
	[]*pb.LocalPackageFileGrant, *exit.Error,
) {
	wheels := make([]LocalWheel, 0, len(ordered))
	for _, selected := range ordered {
		spelled, err := canonical.Spell(selected.digest)
		if err != nil {
			return nil, exit.Internalf("cannot spell a local package wheel digest: %s", err)
		}
		kind := "dependency_wheel"
		if selected.project {
			kind = "project_wheel"
		}
		wheels = append(wheels, LocalWheel{Digest: spelled, Filename: selected.filename,
			Kind: kind, Path: selected.path, Length: int64(selected.length)})
	}
	urls, problem := c.opt.LocalWheels(current.ctx, wheels)
	if problem != nil {
		return nil, problem
	}
	if len(urls) != len(ordered) {
		return nil, exit.Internalf("local package grants answered for %d of %d wheels",
			len(urls), len(ordered))
	}
	grants := make([]*pb.LocalPackageFileGrant, 0, len(ordered))
	for i, selected := range ordered {
		if urls[i] == "" || len(urls[i]) > pb.MaxLocalPackageGrantURLBytes {
			return nil, exit.Named(exit.Internal, "local_package_grant_url_invalid",
				"the read capability for %s is empty or over its bound", selected.filename)
		}
		grants = append(grants, &pb.LocalPackageFileGrant{Digest: selected.digest,
			Filename: selected.filename, Length: selected.length,
			Url: urls[i]})
	}
	return grants, nil
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

// awaitLocalTransfer resolves ONE fetch request. It answers true when every wheel the pod
// reported is verified, false when the set stalled and the caller should re-grant, and a
// problem when the pod refused a wheel outright -- which the schema reserves for bytes that are
// not the named object, and which a fresh URL would never fix.
func (c *Orchestrator) awaitLocalTransfer(instanceID, operationID string,
	ordered []localTransferFile, sent *session,
) (bool, *exit.Error) {
	for {
		verified, stalled := 0, false
		for _, selected := range ordered {
			spelled, err := canonical.Spell(selected.digest)
			if err != nil {
				return false, exit.Internalf("cannot spell a local package wheel digest: %s", err)
			}
			status := c.localStatus(operationID, spelled)
			switch status.state {
			case pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED:
				if status.received != selected.length {
					return false, exit.Named(exit.Structural, "local_package_status_invalid",
						"worker verified %s at %d of %d bytes", selected.filename,
						status.received, selected.length)
				}
				verified++
			case pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_REFUSED:
				return false, exit.Named(exit.Failed, status.safeCode,
					"worker refused local package file %s: %s",
					selected.filename, status.safeDetail)
			case pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING:
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
				"the rented worker exited during local package transfer")
		}
		select {
		case <-c.done:
			return false, exit.Unavailablef("the daemon stopped during local package transfer")
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
	return w.desiredPrivatePlacement != nil &&
		bytes.Equal(w.desiredPrivatePlacement.LocalRevisionDigest, raw)
}

// localOperation names the operation the pod prepared revision under, when this owner
// issued one: a model placement must bind over that operation, whoever converges it.
func (c *Orchestrator) localOperation(w *worker, revision []byte, fallback string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if w.desiredLocal != nil && bytes.Equal(w.desiredLocal.Package.GetLocalRevisionDigest(), revision) {
		return w.desiredLocal.OperationId
	}
	if w.desiredPrivatePlacement != nil && bytes.Equal(w.desiredPrivatePlacement.LocalRevisionDigest, revision) {
		return w.desiredPrivatePlacement.OperationId
	}
	return fallback
}

func (c *Orchestrator) issueLocalPackageSet(s *session, w *worker,
	selected *pb.DesiredLocalPackageSet,
) *exit.Error {
	if selected == nil {
		return exit.Internalf("cannot issue an empty local package set")
	}
	revision, err := canonical.Spell(selected.Package.GetLocalRevisionDigest())
	if err != nil {
		return exit.Internalf("cannot spell the local revision digest: %s", err)
	}
	c.mu.Lock()
	w.desiredLocal = cloneLocalPackageSet(selected)
	w.desiredPrivatePlacement = nil
	w.desiredPackages, w.desiredModels, w.desiredDownloadSets = nil, nil, nil
	w.desiredEpoch = s.epoch
	_, held := w.observedRemote[revision]
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

// ConvergePrivatePlacement binds exact downloaded models to code already admitted under one
// local revision. Creator names logical refs only; Runtime authors the joined PlacementSet.
func (c *Orchestrator) ConvergePrivatePlacement(instanceID, operationID,
	localRevisionDigest string, models []*pb.DownloadModelRef,
) *exit.Error {
	if operationID == "" || !validDigest(localRevisionDigest) || len(models) == 0 {
		return exit.Named(exit.Validation, "private_placement_incomplete",
			"local package placement requires operation, exact revision, and models")
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
	// The models bind OVER the local revision, so the pod must hold that revision first:
	// `ConvergeLocalPackage` only ISSUES the prepare, and a private placement sent on its
	// heels is refused `private_placement_invalid: private placement revision is not
	// prepared` (found live, L4 `shiranui`, cl-101). Wait on the pod's own report of it.
	if problem := c.awaitLocalRevision(instanceID, localRevisionDigest); problem != nil {
		return problem
	}
	downloadSet, problem := c.opt.RentalPackageSet(nil, models)
	if problem != nil {
		return problem
	}
	revision, err := canonical.Raw(localRevisionDigest)
	if err != nil || len(downloadSet) == 0 {
		return exit.Named(exit.Validation, "private_placement_download_set_incomplete",
			"local package placement download set is incomplete")
	}
	operationID = c.localOperation(w, revision, operationID)
	selected := &pb.DesiredPrivatePlacementSet{OperationId: operationID,
		LocalRevisionDigest: revision, DownloadDelegation: downloadSet}
	return c.issuePrivatePlacementSet(s, w, selected)
}

func (c *Orchestrator) issuePrivatePlacementSet(s *session, w *worker,
	selected *pb.DesiredPrivatePlacementSet,
) *exit.Error {
	if selected == nil {
		return exit.Internalf("cannot issue an empty private placement set")
	}
	c.mu.Lock()
	w.desiredLocal, w.desiredPackages, w.desiredModels, w.desiredDownloadSets = nil, nil, nil, nil
	w.desiredPrivatePlacement = clonePrivatePlacementSet(selected)
	w.desiredEpoch = s.epoch
	c.mu.Unlock()
	call := &pb.PreparePrivatePlacementCall{Claim: s.claim, PrivatePlacementSet: clonePrivatePlacementSet(selected)}
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

func clonePrivatePlacementSet(in *pb.DesiredPrivatePlacementSet) *pb.DesiredPrivatePlacementSet {
	if in == nil {
		return nil
	}
	return &pb.DesiredPrivatePlacementSet{OperationId: in.OperationId,
		LocalRevisionDigest: append([]byte(nil), in.LocalRevisionDigest...),
		DownloadDelegation:  append([]byte(nil), in.DownloadDelegation...)}
}
