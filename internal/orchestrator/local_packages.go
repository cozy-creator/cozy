package orchestrator

import (
	"bytes"
	"context"
	"time"

	"google.golang.org/grpc"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// localWheel is one wheel of an unpublished revision as this daemon holds it on disk, in the
// only order that matters: by digest, which is the order the revision's own identity fixed.
type localWheel struct {
	digest         string // `sha256:<64 hex>`, exactly as the object is named on the pod
	filename, path string
	length         int64
}

// Object bounds, not control-stream bounds. They are what one revision may cost the pod's
// disk. Nothing on the wire carries them: the frame that names these wheels carries digests,
// filenames and lengths, and the bytes went up the file plane before it was sent.
const (
	maxLocalWheelBytes    = int64(512 << 20)
	maxLocalWheelSetBytes = int64(1 << 30)
)

// ConvergeLocalPackage puts one sealed editable revision on the pod and then selects it. The
// request row is the recovery authority: once the wheels are durably recorded as pushed
// (`uploaded` records the boot that took them), reconnect sends only the exact
// DesiredLocalPackageSet. The editable refresh converges with no request and no durable
// marker: its next run pushes the bytes again if it must, which costs one round trip per
// object the pod already holds.
func (c *Orchestrator) ConvergeLocalPackage(instanceID, operationID string,
	revision localpackage.Revision, uploadedBootID string, uploaded func(bootID string) *exit.Error,
) *exit.Error {
	selected, wheels, problem := localSelection(operationID, revision)
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
		// The exact revision remains durable, but the wheels are boot-scoped: they live in
		// the media subtree of the pod that held them. Push them to the replacement.
		uploadedBootID = ""
	}
	for uploadedBootID == "" {
		if problem := c.pushLocalPackage(w, operationID, wheels); problem != nil {
			return problem
		}
		if uploaded != nil {
			if problem := uploaded(s.bootID); problem != nil {
				return problem
			}
		}
		uploadedBootID = s.bootID
		// The pod can be replaced while the bytes are moving. Re-read the session and, if
		// it is a different boot, push again: the durable marker names one boot's disk.
		w, s, problem = c.localControl(instanceID)
		if problem != nil {
			return problem
		}
		if uploadedBootID != s.bootID {
			uploadedBootID = ""
		}
	}
	if problem := c.hostNothing(instanceID, revision.Digest); problem != nil {
		return problem
	}
	for {
		if problem := c.issueLocalPackageSet(s, w, selected); problem == nil {
			return nil
		}
		w, s, problem = c.localControl(instanceID)
		if problem != nil {
			return problem
		}
	}
}

// pushLocalPackage puts one revision's wheels on the pod's FILE PLANE and returns when the
// pod holds every one of them (th-142 child 4, decision 697).
//
// This used to be a protocol. The owner PUT each wheel to an object store, minted a
// short-lived read capability per object, sent them on the control stream, and then followed
// a per-file status stream through stalls, re-grants and a bounded no-progress count -- because
// a capability expires and the POD was the party spending it. None of that survives the
// inversion: the owner holds the bytes, the pod's file plane takes them, and a push either
// lands or reports why. There is no capability to age out, so there is nothing to resume.
//
// The object is named by its own sha256, so the pod proves each body against its name before
// it commits it, and a re-push of one it already holds costs a round trip and no bytes.
func (c *Orchestrator) pushLocalPackage(w *worker, operationID string, wheels []localWheel) *exit.Error {
	if w.media == nil {
		return exit.Named(exit.Unavailable, "local_package_media_unwired",
			"worker %s has no media plane to push an editable revision to", w.instanceID)
	}
	for _, wheel := range wheels {
		path, problem := w.media.PushPackage(operationID, wheel.path, wheel.digest, wheel.length)
		if problem != nil {
			return problem
		}
		c.logf("pushed local wheel %s (%d B, %s) to %s at %s", wheel.filename, wheel.length,
			shortDigest(wheel.digest), w.media.Addr(), path)
	}
	return nil
}

// dropLocalPackages removes one operation's pushed wheels from every pod this host holds. It
// replaces the abort frame the pod used to answer with a durable tombstone: the wheels are
// files inside the pod's own quota-bounded media subtree, so dropping them is a DELETE, and a
// DELETE for an operation a pod never held is not an error there.
func (c *Orchestrator) dropLocalPackages(operationID string) *exit.Error {
	c.mu.Lock()
	holders := make([]*worker, 0, len(c.workers))
	for _, w := range c.workers {
		if w != nil && w.media != nil && !w.exited {
			holders = append(holders, w)
		}
	}
	c.mu.Unlock()
	var first *exit.Error
	for _, w := range holders {
		if problem := w.media.DropPackages(operationID); problem != nil && first == nil {
			first = problem
		}
	}
	return first
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

// localSelection turns one sealed revision into the exact set the pod is sent and the exact
// wheels this host pushes to it, in digest order both times. It reads no file: what the wheels
// weigh and hash to is the revision's own identity, and the push proves both again.
func localSelection(operationID string, revision localpackage.Revision) (
	*pb.DesiredLocalPackageSet, []localWheel, *exit.Error,
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
	wheels := make([]localWheel, 0, len(revision.Files))
	seen := make(map[string]bool, len(revision.Files))
	var prior []byte
	var total int64
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
			file.Length > maxLocalWheelSetBytes-total || seen[file.Digest] {
			return nil, nil, exit.Named(exit.Structural, "local_package_file_invalid",
				"local package file %s has invalid identity or bounds", file.Filename)
		}
		total += file.Length
		seen[file.Digest] = true
		wheels = append(wheels, localWheel{digest: file.Digest, filename: file.Filename,
			path: file.Path, length: file.Length})
		selected.Files = append(selected.Files, &pb.LocalPackageFileRef{Digest: digest,
			Filename: file.Filename, Length: uint64(file.Length)})
		prior = digest
	}
	if projects != 1 {
		return nil, nil, exit.Named(exit.Structural, "local_package_project_wheel_count",
			"local package revision must carry one project wheel and unique files")
	}
	return selected, wheels, nil
}

// localControl waits for the instance's claimed, snapshot-acknowledged control session.
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

// replayLocalAborts drops the pushed wheels of every request this owner canceled while the
// pod was away. The pod used to answer an abort frame with a durable tombstone of its own;
// now the wheels are files in its media subtree and dropping them is a DELETE, so what is
// replayed is one idempotent removal per canceled operation.
func (c *Orchestrator) replayLocalAborts(current *session, workerID string) {
	rows, problem := c.opt.Store.CanceledLocalPackages(workerID)
	if problem != nil {
		c.logf("cannot replay local package drops for %s: %s", workerID, problem.Message)
		return
	}
	for _, row := range rows {
		if problem := c.dropLocalPackages(row.ID); problem != nil {
			c.logf("canceled request %s still holds pushed wheels on %s: %s",
				row.ID, workerID, problem.Message)
		}
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
	w.delegationExpiry = time.Time{} // local wheels travel as minted capabilities, not a delegation
	w.desiredLocal = cloneLocalPackageSet(selected)
	w.desiredPrivatePlacement = nil
	w.desiredPackages, w.desiredModels, w.desiredDelegations = nil, nil, nil
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
// local revision. Creator signs logical refs only; Runtime authors the joined PlacementSet.
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
	delegation, signature, problem := c.opt.RentalPackageSet(w.spec.Connection, nil, models)
	if problem != nil {
		return problem
	}
	revision, err := canonical.Raw(localRevisionDigest)
	if err != nil || len(delegation) == 0 || len(signature) != 64 {
		return exit.Named(exit.Validation, "private_placement_delegation_incomplete",
			"local package placement delegation is incomplete")
	}
	operationID = c.localOperation(w, revision, operationID)
	selected := &pb.DesiredPrivatePlacementSet{OperationId: operationID,
		LocalRevisionDigest: revision, DownloadDelegation: delegation,
		DownloadDelegationSignature: signature}
	return c.issuePrivatePlacementSet(s, w, selected)
}

func (c *Orchestrator) issuePrivatePlacementSet(s *session, w *worker,
	selected *pb.DesiredPrivatePlacementSet,
) *exit.Error {
	if selected == nil {
		return exit.Internalf("cannot issue an empty private placement set")
	}
	c.mu.Lock()
	w.delegationExpiry = delegationExpiryOf(selected.DownloadDelegation)
	w.desiredLocal, w.desiredPackages, w.desiredModels, w.desiredDelegations = nil, nil, nil, nil
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
		LocalRevisionDigest:         append([]byte(nil), in.LocalRevisionDigest...),
		DownloadDelegation:          append([]byte(nil), in.DownloadDelegation...),
		DownloadDelegationSignature: append([]byte(nil), in.DownloadDelegationSignature...)}
}
