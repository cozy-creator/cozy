package orchestrator

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// The owner-side FRAME HANDLERS for one claimed stream (owner.go runs the conversation;
// #436 flipped the dial direction, so the old worker-dials hub/Register machinery is
// gone — ClaimAck/snapshot are its successors, handled in owner.go).

func (c *Orchestrator) dropSession(s *session) {
	c.mu.Lock()
	if c.sessions[s.bootID] == s {
		delete(c.sessions, s.bootID)
	}
	c.mu.Unlock()
	// The out channel is closed by `converse`'s own defer — one owner, one close.
	c.logf("control stream for boot %s closed", s.bootID)
	c.signalAllTransfers()
}

// jobMode answers whether the worker behind this session was launched in job mode.
func (c *Orchestrator) jobMode(s *session) bool {
	if s == nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	w := c.workers[s.instanceID]
	return w != nil && w.spec.IsJob()
}

// ConvergePlacementSet is the ONE issuer of a serving-mode desired state: the full-replace
// DesiredWorkerState carrying a GENUINELY content-addressed DesiredPlacementSet (§4).
//
// The frozen wire carried both a structured set and a digest this owner authored for it,
// and the worker echoed the digest as the set's NAME — one digester, two representations,
// and nothing that could ever disagree out loud. rev-2 deletes the structured copy: the
// set travels as its exact canonical bytes, the worker RECOMPUTES sha256 over them before
// parsing a single field, and journals the bytes it accepted. So this side authors bytes
// once and keeps them; there is no second place for the set to disagree with itself.
func (c *Orchestrator) ConvergePlacementSet(instanceID string, placements []DesiredPlacement) *exit.Error {
	c.mu.Lock()
	w := c.workers[instanceID]
	var s *session
	if w != nil {
		s = c.sessions[w.bootID]
	}
	c.mu.Unlock()
	if w == nil {
		return exit.New(exit.NotFound, "no worker %s on this host", instanceID)
	}
	if s == nil {
		return exit.Unavailablef("worker %s holds no claimed control stream to converge", instanceID)
	}
	return c.converge(s, w, placements)
}

// ConvergePackageSet sends Creator's signed logical package/model authority to a
// private pod. The refs never pass through PlacementSet or local platform resolution;
// pod-supervisor verifies the signature and resolves downloads.
func (c *Orchestrator) ConvergePackageSet(instanceID string, packages []*pb.DownloadPackageRef,
	models []*pb.DownloadModelRef) *exit.Error {
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
	// DesiredWorkerState is a complete replacement. A request for package B therefore
	// carries the already-selected package A as well; sending only B unloads A and turns
	// one rented machine into a single-package slot again. Serialize the read/merge/send so
	// concurrent requests cannot each erase the other's addition.
	w.desiredMu.Lock()
	defer w.desiredMu.Unlock()
	c.mu.Lock()
	packages, models = mergePackageSet(w.desiredPackages, w.desiredModels, packages, models)
	c.mu.Unlock()
	return c.issuePackageSet(s, w, packages, models)
}

func mergePackageSet(currentPackages []*pb.DownloadPackageRef, currentModels []*pb.DownloadModelRef,
	requestedPackages []*pb.DownloadPackageRef, requestedModels []*pb.DownloadModelRef,
) ([]*pb.DownloadPackageRef, []*pb.DownloadModelRef) {
	replaced := make(map[string]bool, len(requestedPackages))
	for _, row := range requestedPackages {
		if row != nil {
			replaced[row.Package] = true
		}
	}
	packages := make([]*pb.DownloadPackageRef, 0, len(currentPackages)+len(requestedPackages))
	for _, row := range currentPackages {
		if row != nil && !replaced[row.Package] {
			packages = append(packages, row)
		}
	}
	for _, row := range requestedPackages {
		if row != nil {
			packages = append(packages, row)
		}
	}
	models := make([]*pb.DownloadModelRef, 0, len(currentModels)+len(requestedModels))
	for _, row := range currentModels {
		if row != nil && !replaced[row.Package] {
			models = append(models, row)
		}
	}
	for _, row := range requestedModels {
		if row != nil {
			models = append(models, row)
		}
	}
	sort.Slice(packages, func(i, j int) bool {
		return packages[i].Package+"\x00"+packages[i].Release < packages[j].Package+"\x00"+packages[j].Release
	})
	sort.Slice(models, func(i, j int) bool {
		left := models[i].Package + "\x00" + models[i].Slot + "\x00" + models[i].Model + "\x00" +
			models[i].Release + "\x00" + models[i].Manifest
		right := models[j].Package + "\x00" + models[j].Slot + "\x00" + models[j].Model + "\x00" +
			models[j].Release + "\x00" + models[j].Manifest
		return left < right
	})
	return clonePackageRefs(packages), cloneModelRefs(models)
}

// issuePackageSet prepares the desired logical set ONE PACKAGE PER PREPARE — the
// Runtime's contract (package_prepare: exactly one package and a model array) — and sends
// every prepared placement as the ONE full-replace set a rental hosting several package
// environments converges to. Each package's delegation is signed once and reused
// byte-for-byte while it lives and the boot stands: the pod's preparation ledger answers
// a known delegation without re-downloading, and the Runtime seeds the placement_id from
// the exact delegation bytes, so re-signing package A when package B joins would retire
// a serving placement that changed in nothing but its credential.
func (c *Orchestrator) issuePackageSet(s *session, w *worker, packages []*pb.DownloadPackageRef,
	models []*pb.DownloadModelRef) *exit.Error {
	if c.opt.RentalPackageSet == nil {
		return exit.Named(exit.Unavailable, "rental.package_set_signer_missing",
			"this Cozy daemon has no package_set signer")
	}
	selections, problem := splitPackageSet(packages, models)
	if problem != nil {
		return problem
	}
	c.mu.Lock()
	retained := w.desiredDelegations
	bootID := ""
	if w.spec.Connection != nil {
		bootID = w.spec.Connection.WorkerBootID
	}
	c.mu.Unlock()
	now := time.Now()
	signed := make(map[string]signedPackageDelegation, len(selections))
	prepares := make([]packagePrepare, 0, len(selections))
	var earliest time.Time
	for _, selection := range selections {
		key := selection.contentKey()
		delegation, held := retained[selection.name]
		if !held || delegation.contentKey != key || delegation.bootID != bootID ||
			!now.Before(delegation.expiry) {
			body, signature, problem := c.opt.RentalPackageSet(w.spec.Connection,
				selection.packages, selection.models)
			if problem != nil {
				return problem
			}
			if len(body) == 0 || len(signature) != 64 {
				return exit.Named(exit.Validation, "rental.delegation_incomplete",
					"package_set requires canonical delegation bytes and one Ed25519 signature")
			}
			document, err := canonical.Read(body, &pb.DownloadDelegation{})
			if err != nil {
				return exit.Named(exit.Validation, "rental.delegation_invalid",
					"package_set delegation is not canonical: %s", err)
			}
			delegation = signedPackageDelegation{
				contentKey: key, bootID: bootID,
				expiry:     time.Unix(document.Int("expires_at_unix"), 0),
				delegation: body, signature: signature,
			}
		}
		signed[selection.name] = delegation
		if earliest.IsZero() || delegation.expiry.Before(earliest) {
			earliest = delegation.expiry
		}
		prepares = append(prepares, packagePrepare{
			label:      hostLabel("package_set", selection.name),
			pkg:        selection.name,
			delegation: append([]byte(nil), delegation.delegation...),
			signature:  append([]byte(nil), delegation.signature...),
		})
	}
	c.mu.Lock()
	w.delegationExpiry = earliest
	w.desiredPackages = clonePackageRefs(packages)
	w.desiredModels = cloneModelRefs(models)
	w.desiredDelegations = signed
	w.desiredLocal = nil
	w.desiredPrivatePlacement = nil
	c.mu.Unlock()
	return c.issuePackagePrepares(s, w,
		hostLabel("package_set", fmt.Sprintf("%d packages, %d models", len(packages), len(models))), prepares)
}

// packageSelection is one package's slice of the desired logical set: its ref and the
// models bound to it. Prepare is a per-package operation, so this is exactly what one
// delegation authorizes and one PreparePackageSet call presents.
type packageSelection struct {
	name     string
	packages []*pb.DownloadPackageRef
	models   []*pb.DownloadModelRef
}

// contentKey is the selection's logical content — what a signed delegation binds beside
// its expiry and worker identity. Equal keys authorize the same bytes.
func (g packageSelection) contentKey() string {
	var sb strings.Builder
	for _, row := range g.packages {
		sb.WriteString("p\x00" + row.Package + "\x00" + row.Release + "\x01")
	}
	for _, row := range g.models {
		sb.WriteString("m\x00" + row.Package + "\x00" + row.Slot + "\x00" + row.Model + "\x00" +
			row.Release + "\x00" + row.Manifest + "\x01")
	}
	return sb.String()
}

// splitPackageSet gives every package of the merged logical set its own selection,
// ordered by package name. Every model rides with its named package; a model naming no
// selected package has no prepare to ride and is refused here, where the defect is
// legible, rather than as a pod refusal.
func splitPackageSet(packages []*pb.DownloadPackageRef,
	models []*pb.DownloadModelRef) ([]packageSelection, *exit.Error) {
	byName := map[string]*packageSelection{}
	names := []string(nil)
	for _, row := range packages {
		if row == nil {
			continue
		}
		if _, ok := byName[row.Package]; !ok {
			byName[row.Package] = &packageSelection{name: row.Package}
			names = append(names, row.Package)
		}
		byName[row.Package].packages = append(byName[row.Package].packages, row)
	}
	for _, row := range models {
		if row == nil {
			continue
		}
		selection, ok := byName[row.Package]
		if !ok {
			return nil, exit.Named(exit.Validation, "rental.package_set_model_orphaned",
				"model %s names package %s, which the package_set does not select",
				row.Model, row.Package)
		}
		selection.models = append(selection.models, row)
	}
	if len(names) == 0 {
		return nil, exit.Named(exit.Validation, "rental.package_set_empty",
			"package_set selects no package")
	}
	sort.Strings(names)
	out := make([]packageSelection, 0, len(names))
	for _, name := range names {
		out = append(out, *byName[name])
	}
	return out, nil
}

// unitePreparedPlacementSets joins the per-package prepared sets into the ONE
// full-replace PlacementSet this owner sends. A single preparation passes through
// byte-exact — the Runtime's own authored bytes. A joined set is re-authored canonically
// over the exact placement entries each preparation returned, sorted by placement_id so
// the united document's identity does not depend on prepare order; the worker recomputes
// the digest over these bytes before parsing (§4).
func unitePreparedPlacementSets(sets []*pb.DesiredPlacementSet) (*pb.DesiredPlacementSet, error) {
	if len(sets) == 1 {
		return sets[0], nil
	}
	placements := make([]canonical.Value, 0, len(sets))
	seen := map[string]bool{}
	for _, set := range sets {
		doc, err := canonical.Read(set.PlacementSetCanonicalBytes, &pb.PlacementSet{})
		if err != nil {
			return nil, err
		}
		for _, row := range doc.List("placements") {
			id := row.Str("placement_id")
			if id == "" || seen[id] {
				return nil, fmt.Errorf("prepared placement id %q is empty or duplicated", id)
			}
			seen[id] = true
			placements = append(placements, row)
		}
	}
	if len(placements) == 0 {
		return nil, fmt.Errorf("the prepared sets carry no placements")
	}
	sort.Slice(placements, func(i, j int) bool {
		return placements[i].(canonical.Doc).Str("placement_id") <
			placements[j].(canonical.Doc).Str("placement_id")
	})
	data, err := canonical.Write(map[string]canonical.Value{
		"format": canonical.Format(&pb.PlacementSet{}), "placements": placements,
	})
	if err != nil {
		return nil, err
	}
	return &pb.DesiredPlacementSet{
		PlacementSetDigest: canonical.Digest(data), PlacementSetCanonicalBytes: data,
	}, nil
}

// delegationExpiryOf reads the lifetime out of the exact delegation bytes this owner is
// about to send. It is kept on the worker so a pod reporting the credential lapsed can be
// told apart from a pod reporting a lapse that could not have happened yet (owner.go's
// refusePendingDesiredState). A desired state carrying no delegation clears it: a stale
// expiry must never excuse a refusal of some other mode.
func delegationExpiryOf(delegation []byte) time.Time {
	document, err := canonical.Read(delegation, &pb.DownloadDelegation{})
	if err != nil {
		return time.Time{}
	}
	return time.Unix(document.Int("expires_at_unix"), 0)
}

// signedPackageDelegation is one package's exact signed download authority: the
// canonical DownloadDelegation bytes, their signature, and the facts that decide reuse —
// the logical content signed, the worker boot it binds, and its expiry.
type signedPackageDelegation struct {
	contentKey string
	bootID     string
	expiry     time.Time
	delegation []byte
	signature  []byte
}

func clonePackageRefs(in []*pb.DownloadPackageRef) []*pb.DownloadPackageRef {
	out := make([]*pb.DownloadPackageRef, 0, len(in))
	for _, ref := range in {
		if ref != nil {
			out = append(out, &pb.DownloadPackageRef{
				Package: ref.Package, Release: ref.Release,
			})
		}
	}
	return out
}

func cloneModelRefs(in []*pb.DownloadModelRef) []*pb.DownloadModelRef {
	out := make([]*pb.DownloadModelRef, 0, len(in))
	for _, ref := range in {
		if ref != nil {
			out = append(out, &pb.DownloadModelRef{
				Manifest: ref.Manifest, Model: ref.Model, Release: ref.Release,
				Package: ref.Package, Slot: ref.Slot,
			})
		}
	}
	return out
}

func (c *Orchestrator) converge(s *session, w *worker, placements []DesiredPlacement) *exit.Error {
	var setBytes, digest []byte
	if len(placements) == 1 {
		p := placements[0]
		declared, err := canonical.Raw(p.PlacementSetDigest)
		if err != nil || !bytes.Equal(canonical.Digest(p.PlacementSetBytes), declared) {
			return exit.Named(exit.Conflict, "placement_set_identity_mismatch",
				"the persisted PlacementSet bytes do not match %s", p.PlacementSetDigest)
		}
		doc, err := canonical.Read(p.PlacementSetBytes, &pb.PlacementSet{})
		if err != nil || len(doc.List("placements")) != 1 ||
			doc.List("placements")[0].Str("placement_id") != p.PlacementID() {
			return exit.Named(exit.Conflict, "placement_set_closure_mismatch",
				"the persisted PlacementSet does not name placement %s: %v", p.PlacementID(), err)
		}
		setBytes, digest = append([]byte(nil), p.PlacementSetBytes...), append([]byte(nil), declared...)
	} else {
		var err error
		setBytes, digest, err = canonical.Identity(&pb.PlacementSet{})
		if err != nil {
			return exit.Internalf("cannot mint the empty PlacementSet for %s: %s", w.instanceID, err)
		}
	}
	rev := c.nextRevision()
	c.mu.Lock()
	w.revision, w.setDigest, w.setBytes = rev, digest, setBytes
	w.delegationExpiry = time.Time{}
	w.desiredRefusal = nil
	c.mu.Unlock()

	d := &pb.DesiredWorkerState{
		Revision: rev, WireMinor: pb.WireMinor,
		Posture: pb.Posture_POSTURE_ACCEPTING,
		// No `device_pins` (proto-024): this owner grants a one-device envelope, and width
		// 1 pins nothing — the worker assigns the lane by measured fit. Where the pin lives
		// and who authors a wider one are open group-lanes rulings, not this owner's call.
		Mode: &pb.DesiredWorkerState_PlacementSet{PlacementSet: &pb.DesiredPlacementSet{
			PlacementSetDigest:         digest,
			PlacementSetCanonicalBytes: setBytes,
		}},
	}
	if len(placements) == 0 {
		// An empty set is a worker asked to host nothing. It DRAINS: the posture is the
		// only thing that says "finish what you hold and take no more", and a retirement
		// that left the posture ACCEPTING would be asking for work it has nowhere to run.
		d.Posture = pb.Posture_POSTURE_DRAINING
	}
	d.RecordOwnerEpoch, d.ControlStreamEpoch, d.WorkerBootId = recordOwnerEpoch, s.epoch, s.bootID
	s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_DesiredState{DesiredState: d}})
	c.logf("DesiredWorkerState revision=%d posture=%s placements=%d set=%s (%d canonical bytes) -> %s",
		rev, trimEnum(pb.Posture_name[int32(d.Posture)], "POSTURE_"), len(placements),
		shortDigest(shortNone(digest)), len(setBytes), s.bootID)
	return nil
}

// ------------------------------------------------------------------- observed state

// onObserved applies one ObservedWorkerState. Nothing here infers: every field is the
// worker's own last word, and the two facts the frozen wire could not tell apart —
// "your message arrived" and "your intent is satisfied" — are now two separate readable
// numbers (#473).
func (c *Orchestrator) onObserved(s *session, r *pb.ObservedWorkerState) {
	var status *pb.PlacementStatus
	var desiredRevision uint64
	var laneBreaches []string
	var heldVerdicts []heldVerdict
	workerTerminal, repeatedFault, verdict := false, false, ""
	c.mu.Lock()
	w := c.workers[s.instanceID]
	if w != nil && w.bootID != s.bootID {
		// A superseded stream's report is a fact about a worker that no longer exists.
		c.mu.Unlock()
		c.logf("DROPPED: observed state from superseded boot %s (the live one is %s)",
			s.bootID, w.bootID)
		return
	}
	if w != nil {
		desiredRevision = w.revision
		w.lastReport = time.Now()
		w.phase = r.WorkerPhase
		// THE ADMISSION FENCE, one counter per serialized resource. Per-placement credits
		// are gone: `available_attempt_slots` is the machine's sum and already counts both
		// running attempts and outcomes this owner has not acked (#480d). The lanes beside
		// it (proto-024) are the resources that sum is over; an offer draws from ONE.
		w.admission, w.admissionEpoch = r.AdmissionState, r.AdmissionEpoch
		w.observeSlots(int(r.AvailableAttemptSlots))
		laneBreaches = w.lanes.observe(laneReportsOf(r.Lanes), w.spec.Devices)
		for _, p := range r.Placements {
			if p != nil {
				w.lanes.route(p.PlacementId, p.DeviceLaneId)
			}
		}
		w.observeHeld(heldPlacements(r.HeldAttempts))
		w.heldManifests = setOf(r.HeldManifests)
		heldVerdicts = c.observeHeldOutcomes(w, r.HeldAttempts)
		w.acceptedRevision = r.AcceptedDesiredStateRevision
		w.convergedRevision = r.ConvergedRevision
		w.acceptedSetDigest = r.AcceptedPlacementSetDigest
		dispatchable, materializable := map[string]bool{}, map[string]bool{}
		if w.spec.IsJob() {
			// THE JOB LANE READS JOB CAPACITY: a job worker is in JobDirective mode and
			// hosts no placement at all, so it has no serving axis. `jobs_available` IS
			// the job credit, and this lane's dispatchability is that number.
			avail := r.GetJobCapacity().GetJobsAvailable()
			w.observeJobs(int(avail))
			for _, p := range w.spec.Placement.Jobs {
				dispatchable[p.DescriptorID] = avail > 0
			}
			// A job worker's own admission fence is its job capacity: the placement-set
			// machinery does not run, so the seats are what it says they are.
			if w.admission == pb.AdmissionState_ADMISSION_STATE_UNSPECIFIED {
				w.admission = pb.AdmissionState_ADMISSION_STATE_OPEN
			}
		} else if w.spec.Connection != nil && w.spec.Placement.PlacementSetDigest == "" {
			// A private rental reports every co-fitting placement on the one worker. Keep
			// all of them: collapsing this list to one status made adding package B erase
			// Creator's ability to route the still-live package A.
			observed := make(map[string]remotePlacementObservation, len(r.Placements))
			for _, p := range r.Placements {
				if p == nil {
					continue
				}
				if status == nil || p.Serving == pb.ServingState_SERVING_STATE_DISPATCHABLE {
					status = p
				}
				row := remotePlacementObservation{
					placementID: p.PlacementId, placementSetDigest: spellOf(p.PlacementSetDigest),
					environmentDigest: p.EnvironmentDigest,
					materialization:   p.Materialization, serving: p.Serving,
					dispatchablePlanIDs: map[string]bool{}, knownPlanIDs: map[string]bool{},
				}
				for _, digest := range p.DispatchableBindingDigests {
					planID := spellOf(digest)
					row.dispatchablePlanIDs[planID] = true
					row.knownPlanIDs[planID] = true
					if p.Serving == pb.ServingState_SERVING_STATE_DISPATCHABLE {
						dispatchable[planID] = true
					}
				}
				for _, digest := range p.MaterializableBindingDigests {
					planID := spellOf(digest)
					row.knownPlanIDs[planID] = true
					materializable[planID] = true
				}
				observed[p.PlacementId] = row
				for _, f := range p.Faults {
					w.fault = fmt.Sprintf("%s: %s", f.Reason, brief(f.Detail, 240))
				}
			}
			w.observedRemote = observed
			w.planIDs = keysOf(dispatchable)
			for planID := range materializable {
				if !dispatchable[planID] {
					w.planIDs = append(w.planIDs, planID)
				}
			}
			sort.Strings(w.planIDs)
			if status != nil {
				w.placementID = status.PlacementId
				w.materialization, w.serving = status.Materialization, status.Serving
				w.environmentDigest = status.EnvironmentDigest
				w.executorEpoch = status.ExecutorEpoch
				w.heldSetDigest = status.PlacementSetDigest
				w.fallbackSetDigest = status.RetainedFallbackPlacementSetDigest
				w.acquisition = placementAcquisitionOf(status)
			} else {
				w.materialization = pb.MaterializationState_MATERIALIZATION_STATE_UNSPECIFIED
				w.serving = pb.ServingState_SERVING_STATE_UNSPECIFIED
				w.environmentDigest = ""
			}
		} else {
			for _, p := range r.Placements {
				if p.PlacementId == w.placementID {
					status = p
				}
			}
			if status != nil {
				w.materialization, w.serving = status.Materialization, status.Serving
				w.environmentDigest = status.EnvironmentDigest
				w.executorEpoch = status.ExecutorEpoch
				w.heldSetDigest = status.PlacementSetDigest
				w.fallbackSetDigest = status.RetainedFallbackPlacementSetDigest
				w.acquisition = placementAcquisitionOf(status)
				for _, digest := range status.DispatchableBindingDigests {
					dispatchable[spellOf(digest)] = true
				}
				for _, digest := range status.MaterializableBindingDigests {
					materializable[spellOf(digest)] = true
				}
				for _, f := range status.Faults {
					w.fault = fmt.Sprintf("%s: %s", f.Reason, brief(f.Detail, 240))
				}
			} else {
				// No placement yet — the desired state has not been accepted, or it has and
				// nothing has been materialized. Both axes stay UNSPECIFIED and nothing is
				// dispatchable, which is exactly true.
				w.materialization = pb.MaterializationState_MATERIALIZATION_STATE_UNSPECIFIED
				w.serving = pb.ServingState_SERVING_STATE_UNSPECIFIED
				w.environmentDigest = ""
			}
		}
		w.dispatchable, w.materializable = dispatchable, materializable
		// Fault rows explain state; FAILED axes decide terminality. In particular,
		// BINDING_DEGRADED explicitly means "the worker still serves" and must never become
		// kill authority merely because it shares the diagnostic list with fatal faults.
		w.faulted = faulted(status, r)
		if w.spec.Connection != nil && !w.spec.IsJob() {
			w.faulted = r.WorkerPhase == pb.WorkerPhase_WORKER_PHASE_FAILED
		}
		if w.faulted {
			if w.errorSince.IsZero() {
				w.errorSince = time.Now()
			}
		} else {
			w.errorSince = time.Time{}
			if status != nil && len(status.Faults) == 0 && len(r.Faults) == 0 {
				w.fault = ""
			}
		}
		for _, f := range r.Faults {
			w.fault = fmt.Sprintf("%s: %s", f.Reason, brief(f.Detail, 240))
			if r.AcceptedDesiredStateRevision < desiredRevision &&
				f.Subject == fmt.Sprintf("revision %d", desiredRevision) && permanentDesiredRefusal(f.Kind) {
				w.desiredRefusal = exit.Named(exit.Structural, "placement_config_refused",
					"the package worker refused its placement: %s — %s", f.Reason, brief(f.Detail, 240)).
					WithRemedy("the installed package Runtime is incompatible with this Cozy build")
				c.relayDescriptorDefect(w, desiredRevision, f)
			}
		}
		if r.AcceptedDesiredStateRevision >= desiredRevision {
			w.desiredRefusal = nil
		}
		if len(r.Faults) == 0 && status != nil {
			for _, f := range status.Faults {
				w.fault = fmt.Sprintf("%s: %s", f.Reason, brief(f.Detail, 240))
			}
		}
		if refused := w.observeLatchedFault(r); refused != nil {
			if w.desiredRefusal == nil {
				verdict = fmt.Sprintf("worker %s: desired revision %d REFUSED — the placement fault "+
					"repeated unchanged on %d consecutive reports and nothing here can answer it: %s",
					w.instanceID, desiredRevision, w.latchedFaultReports, refused.Message)
			}
			w.desiredRefusal = refused
		}
		repeatedFault = w.latchedFaultReports > 1
		workerTerminal = retirementGround(w) != ""
	}
	phase := trimEnum(pb.WorkerPhase_name[int32(r.WorkerPhase)], "WORKER_PHASE_")
	c.mu.Unlock()
	if w != nil && w.media != nil {
		go c.retryMediaCleanup(w)
	}
	for _, breach := range laneBreaches {
		c.logf("worker breach on %s: %s", s.instanceID, breach)
	}
	if verdict != "" {
		c.logf("%s", verdict)
	}
	for _, held := range heldVerdicts {
		c.settleRefusedOutcome(s, held.requestID, held.ordinal, held.outcome)
	}
	// A fault rides every ReportCadence until it clears; it is logged when it first
	// appears and again only after it has been absent. The worker likewise puts its last
	// few activity entries on EVERY report; an entry is logged once, when its seq first
	// appears (cl-099: one executor spawn re-logged on every 2 s report read as a boot
	// loop of hundreds of spawns on the owner's box).
	if w != nil {
		var placementFaults []*pb.Fault
		if status != nil {
			placementFaults = status.Faults
		}
		c.mu.Lock()
		workerFaults, placementFaults := w.freshFaults(r.Faults, placementFaults)
		fresh := w.freshActivity(r.Activity)
		c.mu.Unlock()
		if !repeatedFault {
			for _, f := range workerFaults {
				c.logf("worker fault %s on %s: %s (%s)", pb.FaultKind_name[int32(f.Kind)],
					f.Subject, f.Reason, f.Detail)
			}
		}
		for _, a := range fresh {
			c.logf("activity seq=%d %s: %s", a.Seq, a.Kind, a.Step)
		}
		if !repeatedFault {
			for _, f := range placementFaults {
				c.logf("placement fault %s on %s: %s (%s)", pb.FaultKind_name[int32(f.Kind)],
					f.Subject, f.Reason, f.Detail)
			}
		}
	}
	// DISPATCHABLE is the only state that can change the queue's answer. A free local
	// seat also re-asks the FIFO head's residency question: the head may name a different
	// package whose launch was parked while this worker held the same device envelope.
	// `selectOrStart` still passes through the durable holder and idle-worker fences, so a
	// report can wake scheduling but can never authorize preemption by itself.
	if status != nil && status.Serving == pb.ServingState_SERVING_STATE_DISPATCHABLE {
		go c.drain()
		if r.AvailableAttemptSlots > 0 {
			go c.reviveQueue()
		}
	} else if w != nil && w.spec.IsJob() && r.GetJobCapacity().GetJobsAvailable() > 0 {
		go c.drain()
	} else if workerTerminal {
		// A FAILED axis or permanent desired-state refusal is the worker's answer, not a
		// stall timer to begin. Re-ask the queued head immediately so it can replace a
		// stale process or receive the replacement's typed refusal.
		go c.checkRetirement()
	}
	if r.GetJobCapacity() != nil {
		c.logf("observed phase=%s accepted=%d converged=%d jobs_available=%d jobs_in_flight=%d",
			phase, r.AcceptedDesiredStateRevision, r.ConvergedRevision,
			r.GetJobCapacity().GetJobsAvailable(), r.GetJobCapacity().GetJobsInFlight())
		return
	}
	if status != nil {
		c.logf("observed phase=%s accepted=%d converged=%d placement=%s %s/%s epoch=%d "+
			"admission=%s/%d slots=%d lanes=%s placement_lane=%s dispatchable=%d held=%d held_manifests=%v",
			phase, r.AcceptedDesiredStateRevision, r.ConvergedRevision, status.PlacementId,
			trimEnum(pb.MaterializationState_name[int32(status.Materialization)], "MATERIALIZATION_STATE_"),
			trimEnum(pb.ServingState_name[int32(status.Serving)], "SERVING_STATE_"),
			status.ExecutorEpoch,
			trimEnum(pb.AdmissionState_name[int32(r.AdmissionState)], "ADMISSION_STATE_"),
			r.AdmissionEpoch, r.AvailableAttemptSlots, laneSummary(r.Lanes),
			orNone(status.DeviceLaneId), len(status.DispatchableBindingDigests), len(r.HeldAttempts),
			r.HeldManifests)
		return
	}
	c.logf("observed phase=%s accepted=%d converged=%d (no placement applied yet)",
		phase, r.AcceptedDesiredStateRevision, r.ConvergedRevision)
}

func heldPlacements(rows []*pb.HeldAttempt) []string {
	out := make([]string, 0, len(rows))
	for _, row := range rows {
		if row != nil {
			out = append(out, row.PlacementId)
		}
	}
	return out
}

func permanentDesiredRefusal(kind pb.FaultKind) bool {
	switch kind {
	case pb.FaultKind_FAULT_KIND_CONFIG_REFUSED,
		pb.FaultKind_FAULT_KIND_PLACEMENT_SET_UNSUPPORTED,
		pb.FaultKind_FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH:
		return true
	}
	return false
}

func placementAcquisitionOf(status *pb.PlacementStatus) PlacementAcquisitionFacts {
	var facts PlacementAcquisitionFacts
	if status == nil || status.Acquisition == nil {
		return facts
	}
	leg := func(in *pb.AcquisitionLegObservation) AcquisitionLegFacts {
		if in == nil {
			return AcquisitionLegFacts{}
		}
		return AcquisitionLegFacts{
			StartedNS: in.StartedMonotonicNs, EndedNS: in.EndedMonotonicNs,
			DownloadedBytes: in.DownloadedBytes, ReusedBytes: in.ReusedBytes,
		}
	}
	facts.Package = leg(status.Acquisition.Package)
	facts.Model = leg(status.Acquisition.Model)
	return facts
}

// faulted reads only the protocol's terminal axes. Fault rows are explanations and may
// coexist with a serving placement (BINDING_DEGRADED and a rejected replacement both do);
// treating their mere presence as terminal killed healthy active attempts.
func faulted(status *pb.PlacementStatus, r *pb.ObservedWorkerState) bool {
	if r.WorkerPhase == pb.WorkerPhase_WORKER_PHASE_FAILED {
		return true
	}
	if status == nil {
		return false
	}
	return status.Materialization == pb.MaterializationState_MATERIALIZATION_STATE_FAILED
}

// --------------------------------------------------------------------------- accepted

func (c *Orchestrator) onAccepted(s *session, a *pb.AttemptAccepted) {
	ordinal := a.AttemptOrdinal
	row, e := c.opt.Store.AttemptRow(a.RequestId, int64(ordinal))
	if e != nil || row == nil {
		c.logf("AttemptAccepted for %s#%d REFUSED: no such assigned attempt", a.RequestId, ordinal)
		return
	}
	spelled, err := canonical.Spell(a.InvocationSpecDigest)
	if err != nil || spelled != row.InvocationDigest {
		c.logf("AttemptAccepted for %s#%d REFUSED: it echoes %v, the assignment is %s",
			a.RequestId, ordinal, spelled, row.InvocationDigest)
		return
	}
	// Acceptance is queue ADMISSION (minor 23, proto-026): journaled, hydrated, queued on a
	// lane. It binds neither a plan nor an executor; both bind at device entry and ride
	// HeldAttempt from RUNNING on.
	if e := c.opt.Store.Accepted(a.RequestId, int64(ordinal), s.bootID); e != nil {
		c.logf("AttemptAccepted for %s#%d REFUSED: %s", a.RequestId, ordinal, e.Message)
		return
	}
	c.mu.Lock()
	if w := c.workers[row.InstanceID]; w != nil && w.spec.Connection == nil && !w.spec.IsJob() {
		c.lastUseRevision++
		w.lastUseRevision = c.lastUseRevision
	}
	c.mu.Unlock()
	c.settleDispatch(a.RequestId, ordinal, true)
	c.logf("AttemptAccepted %s#%d placement=%s lane=%s", a.RequestId, ordinal, a.PlacementId, a.LaneId)
	c.emit(a.RequestId, "request.accepted", ordinal, map[string]any{"lane_id": a.LaneId})
	c.signalAccepted(key(a.RequestId, ordinal))
}

// --------------------------------------------------------------------------- outcome

// onOutcome applies one AttemptOutcome — was AttemptTerminal, and the rename is the point
// (#481): "terminal" reads as a device and the thing is an OUTCOME. It is also no longer
// only the end of execution, because a pre-execution REFUSAL is a journaled outcome too.
func (c *Orchestrator) onOutcome(s *session, t *pb.AttemptOutcome) {
	ordinal := t.AttemptOrdinal
	report := func(format string, args ...any) {
		c.logf("AttemptOutcome %s#%d REFUSED: "+format,
			append([]any{t.RequestId, ordinal}, args...)...)
	}
	refuse := report

	// 1. The digest is RECOMPUTED over the resident bytes. A digest never bypasses the
	//    lower check, and a mismatched one is not acked — the worker keeps replaying.
	computed := canonical.Digest(t.OutcomeCanonicalBytes)
	if !bytes.Equal(computed, t.OutcomeDigest) {
		refuse("outcome_digest %x does not hash the %d resident bytes (%x)",
			t.OutcomeDigest, len(t.OutcomeCanonicalBytes), computed)
		return
	}
	// 2. The document is parsed under UNKNOWN-FIELD REFUSAL and the re-emit law. It is the
	//    sole pre-release AttemptOutcomeBody/1 shape, including exact weights receipts.
	doc, err := canonical.Read(t.OutcomeCanonicalBytes, &pb.AttemptOutcomeBody{})
	if err != nil {
		refuse("the outcome document is inadmissible (%s)", err)
		return
	}
	// 3. The routing copies must agree with the DOCUMENT, which is authoritative.
	spelledSpec, _ := canonical.Spell(t.InvocationSpecDigest)
	if doc.Str("request_id") != t.RequestId || uint64(doc.Int("attempt_ordinal")) != ordinal ||
		doc.Str("invocation_spec_digest") != spelledSpec {
		refuse("envelope/document divergence: envelope %s#%d/%s, document %s#%d/%s",
			t.RequestId, ordinal, shortDigest(spelledSpec),
			doc.Str("request_id"), doc.Int("attempt_ordinal"),
			shortDigest(doc.Str("invocation_spec_digest")))
		return
	}

	// THE OUTCOME IS HELD FROM HERE UNTIL THE ACK, and the worker counts it against its own
	// free seats while this owner holds it (#480d). Tracking it here is what makes an owner
	// that stops acking VISIBLE as an owner starving its own admission, rather than as a
	// worker that mysteriously stopped taking work.
	c.mu.Lock()
	if holder := c.workers[s.instanceID]; holder != nil {
		holder.unacked++
	}
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		if holder := c.workers[s.instanceID]; holder != nil && holder.unacked > 0 {
			holder.unacked--
		}
		c.mu.Unlock()
	}()

	status := outcomeStatus(doc.Int("status"))
	cause := causeCode(doc.Sub("cause").Int("code"))
	origin := causeOrigin(doc.Sub("cause").Int("origin"))
	executionStarted, _ := doc["execution_started"].(bool)
	req, e := c.opt.Store.RequestRow(t.RequestId)
	if e != nil || req == nil {
		refuse("no request row to settle")
		return
	}
	attemptRow, e := c.opt.Store.AttemptRow(t.RequestId, int64(ordinal))
	if e != nil || attemptRow == nil {
		refuse("no assigned attempt row to settle")
		return
	}
	// FROM HERE A REFUSAL IS A JOURNALED FACT ON BOTH SIDES. The outcome is admissible and
	// names an open attempt this session owns; the worker journaled it before sending and
	// will restate it — replayed byte for byte, or held pending this owner's ack on every
	// report — until an ack names it. Each restatement this owner refuses again is one
	// observation, and on StillFactor of them the request settles typed
	// (settleRefusedOutcome) instead of sitting in_progress behind a held outcome.
	if attemptRow.SessionID == s.bootID && openAttempt(attemptRow.State) {
		spelledDigest := shortNone(t.OutcomeDigest)
		refuse = func(format string, args ...any) {
			reason := fmt.Sprintf(format, args...)
			c.logf("AttemptOutcome %s#%d REFUSED: %s", t.RequestId, ordinal, reason)
			verdict := c.observeRefusedOutcome(s, t.RequestId, ordinal, refusedOutcome{
				outcomeID: t.OutcomeId, digest: spelledDigest, reason: reason,
				body: t.OutcomeCanonicalBytes, spec: spelledSpec,
				consumed: status != "REFUSED" || executionStarted,
			})
			if verdict != nil {
				c.settleRefusedOutcome(s, t.RequestId, ordinal, *verdict)
			}
		}
	}
	declaredWeightsOutputs, e := decodeWeightsOutputs(attemptRow.WeightsOutputs)
	if e != nil {
		refuse("%s", e.Message)
		return
	}
	if len(declaredWeightsOutputs) > 0 && len(doc.Sub("output_manifest").List("outputs")) > 0 {
		refuse("weights-only job also returned ordinary output-manifest entries")
		return
	}
	weightsReceipts, receiptsBySlot, e := weightsReceiptsFromOutcome(*req, *attemptRow, doc)
	if e != nil {
		refuse("%s", e.Message)
		return
	}
	// An exact replay of a terminal already in the authority needs only another ack. In
	// particular it must not fetch pod outputs again: after the first mirror and ack this
	// owner is allowed to delete the remote attempt subtree, while the durable local output
	// rows and bytes remain the accepted answer.
	knownReplay := false
	if attemptRow.State == "terminal" || attemptRow.State == "closed" {
		if attemptRow.TerminalDigest != shortNone(t.OutcomeDigest) {
			refuse("the attempt already closed with terminal %s, not %s",
				shortDigest(attemptRow.TerminalDigest), shortDigest(shortNone(t.OutcomeDigest)))
			return
		}
		knownReplay = true
	}
	// THE MIRROR RUNS BEFORE ANYTHING IS ACCEPTED OR ACKED. An output that is not here,
	// or is not what the manifest says it is, means this outcome cannot be honoured: the
	// worker's journaled outcome stays owed and replayable, exactly as a failed job
	// promotion does below, rather than being acked with a row that points at nothing.
	c.mu.Lock()
	holder := c.workers[s.instanceID]
	c.mu.Unlock()
	var outputs []records.Output
	if !knownReplay {
		outputs, e = c.mirrorOutputs(*req, ordinal, doc, holder)
		if e != nil {
			refuse("%s", e.Message)
			return
		}
	}
	// cl-006 OWNS TRIAGE PERSISTENCE (cr-011's seam): the bundle lives in the worker's
	// own root, and a worker root does not outlive its worker. Copy it out and verify it
	// against the outcome document's OWN TriageBundleRef before the transaction runs.
	triage := Triage{}
	if !knownReplay {
		triage = c.captureTriage(s, t.RequestId, ordinal, doc.Sub("triage_bundle"))
	}
	// WHETHER THIS ATTEMPT ENDS THE REQUEST is decided BEFORE the event is written, not
	// after. An ABANDONED attempt that the requeue projection will re-dispatch has ended
	// an ATTEMPT, not a REQUEST — and the contract's terminal-stop rule means a client
	// that saw `request.failed` would close its stream and report a failure for a request
	// that goes on to succeed. Found live by the orchestrator-kill arm: the killed
	// attempt's ABANDONED terminal stopped the client's stream while attempt 2 was still
	// being minted.
	requeuing := requeueable(status, cause, origin, executionStarted)
	kept := triage.keep(status, cause, doc.Str("safe_message"), outputs, requeuing)
	requestState := requeueState(status, requeuing)
	if req.ModelTransfer != nil && !requeuing {
		requestState = "finalizing"
		kept.Type = "request.finalizing"
		kept.Payload = map[string]any{"status": "FINALIZING", "execution_status": status,
			"outputs": []any{}, "requeuing": false}
	}
	weightsFinalizations, e := weightsFinalizationIntents(
		*req, *attemptRow, status, requeuing, receiptsBySlot)
	if e != nil {
		refuse("%s", e.Message)
		return
	}

	// THE PUBLICATION, for a job. The attempt's staged writes are PROMOTED into the
	// addressable publication root first, and the row that makes the publication exist
	// rides the very transaction that accepts the terminal. A kill before that commit
	// therefore exposes no publication at all — not in the authority and not at the
	// addressable path — and a kill after it has already preserved one.
	//
	// A REQUEUEING attempt writes NO publication. Its terminal ended an attempt, not the
	// request, so a row stamped ABANDONED with zero entries would be a claim about a
	// publication that does not exist — and it would be overwritten by the ordinal that
	// goes on to succeed. Observed live: attempt 1 of a killed job committed an empty
	// `local/_job-…` publication seconds before attempt 2 published the real one.
	var publication *records.Publication
	if req.IsJob() && len(declaredWeightsOutputs) == 0 && !requeuing && !knownReplay {
		if e := c.promote(*req, ordinal, outputs); e != nil {
			refuse("%s", e.Message)
			return
		}
		publication = c.publicationOf(*req, ordinal, status, cause, outputs)
	}

	began := time.Now()
	applied, e := c.opt.Store.AcceptTerminal(records.Terminal{
		RequestID: t.RequestId, Attempt: int64(ordinal), SessionID: s.bootID,
		InvocationDigest: spelledSpec, TerminalID: t.OutcomeId,
		TerminalDigest: shortNone(t.OutcomeDigest), Status: status, Cause: cause,
		SafeMessage:   doc.Str("safe_message"),
		TriageSubject: triage.Subject, TriageDigest: triage.Digest,
		TriageLength: triage.Length, TriagePath: triage.Path,
		Body: t.OutcomeCanonicalBytes, Outputs: outputs,
		WeightsFinalizations: weightsFinalizations,
		EventType:            kept.Type, EventPayload: kept.Payload,
		// A requeueing request is QUEUED for its next ordinal, not failed. Writing the
		// attempt's own status onto the request row would make the status document say
		// `failed` for a request that is still going.
		RequestState: requestState,
		Publication:  publication,
	})
	if e != nil {
		refuse("%s", e.Message)
		return
	}
	// CAPACITY MOVES ONLY AFTER AUTHORITY ACCEPTS THE OUTCOME. In particular, a frame from
	// the wrong session or carrying the wrong InvocationSpec digest must not be able to
	// return another worker's reservation and reopen its seat before AcceptTerminal rejects
	// it. The durable attempt row authenticates the causal answer; the frame cannot do so by
	// naming a request id that happens to hold an offer.
	c.settleDispatch(t.RequestId, ordinal, status != "REFUSED" || executionStarted)
	// The outcome is accepted: whatever this owner refused before is forgotten, and an ack
	// withheld behind weights finalization below is a wait, not a refusal.
	c.forgetRefusedOutcome(s.instanceID, t.RequestId, ordinal)
	refuse = report
	if applied {
		c.logf("AttemptOutcome %s#%d %s/%s(%s) started=%v applied in %.2f ms: %d output(s) "+
			"became visible in the SAME transaction", t.RequestId, ordinal, status, cause,
			origin, executionStarted, float64(time.Since(began).Microseconds())/1000,
			len(outputs))
		if len(weightsReceipts) > 0 {
			c.logf("AttemptOutcome %s#%d durably recorded %s before acknowledgement",
				t.RequestId, ordinal, weightsReceiptSummary(weightsReceipts))
		}
		if publication != nil {
			c.logf("publication %s committed: %d entr(y|ies), %d B, root %s",
				publication.Repo, publication.Entries, publication.Bytes, publication.Root)
		}
	} else {
		c.logf("AttemptOutcome %s#%d is an exact replay of a closed outcome: re-acked, "+
			"nothing applied twice", t.RequestId, ordinal)
	}
	if !requeuing {
		// Export settlement is deliberately separate from the execution terminal. A full
		// destination does not rewrite success into failure, and an exact replay retries
		// this durable row without mirroring worker bytes twice.
		c.RetryOutputExport(t.RequestId)
	}

	// Weights decisions are independent frames, but the worker may reclaim after Ack. Send
	// every persisted first-wins intent now and withhold Ack until their exact results land.
	pending, e := c.sendPendingWeightsFinalizations(s, t.RequestId, int64(ordinal))
	if e != nil {
		refuse("weights finalization remains pending: %s", e.Message)
		return
	}
	if pending > 0 {
		c.logf("AttemptOutcome %s#%d remains unacked behind %d weights finalization(s)",
			t.RequestId, ordinal, pending)
		return
	}
	c.ackSettledOutcome(s, t.RequestId, ordinal)
}

// afterAck is the one post-terminal continuation, shared by the live frame and snapshot
// recovery. It is intentionally idempotent: cleanup and BeginRequeue both have durable
// guards, so a replay cannot spend twice or delete a still-owned asset.
func (c *Orchestrator) afterAck(req records.Request, attempt records.Attempt, holder *worker) {
	requeue := req.State == "requeue_pending"
	c.cleanupAttempt(req, uint64(attempt.Attempt), holder, !requeue)
	verdict := outcomeError(attempt.TerminalStatus, attempt.TerminalCause, attempt.SafeMessage)
	if req.ModelTransfer != nil && req.State == "failed" {
		if transfer, problem := c.opt.Store.ModelTransferOf(req.ID); problem == nil && transfer != nil {
			verdict = exit.Named(exit.Failed, transfer.ErrorCode, "%s", transfer.SafeError)
		}
	}
	if req.ModelTransfer != nil && req.State == "canceled" {
		verdict = exit.New(exit.Canceled, "model transfer %s finalization was canceled", req.ID)
	}
	c.signalClosed(key(req.ID, uint64(attempt.Attempt)), verdict)
	if requeue {
		c.Requeue(req.ID, attempt.TerminalStatus+"/"+attempt.TerminalCause)
		return
	}
	if req.ModelTransfer != nil {
		c.forgetTransferProgress(req.ID)
	} else {
		c.frames.forget(req.ID)
	}
	c.signalClosed(requestWaitKey(req.ID), verdict)
	c.releaseManaged(req)
}

// outcomeRefusedCause is the record owner's OWN terminal cause: the worker's journaled
// outcome stands refused, and the request that waited on it is failed under this name.
const outcomeRefusedCause = "worker.outcome_refused"

// refusedOutcome is one outcome this owner could not honour, as the worker keeps
// restating it: the identity refused, this owner's reason, the bytes the verdict will
// journal, and how many consecutive worker reports have carried it unchanged.
type refusedOutcome struct {
	outcomeID string
	digest    string
	reason    string
	body      []byte
	spec      string
	consumed  bool // the seat was spent: execution started, or the worker did not refuse
	reports   int
}

func openAttempt(state string) bool {
	return state == "offered" || state == "accepted" || state == "recovered_open"
}

// observeRefusedOutcome counts one worker report that restates a refused outcome. The
// report is either the AttemptOutcome replayed byte for byte (the caller refused it
// again) or a held_attempts row pending this owner's ack that names the same outcome id
// and digest. A different outcome for the same attempt is a changed fact and restarts
// the count; an accepted one forgets it (forgetRefusedOutcome).
//
// COUNTED, NOT TIMED, on StillFactor exactly as a latched placement fault is
// (observeLatchedFault): the Runtime restates a held outcome on every ReportCadence, so
// StillFactor consecutive reports carrying it unchanged is the worker's final word on
// that attempt, and the answer is this owner's to give — nothing the worker can do
// changes an outcome it already journaled.
func (c *Orchestrator) observeRefusedOutcome(s *session, requestID string, ordinal uint64,
	seen refusedOutcome) *refusedOutcome {
	c.mu.Lock()
	defer c.mu.Unlock()
	w := c.workers[s.instanceID]
	if w == nil || w.bootID != s.bootID {
		return nil
	}
	k := key(requestID, ordinal)
	held := w.refused[k]
	if held == nil || held.outcomeID != seen.outcomeID || held.digest != seen.digest {
		seen.reports = 0
		held = &seen
		w.refused[k] = held
	}
	held.reason = seen.reason
	held.reports++
	if held.reports < StillFactor {
		return nil
	}
	delete(w.refused, k)
	verdict := *held
	return &verdict
}

// observeHeldOutcomes reads one report's held_attempts rows against the outcomes this
// owner refused: a row pending ack naming a refused outcome is the worker restating it.
// It returns the verdicts reached, for the caller to settle outside the lock.
func (c *Orchestrator) observeHeldOutcomes(w *worker, rows []*pb.HeldAttempt) []heldVerdict {
	var verdicts []heldVerdict
	for _, row := range rows {
		if row == nil || row.State != pb.AttemptState_ATTEMPT_STATE_OUTCOME_PENDING_ACK {
			continue
		}
		k := key(row.RequestId, row.AttemptOrdinal)
		held := w.refused[k]
		if held == nil || held.outcomeID != row.OutcomeId || held.digest != shortNone(row.OutcomeDigest) {
			continue
		}
		held.reports++
		if held.reports < StillFactor {
			continue
		}
		delete(w.refused, k)
		verdicts = append(verdicts, heldVerdict{requestID: row.RequestId,
			ordinal: row.AttemptOrdinal, outcome: *held})
	}
	return verdicts
}

type heldVerdict struct {
	requestID string
	ordinal   uint64
	outcome   refusedOutcome
}

func (c *Orchestrator) forgetRefusedOutcome(instanceID, requestID string, ordinal uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if w := c.workers[instanceID]; w != nil {
		delete(w.refused, key(requestID, ordinal))
	}
}

// settleRefusedOutcome is this owner's verdict on an outcome the worker restated
// unchanged StillFactor times and this owner refused every time. The attempt closes under
// the worker's OWN outcome identity — its id and digest are what the ack must echo and
// what a later replay is matched against — with this owner's status and cause: FAILED,
// worker.outcome_refused, the refusal as the message. The request fails typed, the seat
// is released, the worker is acked so it drops the held outcome, and nothing waits on a
// stream any more. The protocol has no ack-as-rejected: AttemptOutcomeAck is an echo, so
// the same ack that closes an honoured outcome closes a refused one, and the refusal is
// journaled here, in the attempt row, rather than on the wire.
func (c *Orchestrator) settleRefusedOutcome(s *session, requestID string, ordinal uint64,
	r refusedOutcome) {
	verdict := fmt.Sprintf("%s: %s", outcomeRefusedCause, r.reason)
	c.logf("worker %s: outcome %s for %s#%d REFUSED — restated unchanged on %d consecutive "+
		"reports and nothing here can honour it; the request fails %s",
		s.instanceID, r.outcomeID, requestID, ordinal, StillFactor, verdict)
	applied, e := c.opt.Store.AcceptTerminal(records.Terminal{
		RequestID: requestID, Attempt: int64(ordinal), SessionID: s.bootID,
		InvocationDigest: r.spec, TerminalID: r.outcomeID, TerminalDigest: r.digest,
		Status: "FAILED", Cause: outcomeRefusedCause, SafeMessage: r.reason, Body: r.body,
		EventType: "request.failed",
		EventPayload: map[string]any{"status": "FAILED", "cause": outcomeRefusedCause,
			"error_type": outcomeRefusedCause, "error": r.reason,
			"outputs": []any{}, "requeuing": false},
		RequestState: "failed",
	})
	if e != nil {
		c.logf("%s#%d could not settle %s: %s", requestID, ordinal, outcomeRefusedCause, e.Message)
		return
	}
	c.settleDispatch(requestID, ordinal, r.consumed)
	if applied {
		c.RetryOutputExport(requestID)
	}
	c.ackSettledOutcome(s, requestID, ordinal)
}

// cleanupAttempt runs only after the outcome's bytes were mirrored, its terminal commit
// succeeded, and OutcomeAck was sent. A local attempt leaves nothing behind — its result
// files were written where they live and its inputs were never staged. A remote attempt
// releases its reservation on the pod. Request assets and the request's `tmp/<id>/` are
// dropped only after the final attempt (assets only when no other live request owns the
// same content digest).
func (c *Orchestrator) cleanupAttempt(req records.Request, attempt uint64, holder *worker, final bool) {
	if holder != nil && holder.media != nil {
		c.cleanupRemote(req.ID, attempt, holder)
	}
	if !final {
		return
	}
	c.reclaimTmp(req.ID)
	c.cleanupRequestAssets(req)
}

func (c *Orchestrator) cleanupRequestAssets(req records.Request) {
	unlock := inputasset.Guard()
	if e := inputasset.DropUnowned(c.opt.Layout, c.opt.Store, req.Assets); e != nil {
		c.logf("request %s input asset cleanup deferred: %s", req.ID, e.Message)
	}
	unlock()
	if req.LocalPackageDigest == "" {
		return
	}
	unlockLocal := localpackage.Guard()
	defer unlockLocal()
	if e := localpackage.DropDigestUnowned(c.opt.Layout, c.opt.Store,
		req.LocalPackageDigest); e != nil {
		c.logf("request %s local package cleanup deferred: %s", req.ID, e.Message)
	}
}

func (c *Orchestrator) cleanupRemote(requestID string, attempt uint64, holder *worker) {
	k := key(requestID, attempt)
	c.mu.Lock()
	if c.mediaCleaning[k] {
		c.mu.Unlock()
		return
	}
	c.mediaCleaning[k] = true
	c.mu.Unlock()

	e := holder.media.DropAttempt(media.Slot(requestID, attempt))
	if e == nil {
		e = c.opt.Store.MarkMediaCleaned(requestID, int64(attempt))
	}
	c.mu.Lock()
	delete(c.mediaCleaning, k)
	c.mu.Unlock()
	if e != nil {
		c.logf("%s#%d pod media cleanup deferred: %s", requestID, attempt, e.Message)
	}
}

// Every remote Report retries the durable obligations still assigned to that worker.
// DELETE is idempotent, and media_cleaned stops successful rows from being revisited.
func (c *Orchestrator) retryMediaCleanup(holder *worker) {
	// Local workers have no pod media plane and leave nothing to clean; a recovered
	// local snapshot must never try to call a nil client.
	if holder == nil || holder.media == nil {
		return
	}
	owed, e := c.opt.Store.MediaCleanupOwed(holder.instanceID)
	if e != nil {
		c.logf("cannot read media cleanup obligations of %s: %s", holder.instanceID, e.Message)
		return
	}
	for _, attempt := range owed {
		c.cleanupRemote(attempt.RequestID, uint64(attempt.Attempt), holder)
	}
}

// Triage is what cl-006 persisted for one attempt. An empty Subject means the outcome
// named no bundle, which is a fact, not a failure.
type Triage struct {
	Subject string
	Digest  string
	Length  int64
	Path    string
	Fault   string // why the bytes were not kept, when a subject was named and they were not
}

// outcomeEvent is the durable lifecycle event that rides the outcome transaction.
type outcomeEvent struct {
	Type    string
	Payload map[string]any
}

// requeueState is the REQUEST's state after one attempt ended. A requeueing request is
// queued for its next ordinal; anything else takes the attempt's own status.
func requeueState(status string, requeuing bool) string {
	if requeuing {
		return "requeue_pending"
	}
	return strings.ToLower(status)
}

// keep renders the attempt-end event's closed payload. It carries an OPAQUE handle for
// every asset (media id) and for triage (subject) — never a path, never bytes, and never
// base64. That is what makes "no client-supplied path exists" a property of the schema
// rather than of a validator.
//
// `requeuing` decides whether this is a TERMINAL event at all. `request.attempt_failed`
// is deliberately not in the terminal set: it says an attempt ended and the request did
// not, which is precisely the state the recovered-attempts law creates.
func (tr Triage) keep(status, cause, safeMessage string, outputs []records.Output, requeuing bool) outcomeEvent {
	eventType := "request.failed"
	switch {
	case requeuing:
		eventType = "request.attempt_failed"
	case status == "SUCCEEDED":
		eventType = "request.completed"
	case status == "CANCELED":
		eventType = "request.canceled"
	}
	media := make([]map[string]any, 0, len(outputs))
	for _, o := range outputs {
		media = append(media, map[string]any{
			"output_id": o.OutputID, "media_id": o.MediaID,
			"mime_type": o.MimeType, "length": o.Length, "digest": o.Digest,
		})
	}
	payload := map[string]any{
		"status": status, "cause": cause, "outputs": media, "requeuing": requeuing,
	}
	if status != "SUCCEEDED" {
		payload["error_type"] = cause
		payload["error"] = safeMessage
	}
	if tr.Subject != "" {
		payload["triage_subject"] = tr.Subject
	}
	if tr.Fault != "" {
		payload["triage_fault"] = tr.Fault
	}
	return outcomeEvent{Type: eventType, Payload: payload}
}

// captureTriage copies the worker's bundle into cozy-creator's own store, verified
// against the TERMINAL DOCUMENT's reference rather than against the worker's journal.
//
// That choice is deliberate and stronger than cr-011's own reader: this orchestrator has
// already recomputed the terminal digest over the resident bytes and parsed the document
// under unknown-field refusal, so `triage_bundle.write_receipt_digest` is a fact it
// ACCEPTED. Reading the worker's journal instead would be trusting a file the worker can
// still write. A mismatch is recorded as a fault on the attempt and the bytes are not
// kept — a bundle that does not hash to what the terminal claimed is not evidence.
//
// WHERE THE BYTES ARE depends on the worker, exactly as it does for outputs. A local
// worker shares this filesystem and its bundle is a file under its own root; a POD worker
// wrote it on the pod, and it is FETCHED by subject over the pod's media plane before
// the same verification runs here. Until cl-101 the pod path read the local path and
// recorded `bundle_absent` for every remote failure — a pod's own explanation of its
// failure was the one document the owner could never read.
func (c *Orchestrator) captureTriage(s *session, requestID string, attempt uint64, ref canonical.Doc) Triage {
	tr := Triage{Subject: ref.Str("subject_id")}
	if tr.Subject == "" {
		return tr
	}
	tr.Digest, tr.Length = ref.Str("write_receipt_digest"), ref.Int("length")

	c.mu.Lock()
	w := c.workers[s.instanceID]
	c.mu.Unlock()
	if w == nil {
		tr.Fault = "the worker that wrote it is gone before its bundle could be copied"
		return tr
	}
	var data []byte
	if w.media != nil {
		fetched, e := w.media.GetTriage(tr.Subject, tr.Digest, tr.Length)
		if e != nil {
			tr.Fault = "bundle_unfetched: " + e.Message
			c.logf("triage %s for %s#%d NOT kept from %s: %s", tr.Subject, requestID, attempt,
				w.media.Addr(), e.Message)
			return tr
		}
		data = fetched
	} else {
		source := filepath.Join(c.opt.Layout.WorkerDir(w.instanceID), "run", "triage", tr.Subject+".json")
		read, err := os.ReadFile(source)
		if err != nil {
			tr.Fault = "bundle_absent: the terminal names a bundle that is not on disk"
			c.logf("triage %s for %s#%d NOT kept: %s", tr.Subject, requestID, attempt, err)
			return tr
		}
		data = read
	}
	spelled, _ := canonical.Spell(canonical.Digest(data))
	if int64(len(data)) != tr.Length || spelled != tr.Digest {
		tr.Fault = fmt.Sprintf("bundle_corrupt: %d B hashing to %s does not match the terminal's %d B / %s",
			len(data), shortDigest(spelled), tr.Length, shortDigest(tr.Digest))
		c.logf("triage %s for %s#%d REFUSED: %s", tr.Subject, requestID, attempt, tr.Fault)
		return tr
	}
	dest := c.opt.Layout.TriageFile(tr.Subject)
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		tr.Fault = "the local triage store is unwritable: " + err.Error()
		return tr
	}
	if err := os.WriteFile(dest, data, 0o600); err != nil {
		tr.Fault = "the bundle could not be kept: " + err.Error()
		return tr
	}
	tr.Path = dest
	c.logf("triage %s kept for %s#%d (%d B, %s)", tr.Subject, requestID, attempt,
		tr.Length, shortDigest(tr.Digest))
	return tr
}

// requeueable is the record owner's projection: an accepted-but-incomplete attempt, the
// infra-class failures, and a WORKER pre-execution refusal earn a new ordinal. A
// deterministic body failure and an author/runtime refusal settle — re-running them would
// only fail again.
//
// THE REFUSED PROJECTION SPLITS BY (cause, origin) — #480b, and it is the reason
// retryability cannot be a wire fact. "Refused" used to mean one thing: a judgment about
// the WORK, settle it. rev-2 routes a pre-execution capacity decline through the same
// status, because every offer must get a JOURNALED outcome and there is no
// AttemptDeclined message to add. Those consume the attempt ordinal and ZERO billed
// execution budget, so the next ordinal may go out immediately and elsewhere.
//
// `execution_started` is the structural check, not a cause-code allowlist (#480c): the
// author having run is a bit the worker sets, and inferring it from a code list is exactly
// the fragility the bit exists to remove. A "worker refusal" that claims execution
// started is a contradiction, and it settles rather than being re-dispatched.
//
// The re-dispatch still charges the request's DURABLE requeue budget. Zero BILLED budget
// is a statement about money; the bound on how many times this record owner will try is its own,
// and a worker refusing forever must still terminate.
func requeueable(status, cause, origin string, executionStarted bool) bool {
	if status == "ABANDONED" {
		return true
	}
	if status == "REFUSED" {
		return origin == "WORKER" && !executionStarted && preExecution(cause)
	}
	if status != "FAILED" {
		return false
	}
	switch cause {
	case "EXECUTOR_FAULT", "GRANT_EXPIRED", "WEIGHTS_UNFETCHABLE", "CAPABILITY_UNAVAILABLE":
		return true
	}
	return false
}

// preExecution names the four causes rev-2 §6/§7 defines as worker pre-execution
// refusals. It is a CLOSED list because the protocol's is closed: a cause outside it that
// arrives with origin WORKER is a peer saying something this contract does not define,
// and settling is the conservative answer.
func preExecution(cause string) bool {
	switch cause {
	case "NO_CAPACITY", "ADMISSION_EPOCH_STALE", "UNKNOWN_PLACEMENT",
		"PLACEMENT_NOT_DISPATCHABLE":
		return true
	}
	return false
}

// mirrorOutputs joins the manifest's entries to the destinations THIS record owner granted
// and PROVES the bytes are there before any of them becomes visible. The runtime names
// what it wrote; the record owner names where it was allowed to write and decides that the
// result is visible. Neither half can do the other's job.
//
// THE PROOF IS THE POINT, and it is why this is not a join any more. Composing a local
// path out of the grant and stamping the manifest's own digest onto it made the record a
// RESTATEMENT of the worker's claim rather than an observation: for a local worker the
// two happen to agree, and for a REMOTE one the bytes are on the pod and the row pointed
// at a path with nothing in it. `--out` then wrote a file the client had never received.
// So: the destination is read, its length and its digest are recomputed here, and an
// output that cannot be shown is never acked as one that can.
func (c *Orchestrator) mirrorOutputs(req records.Request, attempt uint64, doc canonical.Doc,
	holder *worker) ([]records.Output, *exit.Error) {
	manifest := doc.Sub("output_manifest")
	list, _ := manifest["outputs"].([]canonical.Value)
	// WHERE THE RECORD OWNER GRANTED. A serving attempt writes into the store directory —
	// the package's own or the caller's --out — under the file's digest name; a job writes
	// into its publication stage, which is the same fact stated at the other end of the
	// same grant.
	var dir string
	if req.IsJob() {
		// Where the orchestrator GRANTED: an attempt in flight writes into its stage, and
		// `promote` rewrites these paths when the bundle crosses into the publication.
		dir = c.opt.Layout.PublicationStage(req.Org, req.ID, attempt)
	} else {
		var e *exit.Error
		if dir, e = c.outputDirectory(req); e != nil {
			return nil, e
		}
	}
	// AND WHERE THE BYTES ACTUALLY ARE. For a pod attempt the granted destination is a
	// directory on the pod, so "mirror" stops being a figure of speech: each declared
	// output is FETCHED over the pod's media plane and landed at its final name here.
	// Everything after that is the local law unchanged — the bytes are at the path, their
	// length and digest are recomputed here, and an output that cannot be shown is never
	// acked.
	if holder != nil && holder.media != nil {
		if e := c.fetchOutputs(req, attempt, list, dir, holder); e != nil {
			return nil, e
		}
	}
	out := make([]records.Output, 0, len(list))
	declared := map[string]bool{}
	for _, item := range list {
		entry, ok := item.(map[string]canonical.Value)
		if !ok {
			continue
		}
		e := canonical.Doc(entry)
		id := e.Str("output_id")
		path, problem := outputDest(req, dir, e)
		if problem != nil {
			return nil, problem
		}
		declared[id] = true
		if err := verifyBytes(path, e.Str("digest"), e.Int("length")); err != nil {
			return nil, err.WithRemedy(
				"a remote worker writes on its own machine; its outputs are MIRRORED here " +
					"before the outcome is acked, and this attempt has nothing to show")
		}
		out = append(out, records.Output{
			OutputID: id,
			// The OPAQUE media id is minted HERE, before the terminal transaction, so
			// the terminal EVENT can announce it in the same commit that publishes the
			// row. Minting it inside the transaction instead left the announcement with
			// an empty handle — found live by the arm that reads the event's own
			// `outputs`, which is exactly the field a UI would render from.
			MediaID:  records.NewID("med"),
			Path:     path,
			Digest:   e.Str("digest"),
			Length:   e.Int("length"),
			MimeType: e.Str("mime_type"),
		})
	}
	if !req.IsJob() {
		// A serving manifest may name only granted ids, and a SUCCEEDED one names all of them.
		granted := map[string]bool{}
		for _, id := range splitList(req.Outputs) {
			granted[id] = true
			if !declared[id] && outcomeStatus(doc.Int("status")) == "SUCCEEDED" {
				return nil, exit.Named(exit.Validation, "output_set_mismatch",
					"the terminal omits granted output %q", id)
			}
		}
		for id := range declared {
			if !granted[id] {
				return nil, exit.Named(exit.Validation, "output_set_mismatch",
					"the terminal names output %q, which was never granted", id)
			}
		}
	}
	return out, nil
}

// outputDest resolves where one declared output may land: a job's publication fence, or
// a serving result's digest name inside the granted store directory. The name is
// recomputed HERE from the manifest's own digest and media type — the same closed
// projection the worker used — so a terminal never names a path; it names bytes.
func outputDest(req records.Request, dir string, entry canonical.Doc) (string, *exit.Error) {
	id := entry.Str("output_id")
	if req.IsJob() {
		return publicationDest(dir, id)
	}
	if e := FenceOutputID(id); e != nil {
		return "", e
	}
	name, e := resultfiles.Filename(entry.Str("digest"), entry.Str("mime_type"))
	if e != nil {
		return "", e
	}
	return filepath.Join(dir, name), nil
}

// fetchOutputs pulls one remote attempt's declared outputs across the pod's media plane
// and lands them where this orchestrator granted, so the verification below runs against
// bytes THIS host holds.
//
// It is deliberately not a verification step: the media client checks only the pod's own
// declared digest against what arrived (a transport's business), and whether an output may
// become visible is decided one function up, against the TERMINAL's manifest. Two checks
// of two different claims, and neither substitutes for the other.
//
// RE-MIRRORING CONVERGES. A orchestrator killed between fetch and commit re-runs this on
// the worker's replayed terminal: the slot name is derived from the attempt's identity
// rather than remembered, the fetch is idempotent, and the write is atomic.
func (c *Orchestrator) fetchOutputs(req records.Request, attempt uint64,
	list []canonical.Value, dir string, holder *worker) *exit.Error {
	slot := media.Slot(req.ID, attempt)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return exit.Internalf("cannot create the local mirror directory %s: %s", dir, err)
	}
	for _, item := range list {
		entry, ok := item.(map[string]canonical.Value)
		if !ok {
			continue
		}
		id := canonical.Doc(entry).Str("output_id")
		destination, e := outputDest(req, dir, canonical.Doc(entry))
		if e != nil {
			return e
		}
		length := canonical.Doc(entry).Int("length")
		digest := canonical.Doc(entry).Str("digest")
		written, e := holder.media.GetOutputTo(slot, id, destination, digest, length)
		if e != nil {
			return e.WithRemedy("the pod declared this output in its terminal and this host "+
				"cannot fetch it from the pod's media plane, so the terminal stays OWED and "+
				"unacked rather than being accepted with a row that points at nothing (%s)",
				e.Remedy)
		}
		c.logf("mirrored %s#%d/%s: %d B from %s", req.ID, attempt, id, written,
			holder.media.Addr())
	}
	return nil
}

// verifyBytes is the mirror's proof: the declared identity, recomputed over the bytes
// that are actually at the granted destination. It reads the file once — the grant's own
// per-output ceiling already bounds how large one can be — because a length that agrees
// with a digest that does not is the interesting failure, not the cheap one.
func verifyBytes(path, digest string, length int64) *exit.Error {
	file, err := os.Open(path)
	if err != nil {
		return exit.New(exit.Failed,
			"the declared output at %s cannot be read here: %s", path, err)
	}
	defer file.Close()
	hash := sha256.New()
	read, err := io.Copy(hash, io.LimitReader(file, length+1))
	if err != nil {
		return exit.New(exit.Failed,
			"the declared output at %s cannot be read here: %s", path, err)
	}
	if read != length {
		return exit.New(exit.Failed,
			"the output at %s is %d B and its manifest declares %d B", path, read, length)
	}
	spelled, serr := canonical.Spell(hash.Sum(nil))
	if serr != nil {
		return exit.Internalf("cannot spell the mirrored output's digest: %s", serr)
	}
	if spelled != digest {
		return exit.New(exit.Failed,
			"the output at %s hashes to %s and its manifest declares %s",
			path, shortDigest(spelled), shortDigest(digest))
	}
	return nil
}

// brief keeps a worker's own words readable in one line without editing them.
func brief(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}

func outcomeStatus(n int64) string {
	return trimEnum(pb.OutcomeStatus_name[int32(n)], "OUTCOME_STATUS_")
}

func causeCode(n int64) string {
	return trimEnum(pb.CauseCode_name[int32(n)], "CAUSE_CODE_")
}

func causeOrigin(n int64) string {
	return trimEnum(pb.CauseOrigin_name[int32(n)], "CAUSE_ORIGIN_")
}

// outcomeError is the record owner's PROJECTION over (status, cause). Retryability is
// never a wire observation; this is the only place the neutral facts become an outcome.
func outcomeError(status, cause, message string) *exit.Error {
	switch status {
	case "SUCCEEDED":
		return nil
	case "REFUSED":
		if preExecution(cause) {
			// A PRE-EXECUTION refusal is not a judgment about the work: nothing about the
			// request was wrong, the worker simply could not take it. Rendering it as a
			// validation failure would tell a user to change a payload that is fine.
			return exit.Named(exit.Unavailable, "attempt_not_admitted",
				"the worker did not admit this attempt (%s): %s", cause, message).
				WithRemedy("the ordinal is consumed and no execution budget was spent; the " +
					"next ordinal may be dispatched immediately, here or elsewhere")
		}
		return exit.New(exit.Validation, "the attempt was refused (%s): %s", cause, message)
	case "CANCELED":
		if cause == "DEADLINE_EXPIRED" {
			return exit.New(exit.Deadline, "the attempt hit its deadline: %s", message)
		}
		return exit.New(exit.Canceled, "the attempt was canceled (%s): %s", cause, message)
	case "ABANDONED":
		return exit.New(exit.Failed, "the attempt was abandoned (%s): %s", cause, message).
			WithRemedy("an abandoned attempt is requeued as a NEW ordinal, never re-executed")
	default:
		if cause == outcomeRefusedCause {
			return exit.Named(exit.Failed, cause, "%s", message)
		}
		return exit.New(exit.Failed, "the attempt failed (%s): %s", cause, message)
	}
}

func briefOf(e *exit.Error) string {
	if e == nil {
		return "NOT REFUSED — the gate is open, which would be the law failing"
	}
	return e.Message
}

func shortDigest(s string) string {
	s = strings.TrimPrefix(s, "sha256:")
	if len(s) > 16 {
		return s[:16]
	}
	return s
}

func shortNone(raw []byte) string {
	s, err := canonical.Spell(raw)
	if err != nil {
		return ""
	}
	return s
}

// descriptorDefectCode is the one pod refusal that falsifies a PUBLISHED
// descriptor (cr-067): derivation on the pod disagreed with the committed
// document. Every other preparation refusal is local to that worker.
const descriptorDefectCode = "package_prepare_interface_disagrees"

// relayDescriptorDefect files one defect report per desired revision when the
// pod's typed refusal falsifies the published package interface. The orchestrator
// holds no hub client, so the report goes through the same kind of entrypoint
// callback the package-set signer uses; the callback carries the exact signed
// delegation this download ran under — the report's whole chain of authority.
func (c *Orchestrator) relayDescriptorDefect(w *worker, revision uint64, f *pb.Fault) {
	if c.opt.ReportReleaseDefect == nil || !strings.Contains(f.Detail, descriptorDefectCode) {
		return
	}
	if w.spec.Connection == nil || w.spec.Connection.RentalID == "" ||
		len(w.desiredPackages) != 1 || w.defectReportedRevision == revision {
		return
	}
	selected := w.desiredPackages[0]
	signed, held := w.desiredDelegations[selected.Package]
	if !held {
		return
	}
	w.defectReportedRevision = revision
	report := ReleaseDefect{
		Package:    selected.Package,
		Release:    selected.Release,
		RentalID:   w.spec.Connection.RentalID,
		Delegation: append([]byte(nil), signed.delegation...),
		Signature:  append([]byte(nil), signed.signature...),
		Code:       descriptorDefectCode,
		Detail:     brief(f.Detail, 2048),
	}
	c.logf("relaying package-interface defect for %s@%s from rental %s",
		report.Package, report.Release, report.RentalID)
	go c.opt.ReportReleaseDefect(report)
}
