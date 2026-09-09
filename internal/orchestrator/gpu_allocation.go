package orchestrator

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// Prepared bytes describe a construction; a placement instance assigns that
// construction to a disjoint physical group. Only the routing ID changes.
type gpuInstance struct {
	row      *pb.Placement
	ordinals []uint32
	claim    string // queued request holding this group while it warms
}

func requestGPUWidth(req records.Request) int {
	if !req.NeedsAccelerator {
		return 0
	}
	return max(1, req.RequestedGPUs)
}

func modelPlacement(row *pb.Placement) bool {
	for _, entrypoint := range row.Entrypoints {
		if len(entrypoint.Slots) > 0 {
			return true
		}
	}
	return false
}

func gpuTemplateID(row *pb.Placement) string {
	copy := proto.Clone(row).(*pb.Placement)
	copy.PlacementId = "template"
	_, digest, _ := canonical.Identity(&pb.PlacementSet{Placements: []*pb.Placement{copy}})
	return fmt.Sprintf("%x", digest)
}

// cacheGPUTemplates runs after native preparation, before any GPU is reserved.
// Jobs/CPU-only preparations continue through their existing execution mode.
func (w *worker) cacheGPUTemplates(data []byte) (bool, *exit.Error) {
	var set pb.PlacementSet
	if err := canonical.Unmarshal(data, &set); err != nil {
		return false, exit.Internalf("cannot read prepared GPU templates: %s", err)
	}
	if len(set.Placements) == 0 {
		return false, nil
	}
	for _, row := range set.Placements {
		if !modelPlacement(row) {
			return false, nil
		}
	}
	if w.gpuTemplates == nil {
		w.gpuTemplates = map[string]*pb.Placement{}
		w.gpuInstances = map[string]*gpuInstance{}
	}
	for _, row := range set.Placements {
		w.gpuTemplates[gpuTemplateID(row)] = proto.Clone(row).(*pb.Placement)
	}
	return true, nil
}

// templateMatches binds a cached construction to the exact callable and model selection.
func templateMatches(row *pb.Placement, logical LogicalPackage) bool {
	pkg, release := row.GetPackage().GetPackage(), row.GetPackage().GetRelease()
	if development := row.GetDevelopment(); development != nil {
		pkg, release = development.Package, development.Release
		if logical.LocalRevision != "" && spellOf(development.LocalRevisionDigest) != logical.LocalRevision {
			return false
		}
	} else if logical.LocalRevision != "" {
		return false
	}
	if pkg != logical.Package || release != logical.Release {
		return false
	}
	models := map[string]*pb.Model{}
	for _, model := range row.Models {
		if model == nil || model.Id == "" || model.Manifest == nil {
			return false
		}
		// Placement model IDs are local names. Duplicate IDs would make the
		// lookup below ambiguous and must never turn into a warm-cache hit.
		if _, exists := models[model.Id]; exists {
			return false
		}
		models[model.Id] = model
	}
	for _, entrypoint := range row.Entrypoints {
		if entrypoint.Name != logical.Function || len(entrypoint.Slots) != len(logical.Models) {
			continue
		}
		if logical.PlanID != "" && spellOf(entrypoint.EntrypointBindingDigest) != logical.PlanID {
			continue
		}
		used := make(map[int]bool, len(entrypoint.Slots))
		for _, slot := range entrypoint.Slots {
			if slot == nil || slot.ReferenceModelId == "" {
				return false
			}
			bound, ok := models[slot.ReferenceModelId]
			if !ok {
				return false
			}
			matched := false
			for i, model := range logical.Models {
				if used[i] {
					continue
				}
				// Match every immutable model-selection fact. A manifest digest alone
				// is insufficient: two lanes/releases can legitimately share a tree,
				// and accepting one would route a request to the wrong execution plan.
				expectedSlot := model.BindingPath
				if expectedSlot == "" {
					expectedSlot = logical.Function + ".models." + slot.Slot
				}
				if model.Slot == expectedSlot &&
					model.Model == bound.Repo &&
					model.Release == bound.Version && model.Lane == bound.Lane &&
					model.Manifest == spellOf(bound.Manifest.Digest) &&
					model.ManifestLength > 0 && uint64(model.ManifestLength) == bound.Manifest.Length {
					matched = true
					used[i] = true
					break
				}
			}
			if !matched {
				return false
			}
		}
		return true
	}
	return false
}

func (w *worker) gpuTemplate(logical LogicalPackage) *pb.Placement {
	for _, row := range w.gpuTemplates {
		if templateMatches(row, logical) {
			return row
		}
	}
	return nil
}

func requestLogical(req records.Request) LogicalPackage {
	return LogicalPackage{Package: req.Package, Release: req.Release, Function: req.Entrypoint,
		PlanID: req.PlanID, Models: req.Models, Outputs: splitList(req.Outputs), LocalRevision: req.LocalPackageDigest}
}

func gpuProjection(row *pb.Placement, logical LogicalPackage, rentalID string) (DesiredPlacement, *exit.Error) {
	data, digest, err := canonical.Identity(&pb.PlacementSet{Placements: []*pb.Placement{row}})
	if err != nil {
		return DesiredPlacement{}, exit.Internalf("cannot encode GPU placement: %s", err)
	}
	placement, problem := PlacementFromExact(logical.Package, "", spellOf(digest), data,
		map[string][]string{logical.Function: logical.Outputs})
	if problem != nil {
		return placement, problem
	}
	for _, entrypoint := range placement.Entrypoints {
		if entrypoint.Name == logical.Function {
			placement.Entrypoints = []Entrypoint{entrypoint}
			placement.Package = pinnedPackage(logical.Package, rentalID)
			placement.Models = append([]ModelRef(nil), logical.Models...)
			return placement, nil
		}
	}
	return placement, exit.Internalf("prepared GPU placement omits callable %s", logical.Function)
}

func (w *worker) gpuBusy(id string) bool {
	lane := w.lanes.of(id)
	return lane != nil && (lane.held > 0 || lane.seats.reserved > 0)
}

// ensureRequestGPUs reserves a whole group while holding the same owner mutex
// that reserves attempt seats. Busy instances are immutable until terminal ACK.
func (c *Orchestrator) ensureRequestGPUs(req records.Request) *exit.Error {
	width := requestGPUWidth(req)
	if req.IsJob() || width == 0 || !req.Rental {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	w := c.workers[rentalInstanceID(req.Worker)]
	if req.Worker == "" {
		if candidate := c.route(req).pick(); candidate != nil && candidate.local() {
			return nil
		}
		var candidates []*worker
		for _, candidate := range c.workers {
			if candidate.spec.Connection != nil && !candidate.exited && !candidate.stopping &&
				len(candidate.spec.Devices) >= width && candidate.gpuTemplate(requestLogical(req)) != nil {
				candidates = append(candidates, candidate)
			}
		}
		sort.Slice(candidates, func(i, j int) bool { return candidates[i].instanceID < candidates[j].instanceID })
		for _, candidate := range candidates {
			if w == nil || len(candidate.freeGPUs(c)) >= width {
				w = candidate
			}
			if len(candidate.freeGPUs(c)) >= width {
				break
			}
		}
		if w != nil {
			req.Worker = w.spec.Connection.RentalID
			pinned, problem := c.opt.Store.PinRental(req.ID, req.Worker, nil)
			if problem != nil {
				return problem
			}
			if !pinned {
				return exit.Unavailablef("request settled while selecting its GPU worker")
			}
		}
	}
	if w == nil || w.exited || w.spec.IsJob() || !w.snapshotAcknowledged {
		return exit.Unavailablef("waiting for the selected GPU worker")
	}
	if width > len(w.spec.Devices) {
		return exit.Named(exit.Capacity, "rental.gpu_count_insufficient", "request needs %d GPUs; rental has %d", width, len(w.spec.Devices))
	}
	knownHeld := 0
	for _, lane := range w.lanes.lanes {
		if (lane.held > 0 || lane.seats.reserved > 0) && (len(lane.ordinals) == 0 || lane.outsideEnvelope) {
			return exit.Unavailablef("waiting for valid physical GPU occupancy of recovered work")
		}
		knownHeld += lane.held
	}
	if knownHeld < w.held {
		return exit.Unavailablef("waiting for physical lane occupancy of recovered work")
	}
	logical := requestLogical(req)
	template := w.gpuTemplate(logical)
	if template == nil {
		return exit.Unavailablef("waiting for the prepared GPU construction")
	}
	if req.PlanID == "" {
		for _, entrypoint := range template.Entrypoints {
			if entrypoint.Name == req.Entrypoint {
				req.PlanID = spellOf(entrypoint.EntrypointBindingDigest)
				if problem := c.opt.Store.BindRequestPlan(req.ID, req.PlanID); problem != nil {
					return problem
				}
				break
			}
		}
	}
	for id, instance := range w.gpuInstances {
		if len(instance.ordinals) == 0 && !w.gpuBusy(id) {
			delete(w.gpuInstances, id)
			continue
		}
		if instance.claim != "" && !c.queued(instance.claim) {
			instance.claim = ""
		}
		if instance.claim == req.ID {
			if observed := w.observedRemote[instance.row.PlacementId]; observed.materialization == pb.MaterializationState_MATERIALIZATION_STATE_FAILED {
				return exit.Named(exit.Failed, "rental.package_materialization_failed", "requested GPU construction failed on the worker")
			}
			return nil
		}
	}
	// Once an earlier gang is waiting, keep free cards for it. Existing running
	// work finishes normally; requests on other workers remain eligible.
	for _, id := range c.pending {
		if id == req.ID {
			break
		}
		prior, problem := c.opt.Store.RequestRow(id)
		if problem != nil {
			return problem
		}
		if prior != nil && prior.Worker == req.Worker && !prior.IsJob() &&
			requestGPUWidth(*prior) > 0 && requestGPUWidth(*prior) <= len(w.spec.Devices) {
			reserved := false
			for _, instance := range w.gpuInstances {
				reserved = reserved || instance.claim == prior.ID
			}
			if !reserved && w.gpuTemplate(requestLogical(*prior)) != nil {
				return exit.Unavailablef("free GPUs are reserved for earlier request %s", prior.ID)
			}
		}
	}
	for _, id := range sortedGPUInstances(w) {
		instance := w.gpuInstances[id]
		if len(instance.ordinals) == width && templateMatches(instance.row, logical) &&
			instance.claim == "" && !w.gpuBusy(id) {
			instance.claim = req.ID
			return nil
		}
	}
	ordinals := w.freeGPUs(c)
	if len(ordinals) > width {
		ordinals = ordinals[:width]
	}
	if len(ordinals) != width {
		return exit.Unavailablef("waiting for %d GPUs to be free together", width)
	}
	for id, instance := range w.gpuInstances {
		for _, held := range instance.ordinals {
			for _, selected := range ordinals {
				if held == selected {
					delete(w.gpuInstances, id)
				}
			}
		}
	}
	parts := make([]string, len(ordinals))
	for i, ordinal := range ordinals {
		parts[i] = strconv.FormatUint(uint64(ordinal), 10)
	}
	row := proto.Clone(template).(*pb.Placement)
	row.PlacementId = "gpu-" + gpuTemplateID(template) + "-" + strings.Join(parts, "-")
	w.gpuInstances[row.PlacementId] = &gpuInstance{row: row, ordinals: ordinals, claim: req.ID}
	problem := c.sendGPUInstancesLocked(w)
	if problem != nil {
		delete(w.gpuInstances, row.PlacementId)
	}
	return problem
}

func sortedGPUInstances(w *worker) []string {
	ids := make([]string, 0, len(w.gpuInstances))
	for id := range w.gpuInstances {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

func (c *Orchestrator) sendGPUInstancesLocked(w *worker) *exit.Error {
	s := c.sessions[w.bootID]
	if s == nil {
		return exit.Unavailablef("GPU worker has no claimed stream")
	}
	set := &pb.PlacementSet{}
	var pins []*pb.PlacementDevicePin
	for _, id := range sortedGPUInstances(w) {
		instance := w.gpuInstances[id]
		set.Placements = append(set.Placements, instance.row)
		pins = append(pins, &pb.PlacementDevicePin{PlacementId: id, DeviceOrdinals: instance.ordinals})
	}
	data, digest, err := canonical.Identity(set)
	if err != nil {
		return exit.Internalf("cannot encode GPU allocation: %s", err)
	}
	c.revision++
	w.revision, w.setBytes, w.setDigest = c.revision, data, digest
	desired := &pb.DesiredWorkerState{RecordOwnerEpoch: recordOwnerEpoch, ControlStreamEpoch: s.epoch,
		WorkerBootId: s.bootID, Revision: w.revision, WireMinor: pb.WireMinor, Posture: pb.Posture_POSTURE_ACCEPTING,
		Mode: &pb.DesiredWorkerState_PlacementSet{PlacementSet: &pb.DesiredPlacementSet{
			PlacementSetDigest: digest, PlacementSetCanonicalBytes: data, DevicePins: pins, OrchestrationParent: w.orchestrationParent}}}
	if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_DesiredState{DesiredState: desired}}) {
		return exit.Unavailablef("GPU allocation control stream closed")
	}
	return nil
}

// The digest-fenced recovery snapshot supplies both exact constructions and
// observed physical lanes. Recover occupancy before allocating any queued work.
func (w *worker) restoreGPUInstances(data []byte) *exit.Error {
	cached, problem := w.cacheGPUTemplates(data)
	if problem != nil || !cached {
		return problem
	}
	var set pb.PlacementSet
	if err := canonical.Unmarshal(data, &set); err != nil {
		return exit.Internalf("cannot restore accepted GPU placements: %s", err)
	}
	instances := map[string]*gpuInstance{}
	for _, row := range set.Placements {
		lane := w.lanes.of(row.PlacementId)
		instance := &gpuInstance{row: row}
		if lane != nil && len(lane.ordinals) > 0 && !lane.outsideEnvelope {
			instance.ordinals = append([]uint32(nil), lane.ordinals...)
		}
		instances[row.PlacementId] = instance
	}
	w.gpuInstances = instances
	w.setBytes, w.setDigest = append([]byte(nil), data...), canonical.Digest(data)
	return nil
}

func (c *Orchestrator) routeGPUInstances(w *worker, req records.Request, out *routing) {
	if w.exited || w.stopping || !w.snapshotAcknowledged || !w.supportsCurrentProtocol() || c.sessions[w.bootID] == nil ||
		!req.Rental || (req.Worker != "" && req.Worker != w.spec.Connection.RentalID) ||
		(req.RequestedRental != "" && req.RequestedRental != w.spec.Connection.RentalID) {
		return
	}
	out.lanes = append(out.lanes, w.laneKeys()...)
	for id, instance := range w.gpuInstances {
		if len(instance.ordinals) != requestGPUWidth(req) || !templateMatches(instance.row, requestLogical(req)) ||
			(instance.claim != "" && instance.claim != req.ID) {
			continue
		}
		observed := w.observedRemote[id]
		if observed.serving != pb.ServingState_SERVING_STATE_DISPATCHABLE || !observed.dispatchablePlanIDs[req.PlanID] {
			continue
		}
		laneID, room, held, why := w.roomFor(id, req.PlanID)
		if room <= 0 {
			out.parked = append(out.parked, why)
			continue
		}
		// ensureRequestGPUs already reserved disjoint physical groups in queue
		// order. A warming group cannot claim another group's independent lane.
		candidate := candidate{worker: w, laneID: laneID, placementID: id, held: held}
		candidate.cost, candidate.resident = w.costOn(laneID, id)
		candidate.score = held + candidate.cost
		out.candidates = append(out.candidates, candidate)
	}
}

func (w *worker) freeGPUs(c *Orchestrator) []uint32 {
	busy := map[uint32]bool{}
	for _, instance := range w.gpuInstances {
		if (instance.claim != "" && c.queued(instance.claim)) || w.gpuBusy(instance.row.PlacementId) {
			for _, ordinal := range instance.ordinals {
				busy[ordinal] = true
			}
		}
	}
	// Recovered attempts remain exclusive even before template reconstruction.
	for _, lane := range w.lanes.lanes {
		if lane.held > 0 || lane.seats.reserved > 0 {
			for _, ordinal := range lane.ordinals {
				busy[ordinal] = true
			}
		}
	}
	var ordinals []uint32
	for ordinal := range w.spec.Devices {
		if !busy[uint32(ordinal)] {
			ordinals = append(ordinals, uint32(ordinal))
		}
	}
	return ordinals
}
