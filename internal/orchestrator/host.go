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
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// ObservePrepareEvent folds one PrepareEvent into the phase lane. The subject is the
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
func (c *Orchestrator) ObservePrepareEvent(instanceID, machine, label string, event *pb.PrepareEvent) {
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
