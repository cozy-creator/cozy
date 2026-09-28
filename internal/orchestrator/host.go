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
	"encoding/json"
	"strings"
	"time"

	"google.golang.org/grpc/metadata"

	"github.com/cozy-creator/cozy/internal/exit"
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

// PrepareStagePayload is one ended Host preparation stage as a `request.preparing` payload.
func PrepareStagePayload(label string, stage pb.PrepareStage, began time.Time, last *pb.PrepareEvent) map[string]any {
	if stage == pb.PrepareStage_PREPARE_STAGE_UNSPECIFIED || last == nil {
		return nil
	}
	payload := map[string]any{
		"stage":           strings.ToLower(trimEnum(pb.PrepareStage_name[int32(stage)], "PREPARE_STAGE_")),
		"label":           label,
		"started_unix_ms": began.UnixMilli(),
		"ms":              time.Since(began).Milliseconds(),
	}
	if last.GetTotalBytes() > 0 || last.GetTransferredBytes() > 0 {
		payload["transferred_bytes"], payload["total_bytes"] = last.GetTransferredBytes(), last.GetTotalBytes()
	}
	var origin, cached uint64
	for _, model := range last.GetModelProgress() {
		origin, cached = origin+model.GetOriginBytes(), cached+model.GetCachedBytes()
	}
	if origin > 0 || cached > 0 {
		payload["origin_bytes"], payload["cached_bytes"] = origin, cached
	}
	return payload
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
