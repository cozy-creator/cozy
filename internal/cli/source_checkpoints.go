package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/transfer"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type sourceCheckpointLink struct {
	checkpoint records.ModelSourceCheckpoint
	previous   *pb.Ref
	objects    []hub.Object
}

func sourceCheckpointOperation(requestID, slot, head string) string {
	sum := sha256.Sum256([]byte(requestID + "\x00" + slot + "\x00" + head))
	return "source-progress-" + hex.EncodeToString(sum[:])
}

func checkpointRef(checkpoint records.ModelSourceCheckpoint) (*pb.Ref, []byte, *exit.Error) {
	head, headErr := canonical.Raw(checkpoint.HeadID)
	plan, planErr := canonical.Raw(checkpoint.PlanDigest)
	if headErr != nil || planErr != nil || checkpoint.HeadLength <= 0 {
		return nil, nil, exit.Internalf("stored source checkpoint has invalid identity")
	}
	return &pb.Ref{Digest: head, Length: uint64(checkpoint.HeadLength)}, plan, nil
}

func (o *modelTransferOwner) sourcePublicationClient(requestID string) (*hub.Client, hub.Ref, *records.ModelTransfer, *exit.Error) {
	intent, problem := o.store.ModelTransferOf(requestID)
	if problem != nil || intent == nil {
		return nil, hub.Ref{}, nil, problem
	}
	ref, problem := hub.ParseRef(intent.Destination)
	if problem != nil {
		return nil, ref, nil, problem
	}
	client, problem := ownedPublication(o.cliContext(intent.ModelTransferIntent, true), ref)
	return client, ref, intent, problem
}

// sourceCheckpointLink reads only one bounded immutable Link. The worker's
// TensorFS decoder owns its format; Creator compares the returned typed subjects.
func readSourceCheckpointLink(ctx context.Context, host orchestrator.SourceCheckpointHost,
	requestID, selection string, checkpoint records.ModelSourceCheckpoint,
) (sourceCheckpointLink, *exit.Error) {
	head, plan, problem := checkpointRef(checkpoint)
	if problem != nil {
		return sourceCheckpointLink{}, problem
	}
	selected, err := canonical.Raw(selection)
	if err != nil {
		return sourceCheckpointLink{}, exit.Internalf("stored source selection is malformed")
	}
	link := sourceCheckpointLink{checkpoint: checkpoint}
	objects := map[string]int64{checkpoint.HeadID: checkpoint.HeadLength}
	add := func(ref *pb.Ref) *exit.Error {
		if ref == nil || len(ref.Digest) != 32 || ref.Length == 0 || ref.Length > uint64(^uint64(0)>>1) {
			return exit.Named(exit.Structural, "model_transfer.source_checkpoint_ref_invalid", "checkpoint page has an invalid ObjectRef")
		}
		id := "sha256:" + hex.EncodeToString(ref.Digest)
		if prior, exists := objects[id]; exists && prior != int64(ref.Length) {
			return exit.Named(exit.Structural, "model_transfer.source_checkpoint_ref_changed", "checkpoint page repeats an object with another length")
		}
		objects[id] = int64(ref.Length)
		return nil
	}
	for offset := uint32(0); ; {
		page, problem := host.Page(ctx, &pb.SourceCheckpointPageRequest{OperationId: requestID,
			SourceSelectionDigest: selected, Slot: checkpoint.Slot, PlanDigest: plan, Head: head,
			Offset: offset, Limit: pb.MaxSourceCheckpointObjects})
		if problem != nil {
			return link, problem
		}
		if page.Index > uint64(^uint64(0)>>1) || page.Bytes > uint64(^uint64(0)>>1) {
			return link, exit.Named(exit.Structural, "model_transfer.source_checkpoint_counter_invalid", "checkpoint page exceeds owner integer bounds")
		}
		if offset == 0 {
			link.checkpoint.Index, link.checkpoint.Bytes = int64(page.Index), int64(page.Bytes)
			link.previous = page.Previous
		} else if int64(page.Index) != link.checkpoint.Index || int64(page.Bytes) != link.checkpoint.Bytes {
			return link, exit.Named(exit.Structural, "model_transfer.source_checkpoint_page_changed", "immutable checkpoint page changed its counters")
		}
		if page.Progress != nil {
			if problem := add(page.Progress); problem != nil {
				return link, problem
			}
		}
		for _, object := range page.Objects {
			if object == nil || object.Manifest {
				return link, exit.Named(exit.Structural, "model_transfer.source_checkpoint_namespace_invalid", "partial source checkpoints contain blobs only")
			}
			if problem := add(object.Ref); problem != nil {
				return link, problem
			}
		}
		if len(objects) > pb.MaxSourceCheckpointObjects+2 {
			return link, exit.Named(exit.Structural, "model_transfer.source_checkpoint_too_large", "source checkpoint Link exceeds its fixed object bound")
		}
		if !page.HasMore {
			break
		}
		if page.NextOffset <= offset || page.NextOffset > pb.MaxSourceCheckpointObjects {
			return link, exit.Named(exit.Structural, "model_transfer.source_checkpoint_cursor_invalid", "source checkpoint page did not advance within its bound")
		}
		offset = page.NextOffset
	}
	for id, length := range objects {
		link.objects = append(link.objects, hub.Object{ID: id, Length: length})
	}
	sort.Slice(link.objects, func(i, j int) bool { return link.objects[i].ID < link.objects[j].ID })
	return link, nil
}

func previousSourceCheckpoint(link sourceCheckpointLink) (records.ModelSourceCheckpoint, *exit.Error) {
	previous := link.previous
	if previous == nil || len(previous.Digest) != 32 || previous.Length == 0 ||
		previous.Length > uint64(^uint64(0)>>1) || link.checkpoint.Index == 0 {
		return records.ModelSourceCheckpoint{}, exit.Named(exit.Structural, "model_transfer.source_checkpoint_chain_invalid", "source checkpoint predecessor is invalid")
	}
	return records.ModelSourceCheckpoint{Slot: link.checkpoint.Slot,
		HeadID: "sha256:" + hex.EncodeToString(previous.Digest), HeadLength: int64(previous.Length),
		PlanDigest: link.checkpoint.PlanDigest, Index: link.checkpoint.Index - 1, Bytes: link.checkpoint.Bytes}, nil
}

func (o *modelTransferOwner) checkpointTransfer(requestID, selection, direction string, checkpoint records.ModelSourceCheckpoint, object hub.Object,
) (*pb.SourceCheckpointTransferRequest, *exit.Error) {
	head, plan, problem := checkpointRef(checkpoint)
	if problem != nil {
		return nil, problem
	}
	selected, err := canonical.Raw(selection)
	if err != nil {
		return nil, exit.Internalf("stored source selection is malformed")
	}
	objectDigest, err := canonical.Raw(object.ID)
	if err != nil || object.Length <= 0 {
		return nil, exit.Internalf("checkpoint transfer has invalid object identity")
	}
	revision, problem := o.store.NextModelSourceGrantRevision(requestID, checkpoint.Slot)
	if problem != nil {
		return nil, problem
	}
	sum := sha256.Sum256([]byte(requestID + "\x00" + checkpoint.Slot + "\x00" + checkpoint.HeadID + "\x00" + object.ID + "\x00" + direction))
	return &pb.SourceCheckpointTransferRequest{OperationId: requestID, SourceSelectionDigest: selected,
		Slot: checkpoint.Slot, PlanDigest: plan, Head: head,
		Object:     &pb.SourceCheckpointObject{Ref: &pb.Ref{Digest: objectDigest, Length: uint64(object.Length)}},
		TransferId: "sha256:" + hex.EncodeToString(sum[:]), GrantRevision: revision}, nil
}

func (o *modelTransferOwner) uploadSourceCheckpointLink(ctx context.Context, client *hub.Client, ref hub.Ref,
	requestID string, intent records.ModelTransferIntent, host orchestrator.SourceCheckpointHost, link sourceCheckpointLink,
) *exit.Error {
	operation := sourceCheckpointOperation(requestID, link.checkpoint.Slot, link.checkpoint.HeadID)
	body, _ := json.Marshal(link.objects)
	if problem := o.store.RecordSourcePublication(requestID, operation, body); problem != nil {
		return problem
	}
	opened, problem := client.OpenPublication(ctx, ref, operation, link.objects, "retain source preparation progress")
	if problem != nil {
		return problem
	}
	if _, problem := transfer.ValidateOpenedPublication(opened, operation, link.objects); problem != nil {
		return problem
	}
	if problem := o.store.OpenedSourcePublication(requestID, operation); problem != nil {
		return problem
	}
	window := orchestrator.NewWeightsGrantWindow(func(ctx context.Context, ids []string) (orchestrator.WeightsGrantMint, *exit.Error) {
		return mintWeightsGrants(ctx, client, ref, operation, intent, ids)
	})
	acceptedBefore := make(map[string]bool, len(opened.Publication.Objects))
	for _, object := range opened.Publication.Objects {
		acceptedBefore[object.ObjectID] = object.State == "accepted"
	}
	ids := make([]string, len(link.objects))
	for i, object := range link.objects {
		ids[i] = object.ID
	}
	for i, object := range link.objects {
		if acceptedBefore[object.ID] {
			continue
		}
		decision, problem := window.Spendable(ctx, object.ID, ids[i:min(i+128, len(ids))], time.Now())
		if problem != nil {
			return problem
		}
		if decision.Length != object.Length {
			return exit.Named(exit.Structural, "model_transfer.source_checkpoint_grant_changed", "checkpoint grant changed the selected object length")
		}
		if decision.Held {
			continue
		}
		request, problem := o.checkpointTransfer(requestID, intent.SourceSelection, "upload", link.checkpoint, object)
		if problem != nil {
			return problem
		}
		grant := &pb.WeightsUploadGrant{ObjectId: object.ID, Length: uint64(object.Length), Url: decision.URL,
			ExpiresAtUnix: decision.ExpiresAtUnix}
		for name, value := range decision.Headers {
			grant.RequiredHeaders = append(grant.RequiredHeaders, &pb.WeightsUploadHeader{Name: name, Value: value})
		}
		sort.Slice(grant.RequiredHeaders, func(i, j int) bool { return grant.RequiredHeaders[i].Name < grant.RequiredHeaders[j].Name })
		request.Decision = &pb.SourceCheckpointTransferRequest_UploadGrant{UploadGrant: grant}
		if _, problem := host.Transfer(ctx, request); problem != nil {
			return problem
		}
	}
	for offset := 0; offset < len(link.objects); offset += 128 {
		batch := link.objects[offset:min(offset+128, len(link.objects))]
		accepted, problem := client.VerifyPublicationObjects(ctx, ref, operation, ids[offset:offset+len(batch)])
		if problem != nil {
			return problem
		}
		if accepted.Operation != operation || accepted.State != "open" || len(accepted.Objects) != len(batch) {
			return exit.Named(exit.Structural, "model_transfer.source_checkpoint_custody_invalid", "Tensorhub did not verify the exact checkpoint object batch")
		}
		for i, object := range accepted.Objects {
			if object.ObjectID != batch[i].ID || object.Length != batch[i].Length || object.State != "accepted" {
				return exit.Named(exit.Structural, "model_transfer.source_checkpoint_custody_invalid", "Tensorhub checkpoint custody differs from the declared objects")
			}
		}
	}
	return nil
}

// SyncSourceCheckpoints walks backwards one bounded Link at a time. Only after
// every new predecessor and object is accepted does the owner advance its head.
func (o *modelTransferOwner) SyncSourceCheckpoints(parent context.Context, requestID string,
	host orchestrator.SourceCheckpointHost,
) *exit.Error {
	ctx, cancel := o.requestContext(parent, requestID)
	defer cancel()
	client, ref, transfer, problem := o.sourcePublicationClient(requestID)
	if problem != nil || transfer == nil {
		return problem
	}
	progress, problem := o.store.ModelSourceProgress(requestID)
	if problem != nil {
		return problem
	}
	for _, slot := range progress {
		if slot.WorkerBootID != host.BootID || (slot.Acknowledged != nil && *slot.Acknowledged == slot.Observed) {
			continue
		}
		stop := ""
		if slot.Acknowledged != nil {
			stop = slot.Acknowledged.HeadID
		}
		current := slot.Observed
		first := true
		for current.HeadID != stop {
			link, problem := readSourceCheckpointLink(ctx, host, requestID, transfer.SourceSelection, current)
			if problem != nil {
				return problem
			}
			if link.checkpoint.Index != current.Index || link.checkpoint.Bytes > current.Bytes ||
				(first && link.checkpoint.Bytes != current.Bytes) ||
				(slot.Acknowledged != nil && link.checkpoint.Index <= slot.Acknowledged.Index) {
				return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_chain_changed", "checkpoint chain does not extend acknowledged progress")
			}
			if problem := o.uploadSourceCheckpointLink(ctx, client, ref, requestID, transfer.ModelTransferIntent, host, link); problem != nil {
				return problem
			}
			if link.previous == nil {
				if stop != "" || link.checkpoint.Index != 0 {
					return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_chain_changed", "checkpoint chain ended before acknowledged progress")
				}
				break
			}
			current, problem = previousSourceCheckpoint(link)
			if problem != nil {
				return problem
			}
			first = false
		}
		if slot.Acknowledged != nil && (current.HeadLength != slot.Acknowledged.HeadLength || current.Index != slot.Acknowledged.Index || current.PlanDigest != slot.Acknowledged.PlanDigest || current.Bytes < slot.Acknowledged.Bytes) {
			return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_chain_changed", "checkpoint predecessor changed the acknowledged ObjectRef")
		}
		if problem := o.store.AcknowledgeModelSourceCheckpoint(requestID, transfer.SourceSelection, host.BootID, stop, slot.Observed); problem != nil {
			return problem
		}
	}
	return nil
}

func (o *modelTransferOwner) downloadSourceCheckpointObjects(ctx context.Context, client *hub.Client, ref hub.Ref,
	requestID, selection string, checkpoint records.ModelSourceCheckpoint, host orchestrator.SourceCheckpointHost, objects []hub.Object,
) *exit.Error {
	operation := sourceCheckpointOperation(requestID, checkpoint.Slot, checkpoint.HeadID)
	ids := make([]string, len(objects))
	for i, object := range objects {
		ids[i] = object.ID
	}
	window := orchestrator.NewWeightsGrantWindow(func(ctx context.Context, selected []string) (orchestrator.WeightsGrantMint, *exit.Error) {
		var mint orchestrator.WeightsGrantMint
		reads, problem := client.ReadPublicationObjects(ctx, ref, operation, selected)
		if problem != nil {
			return mint, problem
		}
		if len(reads.Reads) != len(selected) || reads.ServerTimeUnix <= 0 {
			return mint, exit.Named(exit.Structural, "model_transfer.source_checkpoint_reads_invalid", "Tensorhub did not authorize the exact checkpoint read batch")
		}
		wanted := make(map[string]bool, len(selected))
		for _, id := range selected {
			wanted[id] = true
		}
		mint.ServerTimeUnix = reads.ServerTimeUnix
		for _, read := range reads.Reads {
			expires, err := time.Parse(time.RFC3339Nano, read.Expires)
			if !wanted[read.ObjectID] || read.Length <= 0 || read.URL == "" || err != nil || expires.Unix() <= reads.ServerTimeUnix {
				return mint, exit.Named(exit.Structural, "model_transfer.source_checkpoint_reads_invalid", "Tensorhub changed a checkpoint read or returned expired access")
			}
			delete(wanted, read.ObjectID)
			mint.Decisions = append(mint.Decisions, orchestrator.WeightsTransferDecision{ObjectID: read.ObjectID,
				Length: read.Length, URL: read.URL, ExpiresAtUnix: uint64(expires.Unix())})
		}
		return mint, nil
	})
	for i, object := range objects {
		read, problem := window.Spendable(ctx, object.ID, ids[i:], time.Now())
		if problem != nil {
			return problem
		}
		if read.Length != object.Length {
			return exit.Named(exit.Structural, "model_transfer.source_checkpoint_reads_invalid", "Tensorhub changed a checkpoint read length")
		}
		request, problem := o.checkpointTransfer(requestID, selection, "download", checkpoint, object)
		if problem != nil {
			return problem
		}
		request.Decision = &pb.SourceCheckpointTransferRequest_DownloadUrl{DownloadUrl: read.URL}
		if _, problem := host.Transfer(ctx, request); problem != nil {
			return problem
		}
	}
	return nil
}

func (o *modelTransferOwner) RestoreSourceCheckpoints(parent context.Context, requestID string,
	host orchestrator.SourceCheckpointHost,
) *exit.Error {
	ctx, cancel := o.requestContext(parent, requestID)
	defer cancel()
	progress, problem := o.store.ModelSourceProgress(requestID)
	if problem != nil {
		return problem
	}
	if len(progress) == 0 {
		return nil
	}
	client, ref, transfer, problem := o.sourcePublicationClient(requestID)
	if problem != nil || transfer == nil {
		return problem
	}
	for _, slot := range progress {
		if slot.Acknowledged == nil {
			continue
		}
		current := *slot.Acknowledged
		first := true
		for {
			if problem := o.downloadSourceCheckpointObjects(ctx, client, ref, requestID, transfer.SourceSelection,
				current, host, []hub.Object{{ID: current.HeadID, Length: current.HeadLength}}); problem != nil {
				return problem
			}
			link, problem := readSourceCheckpointLink(ctx, host, requestID, transfer.SourceSelection, current)
			if problem != nil {
				return problem
			}
			if link.checkpoint.Index != current.Index || link.checkpoint.Bytes > current.Bytes ||
				(first && link.checkpoint.Bytes != current.Bytes) {
				return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_restore_changed", "restored checkpoint chain changed its recorded progress")
			}
			if problem := o.downloadSourceCheckpointObjects(ctx, client, ref, requestID, transfer.SourceSelection,
				link.checkpoint, host, link.objects); problem != nil {
				return problem
			}
			if link.previous == nil {
				if link.checkpoint.Index != 0 {
					return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_restore_changed", "restored checkpoint chain has no root")
				}
				break
			}
			current, problem = previousSourceCheckpoint(link)
			if problem != nil {
				return problem
			}
			first = false
		}
	}
	return nil
}

func (o *modelTransferOwner) ReleaseSourceCheckpoints(ctx context.Context, requestID string) *exit.Error {
	pending, problem := o.store.SourcePublications(requestID)
	if problem != nil || len(pending) == 0 {
		return problem
	}
	client, ref, intent, problem := o.sourcePublicationClient(requestID)
	if problem != nil || intent == nil {
		return problem
	}
	for {
		operations, problem := o.store.SourcePublications(requestID)
		if problem != nil || len(operations) == 0 {
			return problem
		}
		for _, publication := range operations {
			if !publication.Opened {
				var objects []hub.Object
				if err := json.Unmarshal(publication.Objects, &objects); err != nil {
					return exit.Internalf("stored source publication intent is malformed")
				}
				if _, problem := client.OpenPublication(ctx, ref, publication.Operation, objects, "reconcile source publication before release"); problem != nil {
					return problem
				}
			}
			if problem := client.AbandonPublication(ctx, ref, publication.Operation); problem != nil && problem.ErrName() != "publication.not_found" {
				return problem
			}
			if problem := o.store.ReleaseSourcePublication(requestID, publication.Operation); problem != nil {
				return problem
			}
		}
	}
}
