package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"sync"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/transfer"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type checkpointLink struct {
	checkpoint records.ModelCheckpoint
	previous   *pb.Ref
	objects    []hub.Object
}

func checkpointOperation(kind, requestID, slot, head string) string {
	sum := sha256.Sum256([]byte(requestID + "\x00" + slot + "\x00" + head))
	return kind + "-progress-" + hex.EncodeToString(sum[:])
}

func checkpointRef(checkpoint records.ModelCheckpoint) (*pb.Ref, []byte, *exit.Error) {
	head, headErr := canonical.Raw(checkpoint.HeadID)
	plan, planErr := canonical.Raw(checkpoint.PlanDigest)
	if headErr != nil || planErr != nil || checkpoint.HeadLength <= 0 {
		return nil, nil, exit.Internalf("stored source checkpoint has invalid identity")
	}
	return &pb.Ref{Digest: head, Length: uint64(checkpoint.HeadLength)}, plan, nil
}

func (o *modelTransferOwner) checkpointPublicationClient(requestID string) (*hub.Client, hub.Ref, *records.ModelTransfer, *exit.Error) {
	intent, problem := o.store.ModelTransferOf(requestID)
	if problem != nil || intent == nil {
		return nil, hub.Ref{}, nil, problem
	}
	ref, problem := hub.ParseRef(intent.Destination)
	if problem != nil {
		return nil, ref, nil, problem
	}
	client, problem := ownedPublication(o.cliContext(o.hubOf(requestID), intent.ModelTransferIntent, true), ref)
	return client, ref, intent, problem
}

// checkpointLink reads only one bounded immutable Link. The worker's
// TensorFS decoder owns its format; Creator compares the returned typed subjects.
func readCheckpointLink(ctx context.Context, host orchestrator.CheckpointHost,
	subject *pb.CheckpointSubject, checkpoint records.ModelCheckpoint,
) (checkpointLink, *exit.Error) {
	head, plan, problem := checkpointRef(checkpoint)
	if problem != nil {
		return checkpointLink{}, problem
	}
	link := checkpointLink{checkpoint: checkpoint}
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
		page, problem := host.Page(ctx, &pb.CheckpointPageRequest{Subject: subject, PlanDigest: plan, Head: head,
			Offset: offset, Limit: pb.MaxCheckpointObjects})
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
		if len(objects) > pb.MaxCheckpointObjects+2 {
			return link, exit.Named(exit.Structural, "model_transfer.source_checkpoint_too_large", "source checkpoint Link exceeds its fixed object bound")
		}
		if !page.HasMore {
			break
		}
		if page.NextOffset <= offset || page.NextOffset > pb.MaxCheckpointObjects {
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

func previousCheckpoint(link checkpointLink) (records.ModelCheckpoint, *exit.Error) {
	previous := link.previous
	if previous == nil || len(previous.Digest) != 32 || previous.Length == 0 ||
		previous.Length > uint64(^uint64(0)>>1) || link.checkpoint.Index == 0 {
		return records.ModelCheckpoint{}, exit.Named(exit.Structural, "model_transfer.source_checkpoint_chain_invalid", "source checkpoint predecessor is invalid")
	}
	return records.ModelCheckpoint{Slot: link.checkpoint.Slot,
		HeadID: "sha256:" + hex.EncodeToString(previous.Digest), HeadLength: int64(previous.Length),
		PlanDigest: link.checkpoint.PlanDigest, Index: link.checkpoint.Index - 1, Bytes: link.checkpoint.Bytes}, nil
}

func (o *modelTransferOwner) checkpointTransfer(requestID string, subject *pb.CheckpointSubject, direction string, checkpoint records.ModelCheckpoint, object hub.Object,
) (*pb.CheckpointTransferRequest, *exit.Error) {
	head, plan, problem := checkpointRef(checkpoint)
	if problem != nil {
		return nil, problem
	}
	objectDigest, err := canonical.Raw(object.ID)
	if err != nil || object.Length <= 0 {
		return nil, exit.Internalf("checkpoint transfer has invalid object identity")
	}
	revision, problem := o.store.NextCheckpointGrantRevision(requestID, checkpointKind(subject), checkpoint.Slot)
	if problem != nil {
		return nil, problem
	}
	sum := sha256.Sum256([]byte(requestID + "\x00" + checkpoint.Slot + "\x00" + checkpoint.HeadID + "\x00" + object.ID + "\x00" + direction))
	return &pb.CheckpointTransferRequest{Subject: subject, PlanDigest: plan, Head: head,
		Object:     &pb.CheckpointObject{Ref: &pb.Ref{Digest: objectDigest, Length: uint64(object.Length)}},
		TransferId: checkpointKind(subject) + "-" + hex.EncodeToString(sum[:]), GrantRevision: revision}, nil
}

func (o *modelTransferOwner) uploadCheckpointLink(ctx context.Context, client *hub.Client, ref hub.Ref,
	requestID string, intent records.ModelTransferIntent, host orchestrator.CheckpointHost, subject *pb.CheckpointSubject, link checkpointLink,
) *exit.Error {
	operation := checkpointOperation(checkpointKind(subject), requestID, link.checkpoint.Slot, link.checkpoint.HeadID)
	body, _ := json.Marshal(link.objects)
	if problem := o.store.RecordCheckpointPublication(requestID, operation, body); problem != nil {
		return problem
	}
	opened, problem := client.OpenPublication(ctx, ref, operation, link.objects, "retain source preparation progress")
	if problem != nil {
		return problem
	}
	if _, problem := transfer.ValidateOpenedPublication(opened, operation, link.objects); problem != nil {
		return problem
	}
	if problem := o.store.OpenedCheckpointPublication(requestID, operation); problem != nil {
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
	pending := make([]hub.Object, 0, len(link.objects))
	for _, object := range link.objects {
		if !acceptedBefore[object.ID] {
			pending = append(pending, object)
		}
	}
	if problem := o.transferCheckpointObjects(ctx, requestID, subject, "upload", link.checkpoint, host, pending, window); problem != nil {
		return problem
	}
	for offset := 0; offset < len(link.objects); offset += 128 {
		batch := link.objects[offset:min(offset+128, len(link.objects))]
		accepted, problem := client.VerifyPublicationObjects(ctx, ref, operation, ids[offset:offset+len(batch)])
		if problem != nil {
			return problem
		}
		if accepted.Operation != operation || accepted.State != "open" {
			return exit.Named(exit.Structural, "model_transfer.source_checkpoint_custody_invalid", "Tensorhub did not verify the checkpoint object batch")
		}
		verified := make(map[string]int64, len(accepted.Objects))
		for _, object := range accepted.Objects {
			if object.State == "accepted" {
				verified[object.ObjectID] = object.Length
			}
		}
		for _, object := range batch {
			if length, ok := verified[object.ID]; !ok || length != object.Length {
				return exit.Named(exit.Structural, "model_transfer.source_checkpoint_custody_invalid", "Tensorhub did not accept checkpoint object %s", object.ID)
			}
		}
	}
	return nil
}

// SyncSourceCheckpoints walks backwards one bounded Link at a time. Only after
// every new predecessor and object is accepted does the owner advance its head.
func checkpointKind(subject *pb.CheckpointSubject) string {
	if subject.GetWeights() != nil {
		return "weights"
	}
	return "source"
}

func sourceCheckpointSubject(requestID, selection, slot string) (*pb.CheckpointSubject, *exit.Error) {
	selected, err := canonical.Raw(selection)
	if err != nil {
		return nil, exit.Internalf("source selection is malformed")
	}
	return &pb.CheckpointSubject{Kind: &pb.CheckpointSubject_Source{Source: &pb.SourceCheckpointSubject{OperationId: requestID, SourceSelectionDigest: selected, Slot: slot}}}, nil
}

func (o *modelTransferOwner) SyncCheckpoints(parent context.Context, requestID string, host orchestrator.CheckpointHost) *exit.Error {
	ctx, cancel := o.requestContext(parent, requestID)
	defer cancel()
	sources, problem := o.store.ModelSourceProgress(requestID)
	if problem != nil {
		return problem
	}
	weights, problem := o.store.ModelWeightsProgress(requestID)
	if problem != nil {
		return problem
	}
	pending := false
	needs := func(row records.ModelCheckpointProgress) bool {
		return row.WorkerBootID == host.BootID && row.Observed.HeadID != "" && (row.Acknowledged == nil || *row.Acknowledged != row.Observed)
	}
	for _, row := range sources {
		pending = pending || needs(row)
	}
	for _, row := range weights {
		pending = pending || needs(row.ModelCheckpointProgress)
	}
	if !pending {
		return nil
	}
	client, ref, transfer, problem := o.checkpointPublicationClient(requestID)
	if problem != nil || transfer == nil {
		return problem
	}
	for _, slot := range sources {
		subject, problem := sourceCheckpointSubject(requestID, transfer.SourceSelection, slot.Observed.Slot)
		if problem != nil {
			return problem
		}
		if problem := o.syncCheckpoint(ctx, client, ref, requestID, transfer.ModelTransferIntent, host, subject, slot,
			func(previous string) *exit.Error {
				return o.store.AcknowledgeModelSourceCheckpoint(requestID, transfer.SourceSelection, host.BootID, previous, slot.Observed)
			}); problem != nil {
			return problem
		}
	}
	for _, slot := range weights {
		subject := &pb.CheckpointSubject{Kind: &pb.CheckpointSubject_Weights{Weights: slot.Subject}}
		if problem := o.syncCheckpoint(ctx, client, ref, requestID, transfer.ModelTransferIntent, host, subject, slot.ModelCheckpointProgress,
			func(previous string) *exit.Error {
				return o.store.AcknowledgeModelWeightsCheckpoint(slot.Subject, slot.Attempt, host.BootID, previous, slot.Observed)
			}); problem != nil {
			return problem
		}
	}
	return nil
}

// Both subject kinds use this one backwards native-Link walk and custody barrier.
func (o *modelTransferOwner) syncCheckpoint(ctx context.Context, client *hub.Client, ref hub.Ref, requestID string, intent records.ModelTransferIntent,
	host orchestrator.CheckpointHost, subject *pb.CheckpointSubject, slot records.ModelCheckpointProgress, acknowledge func(string) *exit.Error) *exit.Error {
	if slot.WorkerBootID != host.BootID || slot.Observed.HeadID == "" || slot.Acknowledged != nil && *slot.Acknowledged == slot.Observed {
		return nil
	}
	stop := ""
	if slot.Acknowledged != nil {
		stop = slot.Acknowledged.HeadID
	}
	current, first := slot.Observed, true
	for current.HeadID != stop {
		link, problem := readCheckpointLink(ctx, host, subject, current)
		if problem != nil {
			return problem
		}
		if link.checkpoint.Index != current.Index || link.checkpoint.Bytes > current.Bytes || first && link.checkpoint.Bytes != current.Bytes || slot.Acknowledged != nil && link.checkpoint.Index <= slot.Acknowledged.Index {
			return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_chain_changed", "checkpoint chain does not extend acknowledged progress")
		}
		if problem := o.uploadCheckpointLink(ctx, client, ref, requestID, intent, host, subject, link); problem != nil {
			return problem
		}
		if link.previous == nil {
			if stop != "" || link.checkpoint.Index != 0 {
				return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_chain_changed", "checkpoint chain ended before acknowledged progress")
			}
			break
		}
		current, problem = previousCheckpoint(link)
		if problem != nil {
			return problem
		}
		first = false
	}
	if slot.Acknowledged != nil && (current.HeadLength != slot.Acknowledged.HeadLength || current.Index != slot.Acknowledged.Index || current.PlanDigest != slot.Acknowledged.PlanDigest || current.Bytes < slot.Acknowledged.Bytes) {
		return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_chain_changed", "checkpoint predecessor changed acknowledged identity")
	}
	return acknowledge(stop)
}

func (o *modelTransferOwner) downloadCheckpointObjects(ctx context.Context, client *hub.Client, ref hub.Ref,
	requestID string, subject *pb.CheckpointSubject, checkpoint records.ModelCheckpoint, host orchestrator.CheckpointHost, objects []hub.Object,
) *exit.Error {
	operation := checkpointOperation(checkpointKind(subject), requestID, checkpoint.Slot, checkpoint.HeadID)
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
	return o.transferCheckpointObjects(ctx, requestID, subject, "download", checkpoint, host, objects, window)
}

// Each transfer owns a permit before its grant is selected. The calling goroutine alone
// advances the existing grant window; only the byte-moving RPC runs concurrently.
const checkpointParallelism = 4

func (o *modelTransferOwner) transferCheckpointObjects(parent context.Context,
	requestID string, subject *pb.CheckpointSubject, direction string, checkpoint records.ModelCheckpoint,
	host orchestrator.CheckpointHost, objects []hub.Object, window *orchestrator.WeightsGrantWindow,
) *exit.Error {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	permits := make(chan struct{}, checkpointParallelism)
	var workers sync.WaitGroup
	var failed sync.Once
	var first *exit.Error
	fail := func(problem *exit.Error) {
		if problem != nil {
			failed.Do(func() { first = problem; cancel() })
		}
	}
	ids := make([]string, len(objects))
	for i, object := range objects {
		ids[i] = object.ID
	}
walk:
	for i, object := range objects {
		select {
		case permits <- struct{}{}:
		case <-ctx.Done():
			break walk
		}
		if ctx.Err() != nil {
			<-permits
			break
		}
		decision, problem := window.Spendable(ctx, object.ID, ids[i:], time.Now())
		if problem == nil && decision.Length != object.Length {
			problem = exit.Named(exit.Structural, "model_transfer.source_checkpoint_grant_changed", "checkpoint grant changed the selected object length")
		}
		if problem != nil {
			<-permits
			fail(problem)
			break
		}
		if decision.Held {
			<-permits
			continue
		}
		request, problem := o.checkpointTransfer(requestID, subject, direction, checkpoint, object)
		if problem != nil {
			<-permits
			fail(problem)
			break
		}
		if direction == "upload" {
			grant := &pb.WeightsUploadGrant{ObjectId: object.ID, Length: uint64(object.Length), Url: decision.URL,
				ExpiresAtUnix: decision.ExpiresAtUnix}
			for name, value := range decision.Headers {
				grant.RequiredHeaders = append(grant.RequiredHeaders, &pb.WeightsUploadHeader{Name: name, Value: value})
			}
			sort.Slice(grant.RequiredHeaders, func(i, j int) bool { return grant.RequiredHeaders[i].Name < grant.RequiredHeaders[j].Name })
			request.Decision = &pb.CheckpointTransferRequest_UploadGrant{UploadGrant: grant}
		} else {
			request.Decision = &pb.CheckpointTransferRequest_DownloadUrl{DownloadUrl: decision.URL}
		}
		workers.Add(1)
		go func() {
			defer workers.Done()
			defer func() { <-permits }()
			// Native transfers can outlive a canceled RPC. Stop dispatch on a sibling
			// failure, but drain already-granted calls before an automatic retry.
			_, problem := host.Transfer(parent, request)
			fail(problem)
		}()
	}
	workers.Wait()
	if parent.Err() != nil {
		code := exit.Canceled
		if parent.Err() == context.DeadlineExceeded {
			code = exit.Deadline
		}
		return exit.New(code, "source checkpoint transfer ended before completion")
	}
	if first != nil {
		return first
	}
	return nil
}

func (o *modelTransferOwner) RestoreSourceCheckpoints(parent context.Context, requestID string, host orchestrator.CheckpointHost) *exit.Error {
	ctx, cancel := o.requestContext(parent, requestID)
	defer cancel()
	progress, problem := o.store.ModelSourceProgress(requestID)
	if problem != nil || len(progress) == 0 {
		return problem
	}
	client, ref, transfer, problem := o.checkpointPublicationClient(requestID)
	if problem != nil || transfer == nil {
		return problem
	}
	for _, slot := range progress {
		if slot.Acknowledged == nil {
			continue
		}
		subject, problem := sourceCheckpointSubject(requestID, transfer.SourceSelection, slot.Acknowledged.Slot)
		if problem != nil {
			return problem
		}
		if problem := o.restoreCheckpoint(ctx, client, ref, requestID, host, subject, *slot.Acknowledged); problem != nil {
			return problem
		}
	}
	return nil
}

func (o *modelTransferOwner) RestoreWeightsCheckpoint(parent context.Context, requestID string, host orchestrator.CheckpointHost, subject *pb.WeightsCheckpointSubject) (*pb.CheckpointRef, *exit.Error) {
	ctx, cancel := o.requestContext(parent, requestID)
	defer cancel()
	rows, problem := o.store.ModelWeightsProgress(requestID)
	if problem != nil {
		return nil, problem
	}
	for _, row := range rows {
		if !proto.Equal(row.Subject, subject) || row.WorkerBootID != host.BootID {
			continue
		}
		if row.Acknowledged == nil {
			return nil, nil
		}
		client, ref, transfer, problem := o.checkpointPublicationClient(requestID)
		if problem != nil || transfer == nil {
			return nil, problem
		}
		scope := &pb.CheckpointSubject{Kind: &pb.CheckpointSubject_Weights{Weights: subject}}
		if problem := o.restoreCheckpoint(ctx, client, ref, requestID, host, scope, *row.Acknowledged); problem != nil {
			return nil, problem
		}
		head, plan, problem := checkpointRef(*row.Acknowledged)
		if problem != nil {
			return nil, problem
		}
		return &pb.CheckpointRef{Head: head, PlanDigest: plan, Index: uint64(row.Acknowledged.Index), Bytes: uint64(row.Acknowledged.Bytes)}, nil
	}
	return nil, exit.Named(exit.Conflict, "model_transfer.checkpoint_superseded", "weights restore no longer matches current owner cursor")
}

func (o *modelTransferOwner) restoreCheckpoint(ctx context.Context, client *hub.Client, ref hub.Ref, requestID string, host orchestrator.CheckpointHost, subject *pb.CheckpointSubject, current records.ModelCheckpoint) *exit.Error {
	first := true
	for {
		if problem := o.downloadCheckpointObjects(ctx, client, ref, requestID, subject, current, host, []hub.Object{{ID: current.HeadID, Length: current.HeadLength}}); problem != nil {
			return problem
		}
		link, problem := readCheckpointLink(ctx, host, subject, current)
		if problem != nil {
			return problem
		}
		if link.checkpoint.Index != current.Index || link.checkpoint.Bytes > current.Bytes || first && link.checkpoint.Bytes != current.Bytes {
			return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_restore_changed", "restored checkpoint changed recorded progress")
		}
		if problem := o.downloadCheckpointObjects(ctx, client, ref, requestID, subject, link.checkpoint, host, link.objects); problem != nil {
			return problem
		}
		if link.previous == nil {
			if link.checkpoint.Index != 0 {
				return exit.Named(exit.Conflict, "model_transfer.source_checkpoint_restore_changed", "restored chain has no root")
			}
			break
		}
		current, problem = previousCheckpoint(link)
		if problem != nil {
			return problem
		}
		first = false
	}
	return nil
}

func (o *modelTransferOwner) ReleaseCheckpoints(ctx context.Context, requestID string) *exit.Error {
	pending, problem := o.store.CheckpointPublications(requestID)
	if problem != nil || len(pending) == 0 {
		return problem
	}
	client, ref, intent, problem := o.checkpointPublicationClient(requestID)
	if problem != nil || intent == nil {
		return problem
	}
	for {
		operations, problem := o.store.CheckpointPublications(requestID)
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
			if problem := o.store.ReleaseCheckpointPublication(requestID, publication.Operation); problem != nil {
				return problem
			}
		}
	}
}
