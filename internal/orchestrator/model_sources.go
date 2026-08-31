package orchestrator

import (
	"context"
	"encoding/hex"
	"sort"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// ModelSourceCapability is one short-lived provider URL for a plan-pinned file. It is
// deliberately not a records type: only the stable ObjectRef/status crosses SQLite.
type ModelSourceCapability struct {
	Member, ObjectID, URL string
	Length                int64
	Provider              pb.ModelSourceProvider
	ExpiresAtUnix         uint64
}

type ProductionSourceFile struct {
	Member, ObjectID string
	Length           int64
}

type ProductionSourcePlan struct {
	SelectionDigest string
	SourceURI       string
	DeclaredLicense string
	Files           []ProductionSourceFile
	Profiles        map[string]string
}

func (c *Orchestrator) productionChannel(operationID string) chan struct{} {
	c.mu.Lock()
	defer c.mu.Unlock()
	ch := c.productionWake[operationID]
	if ch == nil {
		ch = make(chan struct{}, 1)
		c.productionWake[operationID] = ch
	}
	return ch
}

func (c *Orchestrator) signalProduction(operationID string) {
	ch := c.productionChannel(operationID)
	select {
	case ch <- struct{}{}:
	default:
	}
}

func (c *Orchestrator) signalAllProductions() {
	c.mu.Lock()
	channels := make([]chan struct{}, 0, len(c.productionWake))
	for _, ch := range c.productionWake {
		channels = append(channels, ch)
	}
	c.mu.Unlock()
	for _, ch := range channels {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
}

func (c *Orchestrator) waitProduction(ctx context.Context, operationID string) *exit.Error {
	select {
	case <-c.productionChannel(operationID):
		return nil
	case <-ctx.Done():
		return exit.New(exit.Canceled, "model production %s stopped: %s", operationID, ctx.Err())
	case <-c.done:
		return exit.Unavailablef("the daemon stopped while model production %s was active", operationID)
	}
}

func (c *Orchestrator) rentalControl(rentalID string) (*session, *exit.Error) {
	instanceID := rentalInstanceID(rentalID)
	c.mu.Lock()
	w := c.workers[instanceID]
	var s *session
	if w != nil && w.snapshotAcknowledged {
		s = c.sessions[w.bootID]
	}
	c.mu.Unlock()
	if w == nil || w.spec.Connection == nil || w.spec.Connection.RentalID != rentalID {
		return nil, exit.New(exit.NotFound, "rental %s has no attached worker", rentalID)
	}
	if s == nil {
		return nil, exit.Unavailablef("rental %s has no reconciled WorkerControl session", rentalID)
	}
	return s, nil
}

// PrepareProductionSources drives the external owner frames only. Provider resolution
// already happened in the caller; TensorFS interpretation remains inside Runtime.
func (c *Orchestrator) PrepareProductionSources(ctx context.Context, operationID, rentalID string,
	plan ProductionSourcePlan, capabilities []ModelSourceCapability,
) ([]records.PreparedModelSource, *exit.Error) {
	selection, err := canonical.Raw(plan.SelectionDigest)
	if err != nil {
		return nil, exit.Named(exit.Validation, "model_production.source_selection_invalid",
			"model production source selection is not one SHA-256 digest")
	}
	expected := make(map[string]ProductionSourceFile, len(plan.Files))
	seed := make([]records.ModelProductionSourceFile, 0, len(plan.Files))
	for _, file := range plan.Files {
		expected[file.Member] = file
		seed = append(seed, records.ModelProductionSourceFile{OperationID: operationID,
			SelectionDigest: plan.SelectionDigest, Member: file.Member,
			ObjectID: file.ObjectID, Length: file.Length})
	}
	if problem := c.opt.Store.RecordModelProductionSourceFiles(operationID, seed); problem != nil {
		return nil, problem
	}
	byMember := make(map[string]ModelSourceCapability, len(capabilities))
	for _, capability := range capabilities {
		file, ok := expected[capability.Member]
		if !ok || capability.ObjectID != file.ObjectID || capability.Length != file.Length ||
			capability.URL == "" || byMember[capability.Member].Member != "" {
			return nil, exit.Named(exit.Conflict, "model_production.source_capability_conflict",
				"refreshed provider access changed the selected source files")
		}
		byMember[capability.Member] = capability
	}
	if len(byMember) != len(expected) {
		return nil, exit.Named(exit.Conflict, "model_production.source_capability_incomplete",
			"refreshed provider access returned %d of %d selected files", len(byMember), len(expected))
	}

	profiles := make([]records.PreparedModelSource, 0, len(plan.Profiles))
	for slot, profile := range plan.Profiles {
		profiles = append(profiles, records.PreparedModelSource{OperationID: operationID,
			Slot: slot, Profile: profile})
	}
	sort.Slice(profiles, func(i, j int) bool { return profiles[i].Slot < profiles[j].Slot })
	if problem := c.opt.Store.RecordModelProductionProfiles(operationID, profiles); problem != nil {
		return nil, problem
	}
	for {
		prepared, problem := c.opt.Store.PreparedModelSources(operationID)
		if problem != nil {
			return prepared, problem
		}
		complete := len(prepared) == len(plan.Profiles)
		for _, source := range prepared {
			complete = complete && source.ManifestID != ""
		}
		if complete {
			return prepared, nil
		}
		files, problem := c.opt.Store.ModelProductionSourceFiles(operationID)
		if problem != nil {
			return nil, problem
		}
		allVerified := len(files) == len(expected)
		for _, file := range files {
			if file.State == "failed" {
				return nil, exit.Named(exit.Failed, file.SafeCode,
					"worker refused model source %s: %s", file.Member, file.SafeDetail)
			}
			allVerified = allVerified && file.State == "verified"
		}
		s, problem := c.rentalControl(rentalID)
		if problem != nil {
			if wait := c.waitProduction(ctx, operationID); wait != nil {
				return nil, wait
			}
			continue
		}
		if !allVerified {
			for _, file := range files {
				if file.State == "verified" {
					continue
				}
				access := byMember[file.Member]
				revision := file.CapabilityRevision + 1
				request := &pb.ModelSourceFileRequest{
					RecordOwnerEpoch: recordOwnerEpoch, ControlStreamGeneration: s.generation,
					WorkerBootId: s.bootID, OperationId: operationID,
					SourceSelectionDigest: selection, Member: file.Member,
					ObjectId: file.ObjectID, Length: uint64(file.Length), Provider: access.Provider,
					Url: access.URL, ExpiresAtUnix: access.ExpiresAtUnix,
					CapabilityRevision: revision,
				}
				if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_ModelSourceFileRequest{
					ModelSourceFileRequest: request}}) {
					break
				}
			}
		} else {
			wireProfiles := make([]*pb.ModelSourceProfile, 0, len(plan.Profiles))
			for slot, profile := range plan.Profiles {
				wireProfiles = append(wireProfiles, &pb.ModelSourceProfile{Slot: slot, Profile: profile})
			}
			sort.Slice(wireProfiles, func(i, j int) bool { return wireProfiles[i].Slot < wireProfiles[j].Slot })
			s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_ModelSourcePrepareRequest{
				ModelSourcePrepareRequest: &pb.ModelSourcePrepareRequest{
					RecordOwnerEpoch: recordOwnerEpoch, ControlStreamGeneration: s.generation,
					WorkerBootId: s.bootID, OperationId: operationID,
					SourceSelectionDigest: selection, Profiles: wireProfiles,
					SourceUri: plan.SourceURI, DeclaredLicense: plan.DeclaredLicense,
				}}})
		}
		if problem := c.waitProduction(ctx, operationID); problem != nil {
			return nil, problem
		}
	}
}

func (c *Orchestrator) onModelSourceFileStatus(s *session, frame *pb.ModelSourceFileStatus) {
	selection, err := canonical.Spell(frame.SourceSelectionDigest)
	files, problem := c.opt.Store.ModelProductionSourceFiles(frame.OperationId)
	if err != nil || problem != nil {
		c.logf("ModelSourceFileStatus %s/%s refused: unknown operation or digest",
			frame.OperationId, frame.Member)
		return
	}
	var expected *records.ModelProductionSourceFile
	for i := range files {
		if files[i].Member == frame.Member {
			expected = &files[i]
			break
		}
	}
	if expected == nil || selection != expected.SelectionDigest {
		c.logf("ModelSourceFileStatus %s/%s refused: source selection changed",
			frame.OperationId, frame.Member)
		return
	}
	state := trimEnum(pb.ModelSourceFileState_name[int32(frame.State)], "MODEL_SOURCE_FILE_STATE_")
	state = map[string]string{"ACCEPTED": "accepted", "DOWNLOADING": "downloading",
		"VERIFIED": "verified", "FAILED": "failed"}[state]
	if state == "" || frame.Length > uint64(^uint64(0)>>1) ||
		frame.TransferredBytes > frame.Length {
		return
	}
	problem = c.opt.Store.RecordModelProductionSourceStatus(records.ModelProductionSourceFile{
		OperationID: frame.OperationId, Member: frame.Member, ObjectID: frame.ObjectId,
		Length: int64(frame.Length), CapabilityRevision: frame.CapabilityRevision,
		State: state, TransferredBytes: int64(frame.TransferredBytes),
		SafeCode: frame.SafeCode, SafeDetail: frame.SafeDetail,
	})
	if problem != nil {
		c.logf("ModelSourceFileStatus %s/%s refused: %s", frame.OperationId, frame.Member,
			problem.Message)
		return
	}
	c.signalProduction(frame.OperationId)
}

func (c *Orchestrator) onModelSourcePrepared(s *session, frame *pb.ModelSourcePrepared) {
	selection, err := canonical.Spell(frame.SourceSelectionDigest)
	profiles, problem := c.opt.Store.PreparedModelSources(frame.OperationId)
	if err != nil || problem != nil {
		return
	}
	files, problem := c.opt.Store.ModelProductionSourceFiles(frame.OperationId)
	if problem != nil || len(files) == 0 || selection != files[0].SelectionDigest {
		return
	}
	if frame.Outcome == pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REFUSED {
		operation, _ := c.opt.Store.ModelProduction(frame.OperationId)
		if operation != nil {
			_ = c.opt.Store.FailModelProduction(frame.OperationId, operation.State,
				frame.SafeCode, frame.SafeDetail)
		}
		c.signalProduction(frame.OperationId)
		return
	}
	if frame.Outcome != pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_PREPARED &&
		frame.Outcome != pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED {
		return
	}
	rows := make([]records.PreparedModelSource, 0, len(frame.Sources))
	expectedProfiles := make(map[string]string, len(profiles))
	for _, profile := range profiles {
		expectedProfiles[profile.Slot] = profile.Profile
	}
	for _, source := range frame.Sources {
		if source == nil || source.Manifest == nil || len(source.Manifest.Digest) != 32 ||
			source.Manifest.Length == 0 || source.Manifest.Length > uint64(^uint64(0)>>1) ||
			len(source.ReleaseEvidenceCanonicalBytes) == 0 ||
			len(source.ReleaseEvidenceCanonicalBytes) > pb.MaxReleaseEvidenceBytes ||
			expectedProfiles[source.Slot] != source.Profile {
			return
		}
		rows = append(rows, records.PreparedModelSource{OperationID: frame.OperationId,
			Slot: source.Slot, Profile: source.Profile,
			ManifestID:      "sha256:" + hex.EncodeToString(source.Manifest.Digest),
			ManifestLength:  int64(source.Manifest.Length),
			ReleaseEvidence: append([]byte(nil), source.ReleaseEvidenceCanonicalBytes...)})
	}
	if len(rows) != len(expectedProfiles) {
		return
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Slot < rows[j].Slot })
	for i := 1; i < len(rows); i++ {
		if rows[i-1].Slot >= rows[i].Slot {
			return
		}
	}
	if problem := c.opt.Store.RecordPreparedModelSources(frame.OperationId, rows); problem != nil {
		c.logf("ModelSourcePrepared %s refused: %s", frame.OperationId, problem.Message)
		return
	}
	c.signalProduction(frame.OperationId)
}
