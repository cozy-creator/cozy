package cli

import (
	"context"
	"fmt"
	"io"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// SyncStoredSourceCustody is an operator handoff while the ordinary daemon is
// stopped. It retains the existing operation through the same publication owner,
// while a Claim-only WorkerControl hold prevents idle release. It creates no
// attempt or desired state and selects no paid capacity.
func SyncStoredSourceCustody(ctx context.Context, cfg config.Config, requestID, rentalID, bootID string,
	log io.Writer, auth *accountauth.Manager, retained func(context.Context, *SourceCustodyResult),
) (*SourceCustodyResult, *exit.Error) {
	if requestID == "" || rentalID == "" || bootID == "" {
		return nil, exit.Usagef("source custody requires exact request, rental and boot identities")
	}
	layout, problem := home.Open(cfg.Home)
	if problem != nil {
		return nil, problem
	}
	held, problem := daemon.Hold(layout, "", "")
	if problem != nil {
		return nil, problem
	}
	defer held.Release()
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return nil, problem
	}
	defer store.Close()
	_, problem = inspectSourceCustody(store, cfg, requestID, rentalID, bootID)
	if problem != nil {
		return nil, problem
	}
	target, problem := rental.Resolver(layout, store)(rentalID)
	if problem != nil {
		return nil, problem
	}
	connection, problem := orchestrator.DialIdleControl(ctx, target.Connection, rental.ClaimProof(layout), nil)
	if problem != nil {
		return nil, problem
	}
	defer connection.Close()
	fmt.Fprintln(log, "source custody control claimed; snapshot dispatch barrier remains closed")
	owner := NewModelTransferOwner(cfg, store, log, auth)
	if problem := owner.SyncCheckpoints(connection.Context, requestID, connection.Host); problem != nil {
		return nil, problem
	}
	result, problem := inspectSourceCustody(store, cfg, requestID, rentalID, bootID)
	if problem != nil {
		return nil, problem
	}
	for _, slot := range result.Checkpoints {
		if slot.Acknowledged == nil || *slot.Acknowledged != slot.Observed {
			return nil, exit.New(exit.Conflict, "source custody is not yet fully acknowledged")
		}
	}
	result.ControlClaimed = true
	if retained != nil {
		retained(connection.Context, result)
	}
	return result, nil
}

// InspectStoredSourceCustody reads eligibility and exact progress without claiming
// the daemon lock, acquiring hardware, or contacting any remote service.
func InspectStoredSourceCustody(cfg config.Config, requestID, rentalID, bootID string) (*SourceCustodyResult, *exit.Error) {
	layout, problem := home.Open(cfg.Home)
	if problem != nil {
		return nil, problem
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return nil, problem
	}
	defer store.Close()
	return inspectSourceCustody(store, cfg, requestID, rentalID, bootID)
}

func inspectSourceCustody(store *records.Store, cfg config.Config, requestID, rentalID, bootID string) (*SourceCustodyResult, *exit.Error) {
	request, problem := store.RequestRow(requestID)
	if problem != nil {
		return nil, problem
	}
	if request == nil || request.Worker != rentalID || !request.Rental || !request.IsJob() || (request.State != "submitted" && request.State != "queued") {
		return nil, exit.New(exit.Conflict, "source custody request is not unstarted on the exact selected rental")
	}
	attempts, problem := store.Attempts(requestID)
	if problem != nil {
		return nil, problem
	}
	if len(attempts) != 0 {
		return nil, exit.New(exit.Conflict, "source custody handoff requires no producer attempts")
	}
	transfer, problem := store.ModelTransferOf(requestID)
	if problem != nil {
		return nil, problem
	}
	if transfer == nil || !transfer.HasAcquisition() || (transfer.State != "materializing" && transfer.State != "materialized") {
		return nil, exit.New(exit.Conflict, "source custody requires an active source materialization")
	}
	progress, problem := store.ModelSourceProgress(requestID)
	if problem != nil {
		return nil, problem
	}
	if len(progress) == 0 || len(progress) != len(transfer.SourceProfiles) {
		return nil, exit.New(exit.Conflict, "source custody has no complete observed checkpoint roster")
	}
	for _, slot := range progress {
		if slot.WorkerBootID != bootID || transfer.SourceProfiles[slot.Observed.Slot] == "" {
			return nil, exit.New(exit.Conflict, "source custody checkpoint does not match the frozen source selection and boot")
		}
	}
	row, problem := store.RentalRow(rentalID)
	if problem != nil {
		return nil, problem
	}
	if row == nil || row.Hub != cfg.HubURL || row.ExpectedWorkerBootID != bootID {
		return nil, exit.New(exit.Conflict, "source custody rental Hub or worker boot changed")
	}
	result := &SourceCustodyResult{SourcePrepared: transfer.State == "materialized" && transfer.ModelsWorkerBootID == bootID && len(transfer.Models) == len(transfer.SourceProfiles), SourceSelection: transfer.SourceSelection, Models: transfer.Models, Checkpoints: progress}
	for _, file := range transfer.SourceFiles {
		result.SourceBytes += file.Length
	}
	for _, slot := range progress {
		result.ObservedBytes += slot.Observed.Bytes
		if slot.Acknowledged != nil {
			result.AcknowledgedBytes += slot.Acknowledged.Bytes
		}
	}
	return result, nil
}

// SourcePrepared distinguishes final source inputs from an acknowledged partial prefix.
type SourceCustodyResult struct {
	ControlClaimed       bool                              `json:"control_claimed"`
	SnapshotAcknowledged bool                              `json:"snapshot_acknowledged"`
	SourcePrepared       bool                              `json:"source_prepared"`
	SourceSelection      string                            `json:"source_selection"`
	SourceBytes          int64                             `json:"source_bytes"`
	ObservedBytes        int64                             `json:"observed_bytes"`
	AcknowledgedBytes    int64                             `json:"acknowledged_bytes"`
	Models               []records.ModelRef                `json:"models"`
	Checkpoints          []records.ModelCheckpointProgress `json:"source_checkpoints"`
}
