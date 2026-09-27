package cli

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
)

// DevelopmentHoldResult reports eligibility separately from an accepted idle Claim.
// This holder never acknowledges the snapshot dispatch barrier.
type DevelopmentHoldResult struct {
	State                string `json:"state"`
	RentalID             string `json:"rental_id"`
	WorkerID             string `json:"worker_id"`
	WorkerBootID         string `json:"worker_boot_id"`
	ControlStreamEpoch   uint64 `json:"control_stream_epoch"`
	SnapshotAcknowledged bool   `json:"snapshot_acknowledged"`
}

type developmentTarget struct {
	connection  *orchestrator.WorkerConnection
	ownerKey    string
	certificate []byte
}

func inspectDevelopmentTarget(layout home.Layout, store *records.Store, rentalID, bootID string) (*developmentTarget, *exit.Error) {
	if rentalID == "" || bootID == "" {
		return nil, exit.Usagef("development hold requires exact rental and boot identities")
	}
	row, problem := store.RentalRow(rentalID)
	if problem != nil {
		return nil, problem
	}
	// The hold contacts no hub; the rental's own recorded hub is authoritative.
	if row == nil || row.State != "ready" || row.ExpectedWorkerBootID != bootID || row.ExpectedWorkerID == "" {
		return nil, exit.Named(exit.Conflict, "development_hold_identity_changed", "development rental is not ready on the exact worker boot")
	}
	instance := (orchestrator.WorkerLaunchSpec{Connection: &orchestrator.WorkerConnection{RentalID: rentalID}}).InstanceID()
	attempts, problem := store.OpenAttemptsOf(instance)
	if problem != nil {
		return nil, problem
	}
	if len(attempts) != 0 {
		return nil, exit.Named(exit.Conflict, "development_hold_busy", "development worker has unclosed attempts")
	}
	target, problem := rental.Resolver(layout, store)(rentalID)
	if problem != nil {
		return nil, problem
	}
	identity, problem := rental.CreatorIdentityFor(layout, rentalID)
	if problem != nil {
		return nil, problem
	}
	pin, err := workertls.LoadPin(target.Connection.CACert)
	if err != nil {
		return nil, exit.New(exit.Credential, "development worker certificate is unavailable")
	}
	return &developmentTarget{connection: target.Connection, ownerKey: identity.PublicKey(), certificate: pin.Digest()}, nil
}

func (target *developmentTarget) same(other *developmentTarget) bool {
	a, b := target.connection, other.connection
	return a.RentalID == b.RentalID && a.WorkerID == b.WorkerID && a.WorkerBootID == b.WorkerBootID &&
		a.Addr == b.Addr && a.CACert == b.CACert && target.ownerKey == other.ownerKey && bytes.Equal(target.certificate, other.certificate)
}

func (target *developmentTarget) result(state string, epoch uint64) DevelopmentHoldResult {
	return DevelopmentHoldResult{State: state, RentalID: target.connection.RentalID, WorkerID: target.connection.WorkerID,
		WorkerBootID: target.connection.WorkerBootID, ControlStreamEpoch: epoch}
}

// InspectStoredDevelopmentHold is local eligibility only: no claim, remote idle
// assertion, provider action, or daemon lock is acquired.
func InspectStoredDevelopmentHold(cfg config.Config, rentalID, bootID string) (*DevelopmentHoldResult, *exit.Error) {
	layout, problem := home.Open(cfg.Home)
	if problem != nil {
		return nil, problem
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return nil, problem
	}
	defer store.Close()
	target, problem := inspectDevelopmentTarget(layout, store, rentalID, bootID)
	if problem != nil {
		return nil, problem
	}
	result := target.result("eligible", 0)
	return &result, nil
}

// HoldStoredDevelopmentWorker retains one daemon kernel lock while an idle pod's
// Host/Runtime group is replaced. Runtime owns the durable stream counter; this
// holder observes accepted epochs and never sends desired state, admission or work.
func HoldStoredDevelopmentWorker(ctx context.Context, cfg config.Config, rentalID, bootID string, log io.Writer,
	observe func(DevelopmentHoldResult),
) *exit.Error {
	if log == nil {
		log = io.Discard
	}
	layout, problem := home.Open(cfg.Home)
	if problem != nil {
		return problem
	}
	lock, problem := daemon.Hold(layout, "", "")
	if problem != nil {
		return problem
	}
	defer lock.Release()
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return problem
	}
	defer store.Close()
	frozen, problem := inspectDevelopmentTarget(layout, store, rentalID, bootID)
	if problem != nil {
		return problem
	}
	emit := func(state string, epoch uint64) {
		if observe != nil {
			observe(frozen.result(state, epoch))
		}
	}
	sign := rental.ClaimProof(layout)
	var acceptedEpoch uint64
	delay := 250 * time.Millisecond
	for ctx.Err() == nil {
		target, problem := inspectDevelopmentTarget(layout, store, rentalID, bootID)
		if problem != nil {
			return problem
		}
		if !frozen.same(target) {
			return exit.Named(exit.Conflict, "development_hold_identity_changed", "development hold cannot adopt a changed worker, certificate or owner")
		}
		control, problem := orchestrator.DialIdleControl(ctx, target.connection, sign, store)
		if ctx.Err() != nil {
			if control != nil {
				_ = control.Close()
			}
			return nil
		}
		if problem != nil && problem.Code != exit.Unavailable {
			return problem
		}
		if problem == nil {
			if control.ControlStreamEpoch <= acceptedEpoch {
				_ = control.Close()
				return exit.New(exit.Conflict, "development worker did not advance its accepted stream fence")
			}
			acceptedEpoch = control.ControlStreamEpoch
			delay = 250 * time.Millisecond
			fmt.Fprintln(log, "development control claimed; snapshot dispatch barrier remains closed")
			emit("holding", acceptedEpoch)
			<-control.Context.Done()
			_ = control.Close()
		}
		if ctx.Err() != nil {
			return nil
		}
		emit("reconnecting", acceptedEpoch)
		fmt.Fprintln(log, "development control unavailable; retaining owner lock while reconnecting")
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil
		case <-timer.C:
		}
		if delay < 4*time.Second {
			delay *= 2
		}
	}
	return nil
}
