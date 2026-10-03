package cli

import (
	"context"
	"encoding/json"
	"os"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// update sends the recorded selection to the rental's machine as Run kind: update and follows
// it to its outcome. The run's id is this update's id, so a daemon that resumes the update
// attaches to the same run; local wheels go up first with Write. The machine rolls back a
// candidate that never proves ready, so an update ends succeeded or failed with its reason.
func (u *rentalRuntimeUpdates) update(ctx context.Context, row *records.RuntimeUpdate, identity *orchestrator.WorkerConnection) *exit.Error {
	var selection runtimeUpdateSelection
	if len(row.Selection) > 0 && json.Unmarshal(row.Selection, &selection) != nil {
		return exit.New(exit.Structural, "recorded Runtime update selection is unreadable")
	}
	// The maintenance hold is this update's: it dials the rental's pinned machine directly.
	pin, err := workertls.LoadPin(identity.CACert)
	if err != nil {
		return exit.New(exit.Credential, "the machine's TLS identity cannot be read")
	}
	key, problem := u.machines.machines.RentalKey(identity.RentalID)
	if problem != nil {
		return problem
	}
	machine, err := machinev1.Dial(identity.Addr, pin.TLSConfig(), identity.WorkerID, key.Signer())
	if err != nil {
		return machines.Transport(err)
	}
	defer machine.Close()
	member := func(local *runtimeUpdateWheel, version string) (*machinev1.Member, *exit.Error) {
		if local == nil {
			if version == "" {
				return nil, nil
			}
			return &machinev1.Member{Version: version}, nil
		}
		file, err := os.Open(local.Path)
		if err != nil {
			return nil, exit.New(exit.Conflict, "the frozen update wheel %s is gone: %s", local.Filename, err)
		}
		defer file.Close()
		if err := machine.Write(ctx, local.Digest, uint64(local.Length), file); err != nil {
			return nil, machines.Transport(err)
		}
		return &machinev1.Member{Wheel: local.Filename, Digest: local.Digest, Length: uint64(local.Length)}, nil
	}
	cohort := machinev1.Cohort{Agent: "bundled"}
	if cohort.Runtime, problem = member(selection.LocalRuntime, selection.RuntimeVersion); problem != nil {
		return problem
	}
	if cohort.TensorFS, problem = member(selection.LocalTensorFS, selection.TensorFSVersion); problem != nil {
		return problem
	}
	if cohort.Runtime == nil && cohort.TensorFS == nil {
		// Nothing named: the newest published pair.
		for _, pick := range []struct {
			name   string
			member **machinev1.Member
		}{{hostruntime.Distribution, &cohort.Runtime}, {"tensorfs", &cohort.TensorFS}} {
			version, problem := machines.NewestPublished(ctx, pick.name)
			if problem != nil {
				return problem
			}
			*pick.member = &machinev1.Member{Version: version}
		}
	}
	row.State = "updating"
	if problem := u.machines.store.SaveRuntimeUpdate(*row); problem != nil {
		return problem
	}
	var saveProblem *exit.Error
	outcome, err := machine.Update(ctx, row.ID, cohort, func(event *pb.RunEvent) {
		if event.GetProgress().GetStage() == "waiting_activation" && row.State != "waiting_activation" {
			row.State = "waiting_activation"
			saveProblem = u.machines.store.SaveRuntimeUpdate(*row)
		}
	})
	if err != nil {
		return machines.Transport(err)
	}
	if saveProblem != nil {
		return saveProblem
	}
	// Same boot, other software: the next call on the kept connection asks it again.
	u.machines.machines.Forget(row.RentalID)
	type pair struct {
		Runtime  string `json:"runtime"`
		TensorFS string `json:"tensorfs"`
	}
	var result struct{ From, To pair }
	_ = json.Unmarshal(outcome.GetResult(), &result)
	observed := result.To
	row.State, row.Error = "succeeded", ""
	if outcome.GetStatus() != "succeeded" {
		observed = result.From
		row.State, row.Error = "failed", outcome.GetReason().GetMessage()
		if row.Error == "" {
			row.Error = outcome.GetReason().GetCode()
		}
	}
	row.Result, _ = json.Marshal(map[string]any{
		"update":   map[string]pair{"from": result.From, "to": result.To},
		"observed": map[string]any{"runtime": map[string]string{"distribution": observed.Runtime}, "tensorfs": observed.TensorFS},
	})
	return nil
}
