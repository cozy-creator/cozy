package cli

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/workertls"
)

// A machine agent advertising runtime-update/1 updates its own Runtime: this controller
// stages the wheels on it, or names versions it fetches, asks for the
// update and follows it to its end. Missing capability is an upgrade refusal.

// nativeUpdate is the operation recorded on a machine that updates itself.
type nativeUpdate struct {
	Operation string `json:"operation"`
}

type machineMaintenance = machines.Maintenance

func (u *rentalRuntimeUpdates) maintenance(identity *orchestrator.WorkerConnection) (*machineMaintenance, *exit.Error) {
	pin, err := workertls.LoadPin(identity.CACert)
	if err != nil {
		return nil, exit.New(exit.Credential, "the machine's TLS identity cannot be read")
	}
	key, problem := u.machines.machines.RentalKey(identity.RentalID)
	if problem != nil {
		return nil, problem
	}
	public, err := base64.RawURLEncoding.DecodeString(key.PublicKey())
	if err != nil || len(public) != ed25519.PublicKeySize {
		return nil, exit.New(exit.Credential, "the rental's owner key is unreadable")
	}
	return &machineMaintenance{Base: "https://" + identity.Addr, Machine: identity.WorkerID, Public: public, Sign: key.Sign,
		Client: &http.Client{Transport: &http.Transport{TLSClientConfig: pin.TLSConfig()}}, Log: u.machines.context.Out}, nil
}

// updatesItself says whether a rental's machine answers that it updates its own Runtime.
func (u *rentalRuntimeUpdates) updatesItself(id string) bool {
	target, problem := u.machines.machines.Rentals(id)
	if problem != nil || target == nil || target.Connection == nil {
		return false
	}
	machine, problem := u.maintenance(target.Connection)
	if problem != nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	state, problem := machine.State(ctx)
	return state != nil || problem != nil && problem.ErrName() == machines.AuthorityPending
}

// updateNative stages the requested pair on the machine and asks it to update itself.
func (u *rentalRuntimeUpdates) updateNative(ctx context.Context, row *records.RuntimeUpdate, selection *runtimeUpdateSelection, c *machineMaintenance) *exit.Error {
	body := map[string]any{"operation": row.ID, "agent": "bundled", "pin": selection.LocalRuntime != nil || selection.LocalTensorFS != nil || selection.RuntimeVersion != "" || selection.TensorFSVersion != ""}
	choose := func(key string, local *runtimeUpdateWheel, version string) *exit.Error {
		switch {
		case local != nil:
			staged, problem := c.Stage(ctx, local.Path, local.Filename)
			if problem != nil {
				return problem
			}
			if "sha256:"+staged != local.Digest {
				return exit.New(exit.Conflict, "the machine staged other bytes than %s", local.Filename)
			}
			body[key] = map[string]string{"file": local.Filename, "sha256": staged}
		case version != "":
			body[key] = map[string]string{"version": version}
		}
		return nil
	}
	if problem := choose("runtime", selection.LocalRuntime, selection.RuntimeVersion); problem != nil {
		return problem
	}
	if problem := choose("tensorfs", selection.LocalTensorFS, selection.TensorFSVersion); problem != nil {
		return problem
	}
	if body["runtime"] == nil && body["tensorfs"] == nil {
		// Nothing named: the newest published pair.
		for key, name := range map[string]string{"runtime": hostruntime.Distribution, "tensorfs": "tensorfs"} {
			version, problem := machines.NewestPublished(ctx, name)
			if problem != nil {
				return problem
			}
			body[key] = map[string]string{"version": version}
		}
	}
	raw, _ := json.Marshal(body)
	for {
		if _, problem := c.AwaitUpdateAdmission(ctx); problem != nil {
			return problem
		}
		// The request can take effect even if its response is lost. Preserve that
		// uncertainty durably before sending, using this operation's stable identity.
		selection.Native = &nativeUpdate{Operation: row.ID}
		row.Selection, _ = json.Marshal(selection)
		row.State = "updating"
		if problem := u.machines.store.SaveRuntimeUpdate(*row); problem != nil {
			return problem
		}
		code, problem := c.Do(ctx, http.MethodPost, "/v1/machine/runtime/update", bytes.NewReader(raw), nil)
		if problem == nil {
			return nil
		}
		waits := problem.ErrName() == "machine.runtime_starting" || problem.ErrName() == machines.AuthorityPending
		if waits || code >= 400 && code < 500 {
			// These responses are the agent's pre-admission refusals.
			selection.Native = nil
			row.Selection, _ = json.Marshal(selection)
			row.State = "preparing"
			if save := u.machines.store.SaveRuntimeUpdate(*row); save != nil {
				return save
			}
			if waits {
				continue
			}
			return problem
		}
		// Do not label a lost acknowledgement as unsent or release the maintenance
		// fence. The same operation's authoritative state decides its result.
		return nil // update() immediately follows the durable operation below
	}

}

// followNative waits for the machine's update to end and records how it ended.
func (u *rentalRuntimeUpdates) followNative(ctx context.Context, row *records.RuntimeUpdate, identity *orchestrator.WorkerConnection, c *machineMaintenance) *exit.Error {
	operation := ""
	var selection runtimeUpdateSelection
	if json.Unmarshal(row.Selection, &selection) == nil && selection.Native != nil {
		operation = selection.Native.Operation
	}
	for ctx.Err() == nil {
		state, problem := c.State(ctx)
		if problem == nil && state == nil {
			problem = exit.New(exit.Conflict, "the machine no longer updates its own Runtime")
		}
		if problem != nil {
			// An unreachable machine is asked again only while its rental lasts.
			if rented, readProblem := u.machines.store.RentalRow(row.RentalID); readProblem != nil {
				return readProblem
			} else if rented == nil || rented.State != "ready" {
				return exit.Named(exit.Conflict, "rental.ended", "the rental ended during its Runtime update")
			}
		} else if state.Update == nil || state.Update.Operation != operation {
			return exit.Named(exit.Unavailable, "machine.update_observation_lost", "update %s is no longer the machine's current operation; inspect its outcome before another update", operation)
		} else if update := state.Update; update != nil && update.Operation == operation {
			switch update.State {
			case "prepared", "waiting_activation":
				// The candidate is durable and the old pair remains active while
				// current work drains. Keep the daemon's hold and continue observing;
				// this is a successful admission of the update, not activation.
				if row.State != "waiting_activation" {
					row.State = "waiting_activation"
					row.Error = ""
					row.Result, _ = json.Marshal(map[string]any{"native": true, "update": update,
						"observed": map[string]any{"runtime": map[string]string{"distribution": state.Runtime}, "tensorfs": state.TensorFS, "agent": state.Agent, "bootstrap": state.Bootstrap}})
					if save := u.machines.store.SaveRuntimeUpdate(*row); save != nil {
						return save
					}
				}
			case "succeeded", "rolled_back", "failed":
				// The worker boot is the same; its Runtime is not. The next call claims it again.
				u.machines.machines.Forget(row.RentalID)
				row.Result, _ = json.Marshal(map[string]any{"native": true, "update": update,
					"observed": map[string]any{"runtime": map[string]string{"distribution": state.Runtime}, "tensorfs": state.TensorFS, "agent": state.Agent, "bootstrap": state.Bootstrap}})
				switch {
				case update.State == "succeeded":
					row.State, row.Error = "succeeded", ""
				case state.Runtime == update.From.Runtime && state.TensorFS == update.From.TensorFS:
					row.State = "failed"
					row.Error = fmt.Sprintf("the update did not complete (%s); the machine runs Runtime %s / TensorFS %s", update.Error, state.Runtime, state.TensorFS)
				default:
					row.State = "unusable"
					row.Error = fmt.Sprintf("the update and its rollback failed (%s); the machine runs Runtime %s / TensorFS %s", update.Error, state.Runtime, state.TensorFS)
				}
				return nil
			}
		}
		select {
		case <-ctx.Done():
		case <-time.After(time.Second):
		}
	}
	return exit.New(exit.Canceled, "Runtime update observation interrupted")
}
