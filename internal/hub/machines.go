package hub

import (
	"context"
	"net/http"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// OwnedMachine is a user's own machine as the hub registered it (proto-062): the worker
// identity its Host presents, the capability it authenticates with, and the hub-known half
// of its grant (under the names a rental's pod receives them), returned once.
type OwnedMachine struct {
	ID          string            `json:"id"`
	WorkerToken string            `json:"worker_token"`
	Environment map[string]string `json:"environment"`
}

// ownedMachinePrefix is the hub's spelling for an owned machine's worker id; no rental id
// begins with it.
const ownedMachinePrefix = "om-"

func validMachineID(id string) *exit.Error {
	if !strings.HasPrefix(id, ownedMachinePrefix) || len(id) > 64 || strings.ContainsAny(id, "/?#% \t\n") {
		return exit.Named(exit.Validation, "hub.machine_id_invalid", "the hub returned an invalid owned machine id")
	}
	return nil
}

// RegisterMachine registers this machine for the signed-in user.
func (c *Client) RegisterMachine(ctx context.Context) (OwnedMachine, *exit.Error) {
	var out OwnedMachine
	if problem := c.do(ctx, call{method: http.MethodPost, path: "/v1/machines", auth: true,
		reason: "register this machine", body: struct{}{}}, &out); problem != nil {
		return OwnedMachine{}, problem
	}
	if problem := validMachineID(out.ID); problem != nil {
		return OwnedMachine{}, problem
	}
	if len(out.WorkerToken) != 43 {
		return OwnedMachine{}, exit.Named(exit.Conflict, "hub.machine_token_invalid", "the hub returned no worker capability for the registered machine")
	}
	if len(out.Environment) == 0 {
		return OwnedMachine{}, exit.Named(exit.Conflict, "hub.machine_environment_invalid", "the hub returned no environment for the registered machine")
	}
	for name := range out.Environment {
		if !strings.HasPrefix(name, "TENSORHUB_") {
			return OwnedMachine{}, exit.Named(exit.Conflict, "hub.machine_environment_invalid", "the hub's machine environment names %s, which is not a hub fact", name)
		}
	}
	return out, nil
}
