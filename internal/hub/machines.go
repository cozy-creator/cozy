package hub

import (
	"context"
	"net/http"
	"net/url"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// OwnedMachine is a user's own machine as the hub registered it (proto-062): the worker
// identity its Host presents and the capability it authenticates with, returned once.
type OwnedMachine struct {
	ID          string `json:"id"`
	WorkerToken string `json:"worker_token"`
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
	return out, nil
}

// MachineEnvironment is the hub-known half of an owned machine's grant, under the names a
// rental's pod receives them.
func (c *Client) MachineEnvironment(ctx context.Context, id string) (map[string]string, *exit.Error) {
	if problem := validMachineID(id); problem != nil {
		return nil, problem
	}
	var out struct {
		Environment map[string]string `json:"environment"`
	}
	if problem := c.do(ctx, call{method: http.MethodGet, path: "/v1/machines/" + url.PathEscape(id) + "/environment", auth: true}, &out); problem != nil {
		return nil, problem
	}
	return out.Environment, nil
}

// MachinePrepareFacts reads the release facts for preparing one package release on an owned
// machine. It has no registered image, so the inventory is absent.
func (c *Client) MachinePrepareFacts(ctx context.Context, id, pkg, release string) (PrepareFactsView, *exit.Error) {
	if problem := validMachineID(id); problem != nil {
		return PrepareFactsView{}, problem
	}
	query := url.Values{"package": {pkg}, "release": {release}}
	var out PrepareFactsView
	problem := c.do(ctx, call{method: http.MethodGet,
		path: "/v1/machines/" + url.PathEscape(id) + "/prepare-facts?" + query.Encode(),
		auth: true, responseBytes: maxPrepareFactsResponseBytes}, &out)
	return out, problem
}
