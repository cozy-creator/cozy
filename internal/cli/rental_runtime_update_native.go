package cli

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/workertls"
)

// A machine agent advertising runtime-update/1 updates its own Runtime: this controller
// stages the wheels on it, or names versions it fetches, asks for the
// update and follows it to its end. A machine that does not keeps the SSH maintenance path.

const nativeUpdateCapability = "runtime-update/1"

// nativeUpdate is the operation recorded on a machine that updates itself.
type nativeUpdate struct {
	Operation string `json:"operation"`
}

type machineRuntimeState struct {
	Capabilities []string `json:"capabilities"`
	Runtime      string   `json:"runtime"`
	TensorFS     string   `json:"tensorfs"`
	Update       *struct {
		Operation string `json:"operation"`
		State     string `json:"state"`
		Error     string `json:"error"`
		From      struct {
			Runtime  string `json:"runtime"`
			TensorFS string `json:"tensorfs"`
		} `json:"from"`
	} `json:"update"`
}

type machineMaintenance struct {
	base    string
	client  *http.Client
	machine string
	public  ed25519.PublicKey
	sign    func([]byte) []byte
}

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
	return &machineMaintenance{base: "https://" + identity.Addr, machine: identity.WorkerID, public: public, sign: key.Sign,
		client: &http.Client{Transport: &http.Transport{TLSClientConfig: pin.TLSConfig()}}}, nil
}

func (c *machineMaintenance) do(ctx context.Context, method, path string, body io.Reader, into any) (int, *exit.Error) {
	token, err := capability.MintSigned(c.public, c.sign, capability.Grant{Machine: c.machine, Action: capability.Maintenance,
		Expires: time.Now().Add(10 * time.Minute).Unix()})
	if err != nil {
		return 0, exit.Internalf("cannot mint the maintenance capability: %s", err)
	}
	request, err := http.NewRequestWithContext(ctx, method, c.base+path, body)
	if err != nil {
		return 0, exit.Internalf("cannot address the machine: %s", err)
	}
	request.Header.Set("Authorization", "Cozy-Cap "+token)
	response, err := c.client.Do(request)
	if err != nil {
		return 0, exit.Unavailablef("the machine did not answer: %s", err)
	}
	defer response.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if response.StatusCode/100 != 2 {
		return response.StatusCode, exit.New(exit.Failed, "the machine refused %s %s (HTTP %d): %s", method, path, response.StatusCode, strings.TrimSpace(string(raw)))
	}
	if into != nil && json.Unmarshal(raw, into) != nil {
		return response.StatusCode, exit.New(exit.Structural, "the machine answered %s %s with unreadable JSON", method, path)
	}
	return response.StatusCode, nil
}

// state is what the machine says of its Runtime, or nil for a machine that does not update
// itself: an older pod supervisor serves no such route, or no HTTPS at all. A machine that
// answers and refuses the owner's capability is an error.
func (c *machineMaintenance) state(ctx context.Context) (*machineRuntimeState, *exit.Error) {
	var state machineRuntimeState
	code, problem := c.do(ctx, http.MethodGet, "/v1/machine/runtime", nil, &state)
	switch {
	case code == http.StatusForbidden:
		return nil, problem
	case problem != nil || !slices.Contains(state.Capabilities, nativeUpdateCapability):
		return nil, nil
	}
	return &state, nil
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
	state, _ := machine.state(ctx)
	return state != nil
}

// updateNative stages the requested pair on the machine and asks it to update itself.
func (u *rentalRuntimeUpdates) updateNative(ctx context.Context, row *records.RuntimeUpdate, selection *runtimeUpdateSelection, c *machineMaintenance) *exit.Error {
	body := map[string]any{"operation": row.ID}
	choose := func(key string, local *runtimeUpdateWheel, version string) *exit.Error {
		switch {
		case local != nil:
			file, err := os.Open(local.Path)
			if err != nil {
				return exit.New(exit.NotFound, "the frozen update wheel is gone: %s", err)
			}
			defer file.Close()
			var staged struct {
				SHA256 string `json:"sha256"`
			}
			if _, problem := c.do(ctx, http.MethodPut, "/v1/machine/runtime/wheels/"+local.Filename, file, &staged); problem != nil {
				return problem
			}
			if "sha256:"+staged.SHA256 != local.Digest {
				return exit.New(exit.Conflict, "the machine staged other bytes than %s", local.Filename)
			}
			body[key] = map[string]string{"file": local.Filename, "sha256": staged.SHA256}
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
			version, problem := newestPublished(ctx, name)
			if problem != nil {
				return problem
			}
			body[key] = map[string]string{"version": version}
		}
	}
	raw, _ := json.Marshal(body)
	if _, problem := c.do(ctx, http.MethodPost, "/v1/machine/runtime/update", bytes.NewReader(raw), nil); problem != nil {
		return problem
	}
	selection.Native = &nativeUpdate{Operation: row.ID}
	row.Selection, _ = json.Marshal(selection)
	row.State = "updating"
	return u.machines.store.SaveRuntimeUpdate(*row)
}

// followNative waits for the machine's update to end and records how it ended.
func (u *rentalRuntimeUpdates) followNative(ctx context.Context, row *records.RuntimeUpdate, identity *orchestrator.WorkerConnection, c *machineMaintenance) *exit.Error {
	operation := ""
	var selection runtimeUpdateSelection
	if json.Unmarshal(row.Selection, &selection) == nil && selection.Native != nil {
		operation = selection.Native.Operation
	}
	for ctx.Err() == nil {
		state, problem := c.state(ctx)
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
		} else if update := state.Update; update != nil && update.Operation == operation {
			switch update.State {
			case "succeeded", "rolled_back", "failed":
				// The worker boot is the same; its Runtime is not. The next call claims it again.
				u.machines.machines.Forget(row.RentalID)
				row.Result, _ = json.Marshal(map[string]any{"native": true, "update": update,
					"observed": map[string]any{"runtime": map[string]string{"distribution": state.Runtime}, "tensorfs": state.TensorFS}})
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

// newestPublished is a distribution's newest release on the package index.
func newestPublished(ctx context.Context, name string) (string, *exit.Error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://pypi.org/pypi/"+name+"/json", nil)
	if err != nil {
		return "", exit.Internalf("cannot address the package index: %s", err)
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return "", exit.Unavailablef("the package index did not answer: %s", err)
	}
	defer response.Body.Close()
	var project struct {
		Info struct {
			Version string `json:"version"`
		} `json:"info"`
	}
	if response.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(response.Body, 32<<20)).Decode(&project) != nil || project.Info.Version == "" {
		return "", exit.New(exit.Unavailable, "the package index has no release of %s", name)
	}
	return project.Info.Version, nil
}
