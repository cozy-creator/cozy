package cli

import (
	"bytes"
	"context"
	"crypto/rand"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

//go:embed runtime_update_transport.py
var runtimeUpdateTransport string

type rentalRuntimeUpdates struct {
	machines *machineRuns
	running  sync.Map
	observed sync.Map
}

type runtimeUpdateSelection struct {
	Target        hub.RuntimeUpdateTarget `json:"target"`
	Directory     string                  `json:"directory"`
	Stage         string                  `json:"stage"`
	Host          string                  `json:"host"`
	SSHArguments  []string                `json:"ssh_arguments"`
	SFTPArguments []string                `json:"sftp_arguments"`
	Selection     json.RawMessage         `json:"selection,omitempty"`
}

func (u *rentalRuntimeUpdates) Start(id string) (*records.RuntimeUpdate, *exit.Error) {
	return u.startForRequest(id, "")
}

func (u *rentalRuntimeUpdates) startForRequest(id, request string) (*records.RuntimeUpdate, *exit.Error) {
	m := u.machines
	row, problem := m.store.RentalRow(id)
	if problem != nil {
		return nil, problem
	}
	if row == nil || row.State != "ready" {
		return nil, exit.New(exit.Conflict, "this rental is not ready; no machine was purchased or changed")
	}
	current, problem := m.store.RuntimeUpdate(id)
	if problem != nil {
		return nil, problem
	}
	if current == nil || !current.Active() {
		current, problem = m.store.BeginRuntimeUpdate(id, row.ExpectedWorkerBootID, request)
	}
	if problem == nil {
		u.run(*current)
	}
	return current, problem
}

func (u *rentalRuntimeUpdates) Resume() {
	rows, problem := u.machines.store.ActiveRuntimeUpdates()
	if problem != nil {
		fmt.Fprintln(u.machines.context.Out, problem.Message)
		return
	}
	for _, row := range rows {
		u.run(row)
	}
}

func (u *rentalRuntimeUpdates) run(row records.RuntimeUpdate) {
	if _, exists := u.running.LoadOrStore(row.RentalID, true); exists {
		return
	}
	go func() {
		defer u.running.Delete(row.RentalID)
		m := u.machines
		problem := m.fleet.owner.MaintainRental(m.ctx, row.RentalID, func(ctx context.Context, identity *orchestrator.WorkerConnection) *exit.Error {
			if identity.WorkerBootID != row.BootID {
				return exit.New(exit.Conflict, "the rental's worker boot changed before maintenance")
			}
			if row.RequestID != "" {
				if problem := m.store.AppendEvent(row.RequestID, "machine.runtime_update_attempted", 0, map[string]any{"rental": row.RentalID, "update": row.ID}); problem != nil {
					return problem
				}
			}
			return u.update(ctx, &row, identity)
		})
		if problem != nil {
			row.Error = problem.Message
			if row.State != "updating" && row.State != "reconciling" {
				row.State = "failed"
			} else {
				row.State = "reconciling"
			}
		}
		if problem := m.store.SaveRuntimeUpdate(row); problem != nil {
			fmt.Fprintln(m.context.Out, problem.Message)
		}
		u.observed.Delete(row.RentalID)
		m.mu.Lock()
		delete(m.claimed, row.RentalID)
		m.mu.Unlock()
	}()
}

func (u *rentalRuntimeUpdates) selection(ctx context.Context, row records.RuntimeUpdate) (runtimeUpdateSelection, *exit.Error) {
	m := u.machines
	var selection runtimeUpdateSelection
	target, problem := client(m.context).RentalRuntimeUpdateTarget(ctx, row.RentalID)
	if problem != nil {
		return selection, problem
	}
	if target.WorkerBootID != row.BootID {
		return selection, exit.New(exit.Conflict, "approved Runtime update refers to another worker boot")
	}
	remote, problem := client(m.context).Rental(ctx, row.RentalID)
	if problem != nil {
		return selection, problem
	}
	if !remote.Development || !remote.Ready() || remote.WorkerBootID != row.BootID {
		return selection, exit.New(exit.Conflict, "this rental has no compatible private maintenance endpoint")
	}
	host, port, err := net.SplitHostPort(remote.SSHAddress)
	number, portErr := strconv.Atoi(port)
	if err != nil || net.ParseIP(host) == nil || portErr != nil || number < 1 || number > 65535 {
		return selection, exit.New(exit.Unavailable, "the rental has no mapped SSH maintenance endpoint")
	}
	key := m.context.Cfg.RentalsSSHPublicKey
	if key == "" {
		key = filepath.Join(m.context.Cfg.Home, "auth", "rental-ssh.pub")
	}
	if strings.HasPrefix(key, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return selection, exit.New(exit.Credential, "cannot resolve rental SSH identity")
		}
		key = filepath.Join(home, strings.TrimPrefix(key, "~/"))
	}
	if !filepath.IsAbs(key) {
		key = filepath.Join(m.context.Cfg.Home, key)
	}
	key = strings.TrimSuffix(key, ".pub")
	if info, err := os.Stat(key); err != nil || !info.Mode().IsRegular() {
		return selection, exit.New(exit.Credential, "the rental SSH private key is unavailable; restore the key matching its creation public key")
	}
	directory := filepath.Join(m.layout.Tmp, "runtime-updates", row.ID)
	if err := os.MkdirAll(directory, 0700); err != nil {
		return selection, exit.Internalf("cannot create update staging: %s", err)
	}
	knownHosts := filepath.Join(m.layout.Rentals, row.RentalID, "ssh-known-hosts")
	common := []string{"-oBatchMode=yes", "-oStrictHostKeyChecking=accept-new", "-oUserKnownHostsFile=" + knownHosts, "-oIdentitiesOnly=yes", "-i", key}
	var entropy [16]byte
	if _, err := rand.Read(entropy[:]); err != nil {
		return selection, exit.Internalf("cannot name Runtime update staging: %s", err)
	}
	selection = runtimeUpdateSelection{Target: target, Directory: directory, Stage: hex.EncodeToString(entropy[:]), Host: "root@" + host}
	selection.SSHArguments = append(append([]string{}, common...), "-p", port, selection.Host)
	selection.SFTPArguments = append(append([]string{}, common...), "-P", port)
	return selection, nil
}

func (u *rentalRuntimeUpdates) transport(ctx context.Context, selection runtimeUpdateSelection, action string) (json.RawMessage, *exit.Error) {
	input, _ := json.Marshal(selection)
	var fields map[string]any
	_ = json.Unmarshal(input, &fields)
	fields["action"] = action
	input, _ = json.Marshal(fields)
	command := exec.CommandContext(ctx, "uv", "run", "--isolated", "--no-project", "--no-config", "--python", "3.12", "--with", "packaging==26.2", "python", "-I", "-c", runtimeUpdateTransport)
	command.Env = u.machines.context.Cfg.Tool()
	command.Stdin = bytes.NewReader(input) //cozy:stdin-value bounded maintenance metadata, never user package code
	log, err := os.OpenFile(filepath.Join(selection.Directory, "transport.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return nil, exit.Internalf("cannot retain update diagnostics: %s", err)
	}
	defer log.Close()
	command.Stderr = log
	answer, err := command.Output()
	if len(answer) > 1<<20 {
		return nil, exit.New(exit.Structural, "Runtime update response exceeded its bound")
	}
	if err != nil {
		_, _ = log.Write(answer)
		return nil, exit.Named(exit.Failed, "rental.runtime_update_failed", "Runtime update could not %s; diagnostics: %s", action, log.Name())
	}
	if !json.Valid(answer) {
		return nil, exit.New(exit.Structural, "Runtime update returned invalid metadata")
	}
	return answer, nil
}

func (u *rentalRuntimeUpdates) update(ctx context.Context, row *records.RuntimeUpdate, identity *orchestrator.WorkerConnection) *exit.Error {
	var selection runtimeUpdateSelection
	if len(row.Selection) > 0 {
		if json.Unmarshal(row.Selection, &selection) != nil {
			return exit.New(exit.Structural, "recorded Runtime update selection is unreadable")
		}
		if row.State == "updating" || row.State == "reconciling" {
			return u.reconcile(ctx, row, identity, selection)
		}
	} else {
		var problem *exit.Error
		selection, problem = u.selection(ctx, *row)
		if problem != nil {
			return problem
		}
		selection.Selection, problem = u.transport(ctx, selection, "plan")
		if problem != nil {
			return problem
		}
		var planned struct {
			Unchanged bool `json:"unchanged"`
		}
		_ = json.Unmarshal(selection.Selection, &planned)
		if planned.Unchanged {
			row.State = "succeeded"
			row.Result = selection.Selection
			return nil
		}
		row.Selection, _ = json.Marshal(selection)
		if problem := u.machines.store.SaveRuntimeUpdate(*row); problem != nil {
			return problem
		}
	}
	control, problem := orchestrator.DialIdleControl(ctx, identity, rental.ClaimProof(u.machines.layout), u.machines.store)
	if problem != nil {
		return problem
	}
	defer control.Close()
	row.State = "updating"
	if problem := u.machines.store.SaveRuntimeUpdate(*row); problem != nil {
		return problem
	}
	result, problem := u.transport(ctx, selection, "apply")
	if problem != nil {
		return problem
	}
	_ = control.Close()
	for ctx.Err() == nil {
		check, problem := orchestrator.DialIdleControl(ctx, identity, rental.ClaimProof(u.machines.layout), u.machines.store)
		if problem == nil {
			epoch := check.ControlStreamEpoch
			_ = check.Close()
			if epoch <= control.ControlStreamEpoch {
				return exit.New(exit.Conflict, "updated Runtime did not advance its authenticated control stream")
			}
			row.Result = result
			row.State = "succeeded"
			row.Error = ""
			return nil
		}
		if problem.Code != exit.Unavailable {
			return problem
		}
		select {
		case <-ctx.Done():
			return exit.New(exit.Canceled, "Runtime update observer disconnected")
		case <-time.After(time.Second):
		}
	}
	return exit.New(exit.Canceled, "Runtime update observer disconnected")
}

func handleRentalUpdate(ctx *Context) *exit.Error {
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	defer store.Close()
	row, problem := store.RentalByMachine(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	if row == nil {
		return exit.New(exit.NotFound, "no rental %q on this host", ctx.Inv.Args[0])
	}
	c, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	result, problem := c.UpdateRentalRuntime(row.ID)
	if problem != nil {
		return problem
	}
	for result.Active() {
		time.Sleep(time.Second)
		result, problem = c.RentalRuntimeUpdate(row.ID)
		if problem != nil {
			return problem
		}
		if result.State == "reconciling" {
			return exit.Named(exit.Unavailable, "rental.runtime_update_reconciliation_required", "%s: %s", row.MachineName, result.Error)
		}
	}
	if result.State == "failed" {
		return exit.Named(exit.Failed, "rental.runtime_update_failed", "%s: %s", row.MachineName, result.Error)
	}
	return emit(ctx, output.Record{Fields: []output.Field{{K: "machine", V: row.MachineName}, {K: "state", V: result.State}, {K: "update", V: result.Result}}})
}

func (u *rentalRuntimeUpdates) reconcile(ctx context.Context, row *records.RuntimeUpdate, identity *orchestrator.WorkerConnection, selection runtimeUpdateSelection) *exit.Error {
	for ctx.Err() == nil {
		raw, problem := u.transport(ctx, selection, "inspect")
		if problem == nil {
			var actual runtimeObservation
			if json.Unmarshal(raw, &actual) != nil {
				return exit.New(exit.Structural, "maintenance reconciliation returned invalid observed versions")
			}
			if !actual.Observed.Updating {
				control, problem := orchestrator.DialIdleControl(ctx, identity, rental.ClaimProof(u.machines.layout), u.machines.store)
				if problem != nil {
					return problem
				}
				_ = control.Close()
				if actual.Observed.Runtime.Distribution == selection.Target.RuntimeUpdate.Runtime.Version && actual.Observed.TensorFS == selection.Target.RuntimeUpdate.TensorFS.Version {
					row.State = "succeeded"
					row.Result = raw
					row.Error = ""
					return nil
				}
				var prior runtimeObservation
				_ = json.Unmarshal(selection.Selection, &prior)
				if actual.Observed.Runtime.Distribution == prior.Observed.Runtime.Distribution && actual.Observed.TensorFS == prior.Observed.TensorFS {
					row.State = "failed"
					row.Error = "the update did not activate; the previous healthy Runtime/TensorFS pair is retained"
					return nil
				}
				return exit.New(exit.Conflict, "worker software changed outside this recorded update; maintenance remains closed for inspection")
			}
		}
		select {
		case <-ctx.Done():
			return exit.New(exit.Canceled, "Runtime update observation interrupted")
		case <-time.After(time.Second):
		}
	}
	return exit.New(exit.Canceled, "Runtime update observation interrupted")
}
