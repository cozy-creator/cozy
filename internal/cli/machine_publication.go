package cli

import (
	"context"
	"crypto/x509"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machines"
)

// machinePublication is a fresh grant intent binding repositories to one machine's leaf for a
// week, or the leaf's remaining life. A rental grants only while the Hub reads it as this
// exact ready worker.
func machinePublication(ctx context.Context, account *hub.Client, machine, machineID string, owned bool, leafDER []byte,
	workerID, bootID string, names []string) (hub.MachinePublicationGrantIntent, *exit.Error) {
	var none hub.MachinePublicationGrantIntent
	if !owned {
		selected, problem := account.Rental(ctx, machineID)
		if problem != nil {
			return none, problem
		}
		if selected.ID != machineID || selected.State != hub.RentalReady || selected.WorkerID != workerID || selected.WorkerBootID != bootID {
			return none, exit.New(exit.Conflict, "publication authority rental readback differs from the authenticated machine")
		}
	}
	leaf, err := x509.ParseCertificate(leafDER)
	if err != nil {
		return none, exit.New(exit.Conflict, "publication authority machine certificate is invalid")
	}
	now := time.Now()
	expires := now.Add(7 * 24 * time.Hour)
	if leaf.NotAfter.Before(expires) {
		expires = leaf.NotAfter
	}
	return hub.PrepareMachinePublicationGrant(machine, machineID, leafDER, names, now, expires)
}

// authorizeV1Publication grants a cozy.machine.v1 machine publication into repositories; the
// machine renews the grant's bearer itself, with its sender proof (the run's execution access, or
// a rental's own worker capability).
func authorizeV1Publication(ctx context.Context, machine *machines.V1, origin string, repositories []string) (string, *exit.Error) {
	if machine.Account == nil || (!machine.Owned && machine.HubID == "") || len(machine.Leaf) == 0 {
		return "", exit.Named(exit.Structural, "publication.machine_identity_required", "publication authority requires the machine's Hub identity and pinned certificate")
	}
	if origin != "" && machine.Account.Base() != origin {
		return "", exit.Named(exit.Structural, "publication.source_hub_unsupported",
			"this machine's publication authority is at %s, but this command selected %s", machine.Account.Base(), origin).
			WithRemedy("publish from a machine authorized at %s; no publication was sent to another Hub", origin)
	}
	names, problem := hub.NormalizePublicationRepositories(repositories)
	if problem != nil {
		return "", problem
	}
	intent, problem := machinePublication(ctx, machine.Account, machine.Name, machine.HubID, machine.Owned, machine.Leaf, machine.WorkerID, machine.BootID, names)
	if problem != nil {
		return "", problem
	}
	if problem := machine.Account.AuthorizeMachinePublication(ctx, intent); problem != nil {
		return "", problem
	}
	return intent.AuthorizationID, nil
}
