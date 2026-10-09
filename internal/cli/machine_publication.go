package cli

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machines"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// authorizeV1Publication grants a cozy.machine.v1 machine publication into repositories, bound
// to the machine's leaf key; the machine redeems the code and refreshes the grant itself. A
// rental grants only while the Hub reads it as this exact ready worker.
func authorizeV1Publication(ctx context.Context, machine *machines.V1, origin string, repositories []string) (*v1.HubAuthorization, *exit.Error) {
	if machine.Account == nil || (!machine.Owned && machine.HubID == "") || len(machine.Leaf) == 0 {
		return nil, exit.Named(exit.Structural, "publication.machine_identity_required", "publication authority requires the machine's Hub identity and pinned certificate")
	}
	if origin != "" && machine.Account.Base() != origin {
		return nil, exit.Named(exit.Structural, "publication.source_hub_unsupported",
			"this machine's publication authority is at %s, but this command selected %s", machine.Account.Base(), origin).
			WithRemedy("publish from a machine authorized at %s; no publication was sent to another Hub", origin)
	}
	names, problem := hub.NormalizePublicationRepositories(repositories)
	if problem != nil {
		return nil, problem
	}
	rental := ""
	if !machine.Owned {
		selected, problem := machine.Account.Rental(ctx, machine.HubID)
		if problem != nil {
			return nil, problem
		}
		if selected.ID != machine.HubID || selected.State != hub.RentalReady || selected.WorkerID != machine.WorkerID || selected.WorkerBootID != machine.BootID {
			return nil, exit.New(exit.Conflict, "publication authority rental readback differs from the authenticated machine")
		}
		rental = machine.HubID
	}
	environment, problem := machine.Account.ExecutionEnvironment(ctx)
	if problem != nil {
		return nil, problem
	}
	grant, problem := machine.Account.GrantPublication(ctx, machine.Leaf, environment.Environment["TENSORHUB_PUBLIC_ORIGIN"], rental, hub.PublicationRepositories(names))
	if problem != nil {
		return nil, problem
	}
	return hubAuthorizationV1(grant), nil
}

func hubAuthorizationV1(grant hub.MachineGrant) *v1.HubAuthorization {
	return &v1.HubAuthorization{Issuer: grant.Issuer, Code: grant.Code, CodeVerifier: grant.Verifier,
		RedirectUri: grant.RedirectURI, Resource: grant.Resource}
}
