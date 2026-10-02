package machines

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// DialEndpoint uses the controller's existing authority and ordinary machine transport.
// The public descriptor selects a target; it supplies no credential or Hub authority.
func (r *Resolver) DialEndpoint(ctx context.Context, ep machineendpoint.Endpoint) (*Machine, *exit.Error) {
	return r.dialEndpointAt(ctx, ep, "")
}
func (r *Resolver) dialEndpointAt(ctx context.Context, ep machineendpoint.Endpoint, origin string) (*Machine, *exit.Error) {
	if err := ep.Validate(); err != nil {
		return nil, exit.New(exit.Validation, "invalid machine endpoint: %s", err)
	}
	key, problem := r.Host.ExistingOwner()
	if problem != nil {
		return nil, problem
	}
	pin, err := workertls.ParsePin([]byte(ep.CertificatePEM))
	if err != nil {
		return nil, exit.New(exit.Credential, "invalid machine TLS leaf: %s", err)
	}
	m := &Machine{Name: ep.Name(), owned: true}
	if problem = m.dial(ctx, target{name: ep.Name(), addr: ep.Address, workerID: ep.WorkerID, bootID: ep.WorkerBootID, pin: pin, key: key}, true); problem != nil {
		return nil, problem
	}
	if problem = m.ValidateNewWork(); problem != nil {
		m.Close()
		return nil, problem
	}
	workspace, err := m.Host.GetMachineExecutionWorkspace(ctx, &pb.MachineExecutionWorkspaceQuery{Claim: m.Claim})
	if err != nil {
		m.Close()
		return nil, Transport(err)
	}
	if workspace.ExecutionWorkspaceId != ep.WorkspaceID || workspace.WorkerId != ep.WorkerID || workspace.WorkerBootId != ep.WorkerBootID {
		m.Close()
		return nil, exit.New(exit.Conflict, "explicit machine endpoint no longer names its frozen workspace or boot")
	}
	m.Seen.Workspace.Store(workspace)
	if origin != "" {
		if r.Hub == nil || r.Hub(origin) == nil {
			m.Close()
			return nil, exit.Named(exit.Credential, "machine.execution_access_required", "explicit endpoint has no authenticated client for its selected Hub")
		}
		account := r.Hub(origin)
		identity := account.CredentialIdentity()
		if identity == "" {
			m.Close()
			return nil, exit.Named(exit.Credential, "machine.execution_access_required", "explicit endpoint catalog use requires the current Hub login")
		}
		access, problem := account.AuthorizeExecutionAccess(ctx, pin.DER())
		if problem != nil {
			m.Close()
			return nil, problem
		}
		if !deviceBoundAccess(access.Token) {
			m.Close()
			return nil, exit.Named(exit.Credential, "hub.execution_access_device_key_required", "Tensorhub execution access must be bound to the current login device")
		}
		reads := machineOrigin(origin, access.Environment["TENSORHUB_ORIGIN"])
		access.Environment["TENSORHUB_ORIGIN"] = reads
		body := map[string]any{"origin": reads, "token": access.Token, "expires_at": access.ExpiresAt.Unix(), "environment": access.Environment}
		if len(access.TrustRoot) > 0 {
			body["ca_der_b64url"] = base64.RawURLEncoding.EncodeToString(access.TrustRoot)
		}
		code, raw, problem := signedMachineAccess(ctx, ep.Address, ep.WorkerID, pin, key, http.MethodPost, body)
		if problem != nil {
			m.Close()
			return nil, problem
		}
		if code/100 != 2 {
			m.Close()
			return nil, exit.Named(exit.Credential, "machine.execution_access_refused", "explicit machine refused scoped access (HTTP %d)", code)
		}
		var acknowledged struct {
			Origin  string `json:"origin"`
			Expires int64  `json:"expires_at"`
		}
		if json.Unmarshal(raw, &acknowledged) != nil || acknowledged.Origin != reads || acknowledged.Expires != access.ExpiresAt.Unix() {
			m.Close()
			return nil, exit.New(exit.Conflict, "explicit endpoint did not acknowledge selected catalog scope")
		}
		if account.CredentialIdentity() != identity {
			m.Close()
			return nil, exit.Named(exit.Credential, "machine.execution_account_changed", "Hub login changed while endpoint access was authorized")
		}
		m.Hub, m.hub = reads, account
	}
	return m, nil
}
