package machines

import (
	"context"
	"strconv"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// V1 is one machine's cozy.machine.v1 API, opened as its owner. Close ends the connection and
// any rental use the dial took.
type V1 struct {
	*machinev1.Client
	Name, WorkerID, BootID string
	// Leaf is the machine's pinned TLS leaf (DER): a Hub binds the execution access it grants
	// to it. Local is this computer's machine, which reads a loopback Hub there.
	Leaf  []byte
	Local bool
	// Rented is a rental's machine: it reads its own Hub with the pod's capability, so a run
	// sends it no Hub access.
	Rented bool
	// HubID, Owned and Account name the machine at its Hub, as Machine's methods do.
	HubID   string
	Owned   bool
	Account *hub.Client
	release func()
}

func (v *V1) Close() {
	_ = v.Client.Close()
	if v.release != nil {
		v.release()
	}
}

// DialV1 resolves a machine as Dial does (this computer's, a rental, or an explicit endpoint)
// and opens its cozy.machine.v1 API, every call carrying a Cozy-Cap this install's key signs.
// holder names what the caller does there, as a rental's use records it.
func (r *Resolver) DialV1(ctx context.Context, name, holder string) (*V1, *exit.Error) {
	if problem := r.scoped(name); problem != nil {
		return nil, problem
	}
	if machineendpoint.IsName(name) {
		if r.Endpoint == nil {
			return nil, exit.Named(exit.Unavailable, "machine.endpoint_unavailable", "this controller cannot resolve the explicit endpoint")
		}
		ep, problem := r.Endpoint(name)
		if problem == nil && ep == nil {
			problem = exit.New(exit.NotFound, "explicit machine endpoint is not retained")
		}
		if problem != nil {
			return nil, problem
		}
		return r.DialEndpointV1(*ep)
	}
	machine := &Machine{Name: name}
	t, problem := r.resolve(ctx, name, "", holder, true, machine)
	if problem != nil {
		return nil, problem
	}
	client, err := machinev1.Dial(t.addr, t.pin.TLSConfig(), t.workerID, t.key.Signer())
	if err != nil {
		if machine.release != nil {
			machine.release()
		}
		return nil, Transport(err)
	}
	return &V1{Client: client, Name: name, WorkerID: t.workerID, BootID: t.bootID, Leaf: t.pin.DER(),
		Local: t.lifetime != "", Rented: t.lifetime == "", HubID: machine.hubID, Owned: machine.owned, Account: machine.hub,
		release: machine.release}, nil
}

// DialEndpointV1 opens an explicit endpoint's cozy.machine.v1 API. An endpoint that is one of
// this host's rentals is signed with that rental's own key, which its machine authorizes;
// any other with this computer's machine owner key.
func (r *Resolver) DialEndpointV1(ep machineendpoint.Endpoint) (*V1, *exit.Error) {
	rented := false
	if r.EndpointRented != nil {
		var problem *exit.Error
		if rented, problem = r.EndpointRented(&ep); problem != nil {
			return nil, problem
		}
	}
	key, problem := r.Host.Key()
	if problem != nil {
		return nil, problem
	}
	pin, err := workertls.ParsePin([]byte(ep.CertificatePEM))
	if err != nil {
		return nil, exit.New(exit.Credential, "invalid machine TLS leaf: %s", err)
	}
	client, err := machinev1.Dial(ep.Address, pin.TLSConfig(), ep.WorkerID, key.Signer())
	if err != nil {
		return nil, Transport(err)
	}
	return &V1{Client: client, Name: ep.Name(), WorkerID: ep.WorkerID, BootID: ep.WorkerBootID, Leaf: pin.DER(), Rented: rented}, nil
}

// ReadStatus observes this computer's running machine over cozy.machine.v1 without starting
// it; nil when it is not running.
func (h *Host) ReadStatus(ctx context.Context) (*pb.StatusFrame, *exit.Error) {
	record, problem := h.record()
	if problem != nil || record == nil || !h.alive(record.PID) {
		return nil, problem
	}
	pin, problem := h.Pin()
	if problem != nil {
		return nil, problem
	}
	key, problem := h.Authorize()
	if problem != nil {
		return nil, problem
	}
	client, err := machinev1.Dial("127.0.0.1:"+strconv.Itoa(record.WorkerPort), pin.TLSConfig(), record.WorkerID, key.Signer())
	if err != nil {
		return nil, Transport(err)
	}
	defer client.Close()
	frame, err := client.Status(ctx)
	if err != nil {
		return nil, Transport(err)
	}
	return frame, nil
}
