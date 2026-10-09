package machines

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// V1 is one machine's cozy.machine.v1 API, opened as its owner. Close ends the connection and
// any rental use the dial took.
type V1 struct {
	*machinev1.Client
	Name, WorkerID, BootID string
	// Leaf is the machine's pinned TLS leaf (DER): a run's capability binds to its key.
	// Local is this computer's machine, which reads a loopback Hub there.
	Leaf  []byte
	Local bool
	// Rented is a rental's machine: a run that names no Hub reads its own Hub's public content.
	Rented bool
	// HubID is a rental's identity, Owned an owned machine (this computer's), and Account the
	// owner's client at the machine's Hub.
	HubID   string
	Owned   bool
	Account *hub.Client
	release func()
}

// Close ends this use of the machine; its connection stays open for the next.
func (v *V1) Close() {
	if v.release != nil {
		v.release()
	}
}

// connect is the one open connection to a machine identity (name, address, worker, boot,
// process, pinned leaf and signing key): every call to the machine rides it, so only its
// first, or its first after the identity changes, pays a TLS handshake. A connection gRPC
// gave up on is dialed anew; Forget drops one when its rental ends.
func (r *Resolver) connect(name, addr, workerID, bootID, lifetime string, pin *workertls.Pin, signer machinev1.Signer) (*machinev1.Client, error) {
	identity := strings.Join([]string{addr, workerID, bootID, lifetime, string(pin.Digest()), string(signer.Public)}, "\x00")
	r.mu.Lock()
	defer r.mu.Unlock()
	if kept := r.v1[name]; kept != nil {
		if kept.identity == identity && !kept.client.Broken() {
			return kept.client, nil
		}
		_ = kept.client.Close()
		delete(r.v1, name)
	}
	client, err := machinev1.Dial(addr, pin.TLSConfig(), workerID, signer)
	if err != nil {
		return nil, err
	}
	if r.v1 == nil {
		r.v1 = map[string]*keptV1{}
	}
	r.v1[name] = &keptV1{client: client, identity: identity}
	if r.Log != nil {
		fmt.Fprintf(r.Log, "machine %s: connecting to %s (boot %s)\n", name, addr, bootID)
	}
	return client, nil
}

type keptV1 struct {
	client   *machinev1.Client
	identity string
}

// DialV1 resolves a machine as Dial does (this computer's, a rental, or an explicit endpoint)
// and opens its cozy.machine.v1 API, every call carrying a Cozy-Cap the owner key signs.
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
	machine := &placement{}
	t, problem := r.resolve(ctx, name, holder, machine)
	if problem != nil {
		return nil, problem
	}
	client, err := r.connect(name, t.addr, t.workerID, t.bootID, t.lifetime, t.pin, t.key.Signer())
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

// EndpointRental is one of this host's rentals reached as an explicit endpoint.
type EndpointRental struct {
	ID  string
	Key rental.CreatorIdentity
}

// DialEndpointV1 opens an explicit endpoint's cozy.machine.v1 API. An endpoint that is one of
// this host's rentals is signed with that rental's own key, which its machine authorizes, and
// names its Hub identity, as a dialed rental does; any other with this computer's machine
// owner key.
func (r *Resolver) DialEndpointV1(ep machineendpoint.Endpoint) (*V1, *exit.Error) {
	var rented *EndpointRental
	var problem *exit.Error
	if r.EndpointRental != nil {
		if rented, problem = r.EndpointRental(&ep); problem != nil {
			return nil, problem
		}
	}
	key := rental.CreatorIdentity{}
	if rented != nil {
		key = rented.Key
	} else if key, problem = r.Host.ExistingOwner(); problem != nil {
		return nil, problem
	}
	pin, err := workertls.ParsePin([]byte(ep.CertificatePEM))
	if err != nil {
		return nil, exit.New(exit.Credential, "invalid machine TLS leaf: %s", err)
	}
	client, err := r.connect(ep.Name(), ep.Address, ep.WorkerID, ep.WorkerBootID, "", pin, key.Signer())
	if err != nil {
		return nil, Transport(err)
	}
	v := &V1{Client: client, Name: ep.Name(), WorkerID: ep.WorkerID, BootID: ep.WorkerBootID, Leaf: pin.DER(), Rented: rented != nil}
	if rented != nil {
		v.HubID = rented.ID
		if r.RentalHub != nil {
			v.Account = r.RentalHub(rented.ID)
		}
	}
	return v, nil
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
	owner, problem := h.ExistingOwner()
	if problem != nil {
		return nil, problem
	}
	client, err := machinev1.Dial("127.0.0.1:"+strconv.Itoa(record.WorkerPort), pin.TLSConfig(), record.WorkerID, owner.Signer())
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
