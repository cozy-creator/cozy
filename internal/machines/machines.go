// Package machines is where a machine is found: this computer's own Host, or a rented pod's.
// Everything after the dial — Claim, preparation, execution, collection, triage — is one
// path for both, so the name "local" and every venue branch live here and nowhere else.
package machines

import (
	"context"
	"fmt"
	"io"
	"sync"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Local names this computer's machine. Every other machine name is a rental id.
const Local = "local"

// IsLocal answers whether a machine name is this computer's.
func IsLocal(name string) bool { return name == Local }

// Resolver finds machines by name.
type Resolver struct {
	Endpoint func(string) (*machineendpoint.Endpoint, *exit.Error)
	Host     *Host
	// Only, when set, is the one machine this resolver reaches: a foreground run's endpoint.
	// Every other name, this computer's machine included, is refused before anything starts.
	Only string
	// HubOrigin is the hub this computer's machine belongs to unless a call names another,
	// and Hub a client for an origin as the signed-in user.
	HubOrigin string
	Hub       func(origin string) *hub.Client
	// Rentals reads a rental's dial identity; RentalHub is the client for the hub it was
	// bought from; UseRental holds it against release while in use.
	Rentals   func(string) (*orchestrator.RemoteTarget, *exit.Error)
	RentalHub func(string) *hub.Client
	UseRental func(id, holder string) (func(), *exit.Error)
	RentalKey func(string) (rental.CreatorIdentity, *exit.Error)
	// EndpointRental is the rental an explicit endpoint reaches when it is one of this host's:
	// its id and its own creator key, which its machine authorizes. nil: the endpoint is
	// signed with this computer's machine owner key.
	EndpointRental func(*machineendpoint.Endpoint) (*EndpointRental, *exit.Error)

	mu sync.Mutex
	// v1 is each machine's open cozy.machine.v1 connection (connect).
	v1 map[string]*keptV1
	// Log, when set, notes each new machine connection.
	Log io.Writer
}

// Forget drops a machine's open connection: its rental ended or its pod booted anew.
func (r *Resolver) Forget(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if kept := r.v1[name]; kept != nil {
		_ = kept.client.Close()
		delete(r.v1, name)
	}
}

// target is a machine's dial identity before its Claim.
type target struct {
	name, addr, workerID, bootID string
	pin                          *workertls.Pin
	key                          rental.CreatorIdentity
	lifetime                     string // what else ends the worker process: a local Host's pid
}

// scoped refuses a machine outside Only.
func (r *Resolver) scoped(name string) *exit.Error {
	if r.Only == "" || name == r.Only {
		return nil
	}
	return exit.Named(exit.Unavailable, "machine.endpoint_scope", "this foreground run reaches only %s, not machine %s", r.Only, name)
}

// placement is what resolving a machine learns beside its dial identity: its Hub client and
// identity, whether it is owned, and the rental use to release.
type placement struct {
	hub     *hub.Client
	hubID   string
	owned   bool
	release func()
}

// resolve names a machine's dial identity: this computer's (started if needed), or a
// rental's (held for holder until machine.release). The endpoint form is resolved apart.
func (r *Resolver) resolve(ctx context.Context, name, holder string, machine *placement) (target, *exit.Error) {
	if IsLocal(name) {
		var client *hub.Client
		if r.Hub != nil {
			client = r.Hub(r.HubOrigin)
		}
		launch, problem := r.Host.Ensure(ctx, client)
		if problem != nil {
			return target{}, problem
		}
		pin, problem := r.Host.Pin()
		if problem != nil {
			return target{}, problem
		}
		key, problem := r.Host.Owner()
		if problem != nil {
			return target{}, problem
		}
		machine.hub, machine.owned = client, true
		return target{name: name, addr: launch.Addr, workerID: launch.WorkerID, bootID: launch.BootID, pin: pin, key: key,
			lifetime: fmt.Sprint(launch.PID)}, nil
	}
	release, problem := r.UseRental(name, holder)
	if problem != nil {
		return target{}, problem
	}
	machine.release = release
	remote, problem := r.Rentals(name)
	if problem == nil && (remote == nil || remote.Connection == nil) {
		problem = exit.New(exit.Conflict, "rental %s has no dial identity", name)
	}
	if problem != nil {
		release()
		return target{}, problem
	}
	identity := remote.Connection
	pin, err := workertls.LoadPin(identity.CACert)
	if err != nil {
		release()
		return target{}, exit.New(exit.Credential, "machine TLS identity cannot be read")
	}
	key, problem := r.RentalKey(identity.RentalID)
	if problem != nil {
		release()
		return target{}, problem
	}
	machine.hubID = identity.RentalID
	if r.RentalHub != nil {
		machine.hub = r.RentalHub(identity.RentalID)
	}
	return target{name: name, addr: identity.Addr, workerID: identity.WorkerID, bootID: identity.WorkerBootID, pin: pin, key: key}, nil
}

// Transport names a failed machine RPC the way every machine caller reports it.
func Transport(err error) *exit.Error {
	// Redialing never installs a call the peer lacks, so the run ends here.
	if problem := machinev1.Skew(err); problem != nil {
		return problem
	}
	code := status.Code(err)
	if code == codes.Unavailable || code == codes.DeadlineExceeded || code == codes.Canceled || code == codes.ResourceExhausted || code == codes.Aborted {
		return exit.Named(exit.Unavailable, "machine_execution.transport_unavailable", "machine execution observation is unavailable: %s", status.Convert(err).Message())
	}
	return exit.Named(exit.Conflict, "machine_execution.refused", "Runtime refused machine execution: %s", status.Convert(err).Message())
}

// RuntimeUpdate names how the owner updates a machine's Runtime.
func RuntimeUpdate(name string) string {
	if IsLocal(name) {
		return "run `cozy machine install` first"
	}
	return "run `cozy rental update " + name + "` first"
}

// Placement is where a request runs when it names no rental: this computer.
func Placement(request records.Request) string {
	if machineendpoint.IsName(request.Worker) {
		return request.Worker
	}
	if request.Rental {
		return request.Worker
	}
	return Local
}
