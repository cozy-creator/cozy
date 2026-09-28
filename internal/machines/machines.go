// Package machines is where a machine is found: this computer's own Host, or a rented pod's.
// Everything after the dial — Claim, preparation, execution, collection, triage — is one
// path for both, so the name "local" and every venue branch live here and nowhere else.
package machines

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
)

// Local names this computer's machine. Every other machine name is a rental id.
const Local = "local"

// IsLocal answers whether a machine name is this computer's.
func IsLocal(name string) bool { return name == Local }

// Machine is one dialed, claimed machine: a pinned TLS connection to its Host and the
// owner's Claim for its current worker lifetime.
type Machine struct {
	Name      string
	Conn      *grpc.ClientConn
	Host      pb.PodHostClient
	Claim     *pb.Claim
	Protocol  *pb.ProtocolInfoResult
	WireMinor uint32
	// CertificateDigest is the pinned leaf's sha256, the identity publication authority binds.
	CertificateDigest []byte
	// SupportsTriage is whether the Host reads a retained triage bundle for its owner.
	SupportsTriage bool
	// ClaimSurvivesRestart is what this dial's ClaimAck said: the boot's Runtime keeps its
	// owner across Host and Runtime restarts, so later dials need no Control Claim.
	ClaimSurvivesRestart bool

	claimAck  *pb.ClaimAck
	hub       *hub.Client
	hubID     string // the hub's identity for the machine: a rental id or an owned machine id
	owned     bool
	release   func()
	publicOrg string
	kept      bool // the connection is the Resolver's, kept for the machine's next call
	// workspace is what this claimed connection's Runtime reported of itself, shared by
	// every use of the connection and gone with it (Resolver.Forget, a new lifetime).
	workspace *atomic.Pointer[pb.MachineExecutionWorkspace]
}

// Workspace is the execution workspace and capabilities this connection's Runtime
// reported, or nil when none was read yet.
func (m *Machine) Workspace() *pb.MachineExecutionWorkspace { return m.workspace.Load() }

// KeepWorkspace records what the Runtime reported; nil asks it again next time.
func (m *Machine) KeepWorkspace(workspace *pb.MachineExecutionWorkspace) {
	m.workspace.Store(workspace)
}

// Close ends this use of the machine. A kept connection stays open for the next one.
func (m *Machine) Close() error {
	var err error
	if !m.kept {
		err = m.Conn.Close()
	}
	if m.release != nil {
		m.release()
	}
	return err
}

// RentalID is the rental behind a rented machine, and "" for an owned one.
func (m *Machine) RentalID() string {
	if m.owned {
		return ""
	}
	return m.hubID
}

// PrepareFacts are the hub-known release facts for preparing one published package on this
// machine. A rental's join its placed image's inventory; an owned machine runs no registered
// image, so its preparation carries none.
func (m *Machine) PrepareFacts(ctx context.Context, ref *pb.DownloadPackageRef) (orchestrator.PrepareFacts, *exit.Error) {
	if m.owned {
		view, problem := m.hub.MachinePrepareFacts(ctx, m.hubID, ref.Package, ref.Release)
		if problem != nil {
			return orchestrator.PrepareFacts{}, problem
		}
		facts, problem := rental.PrepareFactsFromView(view, ref.Package, ref.Release)
		if problem != nil {
			return orchestrator.PrepareFacts{}, problem
		}
		return withInterface(ctx, m.hub, facts, ref)
	}
	return rental.PrepareFactsSource(m.hub)(ctx, &orchestrator.WorkerConnection{RentalID: m.hubID}, ref)
}

func withInterface(ctx context.Context, client *hub.Client, facts orchestrator.PrepareFacts, ref *pb.DownloadPackageRef) (orchestrator.PrepareFacts, *exit.Error) {
	pkg, problem := hub.ParseRef(ref.Package)
	if problem != nil {
		return orchestrator.PrepareFacts{}, problem
	}
	detail, problem := client.PackageRelease(ctx, pkg, ref.Release)
	if problem != nil {
		return orchestrator.PrepareFacts{}, problem
	}
	facts.PackageInterface, problem = rental.ReleaseInterface(detail, ref.Package, ref.Release)
	return facts, problem
}

// PublicOrigin is where this machine reads public hub bytes: the origin its grant names.
func (m *Machine) PublicOrigin(ctx context.Context) (string, *exit.Error) {
	if m.owned {
		return m.publicOrg, nil
	}
	facts, problem := m.hub.RentalImageInventory(ctx, m.hubID)
	return facts.PublicOrigin, problem
}

// Resolver finds machines by name.
type Resolver struct {
	Host *Host
	// HubOrigin is the hub this computer's machine is registered with, and Hub a client
	// for it as the signed-in user.
	HubOrigin string
	Hub       func() *hub.Client
	// Rentals reads a rental's dial identity; RentalHub is the client for the hub it was
	// bought from; UseRental holds it against release while in use.
	Rentals       func(string) (*orchestrator.RemoteTarget, *exit.Error)
	RentalHub     func(string) *hub.Client
	UseRental     func(id, holder string) (func(), *exit.Error)
	ObserveRental func(orchestrator.RentalObservation) *exit.Error
	RentalKey     func(string) (rental.CreatorIdentity, *exit.Error)
	// Held answers whether this daemon's orchestrator holds the boot's control stream. A
	// worker takes one control stream at a time, so a second Control Claim would fence the
	// orchestrator's; its accepted Claim already names this owner.
	Held func(bootID string) bool

	mu sync.Mutex
	// claimed names the worker lifetimes this daemon has Control Claimed. An older Runtime
	// forgets its owner's protocol level when it restarts, so each lifetime Claims again;
	// durable names the boots whose Runtime reported claim_survives_restart, Claimed once.
	claimed, durable map[string]bool
	// dialing names each machine identity a Dial is connecting to; concurrent Dials wait
	// for it and share its connection and Claim instead of Claiming against each other.
	dialing map[string]chan struct{}
	// kept is one open, claimed connection per machine lifetime: every call to a machine
	// rides it, so a call costs its own round trip and never a TLS handshake or probe.
	kept map[string]*keptMachine
}

type keptMachine struct {
	*Machine
	identity string
}

// Forget drops a machine's kept connection and Claim record. Its next call dials, claims
// and asks the Runtime what it is again: a Runtime update keeps the worker boot, so nothing
// in the dial identity says the Runtime and its capabilities changed.
func (r *Resolver) Forget(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if kept := r.kept[name]; kept != nil {
		kept.Conn.Close()
		delete(r.kept, name)
	}
	for _, record := range []map[string]bool{r.claimed, r.durable} {
		for key := range record {
			if strings.HasPrefix(key, name+"\x00") {
				delete(record, key)
			}
		}
	}
}

// target is a machine's dial identity before its Claim.
type target struct {
	name, addr, workerID, bootID string
	pin                          *workertls.Pin
	key                          rental.CreatorIdentity
	lifetime                     string // what else ends the worker process: a local Host's pid
}

// Dial connects to a machine and authenticates as its owner; holder names what the caller
// is doing there, which a rental's maintenance refusal names. The caller closes it.
func (r *Resolver) Dial(ctx context.Context, name, holder string) (*Machine, *exit.Error) {
	machine := &Machine{Name: name}
	var t target
	if IsLocal(name) {
		var client *hub.Client
		if r.Hub != nil {
			client = r.Hub()
		}
		launch, problem := r.Host.Ensure(ctx, r.HubOrigin, client)
		if problem != nil {
			return nil, problem
		}
		pin, problem := r.Host.Pin()
		if problem != nil {
			return nil, problem
		}
		key, problem := r.Host.Owner()
		if problem != nil {
			return nil, problem
		}
		environment, _ := r.Host.environment(ctx, launch.WorkerID, nil)
		machine.hub, machine.hubID, machine.owned = client, launch.WorkerID, true
		machine.publicOrg = environment["TENSORHUB_PUBLIC_ORIGIN"]
		t = target{name: name, addr: launch.Addr, workerID: launch.WorkerID, bootID: launch.BootID, pin: pin, key: key,
			lifetime: fmt.Sprint(launch.PID)}
	} else {
		release, problem := r.UseRental(name, holder)
		if problem != nil {
			return nil, problem
		}
		machine.release = release
		remote, problem := r.Rentals(name)
		if problem == nil && (remote == nil || remote.Connection == nil) {
			problem = exit.New(exit.Conflict, "rental %s has no dial identity", name)
		}
		if problem != nil {
			release()
			return nil, problem
		}
		identity := remote.Connection
		pin, err := workertls.LoadPin(identity.CACert)
		if err != nil {
			release()
			return nil, exit.New(exit.Credential, "machine TLS identity cannot be read")
		}
		key, problem := r.RentalKey(identity.RentalID)
		if problem != nil {
			release()
			return nil, problem
		}
		machine.hubID = identity.RentalID
		if r.RentalHub != nil {
			machine.hub = r.RentalHub(identity.RentalID)
		}
		t = target{name: name, addr: identity.Addr, workerID: identity.WorkerID, bootID: identity.WorkerBootID, pin: pin, key: key}
	}
	lifetime := name + "\x00" + t.bootID + "\x00" + t.lifetime
	identity := lifetime + "\x00" + t.addr + "\x00" + t.workerID + "\x00" + string(t.pin.Digest())
	r.mu.Lock()
	if r.claimed == nil {
		r.claimed, r.durable, r.kept = map[string]bool{}, map[string]bool{}, map[string]*keptMachine{}
		r.dialing = map[string]chan struct{}{}
	}
	for {
		if kept := r.kept[name]; kept != nil && kept.identity == identity {
			r.mu.Unlock()
			use := *kept.Machine
			use.release, use.kept = machine.release, true
			return &use, nil
		}
		dialing := r.dialing[identity]
		if dialing == nil {
			break
		}
		r.mu.Unlock()
		select {
		case <-dialing:
		case <-ctx.Done():
			if machine.release != nil {
				machine.release()
			}
			return nil, Transport(status.FromContextError(ctx.Err()).Err())
		}
		r.mu.Lock()
	}
	done := make(chan struct{})
	r.dialing[identity] = done
	defer func() {
		r.mu.Lock()
		delete(r.dialing, identity)
		r.mu.Unlock()
		close(done)
	}()
	boot := name + "\x00" + t.bootID
	claimed := r.claimed[lifetime] || r.durable[boot] || !machine.owned && r.Held != nil && r.Held(t.bootID)
	r.mu.Unlock()
	if problem := machine.dial(ctx, t, !claimed); problem != nil {
		if machine.release != nil {
			machine.release()
		}
		return nil, problem
	}
	r.mu.Lock()
	r.claimed[lifetime] = true
	if machine.ClaimSurvivesRestart {
		r.durable[boot] = true
	}
	if previous := r.kept[name]; previous != nil {
		previous.Conn.Close() // the machine's earlier lifetime
	}
	kept := *machine
	kept.release, kept.claimAck = nil, nil
	r.kept[name] = &keptMachine{Machine: &kept, identity: identity}
	machine.kept = true
	r.mu.Unlock()
	if ack := machine.claimAck; ack != nil && !machine.owned && r.ObserveRental != nil {
		// The first Claim of a rented worker's lifetime reads back what the pod is.
		if problem := r.ObserveRental(orchestrator.ObservationFromClaimAck(machine.hubID, ack)); problem != nil {
			machine.Close()
			return nil, problem
		}
	}
	return machine, nil
}

func (m *Machine) dial(ctx context.Context, t target, controlClaim bool) *exit.Error {
	connection, err := grpc.NewClient(t.addr, grpc.WithTransportCredentials(credentials.NewTLS(t.pin.TLSConfig())),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(16<<20), grpc.MaxCallSendMsgSize(16<<20)))
	if err != nil {
		return Transport(err)
	}
	m.Conn, m.Host, m.CertificateDigest = connection, pb.NewPodHostClient(connection), t.pin.Digest()
	m.workspace = &atomic.Pointer[pb.MachineExecutionWorkspace]{}
	probe, cancel := context.WithTimeout(ctx, 10*time.Second)
	info, err := m.Host.ProtocolInfo(probe, &pb.ProtocolInfoRequest{})
	cancel()
	if err != nil {
		connection.Close()
		return Transport(err)
	}
	m.Protocol, m.WireMinor, m.SupportsTriage = info, info.WireMinor, info.SupportsMachineExecutionTriage
	claim, ack, problem := claim(ctx, connection, t, info.WireMinor, controlClaim)
	if problem != nil {
		connection.Close()
		return problem
	}
	m.Claim, m.claimAck, m.ClaimSurvivesRestart = claim, ack, ack.GetClaimSurvivesRestart()
	return nil
}

// claim authenticates this owner to the worker lifetime the target names: the owner key's
// Ed25519 signature over ClaimProof/1, presented on every call. The first connection to a
// lifetime also opens one Control Claim, which records the owner and its protocol level.
// A boot whose Runtime reported claim_survives_restart is Claimed once, not per lifetime.
func claim(ctx context.Context, connection *grpc.ClientConn, t target, wireMinor uint32, control bool) (*pb.Claim, *pb.ClaimAck, *exit.Error) {
	transcript, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: orchestrator.RecordOwnerEpoch,
		WorkerId: t.workerID, WorkerBootId: t.bootID, WorkerTlsCertificateDigest: t.pin.Digest()})
	if err != nil {
		return nil, nil, exit.Internalf("cannot author the machine ClaimProof: %s", err)
	}
	claim := &pb.Claim{RecordOwnerEpoch: orchestrator.RecordOwnerEpoch, RecordOwnerId: orchestrator.RecordOwnerID,
		WorkerId: t.workerID, WorkerBootId: t.bootID, WireMinor: min(pb.WireMinor, wireMinor), Proof: t.key.Sign(transcript)}
	if !control {
		return claim, nil, nil
	}
	ack, problem := controlClaim(ctx, pb.NewWorkerControlClient(connection), claim)
	if problem != nil {
		return nil, nil, problem
	}
	return claim, ack, nil
}

func controlClaim(parent context.Context, client pb.WorkerControlClient, claim *pb.Claim) (*pb.ClaimAck, *exit.Error) {
	ctx, cancel := context.WithTimeout(parent, 30*time.Second)
	defer cancel()
	stream, err := client.Control(ctx)
	if err != nil {
		return nil, Transport(err)
	}
	defer stream.CloseSend()
	if err := stream.Send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_Claim{Claim: claim}}); err != nil {
		return nil, Transport(err)
	}
	for {
		frame, err := stream.Recv()
		if err != nil {
			return nil, Transport(err)
		}
		if failure := frame.GetBootFailure(); failure != nil {
			return nil, exit.Named(exit.Structural, "machine.boot_failed", "the machine's Runtime failed to boot: %s", failure.String())
		}
		ack := frame.GetClaimAck()
		if ack == nil {
			continue
		}
		if !ack.Accepted || ack.WorkerId != claim.WorkerId || ack.WorkerBootId != claim.WorkerBootId {
			return nil, exit.New(exit.Credential, "the machine refused its owner's Claim (%s)", ack.Rejection)
		}
		return ack, nil // no SnapshotAck: this stream only records the owner
	}
}

// ValidateNewWork gates a new preparation or execution on the machine's protocol range and
// idle guard. Observation, collection, cancellation and release reach any peer.
func (m *Machine) ValidateNewWork() *exit.Error {
	if !m.owned {
		return orchestrator.ValidateWorkerProtocol(m.Protocol, m.Name)
	}
	if problem := orchestrator.ValidateWorkerProtocol(m.Protocol, ""); problem != nil {
		return problem.WithRemedy("install the current worker cohort: cozy machine install --host <pod-supervisor>")
	}
	if !m.Protocol.SupportsRentalKeepalive {
		return exit.Named(exit.Conflict, "worker.rental_idle_guard_required", "this machine's Host reports no idle guard").
			WithRemedy("install the current worker cohort: cozy machine install --host <pod-supervisor>")
	}
	return nil
}

// Transport names a failed machine RPC the way every machine caller reports it.
func Transport(err error) *exit.Error {
	code := status.Code(err)
	if code == codes.Unimplemented {
		return exit.Named(exit.Unavailable, "machine_execution.worker_upgrade_required", "worker does not implement workspace-fenced execution; worker protocol 59 is required")
	}
	if code == codes.Unavailable || code == codes.DeadlineExceeded || code == codes.Canceled || code == codes.ResourceExhausted || code == codes.Aborted {
		return exit.Named(exit.Unavailable, "machine_execution.transport_unavailable", "machine execution observation is unavailable: %s", status.Convert(err).Message())
	}
	return exit.Named(exit.Conflict, "machine_execution.refused", "Runtime refused machine execution: %s", status.Convert(err).Message())
}

// RuntimeUpdate names how the owner updates a machine's Runtime.
func RuntimeUpdate(name string) string {
	if IsLocal(name) {
		return "run `cozy machine install --host <pod-supervisor>` first"
	}
	return "run `cozy rental update " + name + "` first"
}

// Placement is where a request runs when it names no rental: this computer.
func Placement(request records.Request) string {
	if request.Rental {
		return request.Worker
	}
	return Local
}
