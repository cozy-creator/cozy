package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net"
	"net/url"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// Playback is observation on the recorded machine. Its endpoint/pin and the locally held
// owner key suffice; no Hub or default-daemon upgrade is required.
func handleRunPlay(ctx *Context) *exit.Error {
	layout, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	defer store.Close()
	row, problem := store.RequestByReference(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	if row == nil {
		return exit.New(exit.NotFound, "no recorded run %q", ctx.Inv.Args[0])
	}
	link, problem := store.MachineExecution(row.ID)
	if problem != nil {
		return problem
	}
	accepted, problem := store.RunV1(row.ID)
	if problem != nil {
		return problem
	}
	if link == nil || !accepted {
		return exit.Named(exit.Conflict, "play.not_started", "run %s has no accepted v1 machine execution", runReference(row.Number, row.ID))
	}
	name := strings.TrimSpace(ctx.Inv.Value("--output"))
	lifetime, err := time.ParseDuration(ctx.Inv.Value("--expires"))
	if name == "" || strings.ContainsAny(name, "#&= \t\r\n") {
		return exit.Usagef("--output names one output or name/index")
	}
	if err != nil || lifetime <= 0 {
		return exit.Usagef("--expires is a positive duration")
	}
	machine, frame, problem := playMachine(ctx, layout, store, *row, link)
	if problem != nil {
		return problem
	}
	defer machine.Close()
	media := frame.GetWebrtc()
	if media == nil || len(media.Addresses) == 0 {
		return exit.Named(exit.Unavailable, "play.endpoint_unavailable", "this machine reports no direct playback endpoint")
	}
	// The Status is authenticated against the TLS pin. Its media descriptor names that
	// same machine certificate, preventing a malformed descriptor from redirecting trust.
	address := ""
	for _, candidate := range media.Addresses {
		if _, _, err := net.SplitHostPort(candidate); err == nil {
			address = candidate
			break
		}
	}
	if address == "" {
		return exit.Named(exit.Unavailable, "play.endpoint_unreachable", "the machine reports no usable direct address; NAT requires a mapped endpoint")
	}
	token, err := machine.Cap(row.ID, []string{name}, lifetime)
	if err != nil {
		return exit.New(exit.Credential, "cannot sign this run's output grant: %s", err)
	}
	fragment := url.Values{"v": {"1"}, "a": {address}, "f": {media.Fingerprint}, "c": {token}, "r": {row.ID}, "o": {name}}
	player, err := url.Parse(ctx.Cfg.PlayerURL)
	if err != nil {
		return exit.New(exit.Validation, "player URL is invalid: %s", err)
	}
	player.Fragment = fragment.Encode()
	return emit(ctx, compactRecord([]output.Field{{K: "run", V: runReference(row.Number, row.ID)}, {K: "output", V: name},
		{K: "expires", V: time.Now().Add(lifetime).UTC().Format(time.RFC3339)}, {K: "link", V: player.String()}}, "link"))
}
func playMachine(ctx *Context, layout home.Layout, store *records.Store, row records.Request, link *records.MachineExecution) (*machinev1.Client, *pb.StatusFrame, *exit.Error) {
	var address, worker, boot string
	var pin *workertls.Pin
	var key rental.CreatorIdentity
	var problem *exit.Error
	if machineendpoint.IsName(link.MachineID) {
		ep, problem := store.RequestMachineEndpoint(row.ID, link.MachineID)
		if problem != nil {
			return nil, nil, problem
		}
		if ep == nil {
			return nil, nil, exit.New(exit.NotFound, "the recorded machine endpoint is absent")
		}
		var err error
		pin, err = workertls.ParsePin([]byte(ep.CertificatePEM))
		if err != nil {
			return nil, nil, exit.New(exit.Credential, "machine pin is unreadable")
		}
		found, problem := rentalEndpointKey(layout, store)(ep)
		if problem != nil {
			return nil, nil, problem
		}
		if found != nil {
			key = *found
		} else {
			host, p := localMachineHost(ctx)
			if p != nil {
				return nil, nil, p
			}
			key, problem = host.ExistingOwner()
		}
		address, worker, boot = ep.Address, ep.WorkerID, ep.WorkerBootID
	} else if link.MachineID != "" && link.MachineID != machines.Local {
		rented, problem := store.RentalRow(link.MachineID)
		if problem != nil {
			return nil, nil, problem
		}
		if rented == nil || rented.State != "ready" {
			return nil, nil, exit.Named(exit.Unavailable, "play.machine_unavailable", "the recorded rental is not ready")
		}
		var err error
		pin, err = workertls.LoadPin(rented.CertPath)
		if err != nil {
			return nil, nil, exit.New(exit.Credential, "rental pin is unreadable")
		}
		key, problem = rental.CreatorIdentityFor(layout, rented.ID)
		address, worker, boot = rented.Address, rented.ExpectedWorkerID, rented.ExpectedWorkerBootID
	} else {
		host, p := localMachineHost(ctx)
		if p != nil {
			return nil, nil, p
		}
		launch, p := host.Ensure(machines.AttachOnly(context.Background()), "", nil, false)
		if p != nil {
			return nil, nil, p
		}
		pin, problem = host.Pin()
		if problem != nil {
			return nil, nil, exit.New(exit.Credential, "local machine pin is unreadable")
		}
		key, problem = host.ExistingOwner()
		address, worker, boot = launch.Addr, launch.WorkerID, launch.BootID
	}
	if problem != nil {
		return nil, nil, problem
	}
	client, err := machinev1.Dial(address, pin.TLSConfig(), worker, key.Signer())
	if err != nil {
		return nil, nil, machines.Transport(err)
	}
	frame, err := client.Status(context.Background())
	if err != nil {
		client.Close()
		return nil, nil, machines.Transport(err)
	}
	if frame.WorkerId != worker || boot != "" && frame.BootId != boot {
		client.Close()
		return nil, nil, exit.New(exit.Conflict, "playback Status names a different machine or boot")
	}
	digest := sha256.Sum256(pin.DER())
	if media := frame.GetWebrtc(); media != nil && media.Fingerprint != hex.EncodeToString(digest[:]) {
		client.Close()
		return nil, nil, exit.New(exit.Conflict, "playback fingerprint does not match the pinned machine")
	}
	return client, frame, nil
}
