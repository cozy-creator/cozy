package cli

import (
	"cmp"
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

// handleRunPlay prints a link that plays one output of a run in any browser, straight from
// the machine that ran it. The address and the DTLS fingerprint come from that machine's own
// Status, read with the key it authorizes: no Hub and no daemon is asked. The capability
// names the run's output and rides the fragment, which never leaves the browser.
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
	if media != nil && media.UnavailableReason != "" {
		return exit.Named(exit.Unavailable, "play.endpoint_configuration", "%s", media.UnavailableReason)
	}
	if media == nil || len(media.Addresses) == 0 {
		return exit.Named(exit.Unavailable, "play.endpoint_unavailable", "this machine reports no direct playback endpoint")
	}
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
	player.Fragment = ""
	linkURL := player.String() + "#" + fragment.Encode()
	return emit(ctx, compactRecord([]output.Field{{K: "run", V: runReference(row.Number, row.ID)}, {K: "output", V: name},
		{K: "expires", V: time.Now().Add(lifetime).UTC().Format(time.RFC3339)}, {K: "link", V: linkURL}}, "link"))
}

// playMachine dials the run's recorded machine (an explicit endpoint, a rental, or this
// computer's machine, which it never starts) with the key that machine authorizes, and reads
// its Status: the media descriptor must name the pinned certificate.
func playMachine(ctx *Context, layout home.Layout, store *records.Store, row records.Request, link *records.MachineExecution) (*machinev1.Client, *pb.StatusFrame, *exit.Error) {
	var address, worker, boot string
	var pin *workertls.Pin
	var key rental.CreatorIdentity
	owner := func() (rental.CreatorIdentity, *exit.Error) {
		host, problem := localMachineHost(ctx)
		if problem != nil {
			return rental.CreatorIdentity{}, problem
		}
		return host.ExistingOwner()
	}
	var problem *exit.Error
	switch {
	case machineendpoint.IsName(link.MachineID):
		ep, p := store.RequestMachineEndpoint(row.ID, link.MachineID)
		if p != nil || ep == nil {
			return nil, nil, cmp.Or(p, exit.New(exit.NotFound, "the run's recorded machine endpoint is gone"))
		}
		var err error
		if pin, err = workertls.ParsePin([]byte(ep.CertificatePEM)); err != nil {
			return nil, nil, exit.New(exit.Credential, "the machine's pinned certificate is unreadable")
		}
		rented, p := rentalEndpoint(layout, store)(ep)
		switch {
		case p != nil:
			problem = p
		case rented != nil:
			key = rented.Key
		default:
			key, problem = owner()
		}
		address, worker = ep.Address, ep.WorkerID // an endpoint's boot id is not the machine's word
	case link.MachineID != "" && link.MachineID != machines.Local:
		rented, p := store.RentalRow(link.MachineID)
		if p != nil || rented == nil || rented.State != "ready" {
			return nil, nil, cmp.Or(p, exit.Named(exit.Unavailable, "play.machine_unavailable", "the run's rental is not ready"))
		}
		var err error
		if pin, err = workertls.LoadPin(rented.CertPath); err != nil {
			return nil, nil, exit.New(exit.Credential, "the rental's pinned certificate is unreadable")
		}
		key, problem = rental.CreatorIdentityFor(layout, rented.ID)
		address, worker, boot = rented.Address, rented.ExpectedWorkerID, rented.ExpectedWorkerBootID
	default:
		host, p := localMachineHost(ctx)
		if p != nil {
			return nil, nil, p
		}
		launch, p := host.Ensure(machines.AttachOnly(context.Background()), nil)
		if p != nil {
			return nil, nil, p
		}
		if pin, problem = host.Pin(); problem == nil {
			key, problem = host.ExistingOwner()
		}
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
		return nil, nil, exit.New(exit.Conflict, "the machine answering is not the run's machine or boot")
	}
	digest := sha256.Sum256(pin.DER())
	if media := frame.GetWebrtc(); media != nil && media.Fingerprint != hex.EncodeToString(digest[:]) {
		client.Close()
		return nil, nil, exit.New(exit.Conflict, "the machine's media certificate is not its pinned certificate")
	}
	return client, frame, nil
}
