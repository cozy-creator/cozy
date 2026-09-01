package cli

import (
	"context"
	"io"
	"sync"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/transfer"
)

// th-094. Private package wheels used to be relayed to the pod as 1 MiB WorkerControl frames --
// up to 1 GiB an operation, through a supervisor that had no reason to hold them. This is the
// owner half of the replacement: the daemon holds the bytes, so the daemon uploads them, and
// the pod is handed one short-lived read capability per wheel instead.
//
// The loop is: ask Tensorhub for grants, PUT whatever it says is missing, ask again. That second
// round is what turns a write capability into a read capability, and it is also how a stalled
// transfer is answered -- re-granting costs one HTTP round and re-uploads nothing, because the
// store already holds the objects.
type privateWheelOwner struct {
	cfg     config.Config
	auth    *accountauth.Manager
	log     io.Writer
	mu      sync.Mutex
	account string
}

func newPrivateWheelOwner(cfg config.Config, log io.Writer,
	auth *accountauth.Manager,
) *privateWheelOwner {
	if log == nil {
		log = io.Discard
	}
	return &privateWheelOwner{cfg: cfg, log: log, auth: auth}
}

func (o *privateWheelOwner) client() *hub.Client {
	return client(&Context{Out: io.Discard, Err: o.log, Cfg: o.cfg, AccountAuth: o.auth})
}

// org is the one namespace these objects live under. It is read from the authenticated user
// rather than from the revision, because a private revision's `local/...` ref is a device-local
// alias that names no Tensorhub organization at all.
func (o *privateWheelOwner) org(ctx context.Context, c *hub.Client) (string, *exit.Error) {
	o.mu.Lock()
	held := o.account
	o.mu.Unlock()
	if held != "" {
		return held, nil
	}
	account, problem := c.CurrentAccount(ctx)
	if problem != nil {
		return "", problem
	}
	o.mu.Lock()
	o.account = account.Name
	o.mu.Unlock()
	return account.Name, nil
}

const privateWheelReason = "transfer a private package revision to a rented worker"

// grants is the orchestrator's PrivateWheelGrantSource.
func (o *privateWheelOwner) grants(ctx context.Context, wheels []orchestrator.PrivateWheel) (
	[]string, *exit.Error,
) {
	c := o.client()
	org, problem := o.org(ctx, c)
	if problem != nil {
		return nil, problem
	}
	asked := make([]hub.PrivateWheelRequest, 0, len(wheels))
	for _, wheel := range wheels {
		asked = append(asked, hub.PrivateWheelRequest{Digest: wheel.Digest,
			Filename: wheel.Filename, Kind: wheel.Kind, Length: wheel.Length})
	}
	granted, problem := c.GrantPrivateWheels(ctx, org, asked, privateWheelReason)
	if problem != nil {
		return nil, problem
	}
	uploaded := 0
	for i, grant := range granted {
		if grant.Upload == nil {
			continue
		}
		// One object, one PUT, pinned to this wheel's own sha256 by the signed checksum
		// condition and to first-write-wins by if-none-match. The grant cannot write
		// anything else and cannot overwrite what is already there.
		if _, problem := transfer.UploadPresigned(ctx, wheels[i].Filename, wheels[i].Path,
			grant.Upload.URL, grant.Upload.RequiredHeaders); problem != nil {
			return nil, problem
		}
		uploaded++
	}
	if uploaded == 0 {
		return readGrants(granted)
	}
	// The second round is not a retry: it is the round that asks for READ capabilities, now
	// that the store holds every object.
	granted, problem = c.GrantPrivateWheels(ctx, org, asked, privateWheelReason)
	if problem != nil {
		return nil, problem
	}
	return readGrants(granted)
}

func readGrants(granted []hub.PrivateWheelGrant) ([]string, *exit.Error) {
	urls := make([]string, 0, len(granted))
	for _, grant := range granted {
		if !grant.Present || grant.Download == "" {
			return nil, exit.Named(exit.Conflict, "private_package.wheel_absent",
				"Tensorhub does not hold private package wheel %s after it was uploaded",
				grant.Filename)
		}
		urls = append(urls, grant.Download)
	}
	return urls, nil
}
