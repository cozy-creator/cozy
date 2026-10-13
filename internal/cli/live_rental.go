package cli

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func liveRentalResolver(ctx *Context, store *records.Store) rental.LiveResolver {
	return rental.LiveResolver{Store: store,
		Origins: func() ([]string, *exit.Error) { return rentalInventoryOrigins(ctx, store, ctx.Cfg.HubURL) },
		Client:  func(origin string) *hub.Client { return client(ctx.forHub(origin)) },
	}
}

// observedRental refreshes the endpoint from its owning Hub on every use. Only
// the private signing key is local authority; the PEM file is a materialized pin.
func observedRental(ctx *Context, store *records.Store, id string) (*records.Rental, *exit.Error) {
	call, cancel := hub.Context()
	defer cancel()
	resolver := liveRentalResolver(ctx, store)
	live, problem := resolver.Read(call, id)
	if problem != nil {
		return nil, problem
	}
	remote := live.Rental
	if !remote.Ready() {
		return nil, exit.Named(exit.Unavailable, "rental.not_ready", "rental %s is %s; its Hub has not supplied a ready endpoint", id, remote.State)
	}
	layout := home.Paths(ctx.Cfg.Home)
	key, problem := rental.CreatorIdentityFor(layout, id)
	if problem != nil {
		return nil, problem
	}
	if key.PublicKey() != remote.CreatorPublicKey {
		return nil, exit.Named(exit.Credential, "rental.creator_identity_conflict", "rental %s does not authorize this computer's retained key", id)
	}
	row, problem := resolver.Record(live)
	if problem != nil {
		return nil, problem
	}
	row.CertificatePEM = remote.CertPEM
	return row, nil
}

func resolveLiveRental(ctx *Context, store *records.Store, name string) (rental.Live, *exit.Error) {
	call, cancel := hub.Context()
	defer cancel()
	resolver := liveRentalResolver(ctx, store)
	live, problem := resolver.Resolve(call, name)
	if problem == nil {
		_, problem = resolver.Record(live)
	}
	return live, problem
}
