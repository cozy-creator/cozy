package cli

import (
	"slices"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func liveRentalResolver(ctx *Context, store *records.Store) rental.LiveResolver {
	return rental.LiveResolver{Store: store,
		Origins: func() ([]string, *exit.Error) { return selectedHubFirst(ctx, store) },
		Client:  func(origin string) *hub.Client { return client(ctx.forHub(origin)) },
	}
}

// selectedRental is the rental a command sends work to, by ID or name: one this host recorded
// and still holds is used as recorded, asking no Hub (the work's dispatch reads the rental's
// own Hub when it dials it); any other is resolved at the Hubs. Verbs that end or change a
// rental resolve it at its Hub (resolveLiveRental).
func selectedRental(ctx *Context, store *records.Store, name string) (string, *exit.Error) {
	if id := recordedRental(store, name); id != "" {
		return id, nil
	}
	live, problem := resolveLiveRental(ctx, store, name)
	if problem != nil {
		return "", problem
	}
	if hub.RentalAbsent(live.Rental.State) || live.Rental.State == hub.RentalReleaseRequested {
		return "", exit.Named(exit.NotFound, "rental.selection_unavailable", "--rental=%s does not name an existing usable rental", name)
	}
	return live.Rental.ID, nil
}

// recordedRental is the ID of the rental this host recorded and still holds by that ID or name;
// "" for any other.
func recordedRental(store *records.Store, name string) string {
	if row, _ := store.RentalByMachine(name); row != nil && !hub.RentalAbsent(row.State) && row.State != hub.RentalReleaseRequested {
		return row.ID
	}
	return ""
}

// selectedHubFirst is every hub a rental may live on, the command's own first.
func selectedHubFirst(ctx *Context, store *records.Store) ([]string, *exit.Error) {
	origins, problem := rentalInventoryOrigins(ctx, store, ctx.Cfg.HubURL)
	if i := slices.Index(origins, ctx.Cfg.HubURL); i > 0 {
		origins = append([]string{ctx.Cfg.HubURL}, slices.Delete(origins, i, i+1)...)
	}
	return origins, problem
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
	row, problem := resolver.Record(live)
	if problem != nil {
		return nil, problem
	}
	if !remote.Ready() {
		return nil, exit.Named(exit.Unavailable, "rental.not_ready", "rental %s is %s; its Hub has not supplied a ready endpoint", id, remote.State)
	}
	layout := home.Paths(ctx.Cfg.Home)
	key, problem := rental.CreatorIdentityFor(layout, id)
	if problem != nil && row.CertPath == "" && attaching(store, id) {
		// Rented here and just ready: Calcifer attaches it on its next look, and work waits.
		return nil, exit.Named(exit.Unavailable, "rental.not_ready", "rental %s is ready; this computer is attaching it", id)
	}
	if problem != nil {
		return nil, problem
	}
	if key.PublicKey() != remote.CreatorPublicKey {
		return nil, exit.Named(exit.Credential, "rental.creator_identity_conflict", "rental %s does not authorize this computer's retained key", id)
	}
	row.CertificatePEM = remote.CertPEM
	return row, nil
}

// attaching answers whether this host's acquisition of rental id has not attached it yet.
func attaching(store *records.Store, id string) bool {
	operations, _ := store.ActiveRentalOperations()
	return slices.ContainsFunc(operations, func(op records.RentalOperation) bool { return op.RentalID == id && op.State != "attached" })
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

func liveRentalProjection(ctx *Context, store *records.Store, id string) (*records.Rental, *exit.Error) {
	call, cancel := hub.Context()
	defer cancel()
	resolver := liveRentalResolver(ctx, store)
	live, problem := resolver.Read(call, id)
	if problem != nil {
		return nil, problem
	}
	return resolver.Record(live)
}
