package app

import (
	"fmt"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/hub"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/render"
	"github.com/cozy-creator/cozy-creator-v2/internal/rental"
)

// The rental verbs (cl-015). `cozy rent` asks the hub for a pod, watches it provision,
// and PINS what came back — the address, the certificate to trust, and the owner token —
// so that `cozy run --worker <id>` can dial a worker this host never spawned.
//
// The credential rule is the one every other credential here already keeps: the owner
// token is written to a 0600 file and is never printed, logged, or put on argv. What
// these verbs render is its DIGEST, which is comparable against the pod's own without
// either end saying the value.

// pollCadence is how often a provisioning rental is re-read. It is a SAMPLING
// RESOLUTION, not a bound: the wait ends when the hub says `ready` or `failed`, when the
// hub stops answering, or at the caller's own --timeout — never at a number chosen here
// about how long someone else's provider takes to boot a machine.
const pollCadence = 2 * time.Second

func rentalStores(ctx *Context) (home.Layout, *records.Store, *exit.Error) {
	l, e := home.Open(ctx.Cfg.Home)
	if e != nil {
		return home.Layout{}, nil, e
	}
	st, e := records.Open(l.DB)
	if e != nil {
		return home.Layout{}, nil, e
	}
	return l, st, nil
}

func handleRent(ctx *Context) *exit.Error {
	endpoint := strings.TrimSpace(ctx.Inv.Args[0])
	card := strings.TrimSpace(ctx.Inv.Value("--card"))
	if card == "" {
		return exit.Usagef("`cozy rent` names the accelerator to provision").
			WithRemedy("--card is required; the hub decides nothing about hardware for you").
			WithNext("cozy rent " + endpoint + " --card H200 --reason <why>")
	}
	reason := strings.TrimSpace(ctx.Inv.Value("--reason"))
	if reason == "" {
		return exit.Usagef("`cozy rent` spends money and the hub records why before it acts").
			WithRemedy("--reason is required, exactly as it is for every first-party write").
			WithNext("cozy rent " + endpoint + " --card " + card + " --reason <why>")
	}
	// The wait's ONLY caller-supplied bound. Absent, the wait ends on what the hub says
	// rather than on a clock: a pod that is still booting is not a pod that has failed.
	deadline := time.Time{}
	if v := ctx.Inv.Value("--timeout"); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil || d <= 0 {
			return exit.Usagef("--timeout %q is not a positive duration", v)
		}
		deadline = time.Now().Add(d)
	}

	l, st, e := rentalStores(ctx)
	if e != nil {
		return e
	}
	defer st.Close()

	c := client(ctx)
	hctx, cancel := hub.Context()
	r, e := c.Rent(hctx, endpoint, card, reason)
	cancel()
	if e != nil {
		return e
	}
	row := records.Rental{
		ID: r.ID, Endpoint: endpoint, Card: card, PodID: r.PodID,
		State: r.State, Hub: c.Base(),
	}
	if e := st.RecordRental(row); e != nil {
		return e
	}
	ready, e := waitProvisioned(ctx, c, r.ID, deadline)
	if e != nil {
		return e
	}
	row.PodID, row.Address, row.State = ready.PodID, ready.Address, ready.State
	row.MediaAddress = ready.MediaAddress
	if e := rental.Attach(l, st, row, ready.CertPEM, ready.Token); e != nil {
		return e
	}
	return emit(ctx, render.Record{Kind: "rental", Fields: []render.Field{
		{K: "rental", V: ready.ID},
		{K: "state", V: ready.State},
		{K: "address", V: ready.Address},
		{K: "media", V: ready.MediaAddress},
		{K: "pod", V: ready.PodID},
		{K: "endpoint", V: endpoint},
		{K: "card", V: card},
		// The DIGEST, which is the only rendering a credential has here: it is
		// comparable against the pod's own without either end printing the value.
		{K: "owner_token", V: ready.Token.Digest()},
		{K: "pinned_cert", V: l.RentalCert(ready.ID)},
	}, Notes: []string{
		"the owner token is on this disk at mode 0600 and is never printed; the pod holds the same one"},
		Next: []string{"cozy run <org/endpoint/vN/function> --worker " + ready.ID}})
}

// waitProvisioned polls one rental to a settled state. Every wait here is bounded by something
// OBSERVED: the hub's own verdict, the hub failing to answer at all (each call carries
// hub.Timeout), or the caller's --timeout. A rental that is still provisioning is none of
// those, however long the provider takes.
func waitProvisioned(ctx *Context, c *hub.Client, id string, deadline time.Time) (hub.Rental, *exit.Error) {
	said := ""
	for {
		hctx, cancel := hub.Context()
		r, e := c.Rental(hctx, id)
		cancel()
		if e != nil {
			return hub.Rental{}, e
		}
		switch {
		case r.Ready():
			return r, nil
		case r.State == hub.RentalFailed:
			return hub.Rental{}, exit.New(exit.Failed,
				"rental %s failed to provision: %s", id, detailOr(r.Detail)).
				WithRemedy("the pod is the hub's to reclaim; `cozy rent release %s --yes` closes it out", id).
				WithNext("cozy rent release " + id + " --yes")
		case r.State == hub.RentalReady:
			// READY without a whole triple is the hub contradicting itself, and dialling
			// on a partial one would fail later as something that looks like a network
			// fault. Say which piece is missing instead.
			return hub.Rental{}, exit.New(exit.Failed,
				"rental %s is ready and carries no %s", id, missingOf(r)).
				WithRemedy("this hub build may not provision the worker's TLS leg; `cozy hub status` names it")
		}
		// The hub's own words about what is happening, printed when they CHANGE. A line
		// per poll would be a progress bar for someone else's work.
		if r.Detail != "" && r.Detail != said {
			said = r.Detail
			fmt.Fprintf(ctx.Err, "  %s: %s\n", r.State, r.Detail)
		}
		if !deadline.IsZero() && time.Now().After(deadline) {
			return hub.Rental{}, exit.New(exit.Deadline,
				"rental %s was still %s at the --timeout you set", id, r.State).
				WithRemedy("the pod is NOT released; `cozy rent ls` still names it and release destroys it").
				WithNext("cozy rent ls", "cozy rent release "+id+" --yes")
		}
		time.Sleep(pollCadence)
	}
}

func detailOr(detail string) string {
	if detail == "" {
		return "the hub gave no detail"
	}
	return detail
}

// missingOf names the first piece of the dial triple a `ready` rental did not carry.
func missingOf(r hub.Rental) string {
	switch {
	case r.Address == "":
		return "address"
	case r.CertPEM == "":
		return "certificate to pin"
	default:
		return "owner token"
	}
}

func handleRentLs(ctx *Context) *exit.Error {
	l, st, e := rentalStores(ctx)
	if e != nil {
		return e
	}
	defer st.Close()
	rows, e := st.Rentals()
	if e != nil {
		return e
	}
	list := render.List{
		Kind:      "rentals",
		Fields:    []string{"rental", "state", "endpoint", "card", "address"},
		AllFields: []string{"rental", "state", "endpoint", "card", "address", "media", "pod", "hub", "owner_token", "rented"},
		Empty:     "0 rentals on this host",
		Next:      []string{"cozy rent <hub-endpoint> --card <name> --reason <why>"},
	}
	attached := 0
	for _, r := range rows {
		// The token is read only to DIGEST it: a rental whose credential went missing is
		// worth seeing in the listing, because it is a pod that still costs money and can
		// no longer be dialled.
		digest := "unset"
		if v, e := rental.Token(l, r.ID); e == nil {
			digest = v.Digest()
			attached++
		}
		list.Rows = append(list.Rows, map[string]string{
			"rental": r.ID, "state": r.State, "endpoint": r.Endpoint, "card": r.Card,
			"address": r.Address, "media": r.MediaAddress, "pod": r.PodID, "hub": r.Hub,
			"owner_token": digest, "rented": stamp(r.RentedAt),
		})
	}
	if len(list.Rows) > 0 {
		list.Aggregates = []render.Field{
			{K: "rentals", V: len(list.Rows)}, {K: "dialable", V: attached},
		}
		list.Next = []string{"cozy rent release " + list.Rows[0]["rental"] + " --yes"}
	}
	return emit(ctx, list)
}

func handleRentRelease(ctx *Context) *exit.Error {
	id := strings.TrimSpace(ctx.Inv.Args[0])
	l, st, e := rentalStores(ctx)
	if e != nil {
		return e
	}
	defer st.Close()
	row, e := st.RentalRow(id)
	if e != nil {
		return e
	}
	if row == nil {
		return exit.New(exit.NotFound, "no rental %s on this host", id).
			WithRemedy("`cozy rent ls` names the pods this host holds").
			WithNext("cozy rent ls")
	}
	// PLAN FIRST: what release costs is stated before it happens, because the pod and
	// everything resident on it are gone afterwards and there is no undo and no prompt.
	if !ctx.Inv.Bool("--yes") {
		return emit(ctx, render.Record{Kind: "release-plan", Fields: []render.Field{
			{K: "rental", V: row.ID}, {K: "state", V: row.State}, {K: "pod", V: row.PodID},
			{K: "address", V: row.Address}, {K: "media", V: row.MediaAddress},
			{K: "hub", V: row.Hub},
		}, Notes: []string{
			"the hub DESTROYS the pod: anything resident on it is lost and any run pinned to it stops being placeable",
			"the pinned certificate and the owner token are removed from this host with the row",
			"this printed the plan and changed nothing — re-run with --yes",
		}, Next: []string{"cozy rent release " + id + " --yes"}})
	}

	c := client(ctx)
	hctx, cancel := hub.Context()
	e = c.Release(hctx, id, "cozy rent release")
	cancel()
	if e != nil {
		// THE LOCAL HALF STAYS. A pod this host could not reach is a pod that may still be
		// running and still billing; forgetting the row here would leave it with no name
		// anybody could release it by.
		return e.WithRemedy("the local record is KEPT: the pod may still be running, and this row is its name here").
			WithNext("cozy rent ls", "cozy rent release "+id+" --yes")
	}
	forgotten, e := rental.Forget(l, st, id)
	if e != nil {
		return e
	}
	return emit(ctx, render.Record{Kind: "release", Fields: []render.Field{
		{K: "rental", V: id}, {K: "released", V: forgotten}, {K: "pod", V: row.PodID},
	}, Notes: []string{
		"the hub destroyed the pod; its owner token and pinned certificate are gone from this host"},
		Next: []string{"cozy rent ls"}})
}
