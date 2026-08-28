package app

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/records"
	"github.com/cozy-creator/cozy-creator/internal/render"
	"github.com/cozy-creator/cozy-creator/internal/rental"
	"github.com/cozy-creator/cozy-creator/internal/secret"
)

// The rental verbs (cl-015). `cozy rent` MINTS the pod's access token, asks the hub for a
// pod while presenting only that token's sha256, watches it provision, and PINS the
// triple — the address, the certificate to trust, and the token it minted — so that
// `cozy run --worker <id>` can dial a worker this host never spawned.
//
// THE MINT IS THE POINT (#495e). The token is generated here, from this host's own
// entropy, and what crosses to the hub is a hash. The hub provisions the pod with that
// hash, the pod checks a presented credential by hashing it, and neither the hub nor the
// provider ever holds a value that would let it authenticate as this host to this host's
// own pod. The old contract had the hub mint the token and hand it back on a GET, which
// made every party in the chain — and every log line along it — a holder of the
// credential.
//
// The credential rule is the one every other credential here already keeps: the token is
// written to a 0600 file and is never printed, logged, or put on argv. What these verbs
// render is its DIGEST, which is comparable against the pod's own without either end
// saying the value.

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
	endpointRef := strings.TrimSpace(ctx.Inv.Args[0])
	acceleratorModel := strings.TrimSpace(ctx.Inv.Value("--accelerator"))
	if acceleratorModel == "" {
		return exit.Usagef("`cozy rent` names the accelerator to provision").
			WithRemedy("--accelerator is required; name a provider-neutral model such as NVIDIA H200").
			WithNext("cozy rent " + endpointRef + " --accelerator 'NVIDIA H200' --reason <why>")
	}
	reason := strings.TrimSpace(ctx.Inv.Value("--reason"))
	if reason == "" {
		return exit.Usagef("`cozy rent` spends money and the hub records why before it acts").
			WithRemedy("--reason is required, exactly as it is for every first-party write").
			WithNext("cozy rent " + endpointRef + " --accelerator '" + acceleratorModel + "' --reason <why>")
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
	if !c.Token().Present() {
		return exit.Named(exit.Credential, "hub.token_missing",
			"POST /v1/private-rentals is a first-party route and no admin token is configured").
			WithRemedy("set TENSORHUB_TOKEN to the hub's admin.token; catalog reads need no credential").
			WithNext("cozy hub status")
	}
	operationKey := strings.TrimSpace(ctx.Inv.Value("--idempotency-key"))
	if len(operationKey) > 200 {
		return exit.Usagef("--idempotency-key is %d bytes; the hub admits at most 200", len(operationKey))
	}
	if operationKey == "" {
		// Identical intent is not identity: a user may deliberately rent two
		// identical pods. Only an explicit key may coalesce two invocations.
		operationKey = mintKey()
	}

	// THE DURABLE MINT, BEFORE THE ASK. O_EXCL makes concurrent same-key callers read one
	// token, and the operation row makes a lost HTTP response resumable. Only the token's
	// hash crosses the wire.
	existing, e := st.RentalOperation(operationKey)
	if e != nil {
		return e
	}
	if existing != nil && (existing.State == "rejected" || existing.State == "released") {
		return exit.Named(exit.Conflict, "rental.operation_settled",
			"rental operation %s is already %s", operationKey, existing.State).
			WithRemedy("use a fresh --idempotency-key for a new paid rental")
	}
	if existing != nil && (existing.State == "failed" || existing.State == "release_requested") {
		return exit.Named(exit.Conflict, "rental.release_required",
			"rental operation %s is %s and still names rental %s", operationKey, existing.State, existing.RentalID).
			WithRemedy("release the existing rental before starting another operation").
			WithNext("cozy rent release " + existing.RentalID + " --yes")
	}
	var token secret.Value
	if existing != nil && existing.State == "attached" && existing.RentalID != "" {
		token, e = rental.Token(l, existing.RentalID)
	} else {
		token, e = rental.PendingToken(l, operationKey)
	}
	if e != nil {
		return e
	}
	tokenHash := secret.HashHex(token)
	requestBody, e := hub.RentalRequestBytes(endpointRef, acceleratorModel, tokenHash)
	if e != nil {
		return e
	}
	digest := rentalRequestDigest(c.Base(), requestBody)
	op, replay, e := st.BeginRentalOperation(records.RentalOperation{
		Key: operationKey, RequestDigest: digest, RequestBody: requestBody,
		Hub: c.Base(), Reason: reason,
	})
	if e != nil {
		return e
	}
	if op.RequestDigest != digest || op.Hub != c.Base() || !bytes.Equal(op.RequestBody, requestBody) {
		return exit.Named(exit.Conflict, "rental.idempotency_conflict",
			"rental operation %s already names a different hub or request body", operationKey).
			WithRemedy("reuse a key only for the exact same hub, endpoint, accelerator, and renter token")
	}
	if !replay {
		fmt.Fprintf(ctx.Err, "  rental operation %s persisted; reuse this key to resume\n", operationKey)
	}
	// The first recorded reason is the paid operation's audit reason. A retry may be typed
	// with different prose, but it must not rewrite why the original purchase was made.
	reason = op.Reason

	hctx, cancel := hub.LongContext()
	r, e := c.Rent(hctx, op.RequestBody, reason, operationKey)
	cancel()
	if e != nil {
		// A typed 4xx answer proves this POST bought nothing. Transport, deadline,
		// unreadable-success, and 5xx failures remain pending because the hub may have
		// committed before the answer was lost.
		if e.Code == exit.Credential || e.Code == exit.Validation ||
			e.Code == exit.NotFound || e.Code == exit.Conflict {
			if advanced := st.AdvanceRentalOperation(operationKey, "", "rejected"); advanced != nil {
				return advanced
			}
			rental.ForgetPending(l, operationKey)
		}
		return e
	}
	if e := st.AdvanceRentalOperation(operationKey, r.ID, r.State); e != nil {
		return e
	}
	row := records.Rental{
		ID: r.ID, EndpointRef: endpointRef, AcceleratorModel: acceleratorModel,
		State: r.State, Hub: c.Base(),
	}
	if e := st.RecordRental(row); e != nil {
		return e
	}
	observe := func(seen hub.Rental) *exit.Error {
		row.Address, row.State = seen.Address, seen.State
		row.MediaAddress = seen.MediaAddress
		captureRentalControl(&row, seen)
		if len(row.ControlSnapshotBytes) > 0 {
			if e := rental.ValidateControl(row); e != nil {
				return e
			}
		}
		if e := st.RecordRental(row); e != nil {
			return e
		}
		return st.AdvanceRentalOperation(operationKey, seen.ID, seen.State)
	}
	ready, e := waitProvisioned(ctx, c, r.ID, deadline, observe)
	if e != nil {
		return e
	}
	row.Address, row.State = ready.Address, ready.State
	row.MediaAddress = ready.MediaAddress
	captureRentalControl(&row, ready)
	if !ready.HoldsHash(secret.HashHex(token)) {
		// The pod was provisioned with a credential set this host's token is not in, so
		// dialling it would 401 and look like a network fault. The hub says which hashes
		// are live; neither end has to say a token to find this out.
		return exit.New(exit.Failed,
			"rental %s is ready and its live credential set does not carry the token this host minted", ready.ID).
			WithRemedy("release it and rent again; a pod nobody can authenticate to still costs money").
			WithNext("cozy rent release " + ready.ID + " --yes")
	}
	if e := rental.Attach(l, st, row, ready.CertPEM, token); e != nil {
		return e
	}
	if e := st.AdvanceRentalOperation(operationKey, ready.ID, "attached"); e != nil {
		return e
	}
	rental.ForgetPending(l, operationKey)
	notes := []string{
		"this host MINTED the owner token and holds it at mode 0600; the hub and the pod hold only its sha256, so neither can dial this pod as you",
		"rental operation " + operationKey,
	}
	if replay {
		notes[1] += " resumed"
	}
	return emit(ctx, render.Record{Kind: "rental", Fields: []render.Field{
		{K: "rental", V: ready.ID},
		{K: "state", V: ready.State},
		{K: "address", V: ready.Address},
		{K: "media", V: ready.MediaAddress},
		{K: "endpoint", V: endpointRef},
		{K: "accelerator", V: acceleratorModel},
		// The DIGEST, which is the only rendering a credential has here: it is
		// comparable against the pod's own without either end printing the value.
		{K: "owner_token", V: token.Digest()},
		{K: "pinned_cert", V: l.RentalCert(ready.ID)},
	}, Notes: notes,
		Next: []string{"cozy run <org/endpoint/vN/function> --worker " + ready.ID}})
}

func rentalRequestDigest(hubAuthority string, requestBody []byte) string {
	h := sha256.New()
	_, _ = h.Write([]byte("cozy.rental_request/1\n"))
	_, _ = h.Write([]byte(strings.TrimRight(hubAuthority, "/")))
	_, _ = h.Write([]byte{'\n'})
	_, _ = h.Write(requestBody)
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

// transient answers whether a hub failure says nothing about the rental: the hub was
// unreachable, stalled, or answered 5xx. A typed 4xx is the hub's verdict and is not.
func transient(e *exit.Error) bool {
	return e.Code == exit.Unavailable || e.Code == exit.Deadline
}

// waitProvisioned polls one rental to a settled state. Every wait here is bounded by something
// OBSERVED: the hub's own verdict, a typed refusal, or the caller's --timeout. A rental
// that is still acquiring or materializing is none of those, however long the provider
// takes, and a hub that is momentarily unreachable is asked again at the same cadence.
func waitProvisioned(ctx *Context, c *hub.Client, id string, deadline time.Time, observe func(hub.Rental) *exit.Error) (hub.Rental, *exit.Error) {
	said := ""
	for {
		hctx, cancel := hub.LongContext()
		r, e := c.Rental(hctx, id)
		cancel()
		if e != nil && !transient(e) {
			return hub.Rental{}, e
		}
		if e != nil {
			if e.Message != said {
				said = e.Message
				fmt.Fprintf(ctx.Err, "  hub: %s; retrying\n", e.Message)
			}
			if timedOut := pastDeadline(id, "unreachable", deadline); timedOut != nil {
				return hub.Rental{}, timedOut
			}
			time.Sleep(pollCadence)
			continue
		}
		if e := observe(r); e != nil {
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
		case r.State == hub.RentalReleaseRequested || r.State == hub.RentalReleased:
			// A rental that is LEAVING is not one that is still coming up, and waiting on
			// it is waiting for a state it will never reach.
			return hub.Rental{}, exit.New(exit.Failed,
				"rental %s is %s: %s", id, r.State, detailOr(r.Detail)).
				WithRemedy("the rental is leaving or gone; rent again if you still need a pod").
				WithNext("cozy rent <endpoint> --accelerator <model> --reason <why>")
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
		if timedOut := pastDeadline(id, r.State, deadline); timedOut != nil {
			return hub.Rental{}, timedOut
		}
		time.Sleep(pollCadence)
	}
}

func pastDeadline(id, state string, deadline time.Time) *exit.Error {
	if deadline.IsZero() || !time.Now().After(deadline) {
		return nil
	}
	return exit.New(exit.Deadline,
		"rental %s was still %s at the --timeout you set", id, state).
		WithRemedy("the pod is NOT released; `cozy rent ls` still names it and release destroys it").
		WithNext("cozy rent ls", "cozy rent release "+id+" --yes")
}

func detailOr(detail string) string {
	if detail == "" {
		return "the hub gave no detail"
	}
	return detail
}

// missingOf names the first piece a `ready` rental did not carry. The hub never carries
// the credential, but it must carry the pod's observed HASH set so this host can compare.
func missingOf(r hub.Rental) string {
	if r.Address == "" {
		return "worker address"
	}
	if r.MediaAddress == "" {
		return "media address"
	}
	if r.CertPEM == "" {
		return "certificate to pin"
	}
	if len(r.TokenSHA256) == 0 {
		return "observed renter-token hash set"
	}
	if r.ControlSnapshot == nil {
		return "attempt-bound control snapshot"
	}
	return "complete ready projection"
}

func captureRentalControl(row *records.Rental, seen hub.Rental) {
	if row == nil || seen.ControlSnapshot == nil {
		return
	}
	row.ControlSnapshotDigest = seen.ControlSnapshot.Digest
	row.ControlSnapshotBytes = append([]byte(nil), seen.ControlSnapshot.CanonicalBytes...)
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
		Fields:    []string{"rental", "state", "endpoint", "accelerator", "address"},
		AllFields: []string{"rental", "state", "endpoint", "accelerator", "address", "media", "hub", "owner_token", "rented"},
		Empty:     "0 rentals on this host",
		Next:      []string{"cozy rent <endpoint-ref> --accelerator <model> --reason <why>"},
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
			"rental": r.ID, "state": r.State, "endpoint": r.EndpointRef,
			"accelerator": r.AcceleratorModel, "address": r.Address,
			"media": r.MediaAddress, "hub": r.Hub,
			"owner_token": digest, "rented": stamp(r.RentedAt),
		})
	}
	if len(list.Rows) > 0 {
		list.Aggregates = []render.Field{
			{K: "rentals", V: len(list.Rows)}, {K: "dialable", V: attached},
		}
		list.Next = []string{"cozy rent show <rental-id>", "cozy rent release <rental-id>"}
	}
	return emit(ctx, list)
}

func handleRentShow(ctx *Context) *exit.Error {
	_, st, e := rentalStores(ctx)
	if e != nil {
		return e
	}
	defer st.Close()
	row, control, e := rental.Inspect(st, strings.TrimSpace(ctx.Inv.Args[0]))
	if e != nil {
		return e
	}
	return emit(ctx, render.Record{Kind: "rental_control", Fields: []render.Field{
		{K: "rental", V: row.ID}, {K: "state", V: row.State},
		{K: "endpoint", V: row.EndpointRef}, {K: "accelerator", V: row.AcceleratorModel},
		{K: "observed_accelerator", V: row.ObservedAccelerator},
		{K: "observed_accelerator_count", V: row.ObservedAcceleratorCount},
		{K: "observed_backend", V: row.ObservedBackend},
		{K: "observed_worker_instance", V: row.ObservedWorkerInstance},
		{K: "observed_worker_boot_id", V: row.ObservedWorkerBootID},
		{K: "observed_at", V: row.ObservedAt},
		{K: "control_snapshot_digest", V: control.ControlSnapshotDigest},
		{K: "endpoint_execution_digest", V: control.EndpointExecutionDigest},
		{K: "endpoint_release_id", V: control.EndpointReleaseID},
		{K: "artifact_object_set_digest", V: control.ArtifactObjectSetDigest},
		{K: "model_root_digests", V: control.ModelRootDigests},
		{K: "descriptor_digest", V: control.DescriptorDigest},
		{K: "environment_spec_digest", V: control.EnvironmentSpecDigest},
		{K: "installed_environment_receipt_digest", V: control.InstalledEnvironmentReceiptDigest},
		{K: "placement_set_digest", V: control.PlacementSetDigest},
		{K: "binding_plan_digests", V: control.BindingPlanDigests},
	}, Notes: []string{
		"observed exact control selected by Tensorhub; none of these fields is a placement input",
	}})
}

func handleRentProbe(ctx *Context) *exit.Error {
	id := strings.TrimSpace(ctx.Inv.Args[0])
	client, e := dial(ctx)
	if e != nil {
		return e
	}
	started, e := client.EnsureRental(id)
	if e != nil {
		return e
	}
	if _, e := waitReady(client, started.InstanceID); e != nil {
		return e
	}
	_, st, e := rentalStores(ctx)
	if e != nil {
		return e
	}
	defer st.Close()
	row, control, e := rental.Inspect(st, id)
	if e != nil {
		return e
	}
	return emit(ctx, render.Record{Kind: "rental_probe", Fields: []render.Field{
		{K: "rental", V: id}, {K: "state", V: row.State},
		{K: "endpoint", V: row.EndpointRef}, {K: "accelerator", V: row.AcceleratorModel},
		{K: "observed_accelerator", V: row.ObservedAccelerator},
		{K: "observed_accelerator_count", V: row.ObservedAcceleratorCount},
		{K: "observed_backend", V: row.ObservedBackend},
		{K: "observed_worker_instance", V: row.ObservedWorkerInstance},
		{K: "observed_worker_boot_id", V: row.ObservedWorkerBootID},
		{K: "endpoint_execution_digest", V: control.EndpointExecutionDigest},
	}, Notes: []string{"no model was invoked; this is the worker ClaimAck readback"}})
}

// handleRentRelease is idempotent and ends only on provider ABSENCE: the hub reporting the
// rental gone (404) or `released`. Nothing local is forgotten before that, because the row
// is the only name this host has for a pod that may still be billing.
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
	c := client(ctx)
	if row != nil && row.Hub != c.Base() {
		return exit.Named(exit.Conflict, "rental.hub_mismatch",
			"rental %s was rented from %s, not the configured hub %s", id, row.Hub, c.Base()).
			WithRemedy("point TENSORHUB_URL at the hub that holds the pod; a 404 from another hub says nothing about it")
	}
	rctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	confirmed := ctx.Inv.Bool("--yes")
	w := releaseWatch{ctx: ctx, c: c, id: id, rctx: rctx, once: !confirmed}

	seen, gone, e := w.observe()
	if e != nil {
		return e
	}
	if gone && confirmed {
		return w.finish(l, st, "", row != nil, "the hub already reported this rental gone")
	}
	if !confirmed {
		state := seen.State
		if gone {
			state = "absent on the hub"
		}
		plan := render.Record{Kind: "release-plan", Fields: []render.Field{
			{K: "rental", V: id}, {K: "state", V: state}, {K: "hub", V: c.Base()},
		}, Notes: []string{
			"the hub DESTROYS the pod: anything resident on it is lost and any run pinned to it stops being placeable",
			"the pinned certificate and the owner token are removed from this host with the row",
			"this printed the plan and changed nothing — re-run with --yes",
		}, Next: []string{"cozy rent release " + id + " --yes"}}
		if row != nil {
			plan.Fields = append(plan.Fields, render.Field{K: "address", V: row.Address},
				render.Field{K: "media", V: row.MediaAddress})
		}
		return emit(ctx, plan)
	}

	operationKey, e := st.RequestRentalRelease(id)
	if e != nil {
		return e
	}
	// A rental the hub already shows leaving needs no second DELETE; the poll settles it.
	if seen.State != hub.RentalReleaseRequested {
		if e := w.request(); e != nil {
			return e
		}
	}
	for {
		_, gone, e := w.observe()
		if e != nil {
			return e
		}
		if gone {
			return w.finish(l, st, operationKey, row != nil, "the hub destroyed the pod")
		}
		select {
		case <-rctx.Done():
			return w.interrupted()
		case <-time.After(pollCadence):
		}
	}
}

type releaseWatch struct {
	ctx  *Context
	c    *hub.Client
	id   string
	rctx context.Context
	said string
	once bool // a plan print asks the hub once, bounded; it never waits out a fault
}

// observe reads the rental until the hub gives a verdict. Transport faults are retried at
// cadence: they say nothing about the pod, and a release that gave up on them would leave
// the local half of a billing pod deleted or orphaned on a guess.
func (w *releaseWatch) observe() (hub.Rental, bool, *exit.Error) {
	for {
		rctx := w.rctx
		if w.once {
			var cancel context.CancelFunc
			rctx, cancel = hub.Context()
			defer cancel()
		}
		r, e := w.c.Rental(rctx, w.id)
		switch {
		case e == nil && r.State == hub.RentalReleased:
			return r, true, nil
		case e == nil:
			w.say(r.State, r.Detail)
			return r, false, nil
		case e.Code == exit.NotFound:
			return hub.Rental{}, true, nil
		case !transient(e), w.once:
			return hub.Rental{}, false, w.kept(e)
		}
		w.say("hub", e.Message+"; retrying")
		select {
		case <-w.rctx.Done():
			return hub.Rental{}, false, w.interrupted()
		case <-time.After(pollCadence):
		}
	}
}

// request sends the DELETE. A 404 is absence; any other typed refusal ends the release.
func (w *releaseWatch) request() *exit.Error {
	for {
		e := w.c.Release(w.rctx, w.id, "cozy rent release")
		switch {
		case e == nil, e.Code == exit.NotFound:
			return nil
		case !transient(e):
			return w.kept(e)
		}
		w.say("hub", e.Message+"; retrying")
		select {
		case <-w.rctx.Done():
			return w.interrupted()
		case <-time.After(pollCadence):
		}
	}
}

func (w *releaseWatch) say(state, detail string) {
	line := state + ": " + detail
	if detail == "" || line == w.said {
		return
	}
	w.said = line
	fmt.Fprintf(w.ctx.Err, "  %s\n", line)
}

func (w *releaseWatch) kept(e *exit.Error) *exit.Error {
	return e.WithRemedy("the local record is KEPT: the pod may still be running, and this row is its name here").
		WithNext("cozy rent ls", "cozy rent release "+w.id+" --yes")
}

func (w *releaseWatch) interrupted() *exit.Error {
	return exit.New(exit.Canceled, "release of rental %s was interrupted before the hub reported it gone", w.id).
		WithRemedy("the local record is KEPT and the hub may still be tearing the pod down; re-run release to resume watching").
		WithNext("cozy rent release " + w.id + " --yes")
}

func (w *releaseWatch) finish(l home.Layout, st *records.Store, operationKey string, had bool, note string) *exit.Error {
	forgotten := false
	if had {
		var e *exit.Error
		if forgotten, e = rental.Forget(l, st, w.id); e != nil {
			return e
		}
	}
	if operationKey != "" {
		rental.ForgetPending(l, operationKey)
	}
	notes := []string{note + "; its owner token and pinned certificate are gone from this host"}
	if !had {
		notes = []string{note + "; this host held no record of it — already released"}
	}
	return emit(w.ctx, render.Record{Kind: "release", Fields: []render.Field{
		{K: "rental", V: w.id}, {K: "released", V: forgotten},
	}, Notes: notes, Next: []string{"cozy rent ls"}})
}
