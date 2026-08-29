package cli

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

	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
)

// The rental verbs (cl-015). `cozy rental new` MINTS the pod's access token, asks the hub for a
// pod while presenting only that token's sha256, watches it provision, and PINS the
// triple — the address, the certificate to trust, and the token it minted — so that
// `cozy invoke run --worker <id>` can dial a worker this host never spawned.
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
	skuName := strings.TrimSpace(ctx.Inv.Args[0])
	packageRef := strings.TrimSpace(ctx.Inv.Args[1])
	if skuName == "" {
		hctx, cancel := hub.Context()
		skus, e := client(ctx).RentalSKUs(hctx)
		cancel()
		if e != nil {
			return e
		}
		return emitRentalCatalog(ctx, skus)
	}
	if packageRef == "" {
		return exit.Usagef("`cozy rental new %s` also needs the exact package to run", skuName).
			WithRemedy("append org/package/vN/function").
			WithNext("cozy rental new " + skuName + " <org/package/vN/function>")
	}
	reason := "cozy rental new " + skuName + " for " + packageRef
	c := client(ctx)
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

	if !c.Token().Present() {
		return exit.Named(exit.Credential, "hub.token_missing",
			"POST /v1/rentals is a first-party route and no admin token is configured").
			WithRemedy("set TENSORHUB_TOKEN to the hub's admin.token; catalog reads need no credential").
			WithNext("cozy package search")
	}
	state, _, problem := ensureDaemon(ctx)
	if problem != nil {
		return problem
	}
	ctx.Daemon = state
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
			WithNext("cozy rental end " + existing.RentalID)
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
	requestBody, e := hub.RentalRequestBytes(packageRef, skuName, tokenHash)
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
			WithRemedy("reuse a key only for the exact same hub, package, Cozy GPU SKU, and renter token")
	}
	if !replay {
		fmt.Fprintf(ctx.Err, "  rental operation %s persisted; reuse this key to resume\n", operationKey)
	}
	// The first recorded reason is derived from the immutable paid intent. Replay reads
	// that durable value rather than accepting a second caller-controlled spelling.
	reason = op.Reason

	hctx, cancel := rentalCallContext(deadline)
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
	row := records.Rental{ID: r.ID, PackageRef: packageRef,
		AcceleratorModel: r.AcceleratorModel, State: r.State, Hub: c.Base()}
	if existing != nil && existing.State == "attached" {
		stored, problem := st.RentalRow(r.ID)
		if problem != nil {
			return problem
		}
		if stored == nil {
			return exit.Named(exit.Conflict, "rental.attached_record_missing",
				"rental operation %s is attached but rental %s has no local row", operationKey, r.ID)
		}
		row = *stored
	} else if e := st.RecordRental(row); e != nil {
		return e
	}
	observe := func(seen hub.Rental) *exit.Error {
		row.Address, row.State = seen.Address, seen.State
		row.MediaAddress = seen.MediaAddress
		if e := captureRentalControl(&row, seen); e != nil {
			return e
		}
		if e := st.RecordRental(row); e != nil {
			return e
		}
		return st.AdvanceRentalOperation(operationKey, seen.ID, seen.State)
	}
	attachable, e := waitRental(ctx, c, r.ID, deadline, observe,
		func(r hub.Rental) bool { return r.Attachable() })
	if e != nil {
		return e
	}
	if attachable.State == hub.RentalReady {
		current, problem := st.RentalOperation(operationKey)
		if problem != nil {
			return problem
		}
		if current == nil || current.State != "attached" {
			return exit.Named(exit.Conflict, "rental.convergence_evidence_missing",
				"rental %s became ready before this host attached its private RecordOwner", attachable.ID).
				WithRemedy("upgrade Tensorhub to the convergence-gated private-rental contract; the Hub cannot declare a fresh private worker ready without the renter's relayed frames")
		}
	}
	row.Address, row.State = attachable.Address, attachable.State
	row.MediaAddress = attachable.MediaAddress
	if e := captureRentalControl(&row, attachable); e != nil {
		return e
	}
	if !attachable.HoldsHash(secret.HashHex(token)) {
		// The pod was provisioned with a credential set this host's token is not in, so
		// dialling it would 401 and look like a network fault. The hub says which hashes
		// are live; neither end has to say a token to find this out.
		return exit.New(exit.Failed,
			"rental %s is attachable and its live credential set does not carry the token this host minted", attachable.ID).
			WithRemedy("release it and rent again; a pod nobody can authenticate to still costs money").
			WithNext("cozy rental end " + attachable.ID)
	}
	if e := rental.Attach(l, st, row, attachable.CertPEM, token); e != nil {
		return e
	}
	row.CertPath = l.RentalCert(attachable.ID)
	if e := st.AdvanceRentalOperation(operationKey, attachable.ID, "attached"); e != nil {
		return e
	}
	rental.ForgetPending(l, operationKey)

	// CONVERGING is the handoff to this host's sole RecordOwner. Tensorhub holds only
	// the token hash and cannot authenticate to WorkerControl; the Cozy daemon claims,
	// acknowledges the snapshot barrier, drives desired state, and relays the resulting
	// authenticated worker frames over the rental-scoped HTTP route.
	ctx.Daemon = daemon.Probe(ctx.Cfg)
	if !ctx.Daemon.Up {
		return ctx.Daemon.Unavailable().WithRemedy(
			"the paid rental remains converging and attached on this host; start `cozy invoke list` and resume with the same --idempotency-key")
	}
	local, e := dial(ctx)
	if e != nil {
		return e.WithRemedy("the paid rental remains converging and attached on this host; start `cozy invoke list` and resume with the same --idempotency-key")
	}
	if _, e := local.EnsureRental(attachable.ID); e != nil {
		return e.WithRemedy("the paid rental remains converging and attached on this host; keep `cozy invoke list` running and resume with the same --idempotency-key")
	}
	observeConvergence := func(seen hub.Rental) *exit.Error {
		if e := sameAttachProjection(attachable, seen, secret.HashHex(token)); e != nil {
			return e
		}
		row.State = seen.State
		if e := st.RecordRental(row); e != nil {
			return e
		}
		if seen.State == hub.RentalReady {
			return st.ClearRentalRelayRefusal(seen.ID)
		}
		refusal, e := st.RentalRelayRefusal(seen.ID)
		if e != nil {
			return e
		}
		if refusal != nil {
			return refusal.Error()
		}
		return nil
	}
	ready, e := waitRental(ctx, c.WithToken(token, "rental owner token"), attachable.ID,
		deadline, observeConvergence, func(r hub.Rental) bool { return r.Ready() })
	if e != nil {
		return e
	}
	notes := []string{"billing continues until `cozy rental end " + ready.ID + "` confirms release"}
	if replay {
		notes = append(notes, "the existing rental operation resumed")
	}
	fields := []output.Field{
		{K: "rental", V: ready.ID},
		{K: "state", V: ready.State},
		{K: "address", V: ready.Address},
		{K: "media", V: ready.MediaAddress},
		{K: "package", V: packageRef},
		{K: "gpu", V: skuName},
		{K: "accelerator", V: ready.AcceleratorModel},
		{K: "changed", V: !replay}, {K: "operation", V: operationKey}, {K: "replayed", V: replay},
	}
	rec := compactRecord(fields, "rental", "state", "gpu", "package", "changed")
	rec.Notes = notes
	rec.Next = []string{
		"cozy invoke run <org/package/vN/function> --worker " + ready.ID,
		"cozy rental end " + ready.ID,
	}
	return emit(ctx, rec)
}

func emitRentalCatalog(ctx *Context, skus []hub.RentalSKU) *exit.Error {
	rows := make([]map[string]string, 0, len(skus))
	for _, sku := range skus {
		rows = append(rows, map[string]string{
			"name": sku.Name, "model": sku.AcceleratorModel,
			"vram":  fmt.Sprintf("%d GB", sku.VRAMGB),
			"price": rentalPrice(sku.PriceUSDMicrosPerHour),
		})
	}
	doc := output.List{
		Name: "gpus", Fields: []string{"name", "model", "vram", "price"},
		Rows: rows, Total: len(rows),
		Next: []string{"cozy rental new <gpu-name> <org/package/vN/function>"},
	}
	return emit(ctx, doc)
}

func rentalPrice(micros int64) string {
	amount := strings.TrimRight(strings.TrimRight(fmt.Sprintf("%.6f", float64(micros)/1_000_000), "0"), ".")
	return "$" + amount + "/hr"
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

// rentalCallContext carries the caller's one explicit wall-clock bound into every Hub
// request in the paid flow. Without it, a response body that kept moving one byte at a
// time could remain live past --timeout because the polling loop checked the deadline
// only after the body finished.
func rentalCallContext(deadline time.Time) (context.Context, context.CancelFunc) {
	if deadline.IsZero() {
		return hub.LongContext()
	}
	return context.WithDeadline(context.Background(), deadline)
}

// waitRental polls one rental to the caller's observed goal. Every wait here is bounded by something
// OBSERVED: the hub's own verdict, a typed refusal, or the caller's --timeout. A rental
// that is still acquiring or materializing is none of those, however long the provider
// takes, and a hub that is momentarily unreachable is asked again at the same cadence.
func waitRental(ctx *Context, c *hub.Client, id string, deadline time.Time,
	observe func(hub.Rental) *exit.Error, done func(hub.Rental) bool) (hub.Rental, *exit.Error) {
	said := ""
	for {
		hctx, cancel := rentalCallContext(deadline)
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
		case done(r):
			return r, nil
		case r.State == hub.RentalFailed:
			return hub.Rental{}, exit.New(exit.Failed,
				"rental %s failed to provision: %s", id, detailOr(r.Detail)).
				WithRemedy("the pod is the hub's to reclaim; `cozy rental end %s` closes it out", id).
				WithNext("cozy rental end " + id)
		case r.State == hub.RentalDegraded:
			return hub.Rental{}, exit.New(exit.Failed,
				"rental %s is degraded: %s", id, detailOr(r.Detail)).
				WithRemedy("release it, then rent again; degraded is a settled loss of ready service, not an in-flight boot state").
				WithNext("cozy rental end " + id)
		case r.State == hub.RentalReleaseRequested || r.State == hub.RentalReleased:
			// A rental that is LEAVING is not one that is still coming up, and waiting on
			// it is waiting for a state it will never reach.
			return hub.Rental{}, exit.New(exit.Failed,
				"rental %s is %s: %s", id, r.State, detailOr(r.Detail)).
				WithRemedy("the rental is leaving or gone; rent again if you still need a pod").
				WithNext("cozy rental new <gpu-name> <org/package/vN/function>")
		case r.State == hub.RentalReady:
			// READY without a whole triple is the hub contradicting itself, and dialling
			// on a partial one would fail later as something that looks like a network
			// fault. Say which piece is missing instead.
			return hub.Rental{}, exit.New(exit.Failed,
				"rental %s is ready and carries no %s", id, missingOf(r)).
				WithRemedy("this hub build may not provision the worker's TLS leg; `cozy package search` names it")
		case r.State == hub.RentalConverging && !r.Attachable():
			return hub.Rental{}, exit.New(exit.Failed,
				"rental %s is converging and carries no %s", id, missingOf(r)).
				WithRemedy("Tensorhub must publish the complete receipt-pinned attach projection before asking this RecordOwner to claim it")
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

func sameAttachProjection(attached, seen hub.Rental, tokenHash string) *exit.Error {
	if !seen.Attachable() || attached.ID != seen.ID || attached.Address != seen.Address ||
		attached.MediaAddress != seen.MediaAddress || attached.CertPEM != seen.CertPEM ||
		attached.PackageRef != seen.PackageRef || attached.AcceleratorModel != seen.AcceleratorModel ||
		attached.PlacementRevision != seen.PlacementRevision ||
		!seen.HoldsHash(tokenHash) || !sameHashSet(attached.TokenSHA256, seen.TokenSHA256) ||
		attached.ControlSnapshot == nil || seen.ControlSnapshot == nil ||
		attached.ControlSnapshot.Digest != seen.ControlSnapshot.Digest ||
		attached.ControlSnapshot.Length != seen.ControlSnapshot.Length ||
		!bytes.Equal(attached.ControlSnapshot.CanonicalBytes, seen.ControlSnapshot.CanonicalBytes) {
		return exit.Named(exit.Conflict, "rental.attach_projection_changed",
			"rental %s changed its receipt-pinned control projection while this host was converging it",
			attached.ID).
			WithRemedy("release it; a different address, certificate, credential set, or control snapshot is not the worker this RecordOwner claimed")
	}
	return nil
}

func sameHashSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	set := make(map[string]bool, len(a))
	for _, value := range a {
		set[strings.TrimPrefix(value, "sha256:")] = true
	}
	for _, value := range b {
		if !set[strings.TrimPrefix(value, "sha256:")] {
			return false
		}
	}
	return len(set) == len(a)
}

func pastDeadline(id, state string, deadline time.Time) *exit.Error {
	if deadline.IsZero() || !time.Now().After(deadline) {
		return nil
	}
	return exit.New(exit.Deadline,
		"rental %s was still %s at the --timeout you set", id, state).
		WithRemedy("the pod is NOT released; `cozy rental list` still names it and release destroys it").
		WithNext("cozy rental list", "cozy rental end "+id)
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
	if r.PlacementRevision == 0 {
		return "active placement revision"
	}
	return "complete ready projection"
}

func captureRentalControl(row *records.Rental, seen hub.Rental) *exit.Error {
	if row == nil || seen.ControlSnapshot == nil {
		return nil
	}
	if row.ControlSnapshotDigest == "" {
		candidate := *row
		candidate.ControlSnapshotDigest = seen.ControlSnapshot.Digest
		candidate.ControlSnapshotBytes = append([]byte(nil), seen.ControlSnapshot.CanonicalBytes...)
		candidate.PlacementRevision = seen.PlacementRevision
		if seen.PackageRef != "" {
			candidate.PackageRef = seen.PackageRef
		}
		if problem := rental.ValidateControl(candidate); problem != nil {
			return problem
		}
		*row = candidate
		return nil
	}
	if row.ControlSnapshotDigest != seen.ControlSnapshot.Digest ||
		len(row.ControlSnapshotBytes) != len(seen.ControlSnapshot.CanonicalBytes) ||
		!bytes.Equal(row.ControlSnapshotBytes, seen.ControlSnapshot.CanonicalBytes) {
		return exit.Named(exit.Conflict, "rental.control_snapshot_changed",
			"rental %s changed its acquisition control snapshot after Cozy validated it", row.ID).
			WithRemedy("release it; a different snapshot is a different execution authority")
	}
	if row.PlacementRevision > 0 && seen.PlacementRevision < row.PlacementRevision {
		return exit.Named(exit.Conflict, "rental.placement_revision_regressed",
			"rental %s placement revision regressed from %d to %d",
			row.ID, row.PlacementRevision, seen.PlacementRevision)
	}
	row.PlacementRevision = seen.PlacementRevision
	if seen.PackageRef != "" {
		row.PackageRef = seen.PackageRef
	}
	return nil
}

func handleRentLs(ctx *Context) *exit.Error {
	_, st, e := rentalStores(ctx)
	if e != nil {
		return e
	}
	defer st.Close()
	rows, e := st.Rentals()
	if e != nil {
		return e
	}
	list := output.List{
		Name:      "rentals",
		Fields:    []string{"rental", "state", "package", "accelerator"},
		AllFields: []string{"rental", "state", "package", "accelerator", "address", "media", "hub", "rented"},
		Next:      []string{"cozy help rental new"},
	}
	for _, r := range rows {
		list.Rows = append(list.Rows, map[string]string{
			"rental": r.ID, "state": r.State, "package": r.PackageRef,
			"accelerator": r.AcceleratorModel, "address": r.Address,
			"media": r.MediaAddress, "hub": r.Hub,
			"rented": stamp(r.RentedAt),
		})
	}
	if len(list.Rows) > 0 {
		list.Next = []string{"cozy rental end " + list.Rows[0]["rental"]}
	}
	return emit(ctx, list)
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
	w := releaseWatch{ctx: ctx, c: c, id: id, rctx: rctx}

	seen, gone, e := w.observe()
	if e != nil {
		return e
	}
	if gone {
		return w.finish(l, st, "", row != nil, "the hub already reported this rental gone")
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
}

// observe reads the rental until the hub gives a verdict. Transport faults are retried at
// cadence: they say nothing about the pod, and a release that gave up on them would leave
// the local half of a billing pod deleted or orphaned on a guess.
func (w *releaseWatch) observe() (hub.Rental, bool, *exit.Error) {
	for {
		r, e := w.c.Rental(w.rctx, w.id)
		switch {
		case e == nil && r.State == hub.RentalReleased:
			return r, true, nil
		case e == nil:
			w.say(r.State, r.Detail)
			return r, false, nil
		case e.Code == exit.NotFound:
			return hub.Rental{}, true, nil
		case !transient(e):
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
		e := w.c.Release(w.rctx, w.id, "cozy rental end")
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
		WithNext("cozy rental list", "cozy rental end "+w.id)
}

func (w *releaseWatch) interrupted() *exit.Error {
	return exit.New(exit.Canceled, "release of rental %s was interrupted before the hub reported it gone", w.id).
		WithRemedy("the local record is KEPT and the hub may still be tearing the pod down; re-run release to resume watching").
		WithNext("cozy rental end " + w.id)
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
	return emit(w.ctx, output.Record{Fields: []output.Field{
		{K: "rental", V: w.id}, {K: "state", V: "ended"}, {K: "changed", V: forgotten},
	}, Notes: notes})
}
