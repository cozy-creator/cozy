package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os/signal"
	"strconv"
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
	"github.com/cozy-creator/cozy/internal/rentalid"
	"github.com/cozy-creator/cozy/internal/secret"
)

// The rental verbs (cl-015). `cozy rental new` MINTS the pod's access token, asks the hub for a
// pod while presenting only that token's sha256, watches it provision, and PINS the
// triple — the address, the certificate to trust, and the token it minted — so that
// `cozy run --machine <name>` can dial a worker this host never spawned.
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
	requestedMachineName := strings.TrimSpace(ctx.Inv.Value("--name"))
	if skuName == "" {
		if requestedMachineName != "" || ctx.Inv.Value("--idempotency-key") != "" ||
			ctx.Inv.Value("--timeout") != "" {
			return exit.Usagef("rental options require a GPU SKU").
				WithRemedy("use `cozy rental new` alone to list available machines")
		}
		hctx, cancel := hub.Context()
		skus, e := client(ctx).RentalSKUs(hctx)
		cancel()
		if e != nil {
			return e
		}
		return emitRentalCatalog(ctx, skus)
	}
	if requestedMachineName != "" && !rentalid.ValidMachineName(requestedMachineName) {
		return exit.Usagef("--name %q is not a machine name", requestedMachineName).
			WithRemedy("use 1-32 lowercase letters, numbers, and hyphens; local is reserved")
	}
	reason := "cozy rental new " + skuName
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
	// media bearer and Creator key; the operation row makes a lost HTTP response resumable.
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
	var creator rental.CreatorIdentity
	if existing != nil && existing.State == "attached" && existing.RentalID != "" {
		token, e = rental.MediaToken(l, existing.RentalID)
		if e == nil {
			creator, e = rental.CreatorIdentityFor(l, existing.RentalID)
		}
	} else {
		token, e = rental.PendingMediaToken(l, operationKey)
		if e == nil {
			creator, e = rental.PendingCreatorIdentity(l, operationKey)
		}
	}
	if e != nil {
		return e
	}
	tokenHash := secret.HashHex(token)
	requestBody, e := hub.RentalRequestBytes(skuName, tokenHash, creator.PublicKey(), 0, 0)
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
			WithRemedy("reuse a key only for the exact same hub, GPU SKU, media token, and Creator key")
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
	machineName := requestedMachineName
	if machineName == "" {
		machineName = rentalid.MachineName(r.ID)
	}
	row := records.Rental{ID: r.ID, MachineName: machineName, SKU: skuName,
		AcceleratorModel: r.AcceleratorModel, State: r.State, Hub: c.Base()}
	stored, problem := st.RentalRow(r.ID)
	if problem != nil {
		return problem
	}
	if stored != nil && stored.MachineName != "" {
		if requestedMachineName != "" && requestedMachineName != stored.MachineName {
			return exit.Named(exit.Conflict, "rental.machine_name_changed",
				"rental %s is already named %s", r.ID, stored.MachineName).
				WithRemedy("resume it without --name, or use --name %s", stored.MachineName)
		}
		row.MachineName = stored.MachineName
	}
	if existing != nil && existing.State == "attached" {
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
		row.ExpectedWorkerID, row.ExpectedWorkerBootID = seen.WorkerID, seen.WorkerBootID
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
	row.Address, row.State = attachable.Address, attachable.State
	row.MediaAddress = attachable.MediaAddress
	row.ExpectedWorkerID, row.ExpectedWorkerBootID = attachable.WorkerID, attachable.WorkerBootID
	if !attachable.HoldsMediaHash(secret.HashHex(token)) {
		// The pod was provisioned with a credential set this host's token is not in, so
		// dialling it would 401 and look like a network fault. The hub says which hashes
		// are live; neither end has to say a token to find this out.
		return exit.New(exit.Failed,
			"rental %s is attachable and its live credential set does not carry the token this host minted", attachable.ID).
			WithRemedy("release it and rent again; a pod nobody can authenticate to still costs money").
			WithNext("cozy rental end " + attachable.ID)
	}
	if attachable.CreatorPublicKey != creator.PublicKey() {
		return exit.Named(exit.Conflict, "rental.creator_key_changed",
			"rental %s did not retain the Creator key sent at create", attachable.ID).
			WithRemedy("release it; this host will not sign for a rental bound to another key")
	}
	if e := rental.Attach(l, st, row, attachable.CertPEM, token, creator); e != nil {
		return e
	}
	row.CertPath = l.RentalCert(attachable.ID)
	if e := st.AdvanceRentalOperation(operationKey, attachable.ID, "attached"); e != nil {
		return e
	}
	rental.ForgetPending(l, operationKey)

	// Tensorhub readiness means only that this host can attach. The daemon claims the
	// empty worker now; queued requests later send Creator-owned desired state directly.
	ctx.Daemon = daemon.Probe(ctx.Cfg)
	if !ctx.Daemon.Up {
		return ctx.Daemon.Unavailable().WithRemedy(
			"the paid rental is attached on this host; start `cozy run list` and resume with the same --idempotency-key")
	}
	local, e := dial(ctx)
	if e != nil {
		return e.WithRemedy("the paid rental is attached on this host; start `cozy run list` and resume with the same --idempotency-key")
	}
	if _, e := local.EnsureRental(attachable.ID); e != nil {
		return e.WithRemedy("the paid rental is attached on this host; keep `cozy run list` running and resume with the same --idempotency-key")
	}
	ready := attachable
	notes := []string{"billing continues until `cozy rental end " + ready.ID + "` confirms release"}
	if replay {
		notes = append(notes, "the existing rental operation resumed")
	}
	fields := []output.Field{
		{K: "machine", V: row.MachineName},
		{K: "rental", V: ready.ID},
		{K: "state", V: ready.State},
		{K: "address", V: ready.Address},
		{K: "media", V: ready.MediaAddress},
		{K: "gpu", V: skuName},
		{K: "accelerator", V: ready.AcceleratorModel},
		{K: "changed", V: !replay}, {K: "operation", V: operationKey}, {K: "replayed", V: replay},
	}
	rec := compactRecord(fields, "machine", "state", "gpu", "changed")
	rec.Notes = notes
	rec.Next = []string{
		"cozy run <org/package/function> --machine " + row.MachineName,
		"cozy rental end " + row.MachineName,
	}
	return emit(ctx, rec)
}

func emitRentalCatalog(ctx *Context, skus []hub.RentalSKU) *exit.Error {
	rows := make([]map[string]string, 0, len(skus))
	for _, sku := range skus {
		rows = append(rows, map[string]string{
			"name": sku.Name, "model": sku.AcceleratorModel,
			"compute": computeCapabilityText(sku.ComputeCapability),
			"vram":    fmt.Sprintf("%d GB", sku.VRAMGB),
			"price":   rentalPrice(sku.PriceUSDMicrosPerHour),
		})
	}
	doc := output.List{
		Name: "gpus", Fields: []string{"name", "model", "compute", "vram", "price"},
		Rows: rows, Total: len(rows),
		Next: []string{"cozy rental new <gpu-name>"},
	}
	return emit(ctx, doc)
}

func computeCapabilityText(value string) string {
	if value == "" {
		return "unknown"
	}
	return "sm_" + strings.ReplaceAll(value, ".", "")
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
				WithNext("cozy rental new <gpu-name>")
		case r.State == hub.RentalReady:
			// READY without a whole triple is the hub contradicting itself, and dialling
			// on a partial one would fail later as something that looks like a network
			// fault. Say which piece is missing instead.
			return hub.Rental{}, exit.New(exit.Failed,
				"rental %s is ready and carries no %s", id, missingOf(r)).
				WithRemedy("this hub build may not provision the worker's TLS leg; `cozy package search` names it")
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
		WithRemedy("the pod is NOT released; `cozy rental` still names it and release destroys it").
		WithNext("cozy rental", "cozy rental end "+id)
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
	if len(r.MediaTokenSHA256) == 0 {
		return "observed media-token hash set"
	}
	if r.WorkerID == "" || r.WorkerBootID == "" {
		return "worker and boot identity"
	}
	if r.CreatorPublicKey == "" {
		return "Creator public key"
	}
	return "complete ready projection"
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
		Fields:    []string{"machine", "sku", "state", "uptime", "queued", "running"},
		AllFields: []string{"machine", "sku", "state", "uptime", "queued", "running", "utilization", "rental", "accelerator", "address", "media", "hub", "rented"},
		Next:      []string{"cozy help rental new"},
	}
	for _, r := range rows {
		queued, running, problem := st.RentalRunCounts(r.ID)
		if problem != nil {
			return problem
		}
		list.Rows = append(list.Rows, map[string]string{
			"machine": r.MachineName, "sku": orNone(r.SKU),
			"state": r.State, "uptime": rentalUptime(r.RentedAt),
			"queued": strconv.Itoa(queued), "running": strconv.Itoa(running),
			"utilization": "not reported", "rental": r.ID,
			"accelerator": r.AcceleratorModel, "address": r.Address,
			"media": r.MediaAddress, "hub": r.Hub,
			"rented": stamp(r.RentedAt),
		})
	}
	if len(list.Rows) > 0 {
		list.Next = []string{"cozy rental end " + list.Rows[0]["machine"]}
	}
	return emit(ctx, list)
}

func rentalUptime(started string) string {
	stamp, err := time.Parse(time.RFC3339Nano, started)
	if err != nil {
		return "unknown"
	}
	d := time.Since(stamp)
	if d < 0 {
		d = 0
	}
	if d < time.Minute {
		return "<1m"
	}
	return d.Round(time.Minute).String()
}

// handleRentRelease is idempotent and ends only on provider ABSENCE: the hub reporting the
// rental gone (404) or `released`. Nothing local is forgotten before that, because the row
// is the only name this host has for a pod that may still be billing.
func handleRentRelease(ctx *Context) *exit.Error {
	subject := strings.TrimSpace(ctx.Inv.Args[0])
	l, st, e := rentalStores(ctx)
	if e != nil {
		return e
	}
	defer st.Close()
	row, e := st.RentalByMachine(subject)
	if e != nil {
		return e
	}
	id := subject
	if row != nil {
		id = row.ID
	}
	c := client(ctx)
	if row != nil && row.Hub != c.Base() {
		return exit.Named(exit.Conflict, "rental.hub_mismatch",
			"rental %s was rented from %s, not the configured hub %s", id, row.Hub, c.Base()).
			WithRemedy("point TENSORHUB_URL at the hub that holds the pod; a 404 from another hub says nothing about it")
	}
	rctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	machine := subject
	if row != nil {
		machine = row.MachineName
	}
	w := releaseWatch{ctx: ctx, c: c, id: id, machine: machine, rctx: rctx}

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
	ctx     *Context
	c       *hub.Client
	id      string
	machine string
	rctx    context.Context
	said    string
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
		WithNext("cozy rental", "cozy rental end "+w.id)
}

func (w *releaseWatch) interrupted() *exit.Error {
	return exit.New(exit.Canceled, "release of rental %s was interrupted before the hub reported it gone", w.id).
		WithRemedy("the local record is KEPT and the hub may still be tearing the pod down; re-run release to resume watching").
		WithNext("cozy rental end " + w.id)
}

func (w *releaseWatch) finish(l home.Layout, st *records.Store, operationKey string, had bool, note string) *exit.Error {
	forgotten := false
	if had {
		local, problem := dial(w.ctx)
		if problem != nil {
			return problem.WithRemedy("the rental is gone at the hub, but its local credentials are kept until the daemon's worker control loop can detach")
		}
		if _, problem := local.DetachRental(w.id); problem != nil {
			return problem.WithRemedy("the rental is gone at the hub, but its local credentials are kept until the daemon's worker control loop can detach")
		}
		if forgotten, problem = rental.Forget(l, st, w.id); problem != nil {
			return problem
		}
	}
	if operationKey != "" {
		rental.ForgetPending(l, operationKey)
	}
	notes := []string{note + "; its media bearer, Creator key, and pinned certificate are gone from this host"}
	if !had {
		notes = []string{note + "; this host held no record of it — already released"}
	}
	return emit(w.ctx, output.Record{Fields: []output.Field{
		{K: "machine", V: w.machine}, {K: "rental", V: w.id},
		{K: "state", V: "ended"}, {K: "changed", V: forgotten},
	}, Notes: notes})
}
