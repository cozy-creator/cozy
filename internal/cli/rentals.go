package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os/signal"
	"sort"
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
	"github.com/cozy-creator/cozy/internal/secret"
)

// The rental verbs (cl-015). `cozy rental new` MINTS the pod's access token, asks the hub for a
// pod while presenting only that token's sha256, watches it provision, and PINS the
// triple — the address, the certificate to trust, and the token it minted — so that
// `cozy run --rental` lets the scheduler use the attached capacity.
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
	if skuName == "" {
		if ctx.Inv.Value("--idempotency-key") != "" ||
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
	reason := "cozy rental new " + skuName
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
	fleet := &managedRentals{ctx: ctx, layout: l, store: st}
	line, sku, e := fleet.admit(skuName)
	if e != nil {
		return e
	}
	fmt.Fprintln(ctx.Err, line)

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

	row, attachable, replay, e := acquireRental(ctx, l, st, skuName,
		operationKey, reason, sku.PriceUSDMicrosPerHour, sku.StorageUSDMicrosPerHour,
		ctx.Cfg.RentalsMaxHourlySpendUSDMicros, deadline, "", nil)
	if e != nil {
		return e
	}

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
	notes := []string{"billing continues until `cozy rental end " + ready.ID + "` confirms release",
		idleReleaseNote(ctx.Cfg.RentalsIdleRelease)}
	if line, problem := fleet.status(); problem == nil {
		notes = append(notes, line)
	}
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
		"cozy run <org/package/function> --rental",
		"cozy rental end " + row.MachineName,
	}
	return emit(ctx, rec)
}

// acquireRental is the one paid mutation used by both `cozy rental new` and
// `cozy run --rental`. It returns only after the immutable retail rate and the
// worker's authenticated attach projection are durable locally.
func acquireRental(ctx *Context, l home.Layout, st *records.Store, skuName,
	operationKey, reason string, hourlyRateUSDMicros, storageUSDMicros, fleetCapUSDMicros int64,
	deadline time.Time, managedRequestID string, phase acquisitionPhase,
) (records.Rental, hub.Rental, bool, *exit.Error) {
	return acquireRentalContext(context.Background(), ctx, l, st, skuName, operationKey,
		reason, hourlyRateUSDMicros, storageUSDMicros, fleetCapUSDMicros, deadline,
		managedRequestID, phase)
}

// acquisitionPhase reports one readiness observation to whoever is waiting on this
// acquisition. It exists so the seconds between "renting" and "attachable" are named
// while they pass instead of being one silent edge (cl-121): the hub is polled the whole
// time and its answer already says which side of the boundary the pod is on. A caller
// with nobody to tell passes nil.
type acquisitionPhase func(hub.Rental)

// hourlyRateUSDMicros is the LOCKED accepted quote — the hub's GPU list rate,
// the figure the fresh-acceptance guard compares. storageUSDMicros is the
// SKU's estimated storage adder (th-126): admission money only, totaled with
// the quote against the fleet cap and never persisted as the rate.
func acquireRentalContext(lifecycle context.Context, ctx *Context, l home.Layout,
	st *records.Store, skuName, operationKey, reason string,
	hourlyRateUSDMicros, storageUSDMicros, fleetCapUSDMicros int64,
	deadline time.Time, managedRequestID string, phase acquisitionPhase,
) (records.Rental, hub.Rental, bool, *exit.Error) {
	c := client(ctx)
	existing, e := st.RentalOperation(operationKey)
	if e != nil {
		return records.Rental{}, hub.Rental{}, false, e
	}
	if existing != nil && (existing.State == "rejected" || existing.State == "released") {
		return records.Rental{}, hub.Rental{}, false, exit.Named(exit.Conflict, "rental.operation_settled",
			"rental operation %s is already %s", operationKey, existing.State).
			WithRemedy("use a fresh operation key for a new paid rental")
	}
	if existing != nil && (existing.State == "failed" || existing.State == "release_requested") {
		return records.Rental{}, hub.Rental{}, false, exit.Named(exit.Conflict, "rental.release_required",
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
		return records.Rental{}, hub.Rental{}, false, e
	}
	// A rental bought FOR a request declares that request's workload, so the hub
	// can size the pod's container disk to the job (th-152). An ingest holds its
	// source objects and the canonical CAS output built from them in one Store
	// on the container disk, so a pod bought at the serving default would block
	// on a full filesystem partway through a paid run. The selection is already
	// resolved here — the request and its source inventory were committed in one
	// transaction before any pod was asked for — so this costs no extra round
	// trip. A rental with no managed request declares nothing and takes the
	// serving default.
	//
	// The SERVING half (th-155/cl-130) rides the same fact and the same trip.
	// A serving pod's models are resolved into this store before the pod is
	// asked for — and the very same rows become its desired download set — so
	// the rental can state WHICH models it will hold and let the hub read what
	// they weigh out of its own catalog. It declares identity rather than a
	// byte total on purpose: two models sharing a component share those bytes
	// on disk exactly once, and only the party holding the digests can take
	// that union. Undeclared, the hub buys the image's serving default, which
	// holds the H3 serve set with 35 GB to spare and no room for a second
	// model.
	var workload hub.DeclaredWorkload
	if managedRequestID != "" {
		workload.SourceBytes, e = st.PlannedSourceBytes(managedRequestID)
		if e != nil {
			return records.Rental{}, hub.Rental{}, false, e
		}
		models, problem := st.DeclaredServingModels(managedRequestID)
		if problem != nil {
			return records.Rental{}, hub.Rental{}, false, problem
		}
		for _, model := range models {
			workload.ServingModels = append(workload.ServingModels, hub.ServingModel{
				Lane: model.Lane, Manifest: model.Manifest,
				Model: model.Model, Release: model.Release})
		}
	}
	// The machine word is the store's to reserve; the request is authored under it.
	author := func(machineName string) ([]byte, string, *exit.Error) {
		body, e := hub.RentalRequestBytes(machineName, skuName, secret.HashHex(token),
			creator.PublicKey(), workload)
		if e != nil {
			return nil, "", e
		}
		return body, rentalRequestDigest(c.Base(), body), nil
	}
	op, replay, e := st.BeginRentalOperation(records.RentalOperation{
		Key: operationKey, Hub: c.Base(), Reason: reason, HourlyRateUSDMicros: hourlyRateUSDMicros,
		ManagedRequestID: managedRequestID,
	}, fleetCapUSDMicros, storageUSDMicros, author)
	if e != nil {
		return records.Rental{}, hub.Rental{}, false, e
	}
	if op.RequestDigest != rentalRequestDigest(c.Base(), op.RequestBody) || op.Hub != c.Base() ||
		op.HourlyRateUSDMicros != hourlyRateUSDMicros || op.ManagedRequestID != managedRequestID {
		return records.Rental{}, hub.Rental{}, false, exit.Named(exit.Conflict, "rental.idempotency_conflict",
			"rental operation %s already names a different hub or request body", operationKey).
			WithRemedy("reuse a key only for the exact same hub, GPU SKU, media token, and Creator key")
	}
	request, e := hub.ParseRentalRequestBytes(op.RequestBody)
	if e != nil {
		return records.Rental{}, hub.Rental{}, false, e
	}
	machineName := request.Name
	if !replay {
		fmt.Fprintf(ctx.Err, "  rental operation %s persisted; reuse this key to resume\n", operationKey)
	}
	// The create request is deliberately not canceled with the operation: a lost
	// create answer can name a billing pod. Cancellation is sampled immediately
	// after its durable verdict, when the rental id can be released exactly.
	hctx, cancel := rentalCallContext(deadline)
	remote, e := c.Rent(hctx, op.RequestBody, op.Reason, operationKey)
	cancel()
	if e != nil {
		if e.Code == exit.Credential || e.Code == exit.Validation ||
			e.Code == exit.NotFound || e.Code == exit.Conflict {
			if advanced := st.AdvanceRentalOperation(operationKey, "", "rejected"); advanced != nil {
				return records.Rental{}, hub.Rental{}, false, advanced
			}
			rental.ForgetPending(l, operationKey)
		}
		return records.Rental{}, hub.Rental{}, false, e
	}
	if e := st.AdvanceRentalOperation(operationKey, remote.ID, remote.State); e != nil {
		return records.Rental{}, hub.Rental{}, false, e
	}
	if remote.Name != machineName {
		return records.Rental{}, remote, false, exit.Named(exit.Conflict, "rental.machine_name_changed",
			"Tensorhub returned private rental name %s, not %s", remote.Name, machineName).
			WithRemedy("do not attach a provider machine under a different local identity")
	}
	if lifecycle.Err() != nil {
		return records.Rental{}, remote, false, exit.New(exit.Canceled,
			"rental %s was acquired after model transfer cancellation", remote.ID)
	}
	// A fresh acceptance must lock the catalog quote the renter agreed to; a
	// replayed ask for a rental already acquiring may answer with the hub's
	// reconciled BILLED rate (th-120), and that is truth to adopt, never a
	// reason to destroy a working pod.
	if remote.State == "pending_acquisition" && remote.HourlyRateUSDMicros != hourlyRateUSDMicros {
		_ = st.AdvanceRentalOperation(operationKey, remote.ID, hub.RentalReleaseRequested)
		hctx, cancel := hub.Context()
		_ = c.Release(hctx, remote.ID, "locked Cozy retail rate changed")
		cancel()
		return records.Rental{}, hub.Rental{}, false, exit.Named(exit.Conflict, "rental.hourly_rate_changed",
			"rental %s locked %d USD micros/hour, not catalog rate %d",
			remote.ID, remote.HourlyRateUSDMicros, hourlyRateUSDMicros).
			WithRemedy("Creator requested immediate release and retained the operation until Tensorhub proves absence")
	}
	row := records.Rental{
		ID: remote.ID, MachineName: machineName, SKU: skuName,
		AcceleratorModel: remote.AcceleratorModel, HourlyRateUSDMicros: remote.HourlyRateUSDMicros,
		ManagedRequestID: managedRequestID, State: remote.State, Hub: c.Base(),
	}
	copyRentalFailure(&row, remote)
	stored, problem := st.RentalRow(remote.ID)
	if problem != nil {
		return records.Rental{}, hub.Rental{}, false, problem
	}
	if stored != nil {
		if machineName != stored.MachineName {
			return records.Rental{}, hub.Rental{}, false, exit.Named(exit.Conflict, "rental.machine_name_changed",
				"rental %s is already named %s", remote.ID, stored.MachineName).
				WithRemedy("resume the original operation; its private rental name is immutable")
		}
		if stored.ManagedRequestID != managedRequestID {
			return records.Rental{}, hub.Rental{}, false, exit.Named(exit.Conflict, "rental.management_changed",
				"rental %s cannot change between manual and Creator-managed", remote.ID)
		}
		row.MachineName = stored.MachineName
	}
	if existing != nil && existing.State == "attached" {
		if stored == nil {
			return records.Rental{}, hub.Rental{}, false, exit.Named(exit.Conflict, "rental.attached_record_missing",
				"rental operation %s is attached but rental %s has no local row", operationKey, remote.ID)
		}
		row = *stored
	} else if e := st.RecordRental(row); e != nil {
		return records.Rental{}, hub.Rental{}, false, e
	}
	observe := func(seen hub.Rental) *exit.Error {
		if phase != nil {
			phase(seen)
		}
		if seen.Name != row.MachineName {
			return exit.Named(exit.Conflict, "rental.machine_name_changed",
				"rental %s changed its name from %s to %s", seen.ID, row.MachineName, seen.Name)
		}
		// The hub's rate is the provider's reconciled billed total once the
		// pod is read back (th-120); the row and burn line adopt it.
		if seen.HourlyRateUSDMicros > 0 {
			row.HourlyRateUSDMicros = seen.HourlyRateUSDMicros
		}
		row.Address, row.State = seen.Address, seen.State
		row.MediaAddress = seen.MediaAddress
		row.ExpectedWorkerID, row.ExpectedWorkerBootID = seen.WorkerID, seen.WorkerBootID
		copyRentalFailure(&row, seen)
		if e := st.RecordRental(row); e != nil {
			return e
		}
		return st.AdvanceRentalOperation(operationKey, seen.ID, seen.State)
	}
	attachable, e := waitRentalContext(lifecycle, ctx, c, remote.ID, deadline, observe,
		func(r hub.Rental) bool { return r.Attachable() })
	if e != nil {
		return records.Rental{}, hub.Rental{}, false, e
	}
	row.Address, row.State = attachable.Address, attachable.State
	row.MediaAddress = attachable.MediaAddress
	row.ExpectedWorkerID, row.ExpectedWorkerBootID = attachable.WorkerID, attachable.WorkerBootID
	if !attachable.HoldsMediaHash(secret.HashHex(token)) {
		return records.Rental{}, hub.Rental{}, false, exit.New(exit.Failed,
			"rental %s is attachable and its live credential set does not carry the token this host minted", attachable.ID).
			WithRemedy("release it and rent again; a pod nobody can authenticate to still costs money").
			WithNext("cozy rental end " + attachable.ID)
	}
	if attachable.CreatorPublicKey != creator.PublicKey() {
		return records.Rental{}, hub.Rental{}, false, exit.Named(exit.Conflict, "rental.creator_key_changed",
			"rental %s did not retain the Creator key sent at create", attachable.ID).
			WithRemedy("release it; this host will not sign for a rental bound to another key")
	}
	if e := rental.Attach(l, st, row, attachable.CertPEM, token, creator); e != nil {
		return records.Rental{}, hub.Rental{}, false, e
	}
	row.CertPath = l.RentalCert(attachable.ID)
	if e := st.AdvanceRentalOperation(operationKey, attachable.ID, "attached"); e != nil {
		return records.Rental{}, hub.Rental{}, false, e
	}
	rental.ForgetPending(l, operationKey)
	return row, attachable, replay, nil
}

func emitRentalCatalog(ctx *Context, skus []hub.RentalSKU) *exit.Error {
	// Cheapest first, on the same key the scheduler ranks by (rental.CheapestCompatibleSKU):
	// the combined micros a renter actually pays, tie-broken by name so equal-priced rows
	// hold still between runs. The sort key is the column the table shows, never its
	// formatted text.
	ladder := append([]hub.RentalSKU(nil), skus...)
	total := func(sku hub.RentalSKU) int64 {
		return sku.PriceUSDMicrosPerHour + sku.StorageUSDMicrosPerHour
	}
	sort.Slice(ladder, func(i, j int) bool {
		if total(ladder[i]) != total(ladder[j]) {
			return total(ladder[i]) < total(ladder[j])
		}
		return ladder[i].Name < ladder[j].Name
	})
	rows := make([]map[string]string, 0, len(ladder))
	for _, sku := range ladder {
		// The ladder speaks ONE price and it is the whole pre-spend rate the pod will
		// bill (th-126): GPU plus the SKU's storage adder, never the GPU rate alone,
		// which is the figure that read $0.49/hr while RunPod billed ~$0.70. The
		// components stay one --full away, where `gpu price` is the rate the accepted
		// quote locks.
		rows = append(rows, map[string]string{
			"name": sku.Name, "gpu": acceleratorName(sku.AcceleratorModel),
			"accelerator model": sku.AcceleratorModel,
			"compute":           computeCapabilityText(sku.ComputeCapability),
			"vram":              fmt.Sprintf("%d GB", sku.VRAMGB),
			"gpu price":         rentalPrice(sku.PriceUSDMicrosPerHour),
			"storage price":     rentalPrice(sku.StorageUSDMicrosPerHour),
			"price":             rentalPrice(total(sku)),
		})
	}
	doc := output.List{
		Name:   "gpus",
		Fields: []string{"name", "gpu", "compute", "vram", "price"},
		AllFields: []string{"name", "gpu", "accelerator model", "compute", "vram",
			"gpu price", "storage price", "price"},
		Rows: rows, Total: len(rows),
		Next: []string{"cozy rental new <gpu-name>"},
	}
	return emit(ctx, doc)
}

// acceleratorName is how a provider accelerator id READS in the GPU column. The id
// itself is never rewritten — Tensorhub matches it byte-for-byte against the offers a
// provider advertises (placement's rejectModel), so a respelling at the source stops
// matching real stock. Only the workstation suffix comes off: NVIDIA names the
// workstation card as the plain product and qualifies the variants ("Server Edition",
// "Max-Q"), so dropping any other suffix would render two distinct cards identically.
func acceleratorName(model string) string {
	name := strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(model), "Workstation Edition"))
	return strings.ReplaceAll(name, "RTX PRO ", "RTX Pro ")
}

// computeCapabilityText leaves a card without a stated compute capability empty so the
// table renders the house dash. A CPU SKU has none to state; "unknown" would claim a
// lookup failed.
func computeCapabilityText(value string) string {
	if value == "" {
		return ""
	}
	return "sm_" + strings.ReplaceAll(value, ".", "")
}

// rentalPrice is the catalog's per-hour figure. It DELEGATES rather than formatting
// money a second way: this and usdPerHourBare rendered the same micros through two
// independent implementations, so rounding one left `$0.46/hour` on the fleet line
// beside `$0.463504/hr` in the ladder. One spelling of money, one place to change it.
func rentalPrice(micros int64) string {
	return usdPerHourBare(micros) + "/hr"
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

func waitRentalContext(lifecycle context.Context, ctx *Context, c *hub.Client, id string,
	deadline time.Time, observe func(hub.Rental) *exit.Error,
	done func(hub.Rental) bool,
) (hub.Rental, *exit.Error) {
	said := ""
	for {
		if lifecycle.Err() != nil {
			return hub.Rental{}, exit.New(exit.Canceled,
				"rental %s acquisition was canceled", id)
		}
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
			select {
			case <-lifecycle.Done():
				return hub.Rental{}, exit.New(exit.Canceled,
					"rental %s acquisition was canceled", id)
			case <-time.After(pollCadence):
			}
			continue
		}
		if e := observe(r); e != nil {
			return hub.Rental{}, e
		}
		switch {
		case done(r):
			return r, nil
		case r.State == hub.RentalFailed:
			return hub.Rental{}, rentalProvisionFailure(id, r)
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
		select {
		case <-lifecycle.Done():
			return hub.Rental{}, exit.New(exit.Canceled,
				"rental %s acquisition was canceled", id)
		case <-time.After(pollCadence):
		}
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

func rentalFailureCode(r hub.Rental) string {
	if r.Failure != nil && r.Failure.Code != "" {
		return r.Failure.Code
	}
	return "rental.provision_failed"
}

func rentalProvisionFailure(id string, r hub.Rental) *exit.Error {
	return exit.Named(exit.Failed, rentalFailureCode(r),
		"rental %s failed to provision: %s", id, detailOr(r.Detail)).
		WithRemedy("the pod is the hub's to reclaim; `cozy rental end %s` closes it out", id).
		WithNext("cozy rental end " + id)
}

func copyRentalFailure(row *records.Rental, remote hub.Rental) {
	if remote.Failure == nil || remote.Failure.Code == "" {
		return
	}
	row.Failure = records.RentalFailure{
		Code: remote.Failure.Code, BaseWorkerImageDigest: remote.Failure.BaseWorkerImageDigest,
		Provider: remote.Failure.Provider, ProviderResourceID: remote.Failure.ProviderResourceID,
		ProviderHostID: remote.Failure.ProviderHostID, ProviderState: remote.Failure.ProviderState,
		ContainerState: remote.Failure.ContainerState,
	}
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

// handleRentalList is `cozy rental list` (cl-114). On a terminal it is the live board:
// the fleet redrawn in place every second, the way `cozy run list` watches runs. Piped
// or --json it is one plain snapshot. Bare `cozy rental` prints the verbs, not this.
func handleRentalList(ctx *Context) *exit.Error {
	explicit, disabled := ctx.Inv.Bool("--watch"), ctx.Inv.Bool("--no-watch")
	if explicit && disabled {
		return exit.Usagef("--watch and --no-watch cannot be used together")
	}
	watching := explicit || ctx.Mode().TTY && !disabled
	if watching && (ctx.Mode().JSON || !ctx.Mode().TTY) {
		return exit.Usagef("--watch requires interactive terminal output").
			WithRemedy("omit --watch for one snapshot, or use --json for automation")
	}
	l, st, e := rentalStores(ctx)
	if e != nil {
		return e
	}
	defer st.Close()
	fleet := &managedRentals{ctx: ctx, layout: l, store: st}
	if watching {
		return watchRentalList(ctx, st, fleet)
	}
	list, e := rentalList(ctx, st, fleet, true)
	if e != nil {
		return e
	}
	return emit(ctx, list)
}

// watchRentalList is the board: redrawn every second so UPTIME and the IDLE countdown
// move, reconciled with the hub only at pollCadence — a redraw is a local read, and the
// hub is asked at the same rate every other rental wait asks it.
func watchRentalList(ctx *Context, st *records.Store, fleet *managedRentals) *exit.Error {
	var reconciled time.Time
	return watchList(ctx, "machine", func(context.Context) (output.List, *exit.Error) {
		reconcile := time.Since(reconciled) >= pollCadence
		if reconcile {
			reconciled = time.Now()
		}
		list, problem := rentalList(ctx, st, fleet, reconcile)
		if problem != nil {
			return output.List{}, problem
		}
		// The board's guidance line is its own trail; per-row verbs stay in the snapshot.
		list.Next = nil
		return list, nil
	})
}

// rentalList is one snapshot of the fleet. reconcile says whether to converge local rows
// with the hub first — every `cozy rental list` invocation does; the live board does at
// pollCadence. The spend lead is th-120's reconciled BILLED burn, never the quote.
func rentalList(ctx *Context, st *records.Store, fleet *managedRentals,
	reconcile bool,
) (output.List, *exit.Error) {
	var count int
	var burn int64
	var e *exit.Error
	if reconcile {
		count, burn, e = fleet.totals()
	} else {
		count, burn, e = st.RentalFleetTotals()
	}
	if e != nil {
		return output.List{}, e
	}
	rows, e := st.Rentals()
	if e != nil {
		return output.List{}, e
	}
	// Provenance: which command bought each pod (cl-132). Without it a `ready` machine
	// beside another `ready` machine is two anonymous charges, and the reader supplies
	// the attribution from memory — which is exactly how a refused explicit ask came to
	// be credited with two pods that auto-placement had bought.
	boughtFor, e := st.RentalProvenance()
	if e != nil {
		return output.List{}, e
	}
	grace := ctx.Cfg.RentalsIdleRelease
	list := output.List{
		Name:   "rentals",
		Fields: []string{"machine", "sku", "state", "uptime", "running", "queued", "idle"},
		AllFields: []string{"machine", "sku", "state", "failure", "uptime", "running", "queued", "idle",
			"rental", "bought for", "accelerator", "address", "media", "hub", "rented", "ready",
			"idle_since", "release_due", "image", "provider", "provider resource",
			"provider host", "provider state", "container state"},
		// The machine document carries the underlying facts, never the table's
		// spellings: counts as numbers, moments as timestamps, absences omitted.
		TypedFields: []string{"machine", "sku", "state", "rental_id", "rented_at",
			"running", "queued", "idle_s", "release_due_at"},
		TypedAllFields: []string{"machine", "sku", "state", "rental_id", "bought_for",
			"accelerator", "address", "media_address", "hub", "rented_at", "ready_at",
			"running", "queued", "idle_s", "idle_since_at", "release_due_at",
			"hourly_rate_usd_micros", "failure_code", "base_worker_image_digest",
			"provider", "provider_resource_id", "provider_host_id", "provider_state",
			"container_state"},
		TypedRows: make([]map[string]any, 0, len(rows)),
		Lead: []string{fmt.Sprintf("Remote machines running: %d", count),
			"Current spend per hour: " + usdPerHourBare(burn)},
		Aggregates: []output.Field{{K: "machines_running", V: jsonFact{count}},
			{K: "hourly_spend_usd_micros", V: jsonFact{burn}}},
		Trail: []string{idleShutdownNote(grace)},
		Next:  []string{"cozy help rental new"},
	}
	haveFailure := false
	for _, r := range rows {
		idle, problem := observeRentalIdle(st, r)
		if problem != nil {
			return output.List{}, problem
		}
		due, eligible := idle.releaseAt(grace)
		idleSince, releaseDue := "", ""
		if eligible {
			idleSince, releaseDue = idle.Since.UTC().Format(time.RFC3339), due.UTC().Format(time.RFC3339)
		}
		list.Rows = append(list.Rows, map[string]string{
			"machine": r.MachineName, "sku": orNone(r.SKU),
			"state": r.State, "failure": orNone(r.Failure.Code), "uptime": rentalUptime(r.RentedAt),
			"running": strconv.Itoa(idle.Running), "queued": strconv.Itoa(idle.Queued),
			"idle":   idleCell(idle, grace),
			"rental": r.ID, "bought for": orNone(boughtFor[r.ID]),
			"accelerator": r.AcceleratorModel, "address": r.Address,
			"media": r.MediaAddress, "hub": r.Hub,
			"rented": stamp(r.RentedAt), "ready": orNone(stamp(r.ReadyAt)),
			"idle_since": idleSince, "release_due": releaseDue,
			"image": r.Failure.BaseWorkerImageDigest, "provider": r.Failure.Provider,
			"provider resource": r.Failure.ProviderResourceID, "provider host": r.Failure.ProviderHostID,
			"provider state": r.Failure.ProviderState, "container state": r.Failure.ContainerState,
		})
		typed := map[string]any{
			"machine": r.MachineName, "state": r.State, "rental_id": r.ID,
			"running": idle.Running, "queued": idle.Queued,
			"hourly_rate_usd_micros": r.HourlyRateUSDMicros,
		}
		for key, value := range map[string]string{"sku": r.SKU, "accelerator": r.AcceleratorModel,
			"address": r.Address, "media_address": r.MediaAddress, "hub": r.Hub,
			"rented_at": r.RentedAt, "ready_at": r.ReadyAt, "bought_for": boughtFor[r.ID],
			"idle_since_at": idleSince, "release_due_at": releaseDue} {
			if value != "" {
				typed[key] = value
			}
		}
		if eligible {
			typed["idle_s"] = int64(time.Since(idle.Since).Seconds())
		}
		if r.Failure.Code != "" {
			haveFailure = true
			typed["failure_code"] = r.Failure.Code
			typed["base_worker_image_digest"], typed["provider"] = r.Failure.BaseWorkerImageDigest, r.Failure.Provider
			typed["provider_resource_id"], typed["provider_host_id"] = r.Failure.ProviderResourceID, r.Failure.ProviderHostID
			typed["provider_state"], typed["container_state"] = r.Failure.ProviderState, r.Failure.ContainerState
		}
		list.TypedRows = append(list.TypedRows, typed)
	}
	if haveFailure {
		list.Fields = []string{"machine", "sku", "state", "failure", "uptime", "running", "queued", "idle"}
		list.TypedFields = []string{"machine", "sku", "state", "rental_id", "rented_at",
			"running", "queued", "idle_s", "release_due_at", "failure_code"}
	}
	if len(list.Rows) > 0 {
		list.Next = []string{"cozy rental end " + list.Rows[0]["machine"]}
	}
	return list, nil
}

// jsonFact is an aggregate only a program reads: the terminal already says it in the lead.
type jsonFact struct{ V any }

func (jsonFact) Human() string                  { return "" }
func (f jsonFact) MarshalJSON() ([]byte, error) { return json.Marshal(f.V) }

// idleCell is the IDLE column (cl-114): how long the machine has sat idle over the grace
// that ends it — `41s / 30m` — from the actual rentals.idle_release_s policy. Blank when
// no idle clock is running: the machine is busy, still booting, leaving, or the policy
// is off.
func idleCell(idle rentalIdleness, grace time.Duration) string {
	due, eligible := idle.releaseAt(grace)
	if !eligible {
		return ""
	}
	return idleClock(time.Since(idle.Since)) + " / " + idleClock(due.Sub(idle.Since))
}

// idleClock spells a countdown duration the way a person reads a clock: `0s`, `41s`,
// `1m30s`, `30m` — whole seconds, no zero units.
func idleClock(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	d = d.Round(time.Second)
	if d == 0 {
		return "0s"
	}
	var b strings.Builder
	if h := d / time.Hour; h > 0 {
		fmt.Fprintf(&b, "%dh", h)
		d -= h * time.Hour
	}
	if m := d / time.Minute; m > 0 {
		fmt.Fprintf(&b, "%dm", m)
		d -= m * time.Minute
	}
	if s := d / time.Second; s > 0 {
		fmt.Fprintf(&b, "%ds", s)
	}
	return b.String()
}

func rentalUptime(started string) string {
	stamp, err := time.Parse(time.RFC3339Nano, started)
	if err != nil {
		return "unknown"
	}
	return roughDuration(time.Since(stamp))
}

func roughDuration(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	if d < time.Minute {
		return "<1m"
	}
	return d.Round(time.Minute).String()
}

// idleShutdownNote is the one sentence the rental list owes: what ends an idle machine.
func idleShutdownNote(grace time.Duration) string {
	if grace <= 0 {
		return "Idle machines are never shut down automatically; end them with `cozy rental end`."
	}
	return "Idle machines shut down after " + plainDuration(grace) + "."
}

// plainDuration spells a grace the way a person would: "5 minutes", "90 seconds", "2 hours".
func plainDuration(d time.Duration) string {
	switch {
	case d%time.Hour == 0:
		return plural(int(d/time.Hour), "hour")
	case d%time.Minute == 0:
		return plural(int(d/time.Minute), "minute")
	}
	return plural(int(d/time.Second), "second")
}

func plural(n int, unit string) string {
	if n == 1 {
		return "1 " + unit
	}
	return fmt.Sprintf("%d %ss", n, unit)
}

func idleReleaseNote(grace time.Duration) string {
	if grace <= 0 {
		return "rentals.idle_release_s is 0: a rental ends only through `cozy rental end`"
	}
	return fmt.Sprintf("the daemon ends a rental once nothing has been queued, running, or owed on it for %s "+
		"(rentals.idle_release_s); running work on it is what keeps it", grace)
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
	line, e := (&managedRentals{ctx: ctx, layout: l, store: st}).status()
	if e != nil {
		return e
	}
	fmt.Fprintln(ctx.Err, line)
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
		operationKey, releaseProblem := st.RequestRentalRelease(id)
		if releaseProblem != nil {
			return releaseProblem
		}
		return w.finish(l, st, operationKey, row != nil,
			"the hub already reported this rental gone")
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
		if !had {
			if problem := st.AdvanceRentalOperation(operationKey, w.id, hub.RentalReleased); problem != nil {
				return problem
			}
		}
		rental.ForgetPending(l, operationKey)
	}
	notes := []string{note + "; its media bearer, Creator key, and pinned certificate are gone from this host"}
	if !had {
		notes = []string{note + "; this host held no record of it — already released"}
	}
	if line, problem := (&managedRentals{ctx: w.ctx, layout: l, store: st}).status(); problem == nil {
		notes = append(notes, line)
	}
	return emit(w.ctx, output.Record{Fields: []output.Field{
		{K: "machine", V: w.machine}, {K: "rental", V: w.id},
		{K: "state", V: "ended"}, {K: "changed", V: forgotten},
	}, Notes: notes})
}
