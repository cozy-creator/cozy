package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
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
	if e != nil && e.ErrName() == "records_schema_upgrade_required" {
		// Only the daemon can migrate. Start it if this root is unowned, or wait
		// for an existing startup to finish. A live older daemon stays in place;
		// the reopened reader then reports its own schema requirement.
		state, _, problem := ensureDaemon(ctx)
		if problem != nil {
			return home.Layout{}, nil, problem
		}
		ctx.Daemon = state
		st, e = records.Open(l.DB)
	}
	if e != nil {
		return home.Layout{}, nil, e
	}
	return l, st, nil
}

func handleRent(ctx *Context) *exit.Error {
	skuName := strings.TrimSpace(ctx.Inv.Args[0])
	if skuName == "" {
		_, developmentSet := ctx.Inv.Bools["--development"]
		if ctx.Inv.Value("--idempotency-key") != "" ||
			ctx.Inv.Value("--timeout") != "" || len(ctx.Inv.Values["--model"]) != 0 || developmentSet || ctx.Inv.Value("--ssh-public-key") != "" {
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
	operationKey := strings.TrimSpace(ctx.Inv.Value("--idempotency-key"))
	if len(operationKey) > 200 {
		return exit.Usagef("--idempotency-key is %d bytes; the hub admits at most 200", len(operationKey))
	}
	operationKey = requestKey(operationKey)
	existing, e := st.RentalOperation(operationKey)
	if e != nil {
		return e
	}
	fleet := &managedRentals{ctx: ctx, layout: l, store: st}
	var sku hub.RentalSKU
	if existing != nil {
		// An accepted operation already owns its quote and provider obligation.
		// Resuming it neither needs current stock nor admits a second purchase.
		sku.PriceUSDMicrosPerHour = existing.HourlyRateUSDMicros
	} else {
		line, admitted, problem := fleet.admit(skuName)
		if problem != nil {
			return problem
		}
		sku = admitted
		if !ctx.Mode().JSON {
			fmt.Fprintln(ctx.Err, line)
		}
	}

	state, _, problem := ensureDaemon(ctx)
	if problem != nil {
		return problem
	}
	ctx.Daemon = state
	watchCtx, restoreInput, _, problem := liveWatchContext(ctx, context.Background(), nil)
	if problem != nil {
		return problem
	}
	defer restoreInput()
	progress := NewProgress(ctx, false, time.Now())
	completed, detached := false, false
	defer func() {
		if !completed && !detached {
			progress.On(localapi.Event{Type: "request.failed"})
		}
		progress.Done()
	}()
	progress.rentalAcquisition(hub.Rental{State: "pending_acquisition",
		AcceleratorModel: sku.AcceleratorModel, AcceleratorCount: sku.AcceleratorCount,
		HourlyRateUSDMicros: sku.PriceUSDMicrosPerHour})

	row, attachable, replay, e := acquireRentalContext(watchCtx, ctx, l, st, skuName,
		operationKey, reason, sku.PriceUSDMicrosPerHour, sku.StorageUSDMicrosPerHour,
		ctx.Cfg.RentalsMaxHourlySpendUSDMicros, deadline, "", progress.rentalAcquisition, rentalRates(fleet.unrecorded))
	if e != nil {
		if e.Code == exit.Canceled && watchCtx.Err() != nil && !ctx.Mode().JSON {
			detached = true
			progress.Done()
			fmt.Fprintln(ctx.Err, "detached from acquisition; `cozy rental list` shows its status")
			return nil
		}
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
	progress.On(localapi.Event{Type: "request.completed"})
	progress.Done()
	completed = true
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
		{K: "base_worker_image_digest", V: ready.BaseWorkerImageDigest},
		{K: "base_worker_image_tag", V: ready.BaseWorkerImageTag},
		{K: "base_worker_profile", V: ready.BaseWorkerProfile},
		{K: "changed", V: !replay}, {K: "operation", V: operationKey}, {K: "replayed", V: replay},
	}
	if ready.Development {
		fields = append(fields, output.Field{K: "ssh_address", V: ready.SSHAddress})
	}
	rec := compactRecord(fields, "machine", "state", "gpu", "changed")
	rec.Notes = notes
	rec.Next = []string{
		"cozy run <org/package/function> --rental=" + row.MachineName,
		"cozy rental end " + row.MachineName,
	}
	return emit(ctx, rec)
}

// acquireRental is the one paid mutation used by both `cozy rental new` and
// `cozy run --rental`. It returns only after the immutable retail rate and the
// worker's authenticated attach projection are durable locally.
func acquireRental(ctx *Context, l home.Layout, st *records.Store, skuName,
	operationKey, reason string, hourlyRateUSDMicros, storageUSDMicros, fleetCapUSDMicros int64,
	deadline time.Time, managedRequestID string, phase acquisitionPhase, observed map[string]int64,
) (records.Rental, hub.Rental, bool, *exit.Error) {
	return acquireRentalContext(context.Background(), ctx, l, st, skuName, operationKey,
		reason, hourlyRateUSDMicros, storageUSDMicros, fleetCapUSDMicros, deadline,
		managedRequestID, phase, observed)
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
	deadline time.Time, managedRequestID string, phase acquisitionPhase, observed map[string]int64,
) (records.Rental, hub.Rental, bool, *exit.Error) {
	observation := lifecycle
	if !deadline.IsZero() {
		var cancel context.CancelFunc
		observation, cancel = context.WithDeadline(lifecycle, deadline)
		defer cancel()
	}
	c := client(ctx)
	existing, e := st.RentalOperation(operationKey)
	if e != nil {
		return records.Rental{}, hub.Rental{}, false, e
	}
	if existing != nil && (existing.State == "rejected" || existing.State == "released") {
		if managedRequestID != "" && existing.State == "released" {
			return records.Rental{}, hub.Rental{}, false, exit.Named(exit.Unavailable, "rental.operation_superseded",
				"the prior managed rental was released before acquisition; retry its current selection")
		}
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
	if existing != nil {
		request, problem := hub.ParseRentalRequestBytes(existing.RequestBody)
		if problem != nil {
			return records.Rental{}, hub.Rental{}, false, problem
		}
		if request.SKU != skuName {
			return records.Rental{}, hub.Rental{}, false, exit.Named(exit.Conflict, "rental.idempotency_conflict",
				"rental operation %s names SKU %s, not %s", operationKey, request.SKU, skuName)
		}
	}
	var workload hub.DeclaredWorkload
	development, e := rentalDevelopment(ctx, existing)
	if e != nil {
		return records.Rental{}, hub.Rental{}, false, e
	}
	if managedRequestID == "" {
		workload.ServingModels, e = manualRentalModels(ctx, existing)
		if e != nil {
			return records.Rental{}, hub.Rental{}, false, e
		}
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
	// trip. A manual rental can declare exact models through --model; without
	// that declaration it takes the serving default.
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
			creator.PublicKey(), workload, development)
		if e != nil {
			return nil, "", e
		}
		return body, rentalRequestDigest(c.Base(), body), nil
	}
	op, replay, e := st.BeginRentalOperation(records.RentalOperation{
		Key: operationKey, Hub: c.Base(), Reason: reason, HourlyRateUSDMicros: hourlyRateUSDMicros,
		ManagedRequestID: managedRequestID,
	}, fleetCapUSDMicros, storageUSDMicros, author, observed)
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
	// The create request is deliberately not canceled with the operation: a lost
	// create answer can name a billing pod. Cancellation is sampled immediately
	// after its durable verdict, when the rental id can be released exactly.
	hctx, cancel := rentalCallContext(deadline)
	remote, answered, e := c.Rent(hctx, op.RequestBody, op.Reason, operationKey)
	cancel()
	if e != nil {
		// AN ANSWERED ASK BOUGHT A POD. Whether this client can use the answer is a
		// separate question from whether a machine is now provisioning and billing, and
		// conflating them is what orphaned six H100s: the hub's create answer omitted one
		// field, the width fence refused it, and the ask was filed as having bought
		// nothing — deleting the only name this host had for a live pod. So the identity
		// the answer DID carry is recorded, the operation stays open, and the refusal says
		// how to end the machine it just paid for.
		if answered {
			if rentalid.Valid(remote.ID) {
				state := remote.State
				if state == "" {
					state = "pending_acquisition"
				}
				if advanced := st.AdvanceRentalOperation(operationKey, remote.ID, state); advanced != nil {
					return records.Rental{}, hub.Rental{}, false, advanced
				}
			}
			return records.Rental{}, hub.Rental{}, false, e.
				WithRemedy("%s ACCEPTED this ask, so a pod may be provisioning and billing under it; "+
					"this host kept the operation and will not treat the answer as a refusal", c.Base()).
				WithNext("cozy rental end "+either(remote.ID, machineName), "cozy rental")
		}
		if e.Code == exit.Credential || e.Code == exit.Validation ||
			e.Code == exit.NotFound || e.Code == exit.Conflict {
			if advanced := st.AdvanceRentalOperation(operationKey, "", "rejected"); advanced != nil {
				return records.Rental{}, hub.Rental{}, false, advanced
			}
			rental.ForgetPending(l, operationKey)
			return records.Rental{}, hub.Rental{}, false, e
		}
		// THE KEY IS ONLY ACTIONABLE HERE (cl-135). It used to be printed on EVERY
		// rental — a minted hex string the caller never chose, naming a resume they were
		// not taking — and withheld on the one outcome where reusing it is the difference
		// between resuming this rental and paying for a second pod. The ask that started
		// this was "that doesn't mean anything for me", and it was right: a fact is noise
		// where it cannot be acted on and help where it can.
		if e.Remedy == "" {
			e = e.WithRemedy("the operation is persisted, so this exact rental can be "+
				"resumed rather than a second one paid for: cozy rental new %s "+
				"--idempotency-key %s", skuName, operationKey)
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
		AcceleratorModel: remote.AcceleratorModel, AcceleratorCount: remote.AcceleratorCount,
		HourlyRateUSDMicros: remote.HourlyRateUSDMicros,
		ManagedRequestID:    managedRequestID, State: remote.State, Hub: c.Base(),
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
	} else if e := st.RecordRentalContext(observation, row); e != nil {
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
		if e := st.RecordRentalContext(observation, row); e != nil {
			return e
		}
		return st.AdvanceRentalOperation(operationKey, seen.ID, seen.State)
	}
	attachable, e := waitRentalContext(lifecycle, ctx, c, remote.ID, deadline, observe,
		func(r hub.Rental) bool { return r.Attachable() })
	if e != nil {
		return records.Rental{}, hub.Rental{}, false, e
	}
	row, e = finishRentalAttachment(l, st, row, attachable, operationKey, token, creator)
	return row, attachable, replay, e
}

// finishRentalAttachment is the shared durable handoff for foreground acquisition
// and daemon recovery. It validates the same retained credentials before the row
// advertises an authenticated worker target.
func finishRentalAttachment(l home.Layout, st *records.Store, row records.Rental,
	attachable hub.Rental, operationKey string, token secret.Value, creator rental.CreatorIdentity,
) (records.Rental, *exit.Error) {
	if row.ID != attachable.ID || row.MachineName != attachable.Name || !attachable.Attachable() {
		return records.Rental{}, exit.Named(exit.Conflict, "rental.attach_projection_conflict",
			"rental attachment does not match the recorded ready machine")
	}
	current, problem := st.RentalOperation(operationKey)
	if problem != nil {
		return records.Rental{}, problem
	}
	if current == nil || current.RentalID != row.ID || current.State == hub.RentalReleased ||
		current.State == hub.RentalFailed || current.State == hub.RentalReleaseRequested {
		return records.Rental{}, exit.Named(exit.Conflict, "rental.operation_moved",
			"rental operation is no longer awaiting this attachment")
	}
	if current.State == "attached" {
		stored, problem := st.RentalRow(row.ID)
		if problem != nil {
			return records.Rental{}, problem
		}
		if stored == nil {
			return records.Rental{}, exit.Named(exit.Conflict, "rental.attached_record_missing", "attached rental has no local row")
		}
		return *stored, nil
	}
	row.Address, row.State = attachable.Address, attachable.State
	row.MediaAddress = attachable.MediaAddress
	row.ExpectedWorkerID, row.ExpectedWorkerBootID = attachable.WorkerID, attachable.WorkerBootID
	if !attachable.HoldsMediaHash(secret.HashHex(token)) {
		return records.Rental{}, exit.New(exit.Failed,
			"rental %s is attachable and its live credential set does not carry the token this host minted", attachable.ID).
			WithRemedy("release it and rent again; a pod nobody can authenticate to still costs money").
			WithNext("cozy rental end " + attachable.ID)
	}
	if attachable.CreatorPublicKey != creator.PublicKey() {
		return records.Rental{}, exit.Named(exit.Conflict, "rental.creator_key_changed",
			"rental %s did not retain the Creator key sent at create", attachable.ID).
			WithRemedy("release it; this host will not sign for a rental bound to another key")
	}
	if e := rental.AttachAcquisition(l, st, row, attachable.CertPEM, token, creator, operationKey); e != nil {
		return records.Rental{}, e
	}
	row.CertPath = l.RentalCert(attachable.ID)
	rental.ForgetPending(l, operationKey)
	return row, nil
}

func emitRentalCatalog(ctx *Context, skus []hub.RentalSKU) *exit.Error {
	// Cheapest first on the combined micros a renter actually pays — the rate the
	// placement decision reads — tie-broken by name so equal-priced rows hold still
	// between runs. The sort key is the column the table shows, never its
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
	human := ctx.Mode().Human && !ctx.Mode().JSON
	acceleratorColumn := "gpu"
	if human {
		acceleratorColumn = "accelerator"
	}
	rows := make([]map[string]string, 0, len(ladder))
	for _, sku := range ladder {
		// The displayed price includes both compute and storage. Structured output
		// retains the components and the provider accelerator identity.
		rows = append(rows, map[string]string{
			"name": sku.Name, acceleratorColumn: acceleratorLabel(sku.AcceleratorModel, sku.AcceleratorCount),
			"accelerator model": sku.AcceleratorModel,
			"accelerator count": strconv.Itoa(sku.AcceleratorCount),
			"compute":           computeCapabilityText(sku.ComputeCapability),
			"vram":              fmt.Sprintf("%d GB", sku.VRAMGB),
			"gpu price":         rentalPrice(sku.PriceUSDMicrosPerHour),
			"storage price":     rentalPrice(sku.StorageUSDMicrosPerHour),
			"price":             rentalPrice(total(sku)),
		})
	}
	doc := output.List{
		Name:   "gpus",
		Fields: []string{"name", acceleratorColumn, "compute", "vram", "price"},
		AllFields: []string{"name", "gpu", "accelerator model", "accelerator count", "compute",
			"vram", "gpu price", "storage price", "price"},
		Rows: rows, Total: len(rows),
		Next: []string{"cozy rental new <machine-slug>"},
	}
	if human {
		doc.AllFields = doc.Fields
	}
	return emit(ctx, doc)
}

// acceleratorLabel is how a machine READS in the GPU column: its card, and — because a
// product is sold at a WIDTH — how many of them one rental delivers. The width belongs
// beside the card rather than in a column of its own: `4x NVIDIA H100 80GB HBM3` is one
// machine with four cards, and the VRAM figure next to it stays the ONE-CARD figure,
// which is the number that decides fit (every rank of a group holds the full weights).
func acceleratorLabel(model string, count int) string {
	name := acceleratorName(model)
	if count > 1 {
		return fmt.Sprintf("%dx %s", count, name)
	}
	return name
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
				"rental %s acquisition was cancelled", id)
		}
		hctx, cancel := rentalCallContext(deadline)
		stopCancel := context.AfterFunc(lifecycle, cancel)
		r, e := c.Rental(hctx, id)
		stopCancel()
		cancel()
		if lifecycle.Err() != nil {
			return hub.Rental{}, exit.New(exit.Canceled, "rental %s acquisition watch stopped", id)
		}
		if e != nil && !transient(e) {
			return hub.Rental{}, e
		}
		if e != nil {
			if e.Message != said && !ctx.Mode().JSON {
				said = e.Message
				fmt.Fprintf(ctx.Err, "  hub: %s; retrying\n", e.Message)
			}
			if timedOut := pastDeadline(id, "unreachable", deadline); timedOut != nil {
				return hub.Rental{}, timedOut
			}
			select {
			case <-lifecycle.Done():
				return hub.Rental{}, exit.New(exit.Canceled,
					"rental %s acquisition was cancelled", id)
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
				WithNext("cozy rental new <machine-slug>")
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
		if ctx.Mode().Full && !ctx.Mode().JSON && r.Detail != "" && r.Detail != said {
			said = r.Detail
			fmt.Fprintf(ctx.Err, "  %s: %s\n", r.State, r.Detail)
		}
		if timedOut := pastDeadline(id, r.State, deadline); timedOut != nil {
			return hub.Rental{}, timedOut
		}
		select {
		case <-lifecycle.Done():
			return hub.Rental{}, exit.New(exit.Canceled,
				"rental %s acquisition was cancelled", id)
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

// either is the first non-empty of two names for the same thing.
func either(preferred, fallback string) string {
	if preferred != "" {
		return preferred
	}
	return fallback
}

func rentalFailureCode(r hub.Rental) string {
	if r.Failure != nil && r.Failure.Code != "" {
		return r.Failure.Code
	}
	return "rental.provision_failed"
}

func rentalProvisionFailure(id string, r hub.Rental) *exit.Error {
	refusal := exit.Named(exit.Failed, rentalFailureCode(r),
		"rental %s failed to provision: %s", id, detailOr(r.Detail))
	if rentalFailureCode(r) == "provider_create_did_not_happen" {
		return refusal.WithRemedy("Please try again later or rent a different GPU.").
			WithNext("cozy rental new")
	}
	return refusal.
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
	if r.Development && r.SSHAddress == "" {
		return "development SSH address"
	}
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
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	var legacy *records.Store
	var legacyFleet *managedRentals
	defer func() {
		if legacy != nil {
			legacy.Close()
		}
	}()
	var last time.Time
	fetch := func(call context.Context) (output.List, *exit.Error) {
		reconcile := last.IsZero() || time.Since(last) >= pollCadence
		var inventory api.RentalInventory
		if legacy == nil {
			inventory, problem = client.RentalInventory(call, reconcile)
			// Older daemons have no inventory route. Their compatible local store
			// remains a migration bridge; a schema mismatch refuses without
			// replacing the owner. All other API errors stay authoritative.
			if problem != nil && problem.Code == exit.NotFound && problem.ErrName() == "unknown_route" {
				var layout home.Layout
				layout, legacy, problem = rentalStores(ctx)
				if problem == nil {
					legacyFleet = &managedRentals{ctx: ctx, layout: layout, store: legacy}
				}
			}
		}
		if legacy != nil {
			inventory, problem = readRentalInventory(legacy, legacyFleet, reconcile)
		}
		if problem != nil {
			return output.List{}, problem
		}
		if reconcile {
			last = time.Now()
		}
		ctx.exitCode = 0
		if inventory.HubUnanswered != nil {
			ctx.exitCode = 1
		}
		list := renderRentalList(inventory)
		if watching {
			list.Next = nil
		}
		return list, nil
	}
	if watching {
		return watchList(ctx, "machine", fetch)
	}
	list, problem := fetch(context.Background())
	if problem != nil {
		return problem
	}
	return emit(ctx, list)
}

// renderRentalList formats the daemon's public read model; it never opens SQLite.
func renderRentalList(inventory api.RentalInventory) output.List {
	inventory = inventory.Current()
	count, burn := inventory.MachinesRunning, inventory.HourlySpendUSDMicros
	rows, unrecorded := inventory.Rentals, inventory.Unrecorded
	grace := time.Duration(inventory.IdleReleaseSeconds) * time.Second
	list := output.List{
		Name:   "rentals",
		Fields: []string{"machine", "sku", "state", "$/hour", "uptime", "running", "queued", "idle"},
		AllFields: []string{"machine", "sku", "state", "$/hour", "failure", "uptime", "running", "queued", "idle",
			"rental", "bought for", "accelerator", "address", "media", "hub", "rented", "ready",
			"idle_since", "release_due", "image", "provider", "provider resource",
			"provider host", "provider state", "container state"},
		// The machine document carries the underlying facts, never the table's
		// spellings: counts as numbers, moments as timestamps, absences omitted.
		TypedFields: []string{"machine", "sku", "state", "rental_id", "rented_at",
			"running", "queued", "idle_s", "release_due_at"},
		TypedAllFields: []string{"machine", "sku", "state", "rental_id", "bought_for",
			"accelerator", "accelerator_count", "address", "media_address", "hub", "rented_at", "ready_at",
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
		Next:  []string{"cozy help rental"},
	}
	if problem := inventory.HubUnanswered; problem != nil {
		list.Lead = []string{problem.Message,
			"Showing this host's last local records, which may be out of date; " +
				"account totals and machines this host never recorded are unknown."}
		list.Aggregates = []output.Field{{K: "live", V: jsonFact{false}}, {K: "hub_error", V: jsonFact{projectError(problem)}}}
		if problem.Remedy != "" {
			list.Trail = append(list.Trail, "Try: "+problem.Remedy)
		}
	} else {
		list.Aggregates = append(list.Aggregates, output.Field{K: "live", V: jsonFact{true}})
	}
	haveFailure := false
	for _, r := range rows {
		activity := api.RentalActivity{}
		if r.Activity != nil {
			activity = *r.Activity
		}
		idleSince, releaseDue := activity.IdleSince, activity.ReleaseDue
		since, sinceErr := time.Parse(time.RFC3339, idleSince)
		due, dueErr := time.Parse(time.RFC3339, releaseDue)
		eligible := sinceErr == nil && dueErr == nil
		idleText := ""
		if eligible {
			idleText = idleClock(time.Since(since)) + " / " + idleClock(due.Sub(since))
		}
		list.Rows = append(list.Rows, map[string]string{
			"machine": r.MachineName, "sku": orNone(r.SKU),
			"state": humanRentalState(r.State), "failure": orNone(r.Failure.Code), "uptime": rentalUptime(r.RentedAt),
			"running": strconv.Itoa(activity.Running), "queued": strconv.Itoa(activity.Queued),
			"idle":   idleText,
			"rental": r.ID, "bought for": orNone(r.BoughtFor),
			"accelerator": acceleratorLabel(r.AcceleratorModel, r.AcceleratorCount),
			"address":     r.Address,
			"media":       r.MediaAddress, "hub": r.Hub,
			"rented": stamp(r.RentedAt), "ready": orNone(stamp(r.ReadyAt)),
			"idle_since": idleSince, "release_due": releaseDue,
			"image": r.Failure.BaseWorkerImageDigest, "provider": r.Failure.Provider,
			"provider resource": r.Failure.ProviderResourceID, "provider host": r.Failure.ProviderHostID,
			"provider state": r.Failure.ProviderState, "container state": r.Failure.ContainerState,
			"$/hour": rentalHourlyRate(r.HourlyRateUSDMicros),
		})
		typed := map[string]any{
			"machine": r.MachineName, "state": r.State, "rental_id": r.ID,
			"running": activity.Running, "queued": activity.Queued,
			"hourly_rate_usd_micros": r.HourlyRateUSDMicros,
			"accelerator_count":      r.AcceleratorCount,
		}
		for key, value := range map[string]string{"sku": r.SKU, "accelerator": r.AcceleratorModel,
			"address": r.Address, "media_address": r.MediaAddress, "hub": r.Hub,
			"rented_at": r.RentedAt, "ready_at": r.ReadyAt, "bought_for": r.BoughtFor,
			"idle_since_at": idleSince, "release_due_at": releaseDue} {
			if value != "" {
				typed[key] = value
			}
		}
		if eligible {
			typed["idle_s"] = int64(time.Since(since).Seconds())
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
		list.Fields = []string{"machine", "sku", "state", "$/hour", "failure", "uptime", "running", "queued", "idle"}
		list.TypedFields = []string{"machine", "sku", "state", "rental_id", "rented_at",
			"running", "queued", "idle_s", "release_due_at", "failure_code"}
	}
	// THE HUB'S HALF (cl-199, on th-199). Everything above is what this host FILED, and
	// the incident of 2026-09-07 is the gap between that and what the account is
	// actually paying for: six H100 NVLs booting at $3.19/hour, none of them filed
	// here, so this board — the one command a person types to ask what they are
	// spending — showed an empty fleet. The hub is asked what it bills this account
	// for, and any live rental with no local row is a row here, marked as such.
	for _, seen := range unrecorded {
		list.Rows = append(list.Rows, map[string]string{
			// The hub publishes no Cozy SKU NAME for a rental, only the accelerator it
			// bought. The card is what the SKU column exists to tell a reader — the
			// difference between an H100 and a 4090 is the difference between $3.19 and
			// $0.74 an hour — so the cell carries the card rather than a dash.
			"machine": seen.MachineName,
			"sku":     orNone(acceleratorLabel(seen.AcceleratorModel, seen.AcceleratorCount)),
			"state":   humanRentalState(seen.State),
			"failure": "—", "uptime": rentalUptime(seen.RentedAt),
			"running": "—", "queued": "—", "idle": "—",
			"rental": seen.ID, "bought for": "—",
			"accelerator": acceleratorLabel(seen.AcceleratorModel, seen.AcceleratorCount),
			"address":     seen.Address, "media": seen.MediaAddress, "hub": seen.Hub,
			"rented": orNone(seen.RentedAt), "ready": "—", "idle_since": "", "release_due": "",
			"image": "", "provider": "", "provider resource": "", "provider host": "",
			"provider state": seen.ProviderState, "container state": seen.ContainerState,
			"$/hour": rentalHourlyRate(seen.HourlyRateUSDMicros),
		})
		typed := map[string]any{
			"machine": seen.MachineName, "state": seen.State, "rental_id": seen.ID,
			"hourly_rate_usd_micros": seen.HourlyRateUSDMicros,
			"accelerator_count":      seen.AcceleratorCount, "recorded": false,
		}
		for key, value := range map[string]string{"accelerator": seen.AcceleratorModel,
			"address": seen.Address, "media_address": seen.MediaAddress,
			"hub": seen.Hub, "rented_at": seen.RentedAt} {
			if value != "" {
				typed[key] = value
			}
		}
		list.TypedRows = append(list.TypedRows, typed)
	}
	if len(unrecorded) > 0 {
		var unrecordedBurn int64
		for _, seen := range unrecorded {
			unrecordedBurn += seen.HourlyRateUSDMicros
		}
		list.TypedFields = append(list.TypedFields, "recorded")
		list.TypedAllFields = append(list.TypedAllFields, "recorded")
		list.Aggregates = append(list.Aggregates,
			output.Field{K: "unrecorded_rentals", V: jsonFact{len(unrecorded)}},
			output.Field{K: "unrecorded_hourly_spend_usd_micros", V: jsonFact{unrecordedBurn}})
	}
	// UNSETTLED PAID ASKS ARE PART OF THE FLEET (cl-193). A board built only from attached
	// rows shows zero machines while an accepted ask provisions a pod that is billing, and
	// that is not a display detail — it is the difference between noticing a runaway charge
	// and finding it in the invoice. An ask the hub has now NAMED is already a row above;
	// what stays here is the half the hub could not or would not answer for.
	unattached := len(inventory.Pending)
	for _, op := range inventory.Pending {
		machine := op.MachineName
		list.Rows = append(list.Rows, map[string]string{
			"machine": machine, "sku": orNone(op.SKU), "state": humanRentalState(op.State),
			"failure": "—", "uptime": rentalUptime(op.RentedAt), "running": "0", "queued": "0",
			"idle": "—", "rental": orNone(op.ID), "bought for": orNone(op.BoughtFor),
			"accelerator": "—", "address": "", "media": "", "hub": op.Hub,
			"rented": stamp(op.RentedAt), "ready": "—", "idle_since": "", "release_due": "",
			"image": "", "provider": "", "provider resource": "", "provider host": "",
			"provider state": "", "container state": "",
			"$/hour": rentalHourlyRate(op.HourlyRateUSDMicros),
		})
		typed := map[string]any{
			"machine": machine, "state": op.State, "rental_id": op.ID,
			"running": 0, "queued": 0, "hourly_rate_usd_micros": op.HourlyRateUSDMicros,
			"accelerator_count": 0, "operation": op.Operation, "attached": false,
		}
		for key, value := range map[string]string{"sku": op.SKU, "hub": op.Hub,
			"rented_at": op.RentedAt, "bought_for": op.BoughtFor} {
			if value != "" {
				typed[key] = value
			}
		}
		list.TypedRows = append(list.TypedRows, typed)
	}
	if unattached > 0 {
		list.AllFields = append(list.AllFields, "operation")
		list.TypedAllFields = append(list.TypedAllFields, "operation", "attached")
		list.Lead = append(list.Lead, fmt.Sprintf(
			"Paid asks with no attached machine: %d (a pod may be provisioning and billing under each)", unattached))
		list.Aggregates = append(list.Aggregates,
			output.Field{K: "unattached_rental_operations", V: jsonFact{unattached}})
	}
	if inventory.HubUnanswered != nil {
		for _, row := range list.Rows {
			row["state"] += " (unverified)"
		}
	}
	return list
}

// rentalHourlyRate uses the known rental rate; a missing quote is not free.
func rentalHourlyRate(micros int64) string {
	if micros <= 0 {
		return "unknown"
	}
	return usdPerHourBare(micros)
}

// jsonFact is an aggregate only a program reads: the terminal already says it in the lead.
type jsonFact struct{ V any }

func (jsonFact) Human() string                  { return "" }
func (f jsonFact) MarshalJSON() ([]byte, error) { return json.Marshal(f.V) }

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
		// The sentence used to read as a product stance. It is a CONFIG READOUT: the
		// shipped default is 300 s, and this host has switched the policy off. Naming
		// the setting is the difference between "that is how it works" and "that is how
		// you set it up", and the reader is the person paying for the difference.
		return "Idle machines are never shut down automatically (rentals.idle_release_s is 0)."
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
//
// ABSENCE HAS TO BE PROVED, and a 404 only proves it for an id the hub can look up
// (cl-193). A machine word is Creator's own alias — `GET /v1/rentals/aeirik` is a
// well-formed request for a key that hub indexes nothing under, so its 404 says exactly
// nothing about the pod. Reading it as `state: ended` is what let six provisioned H100s
// keep billing while this command reported success, so the subject is resolved through
// everything this host recorded FIRST, and an unresolvable subject refuses rather than
// answering about Creator's own empty view.
func handleRentRelease(ctx *Context) *exit.Error {
	subject := strings.TrimSpace(ctx.Inv.Args[0])
	l, st, e := rentalStores(ctx)
	if e != nil {
		return e
	}
	defer st.Close()
	known, e := rental.Resolve(st, subject)
	if e != nil {
		return e
	}
	row := known.Row
	c := client(ctx)
	if known.Hub != "" && known.Hub != c.Base() {
		return exit.Named(exit.Conflict, "rental.hub_mismatch",
			"rental %s was rented from %s, not the configured hub %s",
			either(known.RentalID, subject), known.Hub, c.Base()).
			WithRemedy("point TENSORHUB_URL at the hub that holds the pod; a 404 from another hub says nothing about it")
	}
	// An ask this host recorded but never got an id back from is the exact shape of the
	// live incident: the pod is provisioned and billing, and the only handle on it is the
	// idempotency key the ask was sent under. Replaying that key is not a second purchase —
	// it is the hub's own contract for recovering the identity of the first one.
	if known.RentalID == "" && known.Operation != nil {
		// A REFUSED ask bought nothing, and replaying it would be a purchase rather than a
		// recovery. That is the one absence this host can prove without asking anyone.
		if known.Operation.State == "rejected" {
			return emit(ctx, output.Record{Fields: []output.Field{
				{K: "machine", V: either(known.Machine, subject)}, {K: "rental", V: ""},
				{K: "state", V: "ended"}, {K: "changed", V: false}, {K: "forgotten", V: false},
			}, Summary: []string{"No remote machine was created for " + either(known.Machine, subject) + "."},
				Next: []string{"cozy rental list"}, Notes: []string{known.Operation.Hub + " REFUSED this ask (operation " +
					known.Operation.Key + "), so no pod was ever created under it"}})
		}
		id, problem := learnRentalIdentity(ctx, l, st, known.Operation)
		if problem != nil {
			return problem
		}
		if id == "" {
			return exit.Named(exit.Conflict, "rental.identity_unknown",
				"this host asked %s for machine %s under operation %s and never learned the rental id it was given",
				known.Operation.Hub, either(known.Machine, subject), known.Operation.Key).
				WithRemedy("a pod may be provisioned and billing under that ask, and Creator cannot release what it " +
					"cannot name; have the hub name the rentals this account owns and release that id there").
				WithNext("cozy rental")
		}
		known.RentalID = id
	}
	// THE HUB CAN NOW BE ASKED WHO OWNS WHAT (cl-199, on th-199). A machine word is
	// Creator's own alias, so before this route existed an unrecorded name could not be
	// looked up at all and the command could only refuse. The account listing turns the
	// word back into the id the hub minted, which is what a DELETE takes — and it is the
	// ONLY path to a pod whose local record was never written, was lost, or belongs to
	// another host.
	listed, listing := false, 0
	if known.RentalID == "" {
		hctx, cancel := hub.Context()
		remote, published, problem := c.Rentals(hctx)
		cancel()
		if problem != nil {
			return problem.WithRemedy("the hub could not be asked which rentals this account owns; " +
				"a pod may still be billing under this name and nothing here has been changed")
		}
		listed, listing = published, len(remote)
		for _, seen := range remote {
			if seen.ID != subject && seen.Name != subject {
				continue
			}
			known.RentalID = seen.ID
			if known.Machine == "" {
				known.Machine = seen.Name
			}
			break
		}
	}
	id := known.RentalID
	if id == "" {
		// Nothing local matches. The caller may still be naming a hub id this host never
		// recorded, so ask — but the answer is only believed when the hub AFFIRMS.
		id = subject
	}
	rctx, stop, _, inputProblem := liveWatchContext(ctx, context.Background(), nil)
	if inputProblem != nil {
		return inputProblem
	}
	defer stop()
	machine := known.Machine
	if machine == "" {
		machine = subject
	}
	w := releaseWatch{ctx: ctx, c: c, id: id, machine: machine, rctx: rctx}

	seen, verdict, e := w.observe()
	if e != nil {
		return e
	}
	if verdict == rentalAbsent && !known.Recorded() {
		// A listing CHANGES what this refusal is entitled to say. Without one the 404 is
		// a failed lookup and the honest answer is "I cannot tell"; with one the hub has
		// enumerated every rental this account owns and none of them is this name, which
		// is the proof of absence the whole verb was missing.
		if listed {
			return exit.Named(exit.NotFound, "rental.unknown",
				"%s bills this account for %d rental(s) and none of them is %q",
				c.Base(), listing, subject).
				WithRemedy("`cozy rental list` names every machine this account is billed for; " +
					"nothing is billing under this name").
				WithNext("cozy rental list")
		}
		return exit.Named(exit.NotFound, "rental.unknown",
			"this host holds no record of %q and %s answered 404 for that exact key", subject, c.Base()).
			WithRemedy("that 404 is not proof the machine is gone: the hub identifies a rental by the opaque id it "+
				"minted, which a machine word is not, so an unrecorded name cannot be looked up at all. "+
				"This hub publishes no rental listing (th-199), so name the rental id instead").
			WithNext("cozy rental", "cozy rental end <rental-id>")
	}
	if verdict == rentalAbsent && (known.Operation == nil || known.Operation.State != hub.RentalReleased) {
		return rentalReleaseUnconfirmed(id)
	}
	if problem := st.RequestRetainedRentalAbandonment(id, "cozy rental end"); problem != nil {
		return problem
	}
	if verdict != rentalLive {
		operationKey, releaseProblem := st.RequestRentalRelease(id)
		if releaseProblem != nil {
			return releaseProblem
		}
		return w.finish(l, st, operationKey, row != nil, false, known.Operation,
			"the hub already reported this rental gone")
	}

	operationKey, e := st.RequestRentalRelease(id)
	if e != nil {
		return e
	}
	w.say(hub.RentalReleaseRequested, "")
	// A rental the hub already shows leaving needs no second DELETE; the poll settles it.
	if seen.State != hub.RentalReleaseRequested {
		if e := w.request(); e != nil {
			return e
		}
	}
	for {
		_, verdict, e := w.observe()
		if e != nil {
			return e
		}
		if verdict == rentalAbsent {
			return w.kept(rentalReleaseUnconfirmed(id))
		}
		if verdict != rentalLive {
			return w.finish(l, st, operationKey, row != nil, true, known.Operation,
				"the hub destroyed the pod")
		}
		select {
		case <-rctx.Done():
			return w.interrupted()
		case <-time.After(pollCadence):
		}
	}
}

// learnRentalIdentity recovers the hub identity of a paid ask this host never got one back
// from. It replays the ask under its ORIGINAL idempotency key, which is not a second
// purchase but the hub's own contract for answering with the rental the first ask created.
//
// It is the only way out of the state that cost real money: the create answer was received
// and REFUSED locally, so a pod was provisioned and billing while nothing here held its
// name. It returns "" only when the hub proves the ask created nothing.
func learnRentalIdentity(ctx *Context, l home.Layout, st *records.Store,
	op *records.RentalOperation,
) (string, *exit.Error) {
	sub := *ctx
	sub.Cfg.HubURL = op.Hub
	sub.Cfg.HubURLSource = "rental operation"
	hctx, cancel := hub.LongContext()
	seen, answered, problem := client(&sub).Rent(hctx, op.RequestBody, op.Reason, op.Key)
	cancel()
	// A REFUSAL created nothing, so the ask may be settled as having bought nothing. An
	// ANSWERED ask did create something, whatever this client makes of the body, so its
	// identity is kept and the operation stays open.
	if problem != nil && !answered {
		if problem.Code == exit.Credential || problem.Code == exit.Validation ||
			problem.Code == exit.NotFound || problem.Code == exit.Conflict {
			if advanced := st.AdvanceRentalOperation(op.Key, "", "rejected"); advanced != nil {
				return "", advanced
			}
			rental.ForgetPending(l, op.Key)
			return "", nil
		}
		return "", problem
	}
	if !rentalid.Valid(seen.ID) {
		return "", exit.Named(exit.Conflict, "rental.identity_unnamed",
			"%s answered the replay of operation %s without a usable rental id", op.Hub, op.Key).
			WithRemedy("upgrade Tensorhub; a pod may be provisioned under this ask and only its id can release it")
	}
	state := seen.State
	if state == "" {
		state = "pending_acquisition"
	}
	if advanced := st.AdvanceRentalOperation(op.Key, seen.ID, state); advanced != nil {
		return "", advanced
	}
	op.RentalID = seen.ID
	return seen.ID, nil
}

type releaseWatch struct {
	ctx     *Context
	c       *hub.Client
	id      string
	machine string
	rctx    context.Context
	said    string
}

// rentalVerdict is what the hub said about the id it was asked, and the three answers are
// deliberately not two. `rentalReleased` is the hub AFFIRMING a rental it knows and has
// torn down; `rentalAbsent` is a 404, which affirms nothing — it means only that nothing
// is filed under the key that was sent, whether because the pod is gone or because the key
// was never one the hub could resolve. Collapsing the two is the whole defect (cl-193).
type rentalVerdict int

const (
	rentalLive rentalVerdict = iota
	rentalReleased
	rentalAbsent
)

// observe reads the rental until the hub gives a verdict. Transport faults are retried at
// cadence: they say nothing about the pod, and a release that gave up on them would leave
// the local half of a billing pod deleted or orphaned on a guess.
func (w *releaseWatch) observe() (hub.Rental, rentalVerdict, *exit.Error) {
	for {
		r, e := w.c.RentalView(w.rctx, w.id)
		switch {
		case e == nil && r.State == hub.RentalReleased:
			return r, rentalReleased, nil
		case e == nil:
			w.say(r.State, r.Detail)
			return r, rentalLive, nil
		case e.Code == exit.NotFound:
			return hub.Rental{}, rentalAbsent, nil
		case !transient(e):
			return hub.Rental{}, rentalLive, w.kept(e)
		}
		w.say("hub", e.Message+"; retrying")
		select {
		case <-w.rctx.Done():
			return hub.Rental{}, rentalLive, w.interrupted()
		case <-time.After(pollCadence):
		}
	}
}

func rentalReleaseUnconfirmed(id string) *exit.Error {
	return exit.Named(exit.Conflict, "rental.hub_record_missing",
		"Tensorhub has not confirmed release of rental %s; its missing record is not proof of provider destruction", id).
		WithRemedy("keep the local rental and operation records; reconcile Tensorhub with its provider before forgetting the machine")
}

// request sends the DELETE. Only an affirmative release observation can finish it.
func (w *releaseWatch) request() *exit.Error {
	for {
		e := w.c.Release(w.rctx, w.id, "cozy rental end")
		switch {
		case e == nil:
			return nil
		case e.Code == exit.NotFound:
			return w.kept(rentalReleaseUnconfirmed(w.id))
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
	if w.ctx.Mode().JSON {
		return
	}
	line := ""
	switch state {
	case hub.RentalReleaseRequested:
		line = "Shutting down remote machine..."
	case "hub":
		line = "Waiting for Tensorhub; retrying..."
		if w.ctx.Mode().Full && detail != "" {
			line += " " + detail
		}
	default:
		return
	}
	if line == w.said {
		return
	}
	w.said = line
	fmt.Fprintln(w.ctx.Err, line)
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

// finish states what this command did to the WORLD, not to Creator's filing cabinet.
// `changed` used to mean "a local row was deleted", which is why a command that released
// nothing and forgot nothing could answer `state: ended, changed: false` and read as an
// accomplished teardown. It now means: this command found a live pod and the hub destroyed
// it. An idempotent second `rental end` still answers `changed: false` — and now says why
// it is entitled to, naming the record that proves this host once held the machine.
func (w *releaseWatch) finish(l home.Layout, st *records.Store, operationKey string, had, destroyed bool,
	op *records.RentalOperation, note string) *exit.Error {
	if problem := st.CompleteRetainedRentalAbandonment(w.id, "cozy rental end"); problem != nil {
		return problem
	}
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
		switch {
		case destroyed:
			notes = []string{note + "; this host held no local record of it and released it by the id given"}
		case op != nil:
			notes = []string{note + "; this host bought it under operation " + op.Key + " and holds no live record of it"}
		default:
			notes = []string{note + "; this host holds no live record of it"}
		}
	}
	message := w.machine + " shut down."
	if !destroyed {
		message = w.machine + " is already shut down."
	}
	summary := []string{message,
		"Temporary pod files are gone. Local outputs and uploaded checkpoints remain."}
	if line, problem := (&managedRentals{ctx: w.ctx, layout: l, store: st}).status(); problem == nil {
		notes = append(notes, line)
		summary = append(summary, line)
	} else {
		summary = append(summary, "Current rental count and spend are unavailable.")
	}
	return emit(w.ctx, output.Record{Fields: []output.Field{
		{K: "machine", V: w.machine}, {K: "rental", V: w.id},
		{K: "state", V: "ended"}, {K: "changed", V: destroyed},
		{K: "forgotten", V: forgotten},
	}, Summary: summary, Notes: notes, Next: []string{"cozy rental list"}})
}
