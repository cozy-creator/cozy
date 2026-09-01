package records

import (
	"database/sql"
	"errors"
	"strings"
	"time"
	"unicode"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/rentalid"
)

// The RENTAL half of the one lifecycle authority (cl-015). A rented pod outlives the
// process that rented it — that is the whole difference between it and a spawned worker —
// so what it is lives in the same database as everything else durable here, and NOT in a
// sidecar file that would outlive the fact it describes.
//
// What is deliberately NOT a column: the provisioned media bearer. Every reader of this
// database would be a reader of that credential; the 0600 handoff beside it is the
// boundary (home.Layout.RentalMediaToken), exactly as the CLI's own credential is.

const rentalOperationsDDL = `
CREATE TABLE IF NOT EXISTS rental_operations (
  operation_key    TEXT PRIMARY KEY,
  request_digest   TEXT NOT NULL,
  request_body     BLOB NOT NULL,
  hub              TEXT NOT NULL,
  reason           TEXT NOT NULL,
  hourly_rate_usd_micros INTEGER NOT NULL,
  managed_request_id TEXT NOT NULL DEFAULT '',
  rental_id        TEXT NOT NULL DEFAULT '',
  state            TEXT NOT NULL,
  created_at       TEXT NOT NULL,
  updated_at       TEXT NOT NULL
)`

const rentalsDDL = `
CREATE TABLE IF NOT EXISTS rentals (
  id                TEXT PRIMARY KEY,
  machine_name      TEXT NOT NULL DEFAULT '',
  sku               TEXT NOT NULL DEFAULT '',
  accelerator_model TEXT NOT NULL,
  hourly_rate_usd_micros INTEGER NOT NULL,
  managed_request_id TEXT NOT NULL DEFAULT '',
  address           TEXT NOT NULL,
  cert_path         TEXT NOT NULL,
  state             TEXT NOT NULL,
  hub               TEXT NOT NULL,
  rented_at         TEXT NOT NULL,
  media_address     TEXT NOT NULL DEFAULT '',
  expected_worker_id         TEXT NOT NULL DEFAULT '',
  expected_worker_boot_id    TEXT NOT NULL DEFAULT ''
)`

var rentalSchema = []string{rentalOperationsDDL, rentalsDDL, `
CREATE UNIQUE INDEX IF NOT EXISTS rental_operation_remote
  ON rental_operations(rental_id) WHERE rental_id <> ''`, `
CREATE UNIQUE INDEX IF NOT EXISTS rentals_machine_name
  ON rentals(machine_name) WHERE machine_name<>''`}

// rentalStateRank orders the hub's lifecycle words so a delayed observation never moves a
// row backward. Unknown words are opaque: they neither advance nor regress anything.
var rentalStateRank = map[string]int{
	"pending_acquisition": 0, "acquiring": 1, "materializing": 2, "converging": 3,
	"ready": 4, "attached": 5, "failed": 6, "release_requested": 7,
	"released": 8, "rejected": 8,
}

func rentalStateForward(current, next string) string {
	cr, ck := rentalStateRank[current]
	nr, nk := rentalStateRank[next]
	if nk && (!ck || nr > cr) {
		return next
	}
	return current
}

// RentalOperation is the renter-owned half of one paid acquisition. It exists before
// the POST: the operation key and its 0600 token survive a lost response, so retry can
// ask for the same provider obligation instead of buying a second one.
type RentalOperation struct {
	Key                 string
	RequestDigest       string
	RequestBody         []byte
	Hub                 string
	Reason              string
	HourlyRateUSDMicros int64
	ManagedRequestID    string
	RentalID            string
	State               string
	CreatedAt           string
	UpdatedAt           string
}

const rentalOperationCols = `operation_key,request_digest,request_body,hub,reason,hourly_rate_usd_micros,managed_request_id,rental_id,state,created_at,updated_at`

func scanRentalOperation(row interface{ Scan(...any) error }) (RentalOperation, error) {
	var op RentalOperation
	err := row.Scan(&op.Key, &op.RequestDigest, &op.RequestBody, &op.Hub, &op.Reason,
		&op.HourlyRateUSDMicros, &op.ManagedRequestID,
		&op.RentalID, &op.State, &op.CreatedAt, &op.UpdatedAt)
	return op, err
}

// BeginRentalOperation durably installs the caller's operation identity. A replay returns
// the existing row; the caller compares RequestDigest before making any network call.
func (s *Store) BeginRentalOperation(op RentalOperation, fleetCapUSDMicros int64) (RentalOperation, bool, *exit.Error) {
	stamp := now()
	tx, err := s.db.Begin()
	if err != nil {
		return RentalOperation{}, false, exit.Internalf("cannot begin rental operation: %s", err)
	}
	defer tx.Rollback()
	stored, err := scanRentalOperation(tx.QueryRow(
		`SELECT `+rentalOperationCols+` FROM rental_operations WHERE operation_key=?`, op.Key))
	if err == nil {
		if err := tx.Commit(); err != nil {
			return RentalOperation{}, false, exit.Internalf("cannot read replayed rental operation: %s", err)
		}
		return stored, true, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return RentalOperation{}, false, exit.Internalf("cannot read rental operation: %s", err)
	}
	if op.HourlyRateUSDMicros <= 0 || fleetCapUSDMicros <= 0 {
		return RentalOperation{}, false, exit.Named(exit.Usage, "rental.spend_cap_required",
			"a positive locked hourly rate and rentals.max_hourly_spend_usd are required")
	}
	count, burn, problem := rentalFleetTotals(tx)
	if problem != nil {
		return RentalOperation{}, false, problem
	}
	if burn > fleetCapUSDMicros || op.HourlyRateUSDMicros > fleetCapUSDMicros-burn {
		return RentalOperation{}, false, exit.Named(exit.Capacity, "rental.fleet_spend_cap",
			"%d potentially billing rental(s) already reserve %d USD micros/hour; the next %d would exceed %d",
			count, burn, op.HourlyRateUSDMicros, fleetCapUSDMicros)
	}
	if _, err := tx.Exec(`INSERT INTO rental_operations(`+rentalOperationCols+`)
		VALUES(?,?,?,?,?,?,?,?,?,?,?)`, op.Key, op.RequestDigest, op.RequestBody, op.Hub, op.Reason,
		op.HourlyRateUSDMicros, op.ManagedRequestID, op.RentalID, "pending_acquisition", stamp, stamp); err != nil {
		return RentalOperation{}, false, exit.Internalf("cannot record rental operation: %s", err)
	}
	stored, err = scanRentalOperation(tx.QueryRow(
		`SELECT `+rentalOperationCols+` FROM rental_operations WHERE operation_key=?`, op.Key))
	if err != nil {
		return RentalOperation{}, false, exit.Internalf("cannot read rental operation: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return RentalOperation{}, false, exit.Internalf("cannot commit rental operation: %s", err)
	}
	return stored, false, nil
}

func (s *Store) RentalOperation(key string) (*RentalOperation, *exit.Error) {
	op, err := scanRentalOperation(s.db.QueryRow(
		`SELECT `+rentalOperationCols+` FROM rental_operations WHERE operation_key=?`, key))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read rental operation %s: %s", key, err)
	}
	return &op, nil
}

// ActiveRentalOperations is every paid acquisition/release operation whose absence has
// not been proved. A row may precede its provider rental id, so a safe daemon exit
// must fence on the operation key as well as on attached rental rows.
func (s *Store) ActiveRentalOperations() ([]RentalOperation, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + rentalOperationCols + ` FROM rental_operations
		WHERE state NOT IN ('released','rejected') ORDER BY created_at,operation_key`)
	if err != nil {
		return nil, exit.Internalf("cannot list active rental operations: %s", err)
	}
	defer rows.Close()
	var out []RentalOperation
	for rows.Next() {
		op, err := scanRentalOperation(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read an active rental operation: %s", err)
		}
		out = append(out, op)
	}
	return out, nil
}

// AdvanceRentalOperation records facts learned from the hub. Empty rentalID preserves a
// previously learned id, which keeps a later polling update from erasing the join. The
// compare-and-swap keeps a delayed poll from moving an operation behind a fact another
// caller has already committed.
func (s *Store) AdvanceRentalOperation(key, rentalID, state string) *exit.Error {
	for {
		var currentID, currentState string
		err := s.db.QueryRow(`SELECT rental_id,state FROM rental_operations
			WHERE operation_key=?`, key).Scan(&currentID, &currentState)
		if errors.Is(err, sql.ErrNoRows) {
			return exit.Internalf("rental operation %s disappeared while advancing it", key)
		}
		if err != nil {
			return exit.Internalf("cannot read rental operation %s while advancing it: %s", key, err)
		}
		if currentID != "" && rentalID != "" && currentID != rentalID {
			return exit.Named(exit.Conflict, "rental.operation_conflict",
				"rental operation %s already names rental %s, not %s", key, currentID, rentalID)
		}
		if state == "rejected" && currentID != "" {
			return exit.Named(exit.Conflict, "rental.operation_conflict",
				"rental operation %s already names rental %s and cannot be rejected", key, currentID)
		}

		nextState, e := advanceRentalOperationState(key, currentState, state)
		if e != nil {
			return e
		}
		nextID := currentID
		if nextID == "" && rentalID != "" && !rentalOperationFinal(currentState) {
			nextID = rentalID
		}
		if nextID == currentID && nextState == currentState {
			return nil
		}

		res, err := s.db.Exec(`UPDATE rental_operations SET rental_id=?,state=?,updated_at=?
			WHERE operation_key=? AND rental_id=? AND state=?`,
			nextID, nextState, now(), key, currentID, currentState)
		if err != nil {
			return exit.Internalf("cannot advance rental operation: %s", err)
		}
		n, err := res.RowsAffected()
		if err != nil {
			return exit.Internalf("cannot read rental operation advance result: %s", err)
		}
		if n == 1 {
			return nil
		}
		// Another writer won after the read. Re-evaluate against its fact instead of
		// overwriting it with the stale state this caller observed.
	}
}

func rentalOperationFinal(state string) bool {
	switch state {
	case "released", "rejected":
		return true
	}
	return false
}

func advanceRentalOperationState(key, current, next string) (string, *exit.Error) {
	if current == next || rentalOperationFinal(current) {
		return current, nil
	}
	if next == "rejected" {
		if current == "pending_acquisition" {
			return next, nil
		}
		return current, nil
	}
	return rentalStateForward(current, next), nil
}

// RequestRentalRelease records intent before the remote DELETE. It returns the operation
// key so the caller can remove a never-attached operation token after release succeeds;
// an orphan rental row with no operation releases with nothing to mark.
func (s *Store) RequestRentalRelease(id string) (string, *exit.Error) {
	var key string
	err := s.db.QueryRow(`SELECT operation_key FROM rental_operations WHERE rental_id=?`, id).Scan(&key)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", exit.Internalf("cannot read rental operation for %s: %s", id, err)
	}
	if e := s.AdvanceRentalOperation(key, id, "release_requested"); e != nil {
		return "", e
	}
	return key, nil
}

// Rental is one provider-neutral pod rental this client may attach a worker to.
type Rental struct {
	ID                  string
	MachineName         string
	SKU                 string
	AcceleratorModel    string
	HourlyRateUSDMicros int64
	ManagedRequestID    string
	Address             string
	CertPath            string
	State               string
	Hub                 string
	RentedAt            string
	// MediaAddress is where the pod's co-resident media server answers (cl-014). It is a
	// FACT about the pod like the control address is, so it is a row and not a file; the
	// credential it takes is the rental's media bearer, which stays 0600 beside it.
	MediaAddress         string
	ExpectedWorkerID     string
	ExpectedWorkerBootID string
}

const rentalCols = `id,machine_name,sku,accelerator_model,hourly_rate_usd_micros,managed_request_id,address,cert_path,state,hub,rented_at,media_address,expected_worker_id,expected_worker_boot_id`

func scanRental(row interface{ Scan(...any) error }) (Rental, error) {
	var r Rental
	err := row.Scan(&r.ID, &r.MachineName, &r.SKU, &r.AcceleratorModel,
		&r.HourlyRateUSDMicros, &r.ManagedRequestID, &r.Address, &r.CertPath,
		&r.State, &r.Hub, &r.RentedAt, &r.MediaAddress,
		&r.ExpectedWorkerID, &r.ExpectedWorkerBootID)
	return r, err
}

// RecordRental writes what the hub provisioned. It REPLACES on the rental id because the
// id is the hub's, not this host's: re-reading a rental that moved from acquisition to
// ready must land on the same row rather than accumulate one per poll. State only moves
// forward: a delayed poll answering `acquiring` after `ready` was recorded is stale.
func (s *Store) RecordRental(r Rental) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin recording rental %s: %s", r.ID, err)
	}
	defer tx.Rollback()
	if r.MachineName == "" || r.SKU == "" || r.HourlyRateUSDMicros == 0 {
		var existingName, existingSKU string
		var existingRate int64
		var existingManagedRequestID string
		err := tx.QueryRow(`SELECT machine_name,sku,hourly_rate_usd_micros,managed_request_id FROM rentals WHERE id=?`, r.ID).
			Scan(&existingName, &existingSKU, &existingRate, &existingManagedRequestID)
		if err == nil {
			if r.MachineName == "" {
				r.MachineName = existingName
			}
			if r.SKU == "" {
				r.SKU = existingSKU
			}
			if r.HourlyRateUSDMicros == 0 {
				r.HourlyRateUSDMicros = existingRate
			}
			r.ManagedRequestID = existingManagedRequestID
		} else if err != nil && !errors.Is(err, sql.ErrNoRows) {
			return exit.Internalf("cannot read rental %s local identity: %s", r.ID, err)
		}
	}
	if r.HourlyRateUSDMicros <= 0 {
		return exit.Named(exit.Conflict, "rental.hourly_rate_missing",
			"rental %s has no positive locked Cozy retail hourly rate", r.ID).
			WithRemedy("upgrade Tensorhub before accepting a rental")
	}
	if !rentalid.ValidMachineName(r.MachineName) {
		return exit.Named(exit.Validation, "rental.machine_name_invalid",
			"%q is not a machine name", r.MachineName).
			WithRemedy("use 1-32 lowercase letters, numbers, and hyphens; local is reserved")
	}
	var conflictingID string
	err = tx.QueryRow(`SELECT id FROM rentals
		WHERE id<>? AND (id=? OR machine_name=?) LIMIT 1`,
		r.ID, r.MachineName, r.MachineName).Scan(&conflictingID)
	if err == nil {
		return exit.Named(exit.Conflict, "rental.machine_name_conflict",
			"machine name %q already identifies rental %s", r.MachineName, conflictingID).
			WithRemedy("use the immutable name from the original rental operation")
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return exit.Internalf("cannot check rental machine name %s: %s", r.MachineName, err)
	}
	if r.RentedAt == "" {
		r.RentedAt = now()
	}
	var current string
	switch err := tx.QueryRow(`SELECT state FROM rentals WHERE id=?`, r.ID).Scan(&current); {
	case errors.Is(err, sql.ErrNoRows):
	case err != nil:
		return exit.Internalf("cannot read rental %s state: %s", r.ID, err)
	default:
		r.State = rentalStateForward(current, r.State)
	}
	if _, err := tx.Exec(`INSERT INTO rentals(`+rentalCols+`)
			VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
		ON CONFLICT(id) DO UPDATE SET
		  machine_name=CASE WHEN rentals.machine_name<>'' THEN rentals.machine_name ELSE excluded.machine_name END,
		  sku=CASE WHEN rentals.sku<>'' THEN rentals.sku ELSE excluded.sku END,
		  hourly_rate_usd_micros=rentals.hourly_rate_usd_micros,
		  managed_request_id=rentals.managed_request_id,
		  address=CASE WHEN rentals.address<>'' THEN rentals.address ELSE excluded.address END,
		  cert_path=CASE WHEN rentals.cert_path<>'' THEN rentals.cert_path ELSE excluded.cert_path END,
		  state=excluded.state,
		  media_address=CASE WHEN rentals.media_address<>'' THEN rentals.media_address ELSE excluded.media_address END,
		  expected_worker_id=CASE WHEN rentals.expected_worker_id<>''
		    THEN rentals.expected_worker_id ELSE excluded.expected_worker_id END,
		  expected_worker_boot_id=CASE WHEN rentals.expected_worker_boot_id<>''
		    THEN rentals.expected_worker_boot_id ELSE excluded.expected_worker_boot_id END`,
		r.ID, r.MachineName, r.SKU, r.AcceleratorModel, r.HourlyRateUSDMicros, r.ManagedRequestID,
		r.Address, r.CertPath, r.State, r.Hub,
		r.RentedAt, r.MediaAddress, r.ExpectedWorkerID, r.ExpectedWorkerBootID); err != nil {
		return exit.Internalf("cannot record rental %s: %s", r.ID, err)
	}
	stored, err := scanRental(tx.QueryRow(`SELECT `+rentalCols+` FROM rentals WHERE id=?`, r.ID))
	if err != nil {
		return exit.Internalf("cannot read back rental %s: %s", r.ID, err)
	}
	if stored.MachineName != r.MachineName || stored.SKU != r.SKU ||
		stored.HourlyRateUSDMicros != r.HourlyRateUSDMicros || stored.ManagedRequestID != r.ManagedRequestID ||
		r.Address != "" && stored.Address != r.Address ||
		r.MediaAddress != "" && stored.MediaAddress != r.MediaAddress ||
		r.CertPath != "" && stored.CertPath != r.CertPath ||
		r.ExpectedWorkerID != "" && stored.ExpectedWorkerID != r.ExpectedWorkerID ||
		r.ExpectedWorkerBootID != "" && stored.ExpectedWorkerBootID != r.ExpectedWorkerBootID {
		return exit.Named(exit.Conflict, "rental.attach_projection_conflict",
			"rental %s already carries another address, media address, certificate pin, or worker identity", r.ID)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit rental %s: %s", r.ID, err)
	}
	return nil
}

// RentalByMachine resolves either a memorable machine name or the opaque Tensorhub id.
func (s *Store) RentalByMachine(name string) (*Rental, *exit.Error) {
	r, err := scanRental(s.db.QueryRow(
		`SELECT `+rentalCols+` FROM rentals WHERE machine_name=? OR id=?`, name, name))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read rented machine %s: %s", name, err)
	}
	return &r, nil
}

// ObserveRentalWorker validates the direct ClaimAck against Tensorhub's pinned worker
// and requested SKU. Hardware observations are live control facts, not rental-row history.
func (s *Store) ObserveRentalWorker(id, accelerator, backend, driverVersion,
	backendVersion string, deviceMemory uint64, instance, workerID, bootID string, count int) *exit.Error {
	row, err := scanRental(s.db.QueryRow(`SELECT `+rentalCols+` FROM rentals WHERE id=?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return exit.New(exit.NotFound, "no rental %s on this host", id)
	}
	if err != nil {
		return exit.Internalf("cannot read rental %s for worker observation: %s", id, err)
	}
	cpu := strings.EqualFold(row.AcceleratorModel, "CPU")
	complete := row.State == "ready" || row.State == "attached"
	if cpu {
		complete = complete && accelerator == "" && backend == "none" && count == 0 &&
			driverVersion == "" && backendVersion == "" && deviceMemory == 0 &&
			instance != "" && workerID != "" && bootID != ""
	} else {
		complete = complete && accelerator != "" && backend != "" && count == 1 &&
			instance != "" && workerID != "" && bootID != ""
	}
	if !complete {
		return exit.Named(exit.Conflict, "rental.worker_readback_incomplete",
			"rental %s ClaimAck is state=%q backend=%q accelerator=%q count=%d instance=%q boot=%q",
			id, row.State, backend, accelerator, count, instance, bootID)
	}
	if row.ExpectedWorkerID != workerID || row.ExpectedWorkerBootID != bootID {
		return exit.Named(exit.Conflict, "rental.worker_identity_mismatch",
			"rental %s ClaimAck worker/boot identity differs from Tensorhub readiness", id).
			WithRemedy("release it; Creator signs only the worker and boot Tensorhub authenticated")
	}
	if !cpu && !acceleratorMatches(row.AcceleratorModel, accelerator) {
		return exit.Named(exit.Conflict, "rental.accelerator_readback_mismatch",
			"rental %s worker reports %q but the paid request selected %q",
			id, accelerator, row.AcceleratorModel).
			WithRemedy("release it; never invoke a model on hardware that disagrees with the paid selection")
	}
	return nil
}

// acceleratorMatches accepts an observed device name that carries the requested SKU as an
// in-order token subsequence, case-insensitively: "H200" and "NVIDIA H200" both name an
// observed "NVIDIA H200"; "H100" does not.
func acceleratorMatches(requested, observed string) bool {
	split := func(s string) []string {
		return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		})
	}
	want, have := split(requested), split(observed)
	if len(want) == 0 || len(have) == 0 {
		return false
	}
	i := 0
	for _, tok := range have {
		if i < len(want) && tok == want[i] {
			i++
		}
	}
	return i == len(want)
}

// RentalRow reads one rental, or nil when this host rented no such thing.
func (s *Store) RentalRow(id string) (*Rental, *exit.Error) {
	r, err := scanRental(s.db.QueryRow(
		`SELECT `+rentalCols+` FROM rentals WHERE id=?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read rental %s: %s", id, err)
	}
	return &r, nil
}

// Rentals lists what this host holds, newest first.
func (s *Store) Rentals() ([]Rental, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + rentalCols + ` FROM rentals ORDER BY rented_at DESC`)
	if err != nil {
		return nil, exit.Internalf("cannot list rentals: %s", err)
	}
	defer rows.Close()
	out := []Rental{}
	for rows.Next() {
		r, err := scanRental(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read a rental row: %s", err)
		}
		out = append(out, r)
	}
	return out, nil
}

// RentalFleetTotals counts each potentially billing obligation once. A rental
// operation with no local rental row covers the response-loss window; once its
// row exists, the immutable row rate replaces that reservation in the sum.
func (s *Store) RentalFleetTotals() (count int, hourlyRateUSDMicros int64, problem *exit.Error) {
	return rentalFleetTotals(s.db)
}

func rentalFleetTotals(q interface{ QueryRow(string, ...any) *sql.Row }) (int, int64, *exit.Error) {
	var count int
	var burn int64
	err := q.QueryRow(`SELECT COUNT(*),COALESCE(SUM(hourly_rate_usd_micros),0) FROM (
		SELECT id AS identity,hourly_rate_usd_micros FROM rentals
		UNION ALL
		SELECT o.operation_key,o.hourly_rate_usd_micros FROM rental_operations o
		LEFT JOIN rentals r ON r.id=o.rental_id
		WHERE o.state NOT IN ('released','rejected') AND r.id IS NULL
	)`).Scan(&count, &burn)
	if err != nil {
		return 0, 0, exit.Internalf("cannot total the rental fleet: %s", err)
	}
	return count, burn, nil
}

// RentalRunCounts derives the current queue and active-attempt counts for one machine.
func (s *Store) RentalRunCounts(id string) (queued, running int, problem *exit.Error) {
	err := s.db.QueryRow(`SELECT
		COALESCE(SUM(CASE WHEN state IN ('submitted','queued','requeue_pending') THEN 1 ELSE 0 END),0),
		COALESCE(SUM(CASE WHEN state IN ('dispatching','in_progress') THEN 1 ELSE 0 END),0) +
		(SELECT COUNT(*) FROM attempts a JOIN requests held ON held.id=a.request_id
		 WHERE held.worker=? AND a.state='terminal')
		FROM requests WHERE worker=?`, id, id).Scan(&queued, &running)
	if err != nil {
		return 0, 0, exit.Internalf("cannot count runs for rented machine %s: %s", id, err)
	}
	return queued, running, nil
}

// AssignManagedRentalClass batches every queued request that fits the pod class
// Creator already bought. Package and model identities remain local; the only class
// distinction used to select a generic rental is CPU versus GPU.
func (s *Store) AssignManagedRentalClass(id string, cpu bool) *exit.Error {
	predicate := "needs_accelerator=1"
	if cpu {
		predicate = "needs_accelerator=0"
	}
	if _, err := s.db.Exec(`UPDATE requests SET worker=? WHERE rental=1 AND worker=''
		AND state IN ('submitted','queued','requeue_pending') AND (`+predicate+`)`, id); err != nil {
		return exit.Internalf("cannot batch queued requests onto rental %s: %s", id, err)
	}
	return nil
}

// RentalLastSettlement returns the newest settled request assigned to one rental and
// the time its final attempt became durable. A zero ClosedAt means the request settled
// before an attempt crossed the terminal boundary.
type RentalLastSettlement struct {
	RequestID string
	Kind      string
	ClosedAt  time.Time
}

func (s *Store) RentalLastSettlement(id string) (RentalLastSettlement, bool, *exit.Error) {
	var out RentalLastSettlement
	var closedAt string
	err := s.db.QueryRow(`SELECT r.id,r.kind,COALESCE(MAX(a.closed_at),'')
		FROM requests r
		LEFT JOIN attempts a ON a.request_id=r.id AND a.state IN ('terminal','closed')
		WHERE r.worker=? AND r.rental=1
		  AND r.state IN ('succeeded','failed','canceled','refused','abandoned')
		GROUP BY r.id,r.kind,r.created_at
		ORDER BY COALESCE(NULLIF(MAX(a.closed_at),''),r.created_at) DESC,r.id DESC LIMIT 1`, id).
		Scan(&out.RequestID, &out.Kind, &closedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return RentalLastSettlement{}, false, nil
	}
	if err != nil {
		return RentalLastSettlement{}, false, exit.Internalf(
			"cannot read the last settled request for rental %s: %s", id, err)
	}
	if closedAt != "" {
		parsed, err := time.Parse(time.RFC3339Nano, closedAt)
		if err != nil {
			return RentalLastSettlement{}, false, exit.Internalf(
				"rental %s has an invalid terminal timestamp: %s", id, err)
		}
		out.ClosedAt = parsed
	}
	return out, true, nil
}

// ForgetRental removes the row once the hub reports the pod gone and closes whatever
// operation named it. It answers whether a row was there.
func (s *Store) ForgetRental(id string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin forgetting rental %s: %s", id, err)
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`UPDATE rental_operations SET state='released', updated_at=?
		WHERE rental_id=? AND state<>'released'`, now(), id); err != nil {
		return false, exit.Internalf("cannot close rental operation for %s: %s", id, err)
	}
	res, err := tx.Exec(`DELETE FROM rentals WHERE id=?`, id)
	if err != nil {
		return false, exit.Internalf("cannot forget rental %s: %s", id, err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, exit.Internalf("cannot forget rental %s: %s", id, err)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit forgetting rental %s: %s", id, err)
	}
	return n > 0, nil
}
