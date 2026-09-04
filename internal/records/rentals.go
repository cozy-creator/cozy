package records

import (
	"database/sql"
	"encoding/json"
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
  expected_worker_boot_id    TEXT NOT NULL DEFAULT '',
  ready_at          TEXT NOT NULL DEFAULT '',
  failure_code                 TEXT NOT NULL DEFAULT '',
  failure_image_digest         TEXT NOT NULL DEFAULT '',
  failure_provider             TEXT NOT NULL DEFAULT '',
  failure_provider_resource_id TEXT NOT NULL DEFAULT '',
  failure_provider_host_id     TEXT NOT NULL DEFAULT '',
  failure_provider_state       TEXT NOT NULL DEFAULT '',
  failure_container_state      TEXT NOT NULL DEFAULT ''
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

// RentalRequestAuthor renders the exact request bytes, and their digest, under the machine
// name the store reserved for a new operation.
type RentalRequestAuthor func(machineName string) (body []byte, digest string, problem *exit.Error)

// BeginRentalOperation durably installs the caller's operation identity. A replay returns
// the existing row; the caller compares RequestDigest before making any network call. A new
// operation is named here: the store draws a machine word no rental row or unsettled
// operation on this host holds and has the request authored under it, inside the one
// transaction that records it — so two acquisitions can never share a word, and
// `cozy rental end <word>` is never ambiguous.
// storageUSDMicros is the SKU's estimated storage adder (th-126): the spend
// admission totals it with the locked GPU rate — burn is billed-truth money
// (th-120), so the figure admitted is what the pod will actually bill — while
// the operation row keeps the GPU quote alone.
func (s *Store) BeginRentalOperation(op RentalOperation, fleetCapUSDMicros, storageUSDMicros int64,
	author RentalRequestAuthor) (RentalOperation, bool, *exit.Error) {
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
	if op.HourlyRateUSDMicros <= 0 || fleetCapUSDMicros <= 0 || storageUSDMicros < 0 {
		return RentalOperation{}, false, exit.Named(exit.Usage, "rental.spend_cap_required",
			"a positive locked hourly rate, a non-negative storage adder, and rentals.max_hourly_spend_usd are required")
	}
	count, burn, problem := rentalFleetTotals(tx)
	if problem != nil {
		return RentalOperation{}, false, problem
	}
	estimatedTotal := op.HourlyRateUSDMicros + storageUSDMicros
	if burn > fleetCapUSDMicros || estimatedTotal > fleetCapUSDMicros-burn {
		return RentalOperation{}, false, exit.Named(exit.Capacity, "rental.fleet_spend_cap",
			"%d potentially billing rental(s) already reserve %d USD micros/hour; the next %d (%d gpu + %d storage) would exceed %d",
			count, burn, estimatedTotal, op.HourlyRateUSDMicros, storageUSDMicros, fleetCapUSDMicros)
	}
	taken, e := machineNamesInUse(tx)
	if e != nil {
		return RentalOperation{}, false, e
	}
	machineName, err := rentalid.NewMachineName(taken)
	if err != nil {
		return RentalOperation{}, false, exit.Named(exit.Capacity, "rental.machine_names_exhausted",
			"%s", err).WithRemedy("release a rental before renting another")
	}
	op.RequestBody, op.RequestDigest, e = author(machineName)
	if e != nil {
		return RentalOperation{}, false, e
	}
	// The word is bound to the managed request the moment it is minted (cl-107): an
	// acquisition that fails before any claim still names the machine it was buying.
	if op.ManagedRequestID != "" {
		if _, err := tx.Exec(`UPDATE requests SET machine=? WHERE id=?`,
			machineName, op.ManagedRequestID); err != nil {
			return RentalOperation{}, false, exit.Internalf(
				"cannot record machine %s on request %s: %s", machineName, op.ManagedRequestID, err)
		}
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

// MachineNamesInUse is every word a rental row or an unsettled operation on this host
// holds: exactly the set a new operation's word is drawn outside of.
func (s *Store) MachineNamesInUse() (map[string]bool, *exit.Error) {
	return machineNamesInUse(s.db)
}

type querier interface {
	Query(query string, args ...any) (*sql.Rows, error)
}

func machineNamesInUse(q querier) (map[string]bool, *exit.Error) {
	taken := map[string]bool{}
	collect := func(query string, name func([]byte) string) *exit.Error {
		rows, err := q.Query(query)
		if err != nil {
			return exit.Internalf("cannot list machine names in use: %s", err)
		}
		defer rows.Close()
		for rows.Next() {
			var value []byte
			if err := rows.Scan(&value); err != nil {
				return exit.Internalf("cannot read a machine name in use: %s", err)
			}
			if word := name(value); word != "" {
				taken[word] = true
			}
		}
		if err := rows.Err(); err != nil {
			return exit.Internalf("cannot list machine names in use: %s", err)
		}
		return nil
	}
	if e := collect(`SELECT machine_name FROM rentals`, func(v []byte) string { return string(v) }); e != nil {
		return nil, e
	}
	// An unsettled operation holds its word inside the closed request document it replays.
	if e := collect(`SELECT request_body FROM rental_operations WHERE state NOT IN (`+
		finalRentalOperationStates+`)`, func(v []byte) string {
		var request struct {
			Name string `json:"name"`
		}
		_ = json.Unmarshal(v, &request)
		return request.Name
	}); e != nil {
		return nil, e
	}
	return taken, nil
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

// finalRentalOperationStates is the one spelling of "this paid operation is over".
const finalRentalOperationStates = `'released','rejected'`

// ActiveRentalOperations is every paid acquisition/release operation whose absence has
// not been proved. A row may precede its provider rental id, so a safe daemon exit
// must fence on the operation key as well as on attached rental rows.
func (s *Store) ActiveRentalOperations() ([]RentalOperation, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + rentalOperationCols + ` FROM rental_operations
		WHERE state NOT IN (` + finalRentalOperationStates + `) ORDER BY created_at,operation_key`)
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
	// ReadyAt is when this host first recorded the hub saying `ready` — the pod's first
	// observed billing moment as a usable machine, as opposed to RentedAt, which is only
	// when it was asked for. It is the idle clock of a rental that has never run anything.
	ReadyAt string
	Failure RentalFailure
}

// RentalFailure is the local copy of Tensorhub's sanitized terminal boot
// diagnosis. No provider payload, pod log, or credential is stored here.
type RentalFailure struct {
	Code                  string
	BaseWorkerImageDigest string
	Provider              string
	ProviderResourceID    string
	ProviderHostID        string
	ProviderState         string
	ContainerState        string
}

const rentalCols = `id,machine_name,sku,accelerator_model,hourly_rate_usd_micros,managed_request_id,address,cert_path,state,hub,rented_at,media_address,expected_worker_id,expected_worker_boot_id,ready_at,failure_code,failure_image_digest,failure_provider,failure_provider_resource_id,failure_provider_host_id,failure_provider_state,failure_container_state`

const rentalColsPriorTwentyOne = `id,machine_name,sku,accelerator_model,hourly_rate_usd_micros,managed_request_id,address,cert_path,state,hub,rented_at,media_address,expected_worker_id,expected_worker_boot_id,ready_at`

// rentalColsPriorThirteen is the released column list every schema before 13 carried.
var rentalColsPriorThirteen = strings.TrimSuffix(rentalColsPriorTwentyOne, ",ready_at")

func scanRental(row interface{ Scan(...any) error }) (Rental, error) {
	var r Rental
	err := row.Scan(&r.ID, &r.MachineName, &r.SKU, &r.AcceleratorModel,
		&r.HourlyRateUSDMicros, &r.ManagedRequestID, &r.Address, &r.CertPath,
		&r.State, &r.Hub, &r.RentedAt, &r.MediaAddress,
		&r.ExpectedWorkerID, &r.ExpectedWorkerBootID, &r.ReadyAt,
		&r.Failure.Code, &r.Failure.BaseWorkerImageDigest, &r.Failure.Provider,
		&r.Failure.ProviderResourceID, &r.Failure.ProviderHostID,
		&r.Failure.ProviderState, &r.Failure.ContainerState)
	return r, err
}

// RentalReadyState answers whether a hub state means the pod is booted and attachable.
func RentalReadyState(state string) bool { return state == "ready" || state == "attached" }

// RecordRental writes what the hub provisioned. It REPLACES on the rental id because the
// id is the hub's, not this host's: re-reading a rental that moved from acquisition to
// ready must land on the same row rather than accumulate one per poll. State only moves
// forward: a delayed poll answering `acquiring` after `ready` was recorded is stale.
//
// The hourly rate ADOPTS the hub's answer (th-120): once the hub reconciles a rental to
// the provider's actual billed total — GPU plus storage adders — every later read serves
// that figure, and this host's row and burn line must say it too, never a cached quote.
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
	if r.ReadyAt == "" && RentalReadyState(r.State) {
		r.ReadyAt = now()
	}
	if _, err := tx.Exec(`INSERT INTO rentals(`+rentalCols+`)
			VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
		ON CONFLICT(id) DO UPDATE SET
		  machine_name=CASE WHEN rentals.machine_name<>'' THEN rentals.machine_name ELSE excluded.machine_name END,
		  sku=CASE WHEN rentals.sku<>'' THEN rentals.sku ELSE excluded.sku END,
		  hourly_rate_usd_micros=CASE WHEN excluded.hourly_rate_usd_micros>0
		    THEN excluded.hourly_rate_usd_micros ELSE rentals.hourly_rate_usd_micros END,
		  managed_request_id=rentals.managed_request_id,
		  address=CASE WHEN rentals.address<>'' THEN rentals.address ELSE excluded.address END,
		  cert_path=CASE WHEN rentals.cert_path<>'' THEN rentals.cert_path ELSE excluded.cert_path END,
		  state=excluded.state,
		  media_address=CASE WHEN rentals.media_address<>'' THEN rentals.media_address ELSE excluded.media_address END,
		  expected_worker_id=CASE WHEN rentals.expected_worker_id<>''
		    THEN rentals.expected_worker_id ELSE excluded.expected_worker_id END,
		  expected_worker_boot_id=CASE WHEN rentals.expected_worker_boot_id<>''
		    THEN rentals.expected_worker_boot_id ELSE excluded.expected_worker_boot_id END,
		  ready_at=CASE WHEN rentals.ready_at<>'' THEN rentals.ready_at ELSE excluded.ready_at END,
		  failure_code=CASE WHEN rentals.failure_code<>'' THEN rentals.failure_code ELSE excluded.failure_code END,
		  failure_image_digest=CASE WHEN rentals.failure_code<>'' THEN rentals.failure_image_digest ELSE excluded.failure_image_digest END,
		  failure_provider=CASE WHEN rentals.failure_code<>'' THEN rentals.failure_provider ELSE excluded.failure_provider END,
		  failure_provider_resource_id=CASE WHEN rentals.failure_code<>'' THEN rentals.failure_provider_resource_id ELSE excluded.failure_provider_resource_id END,
		  failure_provider_host_id=CASE WHEN rentals.failure_code<>'' THEN rentals.failure_provider_host_id ELSE excluded.failure_provider_host_id END,
		  failure_provider_state=CASE WHEN rentals.failure_code<>'' THEN rentals.failure_provider_state ELSE excluded.failure_provider_state END,
		  failure_container_state=CASE WHEN rentals.failure_code<>'' THEN rentals.failure_container_state ELSE excluded.failure_container_state END`,
		r.ID, r.MachineName, r.SKU, r.AcceleratorModel, r.HourlyRateUSDMicros, r.ManagedRequestID,
		r.Address, r.CertPath, r.State, r.Hub,
		r.RentedAt, r.MediaAddress, r.ExpectedWorkerID, r.ExpectedWorkerBootID, r.ReadyAt,
		r.Failure.Code, r.Failure.BaseWorkerImageDigest, r.Failure.Provider,
		r.Failure.ProviderResourceID, r.Failure.ProviderHostID,
		r.Failure.ProviderState, r.Failure.ContainerState); err != nil {
		return exit.Internalf("cannot record rental %s: %s", r.ID, err)
	}
	stored, err := scanRental(tx.QueryRow(`SELECT `+rentalCols+` FROM rentals WHERE id=?`, r.ID))
	if err != nil {
		return exit.Internalf("cannot read back rental %s: %s", r.ID, err)
	}
	if stored.MachineName != r.MachineName || stored.SKU != r.SKU ||
		stored.HourlyRateUSDMicros != r.HourlyRateUSDMicros ||
		stored.ManagedRequestID != r.ManagedRequestID ||
		r.Address != "" && stored.Address != r.Address ||
		r.MediaAddress != "" && stored.MediaAddress != r.MediaAddress ||
		r.CertPath != "" && stored.CertPath != r.CertPath ||
		r.ExpectedWorkerID != "" && stored.ExpectedWorkerID != r.ExpectedWorkerID ||
		r.ExpectedWorkerBootID != "" && stored.ExpectedWorkerBootID != r.ExpectedWorkerBootID ||
		r.Failure.Code != "" && stored.Failure != r.Failure {
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

// RentalProvenance maps a held rental to the recorded reason it was bought (cl-132):
// `cozy rental new rtx-a4000` for an explicit ask, `cozy run req-... (paul/sdxl)` for
// one auto-placement bought. The reason has always been a column on the operation that
// paid for the pod; nothing ever read it back, so a rental could not be attributed to
// the command that caused it and two pods bought by a job and a run were read as the
// product of a single explicit request that had actually been refused.
//
// An operation predating this, or one whose reason was never written, maps to "" — an
// unknown provenance is reported as unknown, never guessed at from the shape of the id.
func (s *Store) RentalProvenance() (map[string]string, *exit.Error) {
	rows, err := s.db.Query(
		`SELECT rental_id, reason FROM rental_operations WHERE rental_id <> ''`)
	if err != nil {
		return nil, exit.Internalf("cannot list rental provenance: %s", err)
	}
	defer rows.Close()
	out := map[string]string{}
	for rows.Next() {
		var id, reason string
		if err := rows.Scan(&id, &reason); err != nil {
			return nil, exit.Internalf("cannot read a rental operation row: %s", err)
		}
		out[id] = reason
	}
	return out, nil
}

// absentRentalStates is the one spelling of "the hub proved the provider holds
// nothing for this rental": terminal states the hub commits only after
// readback-agreed absence. A row in one is a record to close, never a machine
// that is running or spending.
const absentRentalStates = `'failed','released'`

// HeldRentalIDs is every rental whose secret material this host must still hold:
// absence has not been proven, so its media bearer, pinned certificate and Creator
// identity stay on disk. The boot sweep erases the files of every other id (cl-116).
func (s *Store) HeldRentalIDs() (map[string]bool, *exit.Error) {
	rows, err := s.db.Query(`SELECT id FROM rentals WHERE state NOT IN (` + absentRentalStates + `)`)
	if err != nil {
		return nil, exit.Internalf("cannot list held rentals: %s", err)
	}
	defer rows.Close()
	held := map[string]bool{}
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, exit.Internalf("cannot read a held rental id: %s", err)
		}
		held[id] = true
	}
	return held, nil
}

// RentalFleetTotals counts each potentially billing obligation once. A rental
// operation with no local rental row covers the response-loss window; once its
// row exists, the immutable row rate replaces that reservation in the sum.
// Rentals in a proven-absent terminal state are out of both numbers.
func (s *Store) RentalFleetTotals() (count int, hourlyRateUSDMicros int64, problem *exit.Error) {
	return rentalFleetTotals(s.db)
}

func rentalFleetTotals(q interface{ QueryRow(string, ...any) *sql.Row }) (int, int64, *exit.Error) {
	var count int
	var burn int64
	err := q.QueryRow(`SELECT COUNT(*),COALESCE(SUM(hourly_rate_usd_micros),0) FROM (
		SELECT id AS identity,hourly_rate_usd_micros FROM rentals
		WHERE state NOT IN (`+absentRentalStates+`)
		UNION ALL
		SELECT o.operation_key,o.hourly_rate_usd_micros FROM rental_operations o
		LEFT JOIN rentals r ON r.id=o.rental_id
		WHERE o.state NOT IN (`+finalRentalOperationStates+`)
		  AND o.state NOT IN (`+absentRentalStates+`) AND r.id IS NULL
	)`).Scan(&count, &burn)
	if err != nil {
		return 0, 0, exit.Internalf("cannot total the rental fleet: %s", err)
	}
	return count, burn, nil
}

// RentalRunCounts is the work still pinned to one machine, in the ONE spelling of "still
// owes work or a terminal": queued is every request waiting for the pod, running is every
// request the pod is executing or finalizing plus every attempt whose terminal is still
// owed. The idle release fences on both being zero.
func (s *Store) RentalRunCounts(id string) (queued, running int, problem *exit.Error) {
	err := s.db.QueryRow(`SELECT
		COALESCE(SUM(CASE WHEN state IN ('submitted','queued','requeue_pending') THEN 1 ELSE 0 END),0),
		COALESCE(SUM(CASE WHEN state IN ('dispatching','finalizing') OR EXISTS (
		  SELECT 1 FROM attempts a WHERE a.request_id=requests.id
		    AND a.state IN (`+openAttemptStates+`)) THEN 1 ELSE 0 END),0)
		FROM requests WHERE worker=?`, id).Scan(&queued, &running)
	if err != nil {
		return 0, 0, exit.Internalf("cannot count runs for rented machine %s: %s", id, err)
	}
	return queued, running, nil
}

// QueuedUnpinnedRentalRequests counts the --rental requests that are QUEUED and pinned to
// no rental at all (cl-121). They are the work `RentalRunCounts` cannot see: it counts
// `requests WHERE worker=<rental>`, and an unpinned request belongs to no rental yet
// BY DESIGN — the pin is routing's own output, so an unpinned request stays free to take
// whichever rental frees up first (routing D6). The consequence was that queued work
// counted toward nothing while a warm machine idled out from under it: `selectOrStart`
// returns without pinning whenever some rental has the plan STAGED (`rentalHeld`), while
// `route` will only dispatch, and so only pin, once that placement is DISPATCHABLE. Between
// those two states the request is neither, and observed windows were 21 s and ~24 s — but
// the fix must not depend on the window's length, because a placement that never becomes
// dispatchable never closes it.
//
// The fleet reads this as a reason not to release ANY rental. That is deliberately
// conservative — this request might have gone to a different machine — and the asymmetry
// says why: an extra minute of a warm pod is under a cent, while releasing one out from
// under queued work costs a full cold acquisition, measured at 178-271 s and a 6.93 GB
// re-download of bytes that machine already held.
func (s *Store) QueuedUnpinnedRentalRequests() (int, *exit.Error) {
	var count int
	err := s.db.QueryRow(`SELECT COUNT(*) FROM requests
		WHERE rental=1 AND worker='' AND state IN ('submitted','queued','requeue_pending')`).
		Scan(&count)
	if err != nil {
		return 0, exit.Internalf("cannot count queued unpinned rental requests: %s", err)
	}
	return count, nil
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

// PinnedRentalWork is every unsettled request pinned to one rental, oldest first. It is
// what the daemon asks when that rental reaches a terminal failed state: the pin named a
// machine, the machine is gone, and these rows are the work that went with it.
//
// It is deliberately the SAME predicate `RentalRunCounts` totals — `worker=<rental>` and an
// active state — so the count an operator reads and the set recovery acts on can never
// disagree. A rental shown as holding 1 running and 2 queued must be able to hand over
// exactly those three rows.
func (s *Store) PinnedRentalWork(rentalID string) ([]Request, *exit.Error) {
	if rentalID == "" {
		return nil, nil
	}
	rows, err := s.db.Query(`SELECT `+requestCols+` FROM requests
		WHERE worker=? AND state IN (`+activeRequestStates+`)
		ORDER BY created_at,id`, rentalID)
	if err != nil {
		return nil, exit.Internalf("cannot list work pinned to rental %s: %s", rentalID, err)
	}
	defer rows.Close()
	var out []Request
	for rows.Next() {
		r, err := scanRequest(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read work pinned to rental %s: %s", rentalID, err)
		}
		out = append(out, r)
	}
	return out, nil
}

// UnpinRentalWork releases a still-QUEUED request from a rental that can no longer serve
// it, so routing may replan it onto another machine. It is the exact inverse of PinRental
// and refuses the same rows PinRental would not have written: a request that has reached
// an attempt is not unpinned here, because an attempt that may have run is a different
// question from one that never started (AbandonLostAttempt answers that one).
//
// `machine` is NOT cleared. The machine word is history the run keeps (cl-107); a request
// that waited on a pod which died should still be able to say which pod that was.
func (s *Store) UnpinRentalWork(requestID, rentalID string) (bool, *exit.Error) {
	result, err := s.db.Exec(`UPDATE requests SET worker=''
		WHERE id=? AND worker=? AND rental=1
		  AND state IN ('submitted','queued','requeue_pending')`, requestID, rentalID)
	if err != nil {
		return false, exit.Internalf("cannot release request %s from rental %s: %s",
			requestID, rentalID, err)
	}
	changed, err := result.RowsAffected()
	if err != nil {
		return false, exit.Internalf("cannot read the release of request %s: %s", requestID, err)
	}
	return changed == 1, nil
}

// LostAttemptOutcome says what happens to the REQUEST once its lost attempt is closed.
// The attempt's own fact is the same either way — its execution context is gone — but the
// two callers want opposite things next, and neither may guess.
type LostAttemptOutcome int

const (
	// RequeueAfterLoss hands the request to the ordinary requeue budget. Recovery uses it:
	// the machine died under the work, and the work should run somewhere else.
	RequeueAfterLoss LostAttemptOutcome = iota
	// CancelAfterLoss settles the request as canceled. Teardown uses it: the operator asked
	// for everything to stop, and requeueing into a daemon that is shutting down would be
	// answering a different question than the one they asked.
	CancelAfterLoss
)

// AbandonLostAttempt closes one open attempt whose EXECUTION CONTEXT IS PROVABLY GONE, and
// settles its request the way the caller asked, in one transaction.
//
// This is the one exception to the recovered-attempts law and it needs its reason stated.
// `OpenAttemptsOf` says an unsettled attempt "can only be settled by the supervisor's own
// journal, replayed by a worker in the SAME slot" — which is right for every case where
// that worker may come back. It is FALSE once the pod has been destroyed: there is no slot
// left to replay into and the journal died with it, so waiting is waiting for an event that
// cannot occur. The same is true of a control stream this process no longer holds for a
// worker it no longer has.
//
// The proof is an OBSERVATION — the rental is gone or terminally failed, or the stream that
// held the attempt is gone — never elapsed time. An attempt sitting at 2590 s is not
// evidence of anything; a destroyed pod is.
//
// ONLY an attempt that crossed to the worker and recorded NO terminal is touched. A
// `terminal` attempt already holds a real outcome the worker committed, and overwriting it
// would publish a synthetic failure over a result that may have succeeded — the worst
// failure class this system has. `preparing`/`offered` never crossed at all and belong to
// the abort path, which charges nothing.
func (s *Store) AbandonLostAttempt(requestID string, attempt int64, reason string,
	outcome LostAttemptOutcome,
) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin abandoning %s#%d: %s", requestID, attempt, err)
	}
	defer tx.Rollback()
	res, err := tx.Exec(`UPDATE attempts
		SET state='closed',terminal_status='ABANDONED',terminal_cause='EXECUTION_CONTEXT_LOST',
		    safe_message=?,closed_at=?
		WHERE request_id=? AND attempt=? AND state IN ('accepted','recovered_open')`,
		reason, now(), requestID, attempt)
	if err != nil {
		return false, exit.Internalf("cannot abandon attempt %s#%d: %s", requestID, attempt, err)
	}
	if n, _ := res.RowsAffected(); n != 1 {
		return false, nil
	}
	switch outcome {
	case CancelAfterLoss:
		// The operator asked for everything to stop. The attempt's closure above is what
		// makes this reachable at all: `CancelQueuedRequest` refuses a request that holds
		// an open attempt, which is exactly how one lost attempt made the whole daemon
		// unstoppable.
		if _, err := tx.Exec(`UPDATE requests SET state='canceled'
			WHERE id=? AND state NOT IN (`+settledRequestStates+`)`, requestID); err != nil {
			return false, exit.Internalf("cannot cancel %s after its attempt was lost: %s",
				requestID, err)
		}
		if err := appendEventTx(tx, requestID, "request.canceled", attempt, map[string]any{
			"status": "CANCELED", "cause": "EXECUTION_CONTEXT_LOST",
			"error_type": "EXECUTION_CONTEXT_LOST", "error": reason,
			"outputs": []any{}, "requeuing": false,
		}); err != nil {
			return false, exit.Internalf("cannot record the cancellation of %s: %s", requestID, err)
		}
	default:
		// Only the request's CURRENT ordinal moves it; a stale attempt of an already-
		// advanced request settles alone.
		if _, err := tx.Exec(`UPDATE requests SET state='requeue_pending'
			WHERE id=? AND ordinal=? AND state IN ('dispatching','finalizing')`,
			requestID, attempt); err != nil {
			return false, exit.Internalf("cannot move %s to the requeue path: %s", requestID, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("abandoning %s#%d did not commit: %s", requestID, attempt, err)
	}
	return true, nil
}

// OrphanedRentalWork is every active request pinned to a rental this host can no longer
// use: the row is GONE, or it is there and terminally failed.
//
// The gone case is the one that wedged req-b2df33d1 for sixteen hours. Recovery keyed on
// OBSERVING a rental go `failed`, so `cozy rental end` — which destroys the record — left
// the work pinned to a name nothing would ever look at again. There is no observer for an
// object that does not exist, so the question has to be asked from the REQUEST side, which
// is the side that still has a row.
func (s *Store) OrphanedRentalWork() ([]Request, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + requestCols + ` FROM requests
		WHERE rental=1 AND worker<>'' AND state IN (` + activeRequestStates + `)
		  AND NOT EXISTS (SELECT 1 FROM rentals WHERE rentals.id=requests.worker
		                    AND rentals.state NOT IN ('failed','released'))
		ORDER BY created_at,id`)
	if err != nil {
		return nil, exit.Internalf("cannot list work pinned to lost rentals: %s", err)
	}
	defer rows.Close()
	var out []Request
	for rows.Next() {
		r, err := scanRequest(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read work pinned to a lost rental: %s", err)
		}
		out = append(out, r)
	}
	return out, nil
}
