package records

import (
	"database/sql"
	"errors"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// The RENTAL half of the one lifecycle authority (cl-015). A rented pod outlives the
// process that rented it — that is the whole difference between it and a spawned worker —
// so what it is lives in the same database as everything else durable here, and NOT in a
// sidecar file that would outlive the fact it describes.
//
// What is deliberately NOT a column: the provisioned owner token. Every reader of this
// database would be a reader of that credential; the 0600 handoff beside it is the
// boundary (home.Layout.RentalToken), exactly as the CLI's own credential is.

const rentalOperationsDDL = `
CREATE TABLE IF NOT EXISTS rental_operations (
  operation_key    TEXT PRIMARY KEY,
  request_digest   TEXT NOT NULL,
  request_body     BLOB NOT NULL,
  hub              TEXT NOT NULL,
  reason           TEXT NOT NULL,
  rental_id        TEXT NOT NULL DEFAULT '',
  state            TEXT NOT NULL,
  created_at       TEXT NOT NULL,
  updated_at       TEXT NOT NULL
)`

const rentalsDDL = `
CREATE TABLE IF NOT EXISTS rentals (
  id                TEXT PRIMARY KEY,
  endpoint_ref      TEXT NOT NULL,
  accelerator_model TEXT NOT NULL,
  address           TEXT NOT NULL,
  cert_path         TEXT NOT NULL,
  state             TEXT NOT NULL,
  hub               TEXT NOT NULL,
  rented_at         TEXT NOT NULL,
  released_at       TEXT NOT NULL DEFAULT '',
  media_address     TEXT NOT NULL DEFAULT '',
  control_snapshot_digest TEXT NOT NULL DEFAULT '',
  control_snapshot_length INTEGER NOT NULL DEFAULT 0,
  control_snapshot_bytes  BLOB NOT NULL DEFAULT x''
)`

const migrateRentalOperationState = `CASE state
           WHEN 'pending' THEN 'pending_acquisition'
           WHEN 'provisioning' THEN 'acquiring'
           WHEN 'reclaiming' THEN 'release_requested'
           WHEN 'dead' THEN 'released'
           ELSE state
         END`

var rentalSchema = []string{rentalOperationsDDL, rentalsDDL, `
CREATE UNIQUE INDEX IF NOT EXISTS rental_operation_remote
  ON rental_operations(rental_id) WHERE rental_id <> ''`}

// RentalOperation is the renter-owned half of one paid acquisition. It exists before
// the POST: the operation key and its 0600 token survive a lost response, so retry can
// ask for the same provider obligation instead of buying a second one.
type RentalOperation struct {
	Key           string
	RequestDigest string
	RequestBody   []byte
	Hub           string
	Reason        string
	RentalID      string
	State         string
	CreatedAt     string
	UpdatedAt     string
}

const rentalOperationCols = `operation_key,request_digest,request_body,hub,reason,rental_id,state,created_at,updated_at`

func scanRentalOperation(row interface{ Scan(...any) error }) (RentalOperation, error) {
	var op RentalOperation
	err := row.Scan(&op.Key, &op.RequestDigest, &op.RequestBody, &op.Hub, &op.Reason,
		&op.RentalID, &op.State, &op.CreatedAt, &op.UpdatedAt)
	return op, err
}

// BeginRentalOperation durably installs the caller's operation identity. A replay returns
// the existing row; the caller compares RequestDigest before making any network call.
func (s *Store) BeginRentalOperation(op RentalOperation) (RentalOperation, bool, *exit.Error) {
	stamp := now()
	res, err := s.db.Exec(`INSERT INTO rental_operations(`+rentalOperationCols+`)
		VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(operation_key) DO NOTHING`,
		op.Key, op.RequestDigest, op.RequestBody, op.Hub, op.Reason,
		op.RentalID, "pending_acquisition", stamp, stamp)
	if err != nil {
		return RentalOperation{}, false, exit.Internalf("cannot record rental operation: %s", err)
	}
	inserted, err := res.RowsAffected()
	if err != nil {
		return RentalOperation{}, false, exit.Internalf("cannot read rental operation result: %s", err)
	}
	stored, err := scanRentalOperation(s.db.QueryRow(
		`SELECT `+rentalOperationCols+` FROM rental_operations WHERE operation_key=?`, op.Key))
	if err != nil {
		return RentalOperation{}, false, exit.Internalf("cannot read rental operation: %s", err)
	}
	return stored, inserted == 0, nil
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
	if current == "release_requested" {
		if next == "released" {
			return next, nil
		}
		return current, nil
	}
	if current == "failed" {
		if next == "release_requested" || next == "released" {
			return next, nil
		}
		return current, nil
	}
	if current == "attached" {
		if next == "failed" || next == "release_requested" || next == "released" {
			return next, nil
		}
		return current, nil
	}
	rank := map[string]int{"pending_acquisition": 0, "acquiring": 1, "materializing": 2, "ready": 3}
	currentRank, currentKnown := rank[current]
	nextRank, nextKnown := rank[next]
	if currentKnown && nextKnown {
		if nextRank > currentRank {
			return next, nil
		}
		return current, nil
	}
	if current == "pending_acquisition" && next == "rejected" {
		return next, nil
	}
	if currentKnown && (next == "attached" || next == "failed" || next == "release_requested" ||
		next == "released") {
		return next, nil
	}
	return "", exit.Named(exit.Conflict, "rental.operation_conflict",
		"rental operation %s cannot advance from %q to %q", key, current, next)
}

// RequestRentalRelease records intent before the remote DELETE. It returns the operation
// key so the caller can remove a never-attached operation token after release succeeds.
func (s *Store) RequestRentalRelease(id string) (string, *exit.Error) {
	var key string
	err := s.db.QueryRow(`SELECT operation_key FROM rental_operations WHERE rental_id=?`, id).Scan(&key)
	if errors.Is(err, sql.ErrNoRows) {
		return "", exit.Internalf("rental %s has no operation to release", id)
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
	ID               string
	EndpointRef      string
	AcceleratorModel string
	Address          string
	CertPath         string
	State            string
	Hub              string
	RentedAt         string
	ReleasedAt       string
	// MediaAddress is where the pod's co-resident media server answers (cl-014). It is a
	// FACT about the pod like the control address is, so it is a row and not a file; the
	// credential it takes is the rental's own owner token, which stays 0600 beside it.
	MediaAddress string
	// ControlSnapshotBytes is the exact Tensorhub-authored, attempt-bound
	// snapshot received on the ready view. It remains raw bytes in SQLite so a
	// restart cannot re-render remote execution meaning from a local install.
	ControlSnapshotDigest string
	ControlSnapshotLength int64
	ControlSnapshotBytes  []byte
}

const rentalCols = `id,endpoint_ref,accelerator_model,address,cert_path,state,hub,rented_at,released_at,media_address,control_snapshot_digest,control_snapshot_length,control_snapshot_bytes`

func scanRental(row interface{ Scan(...any) error }) (Rental, error) {
	var r Rental
	err := row.Scan(&r.ID, &r.EndpointRef, &r.AcceleratorModel, &r.Address, &r.CertPath,
		&r.State, &r.Hub, &r.RentedAt, &r.ReleasedAt, &r.MediaAddress,
		&r.ControlSnapshotDigest, &r.ControlSnapshotLength, &r.ControlSnapshotBytes)
	return r, err
}

// RecordRental writes what the hub provisioned. It REPLACES on the rental id because the
// id is the hub's, not this host's: re-reading a rental that moved from acquisition to
// ready must land on the same row rather than accumulate one per poll.
func (s *Store) RecordRental(r Rental) *exit.Error {
	if r.RentedAt == "" {
		r.RentedAt = now()
	}
	if r.ControlSnapshotBytes == nil {
		r.ControlSnapshotBytes = []byte{}
	}
	if _, err := s.db.Exec(`INSERT INTO rentals(`+rentalCols+`)
		VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
		ON CONFLICT(id) DO UPDATE SET address=excluded.address,
		  cert_path=excluded.cert_path, state=excluded.state, released_at=excluded.released_at,
		  media_address=excluded.media_address,
		  control_snapshot_digest=CASE WHEN length(excluded.control_snapshot_bytes)>0
		    THEN excluded.control_snapshot_digest ELSE rentals.control_snapshot_digest END,
		  control_snapshot_length=CASE WHEN length(excluded.control_snapshot_bytes)>0
		    THEN excluded.control_snapshot_length ELSE rentals.control_snapshot_length END,
		  control_snapshot_bytes=CASE WHEN length(excluded.control_snapshot_bytes)>0
		    THEN excluded.control_snapshot_bytes ELSE rentals.control_snapshot_bytes END`,
		r.ID, r.EndpointRef, r.AcceleratorModel, r.Address, r.CertPath, r.State, r.Hub,
		r.RentedAt, r.ReleasedAt, r.MediaAddress, r.ControlSnapshotDigest,
		r.ControlSnapshotLength, r.ControlSnapshotBytes); err != nil {
		return exit.Internalf("cannot record rental %s: %s", r.ID, err)
	}
	return nil
}

// rentalRebuild hardcuts provider-placement vocabulary from pre-launch local
// databases without discarding attached rental handles. An unfinished legacy
// operation has no exact new request body, so it migrates with an empty body and
// is refused before replay rather than being silently re-encoded.
var rentalRebuild = []tableRebuild{{
	table: "rental_operations",
	stale: "region",
	steps: []string{
		`DROP INDEX IF EXISTS rental_operation_remote`,
		`ALTER TABLE rental_operations RENAME TO rental_operations_pre_provider_neutral`,
		rentalOperationsDDL,
		`INSERT INTO rental_operations
  (operation_key,request_digest,request_body,hub,reason,rental_id,state,created_at,updated_at)
  SELECT operation_key,request_digest,x'',hub,reason,rental_id,` + migrateRentalOperationState + `,
         created_at,updated_at
    FROM rental_operations_pre_provider_neutral`,
		`DROP TABLE rental_operations_pre_provider_neutral`,
		`CREATE UNIQUE INDEX rental_operation_remote
  ON rental_operations(rental_id) WHERE rental_id <> ''`,
	},
}, {
	table: "rental_operations",
	stale: "endpoint_ref",
	steps: []string{
		`DROP INDEX IF EXISTS rental_operation_remote`,
		`ALTER TABLE rental_operations RENAME TO rental_operations_pre_operation_field_drop`,
		rentalOperationsDDL,
		`INSERT INTO rental_operations
  (operation_key,request_digest,request_body,hub,reason,rental_id,state,created_at,updated_at)
  SELECT operation_key,request_digest,request_body,hub,reason,rental_id,` + migrateRentalOperationState + `,
         created_at,updated_at
    FROM rental_operations_pre_operation_field_drop`,
		`DROP TABLE rental_operations_pre_operation_field_drop`,
		`CREATE UNIQUE INDEX rental_operation_remote
  ON rental_operations(rental_id) WHERE rental_id <> ''`,
	},
}, {
	table: "rentals",
	stale: "pod_id",
	steps: []string{
		`ALTER TABLE rentals RENAME TO rentals_pre_provider_neutral`,
		rentalsDDL,
		`INSERT INTO rentals
  (id,endpoint_ref,accelerator_model,address,cert_path,state,hub,rented_at,released_at,media_address)
  SELECT id,endpoint,card,address,cert_path,state,hub,rented_at,released_at,media_address
    FROM rentals_pre_provider_neutral`,
		`DROP TABLE rentals_pre_provider_neutral`,
	},
}}

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

// ForgetRental removes the row after the hub has torn the pod down. It answers whether a
// row was there, so a release that names nothing this host holds refuses instead of
// reporting a success it did not have.
func (s *Store) ForgetRental(id string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin forgetting rental %s: %s", id, err)
	}
	defer tx.Rollback()
	advanced, err := tx.Exec(`UPDATE rental_operations SET state='released', updated_at=?
		WHERE rental_id=? AND state='release_requested'`, now(), id)
	if err != nil {
		return false, exit.Internalf("cannot close rental operation for %s: %s", id, err)
	}
	advancedN, err := advanced.RowsAffected()
	if err != nil {
		return false, exit.Internalf("cannot read rental operation close result for %s: %s", id, err)
	}
	if advancedN == 0 {
		var state string
		if err := tx.QueryRow(`SELECT state FROM rental_operations WHERE rental_id=?`, id).Scan(&state); err != nil {
			return false, exit.Internalf("cannot read rental operation while closing %s: %s", id, err)
		}
		if state == "released" {
			advancedN = 1
		}
	}
	if advancedN != 1 {
		return false, exit.Named(exit.Conflict, "rental.release_transition_conflict",
			"rental %s was not release_requested while closing it", id)
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
