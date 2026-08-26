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

var rentalSchema = []string{`
CREATE TABLE IF NOT EXISTS rentals (
  id          TEXT PRIMARY KEY,
  endpoint    TEXT NOT NULL,
  card        TEXT NOT NULL,
  pod_id      TEXT NOT NULL,
  address     TEXT NOT NULL,
  cert_path   TEXT NOT NULL,
  state       TEXT NOT NULL,
  hub         TEXT NOT NULL,
  rented_at   TEXT NOT NULL,
  released_at TEXT NOT NULL DEFAULT '',
  media_address TEXT NOT NULL DEFAULT ''
)`, `
CREATE TABLE IF NOT EXISTS rental_operations (
  operation_key  TEXT PRIMARY KEY,
  request_digest TEXT NOT NULL,
  endpoint       TEXT NOT NULL,
  card           TEXT NOT NULL,
  region         TEXT NOT NULL,
  hub            TEXT NOT NULL,
  reason         TEXT NOT NULL,
  rental_id      TEXT NOT NULL DEFAULT '',
  state          TEXT NOT NULL,
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL
)`, `
CREATE UNIQUE INDEX IF NOT EXISTS rental_operation_remote
  ON rental_operations(rental_id) WHERE rental_id <> ''`}

// RentalOperation is the renter-owned half of one paid acquisition. It exists before
// the POST: the operation key and its 0600 token survive a lost response, so retry can
// ask for the same provider obligation instead of buying a second one.
type RentalOperation struct {
	Key           string
	RequestDigest string
	Endpoint      string
	Card          string
	Region        string
	Hub           string
	Reason        string
	RentalID      string
	State         string
	CreatedAt     string
	UpdatedAt     string
}

const rentalOperationCols = `operation_key,request_digest,endpoint,card,region,hub,reason,rental_id,state,created_at,updated_at`

func scanRentalOperation(row interface{ Scan(...any) error }) (RentalOperation, error) {
	var op RentalOperation
	err := row.Scan(&op.Key, &op.RequestDigest, &op.Endpoint, &op.Card, &op.Region,
		&op.Hub, &op.Reason, &op.RentalID, &op.State, &op.CreatedAt, &op.UpdatedAt)
	return op, err
}

// BeginRentalOperation durably installs the caller's operation identity. A replay returns
// the existing row; the caller compares RequestDigest before making any network call.
func (s *Store) BeginRentalOperation(op RentalOperation) (RentalOperation, bool, *exit.Error) {
	stamp := now()
	res, err := s.db.Exec(`INSERT INTO rental_operations(`+rentalOperationCols+`)
		VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(operation_key) DO NOTHING`,
		op.Key, op.RequestDigest, op.Endpoint, op.Card, op.Region, op.Hub, op.Reason,
		op.RentalID, "pending", stamp, stamp)
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

// PendingRentalOperation finds the one unfinished ask matching an implicit retry. A fresh
// explicit --idempotency-key never calls this; it is what makes re-running the ordinary
// command after SIGINT or response loss resume instead of spend twice.
func (s *Store) PendingRentalOperation(hub, endpoint, card, region string) (*RentalOperation, *exit.Error) {
	op, err := scanRentalOperation(s.db.QueryRow(`SELECT `+rentalOperationCols+`
		FROM rental_operations WHERE hub=? AND endpoint=? AND card=? AND region=?
		  AND state NOT IN ('attached','failed','released')
		ORDER BY created_at LIMIT 1`, hub, endpoint, card, region))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot find a pending rental operation: %s", err)
	}
	return &op, nil
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
// previously learned id, which keeps a later polling update from erasing the join.
func (s *Store) AdvanceRentalOperation(key, rentalID, state string) *exit.Error {
	res, err := s.db.Exec(`UPDATE rental_operations SET
		rental_id=CASE WHEN ?='' THEN rental_id ELSE ? END, state=?, updated_at=?
		WHERE operation_key=?`, rentalID, rentalID, state, now(), key)
	if err != nil {
		return exit.Internalf("cannot advance rental operation: %s", err)
	}
	n, err := res.RowsAffected()
	if err != nil || n != 1 {
		return exit.Internalf("rental operation %s disappeared while advancing it", key)
	}
	return nil
}

// Rental is one pod this client rented and may attach a worker to. `Endpoint` and `Card`
// are the hub's own vocabulary for what was asked for, kept verbatim so a listing says
// what was rented rather than what this host guessed it meant.
type Rental struct {
	ID         string
	Endpoint   string
	Card       string
	PodID      string
	Address    string
	CertPath   string
	State      string
	Hub        string
	RentedAt   string
	ReleasedAt string
	// MediaAddress is where the pod's co-resident media server answers (cl-014). It is a
	// FACT about the pod like the control address is, so it is a row and not a file; the
	// credential it takes is the rental's own owner token, which stays 0600 beside it.
	MediaAddress string
}

const rentalCols = `id,endpoint,card,pod_id,address,cert_path,state,hub,rented_at,released_at,media_address`

func scanRental(row interface{ Scan(...any) error }) (Rental, error) {
	var r Rental
	err := row.Scan(&r.ID, &r.Endpoint, &r.Card, &r.PodID, &r.Address, &r.CertPath,
		&r.State, &r.Hub, &r.RentedAt, &r.ReleasedAt, &r.MediaAddress)
	return r, err
}

// RecordRental writes what the hub provisioned. It REPLACES on the rental id because the
// id is the hub's, not this host's: re-reading a rental that moved from provisioning to
// ready must land on the same row rather than accumulate one per poll.
func (s *Store) RecordRental(r Rental) *exit.Error {
	if r.RentedAt == "" {
		r.RentedAt = now()
	}
	if _, err := s.db.Exec(`INSERT INTO rentals(`+rentalCols+`)
		VALUES(?,?,?,?,?,?,?,?,?,?,?)
		ON CONFLICT(id) DO UPDATE SET address=excluded.address, pod_id=excluded.pod_id,
		  cert_path=excluded.cert_path, state=excluded.state, released_at=excluded.released_at,
		  media_address=excluded.media_address`,
		r.ID, r.Endpoint, r.Card, r.PodID, r.Address, r.CertPath, r.State, r.Hub,
		r.RentedAt, r.ReleasedAt, r.MediaAddress); err != nil {
		return exit.Internalf("cannot record rental %s: %s", r.ID, err)
	}
	return nil
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

// ForgetRental removes the row after the hub has torn the pod down. It answers whether a
// row was there, so a release that names nothing this host holds refuses instead of
// reporting a success it did not have.
func (s *Store) ForgetRental(id string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin forgetting rental %s: %s", id, err)
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`UPDATE rental_operations SET state='released', updated_at=?
		WHERE rental_id=?`, now(), id); err != nil {
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
