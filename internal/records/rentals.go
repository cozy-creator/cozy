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
  released_at TEXT NOT NULL DEFAULT ''
)`}

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
}

const rentalCols = `id,endpoint,card,pod_id,address,cert_path,state,hub,rented_at,released_at`

func scanRental(row interface{ Scan(...any) error }) (Rental, error) {
	var r Rental
	err := row.Scan(&r.ID, &r.Endpoint, &r.Card, &r.PodID, &r.Address, &r.CertPath,
		&r.State, &r.Hub, &r.RentedAt, &r.ReleasedAt)
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
		VALUES(?,?,?,?,?,?,?,?,?,?)
		ON CONFLICT(id) DO UPDATE SET address=excluded.address, pod_id=excluded.pod_id,
		  cert_path=excluded.cert_path, state=excluded.state, released_at=excluded.released_at`,
		r.ID, r.Endpoint, r.Card, r.PodID, r.Address, r.CertPath, r.State, r.Hub,
		r.RentedAt, r.ReleasedAt); err != nil {
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
	res, err := s.db.Exec(`DELETE FROM rentals WHERE id=?`, id)
	if err != nil {
		return false, exit.Internalf("cannot forget rental %s: %s", id, err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, exit.Internalf("cannot forget rental %s: %s", id, err)
	}
	return n > 0, nil
}
