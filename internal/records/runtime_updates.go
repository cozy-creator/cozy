package records

import (
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/cozy-creator/cozy/internal/exit"
)

const runtimeUpdatesDDL = `CREATE TABLE IF NOT EXISTS rental_runtime_updates (
  rental_id TEXT PRIMARY KEY,
  operation_id TEXT NOT NULL,
  request_id TEXT NOT NULL DEFAULT '', -- unused; an older daemon sharing this store writes it
  worker_boot_id TEXT NOT NULL,
  state TEXT NOT NULL,
  selection BLOB NOT NULL DEFAULT x'',
  result BLOB NOT NULL DEFAULT x'',
  error TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
)`

type RuntimeUpdate struct {
	RentalID  string          `json:"rental"`
	ID        string          `json:"id"`
	BootID    string          `json:"worker_boot_id"`
	State     string          `json:"state"`
	Selection json.RawMessage `json:"selection,omitempty"`
	Result    json.RawMessage `json:"result,omitempty"`
	Error     string          `json:"error,omitempty"`
	CreatedAt string          `json:"created_at"`
	UpdatedAt string          `json:"updated_at"`
}

// Active is an update still being carried out: it holds its rental.
func (r RuntimeUpdate) Active() bool {
	return r.State == "preparing" || r.State == "updating" || r.State == "reconciling" || r.State == "waiting_activation"
}

// RuntimeUpdateHold is why a rental takes no work because of its software update, or nil:
// work waits for an update of the rental's current boot to end.
func (s *Store) RuntimeUpdateHold(rental string) *exit.Error {
	r, problem := s.RuntimeUpdate(rental)
	if problem != nil || r == nil || !r.Active() {
		return problem
	}
	if row, problem := s.RentalRow(rental); problem != nil || row == nil || row.ExpectedWorkerBootID != r.BootID {
		return problem // the boot it updated is gone
	}
	return exit.Named(exit.Unavailable, "rental.maintenance", "this rental is updating its software; work waits for it")
}

func (s *Store) RuntimeUpdate(rental string) (*RuntimeUpdate, *exit.Error) {
	var r RuntimeUpdate
	err := s.db.QueryRow(`SELECT rental_id,operation_id,worker_boot_id,state,selection,result,error,created_at,updated_at
	 FROM rental_runtime_updates WHERE rental_id=?`, rental).Scan(&r.RentalID, &r.ID, &r.BootID, &r.State, &r.Selection, &r.Result, &r.Error, &r.CreatedAt, &r.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read rental Runtime update: %s", err)
	}
	return &r, nil
}

func (s *Store) BeginRuntimeUpdate(rental, boot string, selection json.RawMessage) (*RuntimeUpdate, *exit.Error) {
	if len(selection) > 1<<20 || (len(selection) > 0 && !json.Valid(selection)) {
		return nil, exit.New(exit.Validation, "invalid initial Runtime update selection")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return nil, exit.Internalf("cannot begin Runtime update: %s", err)
	}
	defer tx.Rollback()
	var state, currentBoot string
	if err = tx.QueryRow(`SELECT state,expected_worker_boot_id FROM rentals WHERE id=?`, rental).Scan(&state, &currentBoot); err != nil || state != "ready" || currentBoot != boot {
		return nil, exit.New(exit.Conflict, "the rental is no longer ready on its pinned worker boot")
	}
	var active bool
	if err = tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM rental_runtime_updates WHERE rental_id=? AND worker_boot_id=? AND state NOT IN ('succeeded','failed'))`, rental, boot).Scan(&active); err != nil {
		return nil, exit.Internalf("cannot inspect existing Runtime update: %s", err)
	}
	if active {
		return nil, exit.Named(exit.Unavailable, "rental.maintenance", "this rental already has an unfinished Runtime update")
	}
	r := RuntimeUpdate{RentalID: rental, ID: NewID("runtime-update"), BootID: boot, State: "preparing", Selection: append([]byte{}, selection...), CreatedAt: now(), UpdatedAt: now()}
	_, err = tx.Exec(`INSERT INTO rental_runtime_updates(rental_id,operation_id,worker_boot_id,state,selection,created_at,updated_at) VALUES(?,?,?,?,?,?,?)
	 ON CONFLICT(rental_id) DO UPDATE SET operation_id=excluded.operation_id,worker_boot_id=excluded.worker_boot_id,state=excluded.state,selection=excluded.selection,result=x'',error='',created_at=excluded.created_at,updated_at=excluded.updated_at`, r.RentalID, r.ID, r.BootID, r.State, r.Selection, r.CreatedAt, r.UpdatedAt)
	if err != nil {
		return nil, exit.Internalf("cannot record Runtime update: %s", err)
	}
	if err = tx.Commit(); err != nil {
		return nil, exit.Internalf("cannot commit Runtime update: %s", err)
	}
	return &r, nil
}

func (s *Store) SaveRuntimeUpdate(r RuntimeUpdate) *exit.Error {
	switch r.State {
	case "preparing", "updating", "reconciling", "waiting_activation", "succeeded", "failed":
	default:
		return exit.New(exit.Validation, "invalid Runtime update state")
	}
	if len(r.Selection) > 1<<20 || len(r.Result) > 1<<20 || (len(r.Selection) > 0 && !json.Valid(r.Selection)) || (len(r.Result) > 0 && !json.Valid(r.Result)) {
		return exit.New(exit.Validation, "Runtime update metadata is invalid or exceeds its bound")
	}
	selection, resultBytes := append([]byte{}, r.Selection...), append([]byte{}, r.Result...)
	result, err := s.db.Exec(`UPDATE rental_runtime_updates SET state=?,selection=?,result=?,error=?,updated_at=? WHERE rental_id=? AND operation_id=? AND worker_boot_id=?`, r.State, selection, resultBytes, r.Error, now(), r.RentalID, r.ID, r.BootID)
	if err != nil {
		return exit.Internalf("cannot save Runtime update: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict, "Runtime update ownership changed")
	}
	return nil
}

func (s *Store) ActiveRuntimeUpdates() ([]RuntimeUpdate, *exit.Error) {
	rows, err := s.db.Query(`SELECT rental_id FROM rental_runtime_updates WHERE state NOT IN ('succeeded','failed') ORDER BY created_at`)
	if err != nil {
		return nil, exit.Internalf("cannot enumerate unfinished Runtime updates: %s", err)
	}
	var ids []string
	for rows.Next() {
		var id string
		if err = rows.Scan(&id); err != nil {
			rows.Close()
			return nil, exit.Internalf("cannot read unfinished Runtime update: %s", err)
		}
		ids = append(ids, id)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return nil, exit.Internalf("cannot finish unfinished Runtime updates: %s", err)
	}
	var result []RuntimeUpdate
	for _, id := range ids {
		r, problem := s.RuntimeUpdate(id)
		if problem != nil {
			return nil, problem
		}
		if r != nil {
			result = append(result, *r)
		}
	}
	return result, nil
}
