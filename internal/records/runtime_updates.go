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
  request_id TEXT NOT NULL DEFAULT '',
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
	RequestID string          `json:"request_id,omitempty"`
	BootID    string          `json:"worker_boot_id"`
	State     string          `json:"state"`
	Selection json.RawMessage `json:"selection,omitempty"`
	Result    json.RawMessage `json:"result,omitempty"`
	Error     string          `json:"error,omitempty"`
	CreatedAt string          `json:"created_at"`
	UpdatedAt string          `json:"updated_at"`
}

func (r RuntimeUpdate) Active() bool { return r.State != "succeeded" && r.State != "failed" }

func (s *Store) RuntimeUpdate(rental string) (*RuntimeUpdate, *exit.Error) {
	var r RuntimeUpdate
	err := s.db.QueryRow(`SELECT rental_id,operation_id,request_id,worker_boot_id,state,selection,result,error,created_at,updated_at
	 FROM rental_runtime_updates WHERE rental_id=?`, rental).Scan(&r.RentalID, &r.ID, &r.RequestID, &r.BootID, &r.State, &r.Selection, &r.Result, &r.Error, &r.CreatedAt, &r.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read rental Runtime update: %s", err)
	}
	return &r, nil
}

func (s *Store) BeginRuntimeUpdate(rental, boot, request string, selection json.RawMessage) (*RuntimeUpdate, *exit.Error) {
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
	if err = tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM rental_runtime_updates WHERE rental_id=? AND state NOT IN ('succeeded','failed'))`, rental).Scan(&active); err != nil {
		return nil, exit.Internalf("cannot inspect existing Runtime update: %s", err)
	}
	if active {
		return nil, exit.Named(exit.Unavailable, "rental.maintenance", "this rental already has an unfinished Runtime update")
	}
	r := RuntimeUpdate{RentalID: rental, ID: NewID("runtime-update"), RequestID: request, BootID: boot, State: "preparing", Selection: append([]byte{}, selection...), CreatedAt: now(), UpdatedAt: now()}
	_, err = tx.Exec(`INSERT INTO rental_runtime_updates(rental_id,operation_id,request_id,worker_boot_id,state,selection,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)
	 ON CONFLICT(rental_id) DO UPDATE SET operation_id=excluded.operation_id,request_id=excluded.request_id,worker_boot_id=excluded.worker_boot_id,state=excluded.state,selection=excluded.selection,result=x'',error='',created_at=excluded.created_at,updated_at=excluded.updated_at`, r.RentalID, r.ID, r.RequestID, r.BootID, r.State, r.Selection, r.CreatedAt, r.UpdatedAt)
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
	case "preparing", "updating", "reconciling", "succeeded", "failed":
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

func (s *Store) RuntimeUpdateAttempted(request string) (bool, *exit.Error) {
	return s.RequestHasEvent(request, "machine.runtime_update_attempted")
}

func (s *Store) RequestHasEvent(request, kind string) (bool, *exit.Error) {
	var found bool
	err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM request_events WHERE request_id=? AND type=?)`, request, kind).Scan(&found)
	if err != nil {
		return false, exit.Internalf("cannot read automatic Runtime update history: %s", err)
	}
	return found, nil
}
