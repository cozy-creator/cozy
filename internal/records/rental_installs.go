package records

import (
	"cmp"
	"database/sql"
	"encoding/json"
	"errors"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// No foreign key: the accepted intent and its terminal explanation outlive a rental.
const rentalInstallsDDL = `CREATE TABLE IF NOT EXISTS rental_installs (
 id TEXT PRIMARY KEY,
 rental_id TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('queued','installing','succeeded','failed')),
 selection BLOB NOT NULL,
 worker_boot_id TEXT NOT NULL DEFAULT '',
 error_code TEXT NOT NULL DEFAULT '',
 error TEXT NOT NULL DEFAULT '',
 result TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL
)`
const rentalInstallsIndex = `CREATE INDEX IF NOT EXISTS rental_installs_pending ON rental_installs(rental_id,state,created_at)`

type RentalInstallSelection struct {
	Package string     `json:"package,omitempty"`
	Release string     `json:"release,omitempty"`
	Models  []ModelRef `json:"models,omitempty"`
	// Destination is where the one model is put: the org/name it is uploaded to (`cozy model
	// upload`), or the local/name the machine holds it under (`cozy model download … local/`).
	Destination string `json:"destination,omitempty"`
	// Write is a local file this computer writes to the machine before the warm run: the file
	// an upload makes (its model's object:// source).
	Write string `json:"write,omitempty"`
	// Hub is the Tensorhub the selection is read at: this computer's machine prepares it there.
	Hub string `json:"hub,omitempty"`
	// Warm makes the package's Entrypoint a member of the machine's warm set, kept ready up to
	// this level (WarmLevels); "off" takes it out. Models are then its `model.<param>=` choices.
	Entrypoint string `json:"entrypoint,omitempty"`
	Warm       string `json:"warm,omitempty"`
}

// WarmLevels are how far a machine keeps a warm set member ready, each including the ones
// before it.
var WarmLevels = []string{"installed", "downloaded", "imported", "host", "gpu"}

type RentalInstall struct {
	ID           string                 `json:"id"`
	RentalID     string                 `json:"rental"`
	State        string                 `json:"state"`
	Selection    RentalInstallSelection `json:"selection"`
	WorkerBootID string                 `json:"worker_boot_id,omitempty"`
	ErrorCode    string                 `json:"error_code,omitempty"`
	Error        string                 `json:"error,omitempty"`
	// Result is what a succeeded installation produced: an upload's checkpoint.
	Result    json.RawMessage `json:"result,omitempty"`
	CreatedAt string          `json:"created_at"`
	UpdatedAt string          `json:"updated_at"`
}

// HoldsLocally is a selection only a machine that keeps local models takes: a file it is
// written, a local/ alias it reads, or a local/ alias it is put under.
func (s RentalInstallSelection) HoldsLocally() bool {
	local := s.Write != "" || strings.HasPrefix(s.Destination, "local/")
	for _, model := range s.Models {
		local = local || strings.HasPrefix(model.Model, "local/")
	}
	return local
}

// Target names what a selection puts on its machine.
func (s RentalInstallSelection) Target() string {
	if s.Warm != "" {
		return s.Package + "/" + s.Entrypoint + " warm=" + s.Warm
	}
	if s.Package != "" && s.Release == "" {
		return s.Package // the hub's newest, until the machine names it
	}
	if s.Package != "" {
		return s.Package + "@" + s.Release
	}
	models := make([]string, 0, len(s.Models))
	for _, model := range s.Models {
		if s.Destination != "" {
			models = append(models, cmp.Or(model.Source, model.Model, model.Manifest)+" -> "+s.Destination)
			continue
		}
		name := model.Model
		if model.Release != "" {
			name += "@" + model.Release
		}
		if model.Lane != "" {
			name += "/" + model.Lane
		}
		if model.Release == "" {
			name += "#" + model.Manifest
		}
		models = append(models, name)
	}
	return strings.Join(models, ", ")
}

func (r RentalInstall) Active() bool { return r.State == "queued" || r.State == "installing" }

const rentalInstallCols = `id,rental_id,state,selection,worker_boot_id,error_code,error,result,created_at,updated_at`

func scanRentalInstall(row interface{ Scan(...any) error }) (*RentalInstall, *exit.Error) {
	var r RentalInstall
	var raw []byte
	var result string
	err := row.Scan(&r.ID, &r.RentalID, &r.State, &raw, &r.WorkerBootID, &r.ErrorCode, &r.Error, &result, &r.CreatedAt, &r.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read rental installation: %s", err)
	}
	if err := json.Unmarshal(raw, &r.Selection); err != nil {
		return nil, exit.Internalf("cannot decode rental installation: %s", err)
	}
	if result != "" {
		r.Result = json.RawMessage(result)
	}
	return &r, nil
}

func (s *Store) RentalInstall(id string) (*RentalInstall, *exit.Error) {
	return scanRentalInstall(s.db.QueryRow(`SELECT `+rentalInstallCols+` FROM rental_installs WHERE id=?`, id))
}

// BeginRentalInstall persists before acknowledging acceptance. Repeated commands
// converge while the same immutable selection is still pending. A completed
// selection can be requested again, including after a worker reboot.
func (s *Store) BeginRentalInstall(rental string, selection RentalInstallSelection) (*RentalInstall, *exit.Error) {
	raw, err := json.Marshal(selection)
	if err != nil || len(raw) > 4<<20 {
		return nil, exit.New(exit.Validation, "rental installation selection is invalid or too large")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return nil, exit.Internalf("cannot begin rental installation: %s", err)
	}
	defer tx.Rollback()
	prior, problem := scanRentalInstall(tx.QueryRow(`SELECT `+rentalInstallCols+` FROM rental_installs WHERE rental_id=? AND selection=? AND state IN ('queued','installing') ORDER BY created_at,id LIMIT 1`, rental, raw))
	if problem != nil {
		return nil, problem
	}
	if prior != nil {
		return prior, nil
	}
	r := RentalInstall{ID: NewID("rental-install"), RentalID: rental, State: "queued", Selection: selection, CreatedAt: now(), UpdatedAt: now()}
	if _, err := tx.Exec(`INSERT INTO rental_installs(id,rental_id,state,selection,created_at,updated_at) VALUES(?,?,?,?,?,?)`, r.ID, r.RentalID, r.State, raw, r.CreatedAt, r.UpdatedAt); err != nil {
		return nil, exit.Internalf("cannot record rental installation: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return nil, exit.Internalf("cannot commit rental installation: %s", err)
	}
	return &r, nil
}

func RentalInstallStateProblem(rental, state string) *exit.Error {
	switch state {
	case "failed", "rejected":
		return exit.Named(exit.Unavailable, "rental.boot_failed", "rental %s failed; its queued installation cannot run", rental)
	case "released", "release_requested":
		return exit.Named(exit.Unavailable, "rental.ended", "rental %s ended; its queued installation cannot run", rental)
	}
	return nil
}

func (s *Store) PendingRentalInstalls() ([]RentalInstall, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + rentalInstallCols + ` FROM rental_installs WHERE state IN ('queued','installing') ORDER BY created_at,id`)
	if err != nil {
		return nil, exit.Internalf("cannot list rental installations: %s", err)
	}
	defer rows.Close()
	out := []RentalInstall{}
	for rows.Next() {
		row, problem := scanRentalInstall(rows)
		if problem != nil {
			return nil, problem
		}
		out = append(out, *row)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish rental installations: %s", err)
	}
	return out, nil
}

// StartRentalInstall claims the installation on the rental's current ready worker
// boot. A restarted worker re-claims it, and the selection is prepared again there.
func (s *Store) StartRentalInstall(id, boot string) (*RentalInstall, *exit.Error) {
	// An empty boot is a machine without a rented worker boot to fence on.
	result, err := s.db.Exec(`UPDATE rental_installs SET state='installing',worker_boot_id=?,error_code='',error='',updated_at=? WHERE id=? AND state IN ('queued','installing')
		AND (?='' OR EXISTS(SELECT 1 FROM rentals r WHERE r.id=rental_installs.rental_id AND r.state='ready' AND r.expected_worker_boot_id=?))`,
		boot, now(), id, boot, boot)
	if err != nil {
		return nil, exit.Internalf("cannot start rental installation: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		return nil, exit.New(exit.Conflict, "rental installation or worker changed before dispatch")
	}
	return s.RentalInstall(id)
}

func (s *Store) SettleRentalInstall(id, state string, result json.RawMessage, problem *exit.Error) *exit.Error {
	if state != "queued" && state != "succeeded" && state != "failed" || len(result) > 0 && (state != "succeeded" || !json.Valid(result)) {
		return exit.New(exit.Validation, "invalid rental installation settlement")
	}
	code, message := "", ""
	if problem != nil {
		code, message = problem.ErrName(), problem.Message
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot settle rental installation: %s", err)
	}
	defer tx.Rollback()
	stamp := now()
	settled, err := tx.Exec(`UPDATE rental_installs SET state=?,error_code=?,error=?,result=?,updated_at=? WHERE id=? AND state IN ('queued','installing')`, state, code, message, string(result), stamp, id)
	if err != nil {
		return exit.Internalf("cannot save rental installation result: %s", err)
	}
	if n, _ := settled.RowsAffected(); n == 1 && state != "queued" {
		_, err = tx.Exec(`INSERT INTO rental_idle(rental_id,work_finished_at) SELECT r.id,? FROM rentals r JOIN rental_installs i ON i.rental_id=r.id WHERE i.id=? AND r.state='ready' ON CONFLICT(rental_id) DO UPDATE SET work_finished_at=excluded.work_finished_at`, stamp, id)
		if err != nil {
			return exit.Internalf("cannot record rental installation completion: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit rental installation result: %s", err)
	}
	return nil
}
