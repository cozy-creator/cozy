package records

import (
	"database/sql"
	"encoding/json"
	"errors"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// The journal numbers every long-running command on this host in one sequence: runs and
// the operations below. A number is stored when its row is recorded, never derived.
// Kind is what the user sees the row as: run, download, install, update or upload.
const journalDDL = `CREATE TABLE IF NOT EXISTS journal (
 number INTEGER PRIMARY KEY AUTOINCREMENT,
 id TEXT NOT NULL UNIQUE,
 kind TEXT NOT NULL DEFAULT 'run'
)`

// An operation is journaled work that is not a run: a model download or package
// installation on a machine, a rental's Runtime update, or a run output's upload. No foreign key: the accepted
// intent and its terminal explanation outlive a rental.
const operationsDDL = `CREATE TABLE IF NOT EXISTS operations (
 id TEXT PRIMARY KEY,
 kind TEXT NOT NULL CHECK(kind IN ('download','install','update','upload')),
 machine TEXT NOT NULL,
 hub TEXT NOT NULL DEFAULT '',
 state TEXT NOT NULL,
 selection BLOB NOT NULL,
 idem_key TEXT NOT NULL DEFAULT '',
 worker_boot_id TEXT NOT NULL DEFAULT '',
 progress BLOB NOT NULL DEFAULT x'',
 result BLOB NOT NULL DEFAULT x'',
 error_code TEXT NOT NULL DEFAULT '',
 error TEXT NOT NULL DEFAULT '',
 canceled_by TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL
)`

const operationsByMachine = `CREATE INDEX IF NOT EXISTS operations_by_machine ON operations(machine,kind,state)`
const operationsByKey = `CREATE UNIQUE INDEX IF NOT EXISTS operations_by_key ON operations(idem_key) WHERE idem_key<>''`

// A download or installation is queued, installing, then terminal; a machine that goes
// away mid-way returns it to queued. A Runtime update moves through preparing, updating,
// reconciling and waiting_activation; unusable holds its rental until the owner retries
// the update or ends the rental.
const finalOperationStates = `'succeeded','failed','canceled'`

// InstallSelection is what a download or installation lands on its machine, frozen at
// acceptance. Hub is the Tensorhub the machine reads it at.
type InstallSelection struct {
	Package string     `json:"package,omitempty"`
	Release string     `json:"release,omitempty"`
	Models  []ModelRef `json:"models,omitempty"`
	Hub     string     `json:"hub,omitempty"`
}

// ModelProgress is one model's bytes as last observed, kept when the operation settles.
type ModelProgress struct {
	Model    string `json:"model"`
	Release  string `json:"release,omitempty"`
	Lane     string `json:"lane,omitempty"`
	Manifest string `json:"manifest"`
	Moved    uint64 `json:"moved_bytes"`
	Total    uint64 `json:"total_bytes,omitempty"`
}

type Operation struct {
	Number  int64  `json:"number"`
	ID      string `json:"id"`
	Kind    string `json:"kind"`
	Machine string `json:"machine"`
	Hub     string `json:"hub,omitempty"`
	State   string `json:"state"`
	// Install is a download's or installation's selection; Update is a Runtime update's,
	// a document only the updater reads.
	Install    InstallSelection `json:"selection"`
	Update     json.RawMessage  `json:"update,omitempty"`
	Upload     OutputUpload     `json:"upload,omitzero"`
	Result     json.RawMessage  `json:"result,omitempty"`
	IdemKey    string           `json:"-"`
	BootID     string           `json:"worker_boot_id,omitempty"`
	Progress   []ModelProgress  `json:"progress,omitempty"`
	ErrorCode  string           `json:"error_code,omitempty"`
	Error      string           `json:"error,omitempty"`
	CanceledBy string           `json:"canceled_by,omitempty"`
	CreatedAt  string           `json:"created_at"`
	UpdatedAt  string           `json:"updated_at"`
}

// Status is the lifecycle every journaled kind shares.
func (o Operation) Status() string {
	switch o.State {
	case "queued":
		return "queued"
	case "succeeded":
		return "completed"
	case "failed", "unusable":
		return "failed"
	case "canceled":
		return "canceled"
	}
	return "in_progress"
}

// Active is an operation that still holds its machine: in progress, queued, or an update
// that left its rental unusable until the owner acts.
func (o Operation) Active() bool {
	return o.State != "succeeded" && o.State != "failed" && o.State != "canceled"
}

// InProgress is an update the daemon is still carrying out.
func (o Operation) InProgress() bool {
	return o.State == "preparing" || o.State == "updating" || o.State == "reconciling" || o.State == "waiting_activation"
}

// Target names what the operation lands: a package release, models, or the Runtime.
func (o Operation) Target() string {
	switch o.Kind {
	case "update":
		return "runtime"
	case "upload":
		return o.Upload.Output + " to " + o.Upload.Destination
	}
	if o.Install.Package != "" {
		return o.Install.Package + "@" + o.Install.Release
	}
	models := make([]string, 0, len(o.Install.Models))
	for _, model := range o.Install.Models {
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

// Unusable is the refusal of work on the machine name while this update is unusable.
func (o Operation) Unusable(name string) *exit.Error {
	return exit.Named(exit.Conflict, "rental.unusable", "%s is unusable: its Runtime update #%d could not finish: %s", name, o.Number, o.Error).
		WithRemedy("cozy rental update %s retries it; cozy rental end %s ends the rental", name, name)
}

const operationCols = `j.number,o.id,o.kind,o.machine,o.hub,o.state,o.selection,o.idem_key,o.worker_boot_id,o.progress,o.result,o.error_code,o.error,o.canceled_by,o.created_at,o.updated_at`
const operationFrom = ` FROM operations o JOIN journal j ON j.id=o.id`

func scanOperation(row interface{ Scan(...any) error }) (*Operation, *exit.Error) {
	var o Operation
	var selection, progress, result []byte
	err := row.Scan(&o.Number, &o.ID, &o.Kind, &o.Machine, &o.Hub, &o.State, &selection, &o.IdemKey, &o.BootID,
		&progress, &result, &o.ErrorCode, &o.Error, &o.CanceledBy, &o.CreatedAt, &o.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read operation: %s", err)
	}
	switch o.Kind {
	case "update":
		o.Update = selection
	case "upload":
		err = json.Unmarshal(selection, &o.Upload)
	default:
		err = json.Unmarshal(selection, &o.Install)
	}
	if err != nil {
		return nil, exit.Internalf("cannot decode operation %s: %s", o.ID, err)
	}
	if len(progress) > 0 && json.Unmarshal(progress, &o.Progress) != nil {
		return nil, exit.Internalf("cannot decode the progress of operation %s", o.ID)
	}
	if len(result) > 0 {
		o.Result = result
	}
	return &o, nil
}

func (s *Store) queryOperations(where string, args ...any) ([]Operation, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+operationCols+operationFrom+` WHERE `+where, args...)
	if err != nil {
		return nil, exit.Internalf("cannot list operations: %s", err)
	}
	defer rows.Close()
	out := []Operation{}
	for rows.Next() {
		row, problem := scanOperation(rows)
		if problem != nil {
			return nil, problem
		}
		out = append(out, *row)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish operations: %s", err)
	}
	return out, nil
}

// Operation reads one operation by id or journal number; nil when the reference names none.
func (s *Store) Operation(reference string) (*Operation, *exit.Error) {
	reference = strings.TrimSpace(reference)
	if number, err := strconv.ParseInt(reference, 10, 64); err == nil && number > 0 {
		return scanOperation(s.db.QueryRow(`SELECT `+operationCols+operationFrom+` WHERE j.number=?`, number))
	}
	return scanOperation(s.db.QueryRow(`SELECT `+operationCols+operationFrom+` WHERE o.id=?`, reference))
}

// OperationsBefore is one history page, newest first, under the run listing's filters.
func (s *Store) OperationsBefore(status, packageName, hub, fallbackHub string, limit int, before int64) ([]Operation, *exit.Error) {
	where, args := []string{"1"}, []any{}
	if before > 0 {
		where, args = append(where, "j.number<?"), append(args, before)
	}
	if status != "" {
		where, args = append(where, operationStatusSQL+"=?"), append(args, status)
	}
	if packageName != "" {
		where, args = append(where, "json_extract(o.selection,'$.package')=?"), append(args, packageName)
	}
	if hub != "" {
		where = append(where, `COALESCE(NULLIF(o.hub,''),(SELECT r.hub FROM rentals r WHERE r.id=o.machine AND r.hub<>''),?)=?`)
		args = append(args, strings.TrimRight(fallbackHub, "/"), hub)
	}
	return s.queryOperations(strings.Join(where, " AND ")+` ORDER BY j.number DESC LIMIT ?`, append(args, limit)...)
}

const operationStatusSQL = `CASE o.state WHEN 'queued' THEN 'queued' WHEN 'succeeded' THEN 'completed'
 WHEN 'failed' THEN 'failed' WHEN 'unusable' THEN 'failed' WHEN 'canceled' THEN 'canceled' ELSE 'in_progress' END`

// ActiveOperations is every unfinished operation on one machine ("" for every machine).
func (s *Store) ActiveOperations(machine string) ([]Operation, *exit.Error) {
	return s.queryOperations(`o.state NOT IN (`+finalOperationStates+`) AND (?='' OR o.machine=?) ORDER BY j.number`, machine, machine)
}

// journalTx gives a newly recorded row its number.
func journalTx(tx *sql.Tx, id, kind string) (int64, error) {
	result, err := tx.Exec(`INSERT INTO journal(id,kind) VALUES(?,?)`, id, kind)
	if err != nil {
		return 0, err
	}
	return result.LastInsertId()
}

// journalEntry is a recorded row's number and kind.
func journalEntry(q interface{ QueryRow(string, ...any) *sql.Row }, id string) (int64, string, error) {
	var number int64
	var kind string
	err := q.QueryRow(`SELECT number,kind FROM journal WHERE id=?`, id).Scan(&number, &kind)
	return number, kind, err
}

// JournalKind is what a recorded row is listed as.
func (s *Store) JournalKind(id string) string {
	kind := "run"
	_ = s.db.QueryRow(`SELECT kind FROM journal WHERE id=?`, id).Scan(&kind)
	return kind
}

// numberJournal journals runs recorded without a number (before the journal existed, or by
// an older Creator) in their recorded order, so each keeps the number it was shown with.
func numberJournal(db *sql.DB) error {
	var pending bool
	if err := db.QueryRow(`SELECT EXISTS(SELECT 1 FROM requests r WHERE NOT EXISTS(SELECT 1 FROM journal j WHERE j.id=r.id))`).Scan(&pending); err != nil || !pending {
		return err
	}
	_, err := db.Exec(`INSERT OR IGNORE INTO journal(id) SELECT id FROM requests r
 WHERE NOT EXISTS(SELECT 1 FROM journal j WHERE j.id=r.id) ORDER BY created_at,id`)
	return err
}

func (s *Store) insertOperation(tx *sql.Tx, o *Operation, selection []byte) *exit.Error {
	o.CreatedAt, o.UpdatedAt = now(), now()
	if _, err := tx.Exec(`INSERT INTO operations(id,kind,machine,hub,state,selection,idem_key,worker_boot_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)`,
		o.ID, o.Kind, o.Machine, o.Hub, o.State, selection, o.IdemKey, o.BootID, o.CreatedAt, o.UpdatedAt); err != nil {
		return exit.Internalf("cannot record operation: %s", err)
	}
	number, err := journalTx(tx, o.ID, o.Kind)
	if err != nil {
		return exit.Internalf("cannot number operation: %s", err)
	}
	o.Number = number
	return nil
}

// BeginInstall persists a download or installation before acknowledging it. A repeated
// command converges on the same pending selection, and an idempotency key on its first
// answer; a completed selection can be requested again, including after a worker reboot.
func (s *Store) BeginInstall(machine, hub, key string, selection InstallSelection) (*Operation, bool, *exit.Error) {
	raw, err := json.Marshal(selection)
	if err != nil || len(raw) > 4<<20 {
		return nil, false, exit.New(exit.Validation, "installation selection is invalid or too large")
	}
	kind := "download"
	if selection.Package != "" {
		kind = "install"
	}
	tx, err := s.db.Begin()
	if err != nil {
		return nil, false, exit.Internalf("cannot begin installation: %s", err)
	}
	defer tx.Rollback()
	var prior *Operation
	var problem *exit.Error
	if key != "" {
		prior, problem = scanOperation(tx.QueryRow(`SELECT `+operationCols+operationFrom+` WHERE o.idem_key=?`, key))
		if problem == nil && prior != nil && (prior.Machine != machine || prior.Kind != kind || string(mustJSON(prior.Install)) != string(raw)) {
			return nil, false, exit.New(exit.Conflict, "idempotency key %s already names #%d, a different operation", key, prior.Number)
		}
	} else {
		prior, problem = scanOperation(tx.QueryRow(`SELECT `+operationCols+operationFrom+` WHERE o.machine=? AND o.kind=? AND o.selection=? AND o.state IN ('queued','installing') ORDER BY j.number LIMIT 1`, machine, kind, raw))
	}
	if problem != nil || prior != nil {
		return prior, false, problem
	}
	o := Operation{ID: NewID(kind), Kind: kind, Machine: machine, Hub: hub, State: "queued", Install: selection, IdemKey: key}
	if problem := s.insertOperation(tx, &o, raw); problem != nil {
		return nil, false, problem
	}
	if err := tx.Commit(); err != nil {
		return nil, false, exit.Internalf("cannot commit installation: %s", err)
	}
	return &o, true, nil
}

// MachineEndedProblem is why a rental's queued operations cannot run, or nil.
func MachineEndedProblem(rental, state string) *exit.Error {
	switch state {
	case "failed", "rejected":
		return exit.Named(exit.Unavailable, "rental.boot_failed", "rental %s failed; its queued work cannot run", rental)
	case "released", "release_requested":
		return exit.Named(exit.Unavailable, "rental.ended", "rental %s ended; its queued work cannot run", rental)
	}
	return nil
}

// PendingInstalls is every download and installation still owed a machine, oldest first.
func (s *Store) PendingInstalls() ([]Operation, *exit.Error) {
	return s.queryOperations(`o.kind IN ('download','install') AND o.state IN ('queued','installing') ORDER BY j.number`)
}

// StartInstall claims the operation on the machine's current ready worker boot. A
// restarted worker re-claims it, and the selection is prepared again there.
func (s *Store) StartInstall(id, boot string) (*Operation, *exit.Error) {
	// An empty boot is a machine without a rented worker boot to fence on.
	result, err := s.db.Exec(`UPDATE operations SET state='installing',worker_boot_id=?,error_code='',error='',updated_at=? WHERE id=? AND state IN ('queued','installing')
		AND (?='' OR EXISTS(SELECT 1 FROM rentals r WHERE r.id=operations.machine AND r.state='ready' AND r.expected_worker_boot_id=?))`,
		boot, now(), id, boot, boot)
	if err != nil {
		return nil, exit.Internalf("cannot start installation: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		return nil, exit.New(exit.Conflict, "installation or worker changed before dispatch")
	}
	return s.Operation(id)
}

// SettleInstall records where a download or installation ended: back to queued for a
// machine that went away, or terminal with the last observed bytes. A canceled operation
// keeps its cancellation.
func (s *Store) SettleInstall(id, state string, problem *exit.Error, progress []ModelProgress) *exit.Error {
	if state != "queued" && state != "succeeded" && state != "failed" {
		return exit.New(exit.Validation, "invalid installation settlement")
	}
	code, message := "", ""
	if problem != nil {
		code, message = problem.ErrName(), problem.Message
	}
	var observed []byte
	if len(progress) > 0 {
		observed = mustJSON(progress)
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot settle installation: %s", err)
	}
	defer tx.Rollback()
	stamp := now()
	result, err := tx.Exec(`UPDATE operations SET state=?,error_code=?,error=?,progress=COALESCE(?,progress),updated_at=? WHERE id=? AND state IN ('queued','installing')`, state, code, message, observed, stamp, id)
	if err != nil {
		return exit.Internalf("cannot save installation result: %s", err)
	}
	if n, _ := result.RowsAffected(); n == 0 {
		// A cancellation settled first; it keeps the bytes landed and the machine's answer.
		_, err = tx.Exec(`UPDATE operations SET progress=COALESCE(?,progress),error_code=?,error=?,updated_at=? WHERE id=? AND state='canceled'`, observed, code, message, stamp, id)
	} else if state != "queued" {
		err = finishMachineWorkTx(tx, id, stamp)
	}
	if err != nil {
		return exit.Internalf("cannot record installation completion: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit installation result: %s", err)
	}
	return nil
}

func finishMachineWorkTx(tx *sql.Tx, id, stamp string) error {
	_, err := tx.Exec(`INSERT INTO rental_idle(rental_id,work_finished_at) SELECT r.id,? FROM rentals r JOIN operations o ON o.machine=r.id WHERE o.id=? AND r.state='ready' ON CONFLICT(rental_id) DO UPDATE SET work_finished_at=excluded.work_finished_at`, stamp, id)
	return err
}

// CancelOperation ends a queued or installing download or installation, an upload, and
// an update only before it starts. Changed is false when the operation had already settled.
func (s *Store) CancelOperation(id, actor string) (bool, *exit.Error) {
	o, problem := s.Operation(id)
	if problem != nil || o == nil {
		return false, problem
	}
	if o.Kind == "update" && o.Active() && o.State != "queued" {
		return false, exit.Named(exit.Conflict, "operation.update_started",
			"#%d has already started updating the Runtime; an update cannot be canceled once it starts", o.Number).
			WithRemedy("cozy run watch %d follows it to its end", o.Number)
	}
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot cancel operation: %s", err)
	}
	defer tx.Rollback()
	stamp := now()
	result, err := tx.Exec(`UPDATE operations SET state='canceled',canceled_by=?,updated_at=? WHERE id=? AND state IN ('queued','installing','uploading')`, actor, stamp, o.ID)
	if err != nil {
		return false, exit.Internalf("cannot cancel operation: %s", err)
	}
	n, _ := result.RowsAffected()
	if n == 1 {
		if err := finishMachineWorkTx(tx, o.ID, stamp); err != nil {
			return false, exit.Internalf("cannot record operation cancellation: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit operation cancellation: %s", err)
	}
	return n == 1, nil
}

// RuntimeUpdate is the rental's newest Runtime update, or nil.
func (s *Store) RuntimeUpdate(rental string) (*Operation, *exit.Error) {
	return scanOperation(s.db.QueryRow(`SELECT `+operationCols+operationFrom+` WHERE o.kind='update' AND o.machine=? ORDER BY j.number DESC LIMIT 1`, rental))
}

// RuntimeUpdateHold is why a rental takes no work because of its Runtime update, or nil.
// Work waits for an update in progress; an update that ended without a serving worker
// refuses work until the owner resumes it or ends the rental.
func (s *Store) RuntimeUpdateHold(rental string) *exit.Error {
	r, problem := s.RuntimeUpdate(rental)
	if problem != nil || r == nil || !r.Active() {
		return problem
	}
	if r.InProgress() {
		return exit.Named(exit.Unavailable, "rental.maintenance", "this rental is updating its Runtime (#%d); work waits for it", r.Number)
	}
	name := rental
	if row, _ := s.RentalRow(rental); row != nil && row.MachineName != "" {
		name = row.MachineName
	}
	return r.Unusable(name)
}

// BeginRuntimeUpdate journals one update of a ready rental's pinned worker boot.
func (s *Store) BeginRuntimeUpdate(rental, boot string, selection json.RawMessage) (*Operation, *exit.Error) {
	if len(selection) > 1<<20 || (len(selection) > 0 && !json.Valid(selection)) {
		return nil, exit.New(exit.Validation, "invalid initial Runtime update selection")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return nil, exit.Internalf("cannot begin Runtime update: %s", err)
	}
	defer tx.Rollback()
	var state, currentBoot, hub string
	if err = tx.QueryRow(`SELECT state,expected_worker_boot_id,hub FROM rentals WHERE id=?`, rental).Scan(&state, &currentBoot, &hub); err != nil || state != "ready" || currentBoot != boot {
		return nil, exit.New(exit.Conflict, "the rental is no longer ready on its pinned worker boot")
	}
	var active bool
	if err = tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM operations WHERE kind='update' AND machine=? AND state NOT IN (`+finalOperationStates+`))`, rental).Scan(&active); err != nil {
		return nil, exit.Internalf("cannot inspect existing Runtime update: %s", err)
	}
	if active {
		return nil, exit.Named(exit.Unavailable, "rental.maintenance", "this rental already has an unfinished Runtime update")
	}
	o := Operation{ID: NewID("update"), Kind: "update", Machine: rental, Hub: hub, State: "preparing", BootID: boot, Update: append([]byte{}, selection...)}
	if problem := s.insertOperation(tx, &o, o.Update); problem != nil {
		return nil, problem
	}
	if err = tx.Commit(); err != nil {
		return nil, exit.Internalf("cannot commit Runtime update: %s", err)
	}
	return &o, nil
}

func (s *Store) SaveRuntimeUpdate(r Operation) *exit.Error {
	switch r.State {
	case "preparing", "updating", "reconciling", "waiting_activation", "unusable", "succeeded", "failed":
	default:
		return exit.New(exit.Validation, "invalid Runtime update state")
	}
	if len(r.Update) > 1<<20 || len(r.Result) > 1<<20 || (len(r.Update) > 0 && !json.Valid(r.Update)) || (len(r.Result) > 0 && !json.Valid(r.Result)) {
		return exit.New(exit.Validation, "Runtime update metadata is invalid or exceeds its bound")
	}
	result, err := s.db.Exec(`UPDATE operations SET state=?,selection=?,result=?,error=?,updated_at=? WHERE id=? AND kind='update' AND machine=? AND worker_boot_id=?`,
		r.State, append([]byte{}, r.Update...), append([]byte{}, r.Result...), r.Error, now(), r.ID, r.Machine, r.BootID)
	if err != nil {
		return exit.Internalf("cannot save Runtime update: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict, "Runtime update ownership changed")
	}
	return nil
}

// ActiveRuntimeUpdates is every unfinished Runtime update, oldest first.
func (s *Store) ActiveRuntimeUpdates() ([]Operation, *exit.Error) {
	return s.queryOperations(`o.kind='update' AND o.state NOT IN (` + finalOperationStates + `) ORDER BY j.number`)
}
