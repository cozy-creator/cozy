package records

import (
	"bytes"
	"database/sql"
	"errors"
	"strings"
	"unicode"

	"github.com/cozy-creator/cozy-creator/internal/exit"
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
  media_address     TEXT NOT NULL DEFAULT '',
  control_snapshot_digest TEXT NOT NULL DEFAULT '',
  control_snapshot_bytes  BLOB NOT NULL DEFAULT x'',
  observed_accelerator       TEXT NOT NULL DEFAULT '',
  observed_accelerator_count INTEGER NOT NULL DEFAULT 0,
  observed_backend           TEXT NOT NULL DEFAULT '',
  observed_worker_instance   TEXT NOT NULL DEFAULT '',
  observed_worker_boot_id    TEXT NOT NULL DEFAULT '',
  observed_at                TEXT NOT NULL DEFAULT '',
  artifact_grant_revision    INTEGER NOT NULL DEFAULT 0
)`

var rentalSchema = []string{rentalOperationsDDL, rentalsDDL, `
CREATE UNIQUE INDEX IF NOT EXISTS rental_operation_remote
  ON rental_operations(rental_id) WHERE rental_id <> ''`}

// CheckRentalSchema refuses a root whose rental tables were created by another shape of
// this binary. Rentals are paid obligations, so a mismatched table is never rebuilt or
// dropped here; the operator settles it with the binary that wrote it.
func CheckRentalSchema(db *sql.DB, path string) *exit.Error {
	for table, expected := range map[string]string{"rentals": rentalsDDL, "rental_operations": rentalOperationsDDL} {
		var stored string
		err := db.QueryRow(`SELECT COALESCE(sql,'') FROM sqlite_master WHERE type='table' AND name=?`, table).Scan(&stored)
		if err != nil || stored == "" {
			continue
		}
		if normalizeDDL(stored) != normalizeDDL(expected) {
			return exit.Named(exit.Conflict, "records.rental_schema_mismatch",
				"table %s in %s was written by a different Creator build", table, path).
				WithRemedy("release its rentals with the binary that wrote it (`cozy rent ls`, `cozy rent release <id> --yes`), or delete %s if nothing there is still rented", path)
		}
	}
	return nil
}

func normalizeDDL(ddl string) string {
	return strings.Join(strings.Fields(strings.Replace(ddl, "IF NOT EXISTS ", "", 1)), " ")
}

// rentalStateRank orders the hub's lifecycle words so a delayed observation never moves a
// row backward. Unknown words are opaque: they neither advance nor regress anything.
var rentalStateRank = map[string]int{
	"pending_acquisition": 0, "acquiring": 1, "materializing": 2, "ready": 3,
	"attached": 4, "failed": 5, "release_requested": 6, "released": 7, "rejected": 7,
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
	ID               string
	EndpointRef      string
	AcceleratorModel string
	Address          string
	CertPath         string
	State            string
	Hub              string
	RentedAt         string
	// MediaAddress is where the pod's co-resident media server answers (cl-014). It is a
	// FACT about the pod like the control address is, so it is a row and not a file; the
	// credential it takes is the rental's own owner token, which stays 0600 beside it.
	MediaAddress string
	// ControlSnapshotBytes is the exact Tensorhub-authored, attempt-bound
	// snapshot received on the ready view. It remains raw bytes in SQLite so a
	// restart cannot re-render remote execution meaning from a local install.
	ControlSnapshotDigest string
	ControlSnapshotBytes  []byte
	// Observed* is the remote worker's ClaimAck readback. AcceleratorModel above is
	// only the caller's requested SKU; these fields are absent until Creator has
	// actually claimed the rented worker without invoking a model.
	ObservedAccelerator      string
	ObservedAcceleratorCount int
	ObservedBackend          string
	ObservedWorkerInstance   string
	ObservedWorkerBootID     string
	ObservedAt               string
	// ArtifactGrantRevision is the highest renter-owned access revision reserved for
	// this pod. It advances before the HTTP ask so a lost answer can never make a service
	// restart replay an older revision the worker will ignore.
	ArtifactGrantRevision uint64
}

const rentalCols = `id,endpoint_ref,accelerator_model,address,cert_path,state,hub,rented_at,media_address,control_snapshot_digest,control_snapshot_bytes,observed_accelerator,observed_accelerator_count,observed_backend,observed_worker_instance,observed_worker_boot_id,observed_at,artifact_grant_revision`

func scanRental(row interface{ Scan(...any) error }) (Rental, error) {
	var r Rental
	err := row.Scan(&r.ID, &r.EndpointRef, &r.AcceleratorModel, &r.Address, &r.CertPath,
		&r.State, &r.Hub, &r.RentedAt, &r.MediaAddress,
		&r.ControlSnapshotDigest, &r.ControlSnapshotBytes,
		&r.ObservedAccelerator, &r.ObservedAcceleratorCount, &r.ObservedBackend,
		&r.ObservedWorkerInstance, &r.ObservedWorkerBootID, &r.ObservedAt,
		&r.ArtifactGrantRevision)
	return r, err
}

// RecordRental writes what the hub provisioned. It REPLACES on the rental id because the
// id is the hub's, not this host's: re-reading a rental that moved from acquisition to
// ready must land on the same row rather than accumulate one per poll. State only moves
// forward: a delayed poll answering `acquiring` after `ready` was recorded is stale.
func (s *Store) RecordRental(r Rental) *exit.Error {
	if r.RentedAt == "" {
		r.RentedAt = now()
	}
	if r.ControlSnapshotBytes == nil {
		r.ControlSnapshotBytes = []byte{}
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin recording rental %s: %s", r.ID, err)
	}
	defer tx.Rollback()
	var current string
	switch err := tx.QueryRow(`SELECT state FROM rentals WHERE id=?`, r.ID).Scan(&current); {
	case errors.Is(err, sql.ErrNoRows):
	case err != nil:
		return exit.Internalf("cannot read rental %s state: %s", r.ID, err)
	default:
		r.State = rentalStateForward(current, r.State)
	}
	if _, err := tx.Exec(`INSERT INTO rentals(`+rentalCols+`)
		VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
		ON CONFLICT(id) DO UPDATE SET address=excluded.address,
		  cert_path=excluded.cert_path, state=excluded.state,
		  media_address=excluded.media_address,
		  control_snapshot_digest=CASE WHEN length(rentals.control_snapshot_bytes)>0
		    THEN rentals.control_snapshot_digest ELSE excluded.control_snapshot_digest END,
		  control_snapshot_bytes=CASE WHEN length(rentals.control_snapshot_bytes)>0
		    THEN rentals.control_snapshot_bytes ELSE excluded.control_snapshot_bytes END,
		  observed_accelerator=CASE WHEN rentals.observed_accelerator<>''
		    THEN rentals.observed_accelerator ELSE excluded.observed_accelerator END,
		  observed_accelerator_count=CASE WHEN rentals.observed_accelerator_count>0
		    THEN rentals.observed_accelerator_count ELSE excluded.observed_accelerator_count END,
		  observed_backend=CASE WHEN rentals.observed_backend<>''
		    THEN rentals.observed_backend ELSE excluded.observed_backend END,
		  observed_worker_instance=CASE WHEN rentals.observed_worker_instance<>''
		    THEN rentals.observed_worker_instance ELSE excluded.observed_worker_instance END,
		  observed_worker_boot_id=CASE WHEN rentals.observed_worker_boot_id<>''
		    THEN rentals.observed_worker_boot_id ELSE excluded.observed_worker_boot_id END,
		  observed_at=CASE WHEN rentals.observed_at<>''
		    THEN rentals.observed_at ELSE excluded.observed_at END`,
		r.ID, r.EndpointRef, r.AcceleratorModel, r.Address, r.CertPath, r.State, r.Hub,
		r.RentedAt, r.MediaAddress, r.ControlSnapshotDigest,
		r.ControlSnapshotBytes, r.ObservedAccelerator,
		r.ObservedAcceleratorCount, r.ObservedBackend, r.ObservedWorkerInstance,
		r.ObservedWorkerBootID, r.ObservedAt, r.ArtifactGrantRevision); err != nil {
		return exit.Internalf("cannot record rental %s: %s", r.ID, err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit rental %s: %s", r.ID, err)
	}
	stored, e := s.RentalRow(r.ID)
	if e != nil {
		return e
	}
	if len(r.ControlSnapshotBytes) > 0 && (stored == nil ||
		stored.ControlSnapshotDigest != r.ControlSnapshotDigest ||
		!bytes.Equal(stored.ControlSnapshotBytes, r.ControlSnapshotBytes)) {
		return exit.Named(exit.Conflict, "rental.control_snapshot_conflict",
			"rental %s already carries another exact acquisition-attempt control snapshot", r.ID)
	}
	return nil
}

// ReserveArtifactGrantRevision durably allocates the next monotonic access revision.
// Gaps are harmless; reuse is not. The reservation commits before a network call so a
// response lost after Tensorhub committed cannot move the next service process backward.
func (s *Store) ReserveArtifactGrantRevision(id string) (uint64, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return 0, exit.Internalf("cannot begin artifact-grant revision for rental %s: %s", id, err)
	}
	defer tx.Rollback()
	var current uint64
	if err := tx.QueryRow(`SELECT artifact_grant_revision FROM rentals WHERE id=?`, id).Scan(&current); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return 0, exit.New(exit.NotFound, "no rental %s on this host", id)
		}
		return 0, exit.Internalf("cannot read artifact-grant revision for rental %s: %s", id, err)
	}
	if current == ^uint64(0) {
		return 0, exit.Internalf("artifact-grant revision for rental %s is exhausted", id)
	}
	next := current + 1
	result, err := tx.Exec(`UPDATE rentals SET artifact_grant_revision=?
		WHERE id=? AND artifact_grant_revision=?`, next, id, current)
	if err != nil {
		return 0, exit.Internalf("cannot reserve artifact-grant revision for rental %s: %s", id, err)
	}
	changed, err := result.RowsAffected()
	if err != nil || changed != 1 {
		return 0, exit.Internalf("artifact-grant revision for rental %s changed concurrently", id)
	}
	if err := tx.Commit(); err != nil {
		return 0, exit.Internalf("cannot commit artifact-grant revision for rental %s: %s", id, err)
	}
	return next, nil
}

// ObserveRentalWorker records the actual remote worker ClaimAck before any model
// invocation. The requested accelerator is not evidence; the worker's readback must
// exactly agree with the SKU Tensorhub already qualified in its readiness receipt.
// The observation is immutable for one rental so a changed machine identity refuses
// instead of silently rewriting the evidence a workflow is about to freeze.
func (s *Store) ObserveRentalWorker(id, accelerator, backend, instance, bootID string,
	count int) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin rental %s worker observation: %s", id, err)
	}
	defer tx.Rollback()
	row, err := scanRental(tx.QueryRow(`SELECT `+rentalCols+` FROM rentals WHERE id=?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return exit.New(exit.NotFound, "no rental %s on this host", id)
	}
	if err != nil {
		return exit.Internalf("cannot read rental %s for worker observation: %s", id, err)
	}
	if row.State != "ready" || accelerator == "" || backend == "" || instance == "" ||
		bootID == "" || count != 1 {
		return exit.Named(exit.Conflict, "rental.worker_readback_incomplete",
			"rental %s ClaimAck is state=%q backend=%q accelerator=%q count=%d instance=%q boot=%q",
			id, row.State, backend, accelerator, count, instance, bootID)
	}
	if !acceleratorMatches(row.AcceleratorModel, accelerator) {
		return exit.Named(exit.Conflict, "rental.accelerator_readback_mismatch",
			"rental %s worker reports %q but the paid request selected %q",
			id, accelerator, row.AcceleratorModel).
			WithRemedy("release it; never invoke a model on hardware that disagrees with the paid selection")
	}
	if row.ObservedAccelerator != "" && (row.ObservedAccelerator != accelerator ||
		row.ObservedAcceleratorCount != count || row.ObservedBackend != backend ||
		row.ObservedWorkerInstance != instance) {
		return exit.Named(exit.Conflict, "rental.worker_readback_changed",
			"rental %s now claims a different accelerator or worker identity", id).
			WithRemedy("release it; a rental's accepted execution evidence is immutable")
	}
	// The worker process on the pod may restart; its boot id is the latest seen, while
	// the hardware and instance identity above stay write-once.
	if _, err := tx.Exec(`UPDATE rentals SET observed_accelerator=?,
		observed_accelerator_count=?,observed_backend=?,observed_worker_instance=?,
		observed_worker_boot_id=?,observed_at=? WHERE id=?`,
		accelerator, count, backend, instance, bootID, now(), id); err != nil {
		return exit.Internalf("cannot record rental %s worker observation: %s", id, err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit rental %s worker observation: %s", id, err)
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
