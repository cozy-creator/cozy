package records

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"errors"
	"strings"
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
  package_ref      TEXT NOT NULL,
  accelerator_model TEXT NOT NULL,
  address           TEXT NOT NULL,
  cert_path         TEXT NOT NULL,
  state             TEXT NOT NULL,
  hub               TEXT NOT NULL,
  rented_at         TEXT NOT NULL,
  media_address     TEXT NOT NULL DEFAULT '',
  placement_set_digest TEXT NOT NULL DEFAULT '',
  placement_set_bytes  BLOB NOT NULL DEFAULT x'',
  selection_profile TEXT NOT NULL DEFAULT '',
  package_release_digest TEXT NOT NULL DEFAULT '',
  package_release_bytes BLOB NOT NULL DEFAULT x'',
  package_descriptor_digest TEXT NOT NULL DEFAULT '',
  package_descriptor_bytes BLOB NOT NULL DEFAULT x'',
  qualification_digest TEXT NOT NULL DEFAULT '',
  qualification_bytes BLOB NOT NULL DEFAULT x'',
  placement_revision      INTEGER NOT NULL DEFAULT 0,
  observed_accelerator       TEXT NOT NULL DEFAULT '',
  observed_accelerator_count INTEGER NOT NULL DEFAULT 0,
  observed_backend           TEXT NOT NULL DEFAULT '',
  observed_driver_version    TEXT NOT NULL DEFAULT '',
  observed_backend_version   TEXT NOT NULL DEFAULT '',
  observed_device_memory_total_bytes INTEGER NOT NULL DEFAULT 0,
  observed_worker_instance   TEXT NOT NULL DEFAULT '',
  observed_worker_boot_id    TEXT NOT NULL DEFAULT '',
  observed_at                TEXT NOT NULL DEFAULT '',
  expected_worker_id         TEXT NOT NULL DEFAULT '',
  expected_worker_boot_id    TEXT NOT NULL DEFAULT ''
)`

var rentalSchema = []string{rentalOperationsDDL, rentalsDDL, `
CREATE UNIQUE INDEX IF NOT EXISTS rental_operation_remote
  ON rental_operations(rental_id) WHERE rental_id <> ''`, `
CREATE TABLE IF NOT EXISTS rental_relay_refusals (
  rental_id   TEXT PRIMARY KEY REFERENCES rentals(id) ON DELETE CASCADE,
  exit_code   INTEGER NOT NULL,
  name        TEXT NOT NULL,
  message     TEXT NOT NULL,
  remedy      TEXT NOT NULL,
  next_json   TEXT NOT NULL,
  observed_at TEXT NOT NULL
)`}

// migrateRentalSchema moves older local roots forward without tying their data to the
// executable that last wrote it. Paid-rental safety comes from preserving lifecycle rows,
// not from comparing sqlite_master's formatting with a Go string.
func migrateRentalSchema(db *sql.DB, path string) *exit.Error {
	columns, err := tableColumns(db, "rentals")
	if err != nil {
		return exit.Internalf("cannot inspect the rentals schema in %s: %s", path, err)
	}
	if !columns["package_ref"] {
		return exit.Internalf("cannot migrate the rentals schema in %s: package_ref is absent", path)
	}
	for _, column := range []struct {
		name string
		ddl  string
	}{
		{"placement_set_digest", `TEXT NOT NULL DEFAULT ''`},
		{"placement_set_bytes", `BLOB NOT NULL DEFAULT x''`},
		{"placement_revision", `INTEGER NOT NULL DEFAULT 0`},
		{"selection_profile", `TEXT NOT NULL DEFAULT ''`},
		{"package_release_digest", `TEXT NOT NULL DEFAULT ''`},
		{"package_release_bytes", `BLOB NOT NULL DEFAULT x''`},
		{"package_descriptor_digest", `TEXT NOT NULL DEFAULT ''`},
		{"package_descriptor_bytes", `BLOB NOT NULL DEFAULT x''`},
		{"qualification_digest", `TEXT NOT NULL DEFAULT ''`},
		{"qualification_bytes", `BLOB NOT NULL DEFAULT x''`},
		{"observed_accelerator", `TEXT NOT NULL DEFAULT ''`},
		{"observed_accelerator_count", `INTEGER NOT NULL DEFAULT 0`},
		{"observed_backend", `TEXT NOT NULL DEFAULT ''`},
		{"observed_driver_version", `TEXT NOT NULL DEFAULT ''`},
		{"observed_backend_version", `TEXT NOT NULL DEFAULT ''`},
		{"observed_device_memory_total_bytes", `INTEGER NOT NULL DEFAULT 0`},
		{"observed_worker_instance", `TEXT NOT NULL DEFAULT ''`},
		{"observed_worker_boot_id", `TEXT NOT NULL DEFAULT ''`},
		{"observed_at", `TEXT NOT NULL DEFAULT ''`},
		{"expected_worker_id", `TEXT NOT NULL DEFAULT ''`},
		{"expected_worker_boot_id", `TEXT NOT NULL DEFAULT ''`},
		{"machine_name", `TEXT NOT NULL DEFAULT ''`},
		{"sku", `TEXT NOT NULL DEFAULT ''`},
	} {
		if columns[column.name] {
			continue
		}
		if _, err := db.Exec(`ALTER TABLE rentals ADD COLUMN ` + column.name + ` ` + column.ddl); err != nil {
			return exit.Internalf("cannot add rentals.%s in %s: %s", column.name, path, err)
		}
	}
	rows, err := db.Query(`SELECT id,machine_name FROM rentals`)
	if err != nil {
		return exit.Internalf("cannot read rental machine names in %s: %s", path, err)
	}
	type rentalName struct{ id, name string }
	var names []rentalName
	for rows.Next() {
		var row rentalName
		if err := rows.Scan(&row.id, &row.name); err != nil {
			rows.Close()
			return exit.Internalf("cannot read a rental machine name in %s: %s", path, err)
		}
		names = append(names, row)
	}
	if err := rows.Close(); err != nil {
		return exit.Internalf("cannot finish reading rental machine names in %s: %s", path, err)
	}
	for _, row := range names {
		if row.name != "" {
			continue
		}
		if _, err := db.Exec(`UPDATE rentals SET machine_name=? WHERE id=?`,
			rentalid.MachineName(row.id), row.id); err != nil {
			return exit.Internalf("cannot name rental %s in %s: %s", row.id, path, err)
		}
	}
	if _, err := db.Exec(`CREATE UNIQUE INDEX IF NOT EXISTS rentals_machine_name
		ON rentals(machine_name) WHERE machine_name<>''`); err != nil {
		return exit.Internalf("cannot index rental machine names in %s: %s", path, err)
	}
	return nil
}

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
	ID               string
	MachineName      string
	SKU              string
	PackageRef       string
	AcceleratorModel string
	Address          string
	CertPath         string
	State            string
	Hub              string
	RentedAt         string
	// MediaAddress is where the pod's co-resident media server answers (cl-014). It is a
	// FACT about the pod like the control address is, so it is a row and not a file; the
	// credential it takes is the rental's media bearer, which stays 0600 beside it.
	MediaAddress string
	// PlacementSetBytes are Tensorhub's exact selected desired state. They remain
	// raw bytes so a restart relays rather than re-renders execution meaning.
	PlacementSetDigest      string
	PlacementSetBytes       []byte
	SelectionProfile        string
	PackageReleaseDigest    string
	PackageReleaseBytes     []byte
	PackageDescriptorDigest string
	PackageDescriptorBytes  []byte
	QualificationDigest     string
	QualificationBytes      []byte
	PlacementRevision       uint64
	// Observed* is the remote worker's ClaimAck readback. AcceleratorModel above is
	// only the caller's requested SKU; these fields are absent until Cozy has
	// actually claimed the rented worker without invoking a model.
	ObservedAccelerator            string
	ObservedAcceleratorCount       int
	ObservedBackend                string
	ObservedDriverVersion          string
	ObservedBackendVersion         string
	ObservedDeviceMemoryTotalBytes uint64
	ObservedWorkerInstance         string
	ObservedWorkerBootID           string
	ObservedAt                     string
	ExpectedWorkerID               string
	ExpectedWorkerBootID           string
}

const rentalCols = `id,machine_name,sku,package_ref,accelerator_model,address,cert_path,state,hub,rented_at,media_address,placement_set_digest,placement_set_bytes,selection_profile,package_release_digest,package_release_bytes,package_descriptor_digest,package_descriptor_bytes,qualification_digest,qualification_bytes,placement_revision,observed_accelerator,observed_accelerator_count,observed_backend,observed_driver_version,observed_backend_version,observed_device_memory_total_bytes,observed_worker_instance,observed_worker_boot_id,observed_at,expected_worker_id,expected_worker_boot_id`

func scanRental(row interface{ Scan(...any) error }) (Rental, error) {
	var r Rental
	err := row.Scan(&r.ID, &r.MachineName, &r.SKU, &r.PackageRef, &r.AcceleratorModel, &r.Address, &r.CertPath,
		&r.State, &r.Hub, &r.RentedAt, &r.MediaAddress,
		&r.PlacementSetDigest, &r.PlacementSetBytes,
		&r.SelectionProfile, &r.PackageReleaseDigest, &r.PackageReleaseBytes,
		&r.PackageDescriptorDigest, &r.PackageDescriptorBytes,
		&r.QualificationDigest, &r.QualificationBytes,
		&r.PlacementRevision,
		&r.ObservedAccelerator, &r.ObservedAcceleratorCount, &r.ObservedBackend,
		&r.ObservedDriverVersion, &r.ObservedBackendVersion, &r.ObservedDeviceMemoryTotalBytes,
		&r.ObservedWorkerInstance, &r.ObservedWorkerBootID, &r.ObservedAt,
		&r.ExpectedWorkerID, &r.ExpectedWorkerBootID)
	return r, err
}

// RecordRental writes what the hub provisioned. It REPLACES on the rental id because the
// id is the hub's, not this host's: re-reading a rental that moved from acquisition to
// ready must land on the same row rather than accumulate one per poll. State only moves
// forward: a delayed poll answering `acquiring` after `ready` was recorded is stale.
func (s *Store) RecordRental(r Rental) *exit.Error {
	var problem *exit.Error
	for range 3 {
		problem = s.recordRental(r)
		if problem == nil || !strings.Contains(problem.Message, "SQLITE_BUSY") {
			return problem
		}
	}
	return problem
}

func (s *Store) recordRental(r Rental) *exit.Error {
	if r.MachineName == "" || r.SKU == "" {
		var existingName, existingSKU string
		err := s.db.QueryRow(`SELECT machine_name,sku FROM rentals WHERE id=?`, r.ID).
			Scan(&existingName, &existingSKU)
		if err == nil {
			if r.MachineName == "" {
				r.MachineName = existingName
			}
			if r.SKU == "" {
				r.SKU = existingSKU
			}
		} else if err != nil && !errors.Is(err, sql.ErrNoRows) {
			return exit.Internalf("cannot read rental %s local identity: %s", r.ID, err)
		}
	}
	if r.MachineName == "" {
		r.MachineName = rentalid.MachineName(r.ID)
	}
	if !rentalid.ValidMachineName(r.MachineName) {
		return exit.Named(exit.Validation, "rental.machine_name_invalid",
			"%q is not a machine name", r.MachineName).
			WithRemedy("use 1-32 lowercase letters, numbers, and hyphens; local is reserved")
	}
	var conflictingID string
	err := s.db.QueryRow(`SELECT id FROM rentals
		WHERE id<>? AND (id=? OR machine_name=?) LIMIT 1`,
		r.ID, r.MachineName, r.MachineName).Scan(&conflictingID)
	if err == nil {
		return exit.Named(exit.Conflict, "rental.machine_name_conflict",
			"machine name %q already identifies rental %s", r.MachineName, conflictingID).
			WithRemedy("choose another --name")
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return exit.Internalf("cannot check rental machine name %s: %s", r.MachineName, err)
	}
	if r.RentedAt == "" {
		r.RentedAt = now()
	}
	if r.PlacementSetBytes == nil {
		r.PlacementSetBytes = []byte{}
	}
	if r.PackageReleaseBytes == nil {
		r.PackageReleaseBytes = []byte{}
	}
	if r.PackageDescriptorBytes == nil {
		r.PackageDescriptorBytes = []byte{}
	}
	if r.QualificationBytes == nil {
		r.QualificationBytes = []byte{}
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
		VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
		ON CONFLICT(id) DO UPDATE SET
		  machine_name=CASE WHEN rentals.machine_name<>'' THEN rentals.machine_name ELSE excluded.machine_name END,
		  sku=CASE WHEN rentals.sku<>'' THEN rentals.sku ELSE excluded.sku END,
		  address=CASE WHEN rentals.address<>'' THEN rentals.address ELSE excluded.address END,
		  cert_path=CASE WHEN rentals.cert_path<>'' THEN rentals.cert_path ELSE excluded.cert_path END,
		  state=excluded.state,
		  media_address=CASE WHEN rentals.media_address<>'' THEN rentals.media_address ELSE excluded.media_address END,
		  placement_set_digest=CASE WHEN length(rentals.placement_set_bytes)>0
		    THEN rentals.placement_set_digest ELSE excluded.placement_set_digest END,
		  placement_set_bytes=CASE WHEN length(rentals.placement_set_bytes)>0
		    THEN rentals.placement_set_bytes ELSE excluded.placement_set_bytes END,
		  selection_profile=CASE WHEN rentals.selection_profile<>'' THEN rentals.selection_profile ELSE excluded.selection_profile END,
		  package_release_digest=CASE WHEN length(rentals.package_release_bytes)>0 THEN rentals.package_release_digest ELSE excluded.package_release_digest END,
		  package_release_bytes=CASE WHEN length(rentals.package_release_bytes)>0 THEN rentals.package_release_bytes ELSE excluded.package_release_bytes END,
		  package_descriptor_digest=CASE WHEN length(rentals.package_descriptor_bytes)>0 THEN rentals.package_descriptor_digest ELSE excluded.package_descriptor_digest END,
		  package_descriptor_bytes=CASE WHEN length(rentals.package_descriptor_bytes)>0 THEN rentals.package_descriptor_bytes ELSE excluded.package_descriptor_bytes END,
		  qualification_digest=CASE WHEN length(rentals.qualification_bytes)>0 THEN rentals.qualification_digest ELSE excluded.qualification_digest END,
		  qualification_bytes=CASE WHEN length(rentals.qualification_bytes)>0 THEN rentals.qualification_bytes ELSE excluded.qualification_bytes END,
		  placement_revision=CASE WHEN rentals.placement_revision>0
		    THEN rentals.placement_revision ELSE excluded.placement_revision END,
		  observed_accelerator=CASE WHEN rentals.observed_accelerator<>''
		    THEN rentals.observed_accelerator ELSE excluded.observed_accelerator END,
		  observed_accelerator_count=CASE WHEN rentals.observed_accelerator_count>0
		    THEN rentals.observed_accelerator_count ELSE excluded.observed_accelerator_count END,
		  observed_backend=CASE WHEN rentals.observed_backend<>''
		    THEN rentals.observed_backend ELSE excluded.observed_backend END,
		  observed_driver_version=CASE WHEN rentals.observed_driver_version<>''
		    THEN rentals.observed_driver_version ELSE excluded.observed_driver_version END,
		  observed_backend_version=CASE WHEN rentals.observed_backend_version<>''
		    THEN rentals.observed_backend_version ELSE excluded.observed_backend_version END,
		  observed_device_memory_total_bytes=CASE WHEN rentals.observed_device_memory_total_bytes>0
		    THEN rentals.observed_device_memory_total_bytes ELSE excluded.observed_device_memory_total_bytes END,
		  observed_worker_instance=CASE WHEN rentals.observed_worker_instance<>''
		    THEN rentals.observed_worker_instance ELSE excluded.observed_worker_instance END,
		  observed_worker_boot_id=CASE WHEN rentals.observed_worker_boot_id<>''
		    THEN rentals.observed_worker_boot_id ELSE excluded.observed_worker_boot_id END,
		  observed_at=CASE WHEN rentals.observed_at<>''
		    THEN rentals.observed_at ELSE excluded.observed_at END,
		  expected_worker_id=CASE WHEN rentals.expected_worker_id<>''
		    THEN rentals.expected_worker_id ELSE excluded.expected_worker_id END,
		  expected_worker_boot_id=CASE WHEN rentals.expected_worker_boot_id<>''
		    THEN rentals.expected_worker_boot_id ELSE excluded.expected_worker_boot_id END`,
		r.ID, r.MachineName, r.SKU, r.PackageRef, r.AcceleratorModel, r.Address, r.CertPath, r.State, r.Hub,
		r.RentedAt, r.MediaAddress, r.PlacementSetDigest,
		r.PlacementSetBytes, r.SelectionProfile, r.PackageReleaseDigest, r.PackageReleaseBytes,
		r.PackageDescriptorDigest, r.PackageDescriptorBytes, r.QualificationDigest,
		r.QualificationBytes, r.PlacementRevision, r.ObservedAccelerator,
		r.ObservedAcceleratorCount, r.ObservedBackend, r.ObservedDriverVersion,
		r.ObservedBackendVersion, r.ObservedDeviceMemoryTotalBytes, r.ObservedWorkerInstance,
		r.ObservedWorkerBootID, r.ObservedAt, r.ExpectedWorkerID, r.ExpectedWorkerBootID); err != nil {
		return exit.Internalf("cannot record rental %s: %s", r.ID, err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit rental %s: %s", r.ID, err)
	}
	stored, e := s.RentalRow(r.ID)
	if e != nil {
		return e
	}
	if len(r.PlacementSetBytes) > 0 && (stored == nil ||
		stored.PlacementSetDigest != r.PlacementSetDigest ||
		!bytes.Equal(stored.PlacementSetBytes, r.PlacementSetBytes)) {
		return exit.Named(exit.Conflict, "rental.placement_set_conflict",
			"rental %s already carries another exact PlacementSet", r.ID)
	}
	if stored == nil || stored.MachineName != r.MachineName || stored.SKU != r.SKU ||
		r.Address != "" && stored.Address != r.Address ||
		r.MediaAddress != "" && stored.MediaAddress != r.MediaAddress ||
		r.CertPath != "" && stored.CertPath != r.CertPath ||
		r.ExpectedWorkerID != "" && stored.ExpectedWorkerID != r.ExpectedWorkerID ||
		r.ExpectedWorkerBootID != "" && stored.ExpectedWorkerBootID != r.ExpectedWorkerBootID {
		return exit.Named(exit.Conflict, "rental.attach_projection_conflict",
			"rental %s already carries another address, media address, or certificate pin", r.ID)
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

// ReplaceRentalSelection installs the Hub-authored replacement after its exact
// desired revision is known. It is the only path allowed to change PlacementSet.
func (s *Store) ReplaceRentalSelection(next Rental) *exit.Error {
	current, e := s.RentalRow(next.ID)
	if e != nil {
		return e
	}
	if current == nil {
		return exit.New(exit.NotFound, "no rental %s on this host", next.ID)
	}
	same := current.PackageRef == next.PackageRef && current.PlacementRevision == next.PlacementRevision &&
		current.PlacementSetDigest == next.PlacementSetDigest &&
		bytes.Equal(current.PlacementSetBytes, next.PlacementSetBytes)
	if same {
		return nil
	}
	if next.PlacementRevision <= current.PlacementRevision {
		return exit.Named(exit.Conflict, "rental.placement_revision_regressed",
			"rental %s replacement revision %d does not advance %d",
			next.ID, next.PlacementRevision, current.PlacementRevision)
	}
	result, err := s.db.Exec(`UPDATE rentals SET package_ref=?,placement_set_digest=?,placement_set_bytes=?,
		selection_profile=?,package_release_digest=?,package_release_bytes=?,
		package_descriptor_digest=?,package_descriptor_bytes=?,qualification_digest=?,qualification_bytes=?,
		placement_revision=? WHERE id=? AND placement_revision=?`,
		next.PackageRef, next.PlacementSetDigest, next.PlacementSetBytes, next.SelectionProfile,
		next.PackageReleaseDigest, next.PackageReleaseBytes, next.PackageDescriptorDigest,
		next.PackageDescriptorBytes, next.QualificationDigest, next.QualificationBytes,
		next.PlacementRevision, next.ID, current.PlacementRevision)
	if err != nil {
		return exit.Internalf("cannot replace rental %s selection: %s", next.ID, err)
	}
	changed, _ := result.RowsAffected()
	if changed != 1 {
		return exit.Named(exit.Conflict, "rental.placement_update_raced",
			"rental %s selection advanced concurrently", next.ID)
	}
	return nil
}

// ObserveRentalWorker records the actual remote worker ClaimAck before any model
// invocation. The requested accelerator is not evidence; the worker's readback must
// exactly agree with the SKU Tensorhub already qualified in its readiness receipt.
// The observation is immutable for one rental so a changed machine identity refuses
// instead of silently rewriting retained execution evidence.
func (s *Store) ObserveRentalWorker(id, accelerator, backend, driverVersion,
	backendVersion string, deviceMemory uint64, instance, workerID, bootID string, count int) *exit.Error {
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
	cpu := strings.EqualFold(row.AcceleratorModel, "CPU")
	complete := row.State == "converging" || row.State == "ready"
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
	if row.ObservedAt != "" && (row.ObservedAccelerator != accelerator ||
		row.ObservedAcceleratorCount != count || row.ObservedBackend != backend ||
		row.ObservedDriverVersion != driverVersion || row.ObservedBackendVersion != backendVersion ||
		row.ObservedDeviceMemoryTotalBytes != deviceMemory ||
		row.ObservedWorkerInstance != instance) {
		return exit.Named(exit.Conflict, "rental.worker_readback_changed",
			"rental %s now claims a different accelerator or worker identity", id).
			WithRemedy("release it; a rental's accepted execution evidence is immutable")
	}
	// The worker process on the pod may restart; its boot id is the latest seen, while
	// the hardware and instance identity above stay write-once.
	if _, err := tx.Exec(`UPDATE rentals SET observed_accelerator=?,
		observed_accelerator_count=?,observed_backend=?,observed_driver_version=?,
		observed_backend_version=?,observed_device_memory_total_bytes=?,observed_worker_instance=?,
		observed_worker_boot_id=?,observed_at=? WHERE id=?`,
		accelerator, count, backend, driverVersion, backendVersion, deviceMemory,
		instance, bootID, now(), id); err != nil {
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

// RentalRunCounts derives the current queue and active-attempt counts for one machine.
func (s *Store) RentalRunCounts(id string) (queued, running int, problem *exit.Error) {
	err := s.db.QueryRow(`SELECT
		COALESCE(SUM(CASE WHEN state IN ('submitted','queued','requeue_pending') THEN 1 ELSE 0 END),0),
		COALESCE(SUM(CASE WHEN state IN ('dispatching','in_progress') THEN 1 ELSE 0 END),0)
		FROM requests WHERE worker=?`, id).Scan(&queued, &running)
	if err != nil {
		return 0, 0, exit.Internalf("cannot count runs for rented machine %s: %s", id, err)
	}
	return queued, running, nil
}

// RentalRelayRefusal is the last durable non-transient private-control verdict, whether
// the worker rejected Claim or Tensorhub rejected relayed session evidence. Without it
// every client would continue seeing only `converging` after control had already refused.
type RentalRelayRefusal struct {
	RentalID   string
	Code       exit.Code
	Name       string
	Message    string
	Remedy     string
	Next       []string
	ObservedAt string
}

func (r RentalRelayRefusal) Error() *exit.Error {
	e := exit.Named(r.Code, r.Name, "%s", r.Message)
	if r.Remedy != "" {
		e.WithRemedy("%s", r.Remedy)
	}
	if len(r.Next) > 0 {
		e.WithNext(r.Next...)
	}
	return e
}

func (s *Store) RecordRentalRelayRefusal(id string, problem *exit.Error) *exit.Error {
	if id == "" || problem == nil {
		return exit.Internalf("cannot record an empty rental relay refusal")
	}
	next, err := json.Marshal(problem.Next)
	if err != nil {
		return exit.Internalf("cannot encode rental %s relay refusal: %s", id, err)
	}
	if _, err := s.db.Exec(`INSERT INTO rental_relay_refusals(
		rental_id,exit_code,name,message,remedy,next_json,observed_at)
		VALUES(?,?,?,?,?,?,?) ON CONFLICT(rental_id) DO UPDATE SET
		exit_code=excluded.exit_code,name=excluded.name,message=excluded.message,
		remedy=excluded.remedy,next_json=excluded.next_json,observed_at=excluded.observed_at`,
		id, int(problem.Code), problem.ErrName(), problem.Message, problem.Remedy, string(next), now()); err != nil {
		return exit.Internalf("cannot record rental %s relay refusal: %s", id, err)
	}
	return nil
}

func (s *Store) RentalRelayRefusal(id string) (*RentalRelayRefusal, *exit.Error) {
	var row RentalRelayRefusal
	var code int
	var next string
	err := s.db.QueryRow(`SELECT rental_id,exit_code,name,message,remedy,next_json,observed_at
		FROM rental_relay_refusals WHERE rental_id=?`, id).Scan(
		&row.RentalID, &code, &row.Name, &row.Message, &row.Remedy, &next, &row.ObservedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read rental %s relay refusal: %s", id, err)
	}
	row.Code = exit.Code(code)
	if !row.Code.Valid() {
		return nil, exit.Internalf("rental %s relay refusal carries invalid exit code %d", id, code)
	}
	if err := json.Unmarshal([]byte(next), &row.Next); err != nil {
		return nil, exit.Internalf("cannot decode rental %s relay refusal: %s", id, err)
	}
	return &row, nil
}

func (s *Store) ClearRentalRelayRefusal(id string) *exit.Error {
	if _, err := s.db.Exec(`DELETE FROM rental_relay_refusals WHERE rental_id=?`, id); err != nil {
		return exit.Internalf("cannot clear rental %s relay refusal: %s", id, err)
	}
	return nil
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
