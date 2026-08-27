package records

import (
	"bytes"
	"database/sql"
	"path/filepath"
	"sync"
	"testing"
)

func TestOpenDropsWriteOnlyRentalOperationFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "records.db")
	legacy, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	requestBody := []byte(`{"endpoint_ref":"acme/h3/v1/generate","accelerator_model":"NVIDIA H200"}`)
	if _, err := legacy.Exec(`CREATE TABLE rental_operations (
  operation_key TEXT PRIMARY KEY, request_digest TEXT NOT NULL, request_body BLOB NOT NULL,
  endpoint_ref TEXT NOT NULL, accelerator_model TEXT NOT NULL, hub TEXT NOT NULL,
  reason TEXT NOT NULL, rental_id TEXT NOT NULL DEFAULT '', state TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL)`); err != nil {
		legacy.Close()
		t.Fatal(err)
	}
	if _, err := legacy.Exec(`INSERT INTO rental_operations VALUES
  ('op-current','sha256:request',?,?,?,?,'reason','rnt-current','provisioning','then','then')`,
		requestBody, "acme/h3/v1/generate", "NVIDIA H200", "https://hub.invalid"); err != nil {
		legacy.Close()
		t.Fatal(err)
	}
	if err := legacy.Close(); err != nil {
		t.Fatal(err)
	}

	st, e := Open(path)
	if e != nil {
		t.Fatal(e)
	}
	defer st.Close()
	op, e := st.RentalOperation("op-current")
	if e != nil || op == nil || !bytes.Equal(op.RequestBody, requestBody) || op.State != "acquiring" {
		t.Fatalf("migrated operation = %#v, %v", op, e)
	}
	for _, column := range []string{"endpoint_ref", "accelerator_model"} {
		var count int
		if err := st.db.QueryRow(`SELECT COUNT(*) FROM pragma_table_info('rental_operations')
      WHERE name=?`, column).Scan(&count); err != nil {
			t.Fatal(err)
		}
		if count != 0 {
			t.Fatalf("write-only column rental_operations.%s survived", column)
		}
	}
}

func TestAdvanceRentalOperationIsMonotone(t *testing.T) {
	path := filepath.Join(t.TempDir(), "records.db")
	st, e := Open(path)
	if e != nil {
		t.Fatal(e)
	}
	defer st.Close()
	_, _, e = st.BeginRentalOperation(RentalOperation{
		Key: "op", RequestDigest: "sha256:request", RequestBody: []byte(`{}`),
		Hub: "https://hub.invalid", Reason: "test",
	})
	if e != nil {
		t.Fatal(e)
	}

	// These calls may race in production. Whichever writer lands first, a later stale
	// observation cannot move the durable operation behind attached.
	states := []string{"acquiring", "materializing", "ready", "attached", "pending_acquisition"}
	var wg sync.WaitGroup
	for _, state := range states {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if e := st.AdvanceRentalOperation("op", "rnt-one", state); e != nil {
				t.Errorf("advance to %s: %v", state, e)
			}
		}()
	}
	wg.Wait()
	op, e := st.RentalOperation("op")
	if e != nil || op == nil || op.State != "attached" || op.RentalID != "rnt-one" {
		t.Fatalf("operation after concurrent observations = %#v, %v", op, e)
	}
	if e := st.AdvanceRentalOperation("op", "rnt-one", "materializing"); e != nil {
		t.Fatal(e)
	}
	op, e = st.RentalOperation("op")
	if e != nil || op.State != "attached" {
		t.Fatalf("attached operation regressed = %#v, %v", op, e)
	}
	if e := st.AdvanceRentalOperation("op", "rnt-one", "failed"); e != nil {
		t.Fatal(e)
	}
	if e := st.AdvanceRentalOperation("op", "rnt-one", "ready"); e != nil {
		t.Fatal(e)
	}
	op, e = st.RentalOperation("op")
	if e != nil || op.State != "failed" {
		t.Fatalf("failed operation regressed = %#v, %v", op, e)
	}
	if e := st.AdvanceRentalOperation("op", "rnt-other", "ready"); e == nil || e.ErrName() != "rental.operation_conflict" {
		t.Fatalf("foreign rental id conflict = %v", e)
	}
	if e := st.AdvanceRentalOperation("op", "", "rejected"); e == nil || e.ErrName() != "rental.operation_conflict" {
		t.Fatalf("paid operation rejection conflict = %v", e)
	}
	_, _, e = st.BeginRentalOperation(RentalOperation{
		Key: "op-rejected", RequestDigest: "sha256:rejected", RequestBody: []byte(`{}`),
		Hub: "https://hub.invalid", Reason: "test",
	})
	if e != nil {
		t.Fatal(e)
	}
	if e := st.AdvanceRentalOperation("op-rejected", "", "rejected"); e != nil {
		t.Fatal(e)
	}
	if e := st.AdvanceRentalOperation("op-rejected", "rnt-late", "ready"); e != nil {
		t.Fatal(e)
	}
	rejected, e := st.RentalOperation("op-rejected")
	if e != nil || rejected == nil || rejected.State != "rejected" || rejected.RentalID != "" {
		t.Fatalf("rejected operation regressed = %#v, %v", rejected, e)
	}

	if e := st.RecordRental(Rental{ID: "rnt-one", EndpointRef: "acme/h3/v1/generate",
		AcceleratorModel: "NVIDIA H200", State: "ready", Hub: "https://hub.invalid"}); e != nil {
		t.Fatal(e)
	}
	if key, e := st.RequestRentalRelease("rnt-one"); e != nil || key != "op" {
		t.Fatalf("request release = %q, %v", key, e)
	}
	if e := st.AdvanceRentalOperation("op", "rnt-one", "materializing"); e != nil {
		t.Fatal(e)
	}
	op, e = st.RentalOperation("op")
	if e != nil || op.State != "release_requested" {
		t.Fatalf("release request regressed = %#v, %v", op, e)
	}
	if forgotten, e := st.ForgetRental("rnt-one"); e != nil || !forgotten {
		t.Fatalf("forget = %v, %v", forgotten, e)
	}
	if e := st.AdvanceRentalOperation("op", "rnt-one", "acquiring"); e != nil {
		t.Fatal(e)
	}
	op, e = st.RentalOperation("op")
	if e != nil || op.State != "released" {
		t.Fatalf("released operation regressed = %#v, %v", op, e)
	}
}

func TestRentalControlSnapshotPersistsVerbatimAndLatePollCannotEraseIt(t *testing.T) {
	st, e := Open(filepath.Join(t.TempDir(), "records.db"))
	if e != nil {
		t.Fatal(e)
	}
	defer st.Close()
	raw := []byte{0x00, 0x01, 0x02, '{', '}', 0xff}
	row := Rental{ID: "rnt-control", EndpointRef: "acme/h3/v1/generate",
		AcceleratorModel: "NVIDIA H200", Address: "pod.invalid:443", CertPath: "/pinned/cert",
		State: "ready", Hub: "https://hub.invalid", ControlSnapshotDigest: "sha256:exact",
		ControlSnapshotLength: int64(len(raw)), ControlSnapshotBytes: raw}
	if e := st.RecordRental(row); e != nil {
		t.Fatal(e)
	}
	// A later lifecycle observation may omit the ready-only snapshot. It can move
	// state/address fields but cannot erase the exact bytes already persisted.
	row.State, row.ControlSnapshotDigest, row.ControlSnapshotLength, row.ControlSnapshotBytes =
		"release_requested", "", 0, nil
	if e := st.RecordRental(row); e != nil {
		t.Fatal(e)
	}
	got, e := st.RentalRow("rnt-control")
	if e != nil || got == nil || got.ControlSnapshotDigest != "sha256:exact" ||
		got.ControlSnapshotLength != int64(len(raw)) || !bytes.Equal(got.ControlSnapshotBytes, raw) {
		t.Fatalf("persisted snapshot = %#v, %v", got, e)
	}
	conflict := *got
	conflict.ControlSnapshotDigest = "sha256:changed"
	conflict.ControlSnapshotBytes = append([]byte(nil), raw...)
	conflict.ControlSnapshotBytes[0] ^= 0xff
	if e := st.RecordRental(conflict); e == nil || e.Name != "rental.control_snapshot_conflict" {
		t.Fatalf("changed snapshot was not refused: %v", e)
	}
}

func TestOpenHardcutsLegacyProviderRentalFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "records.db")
	legacy, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	for _, stmt := range []string{
		`CREATE TABLE rental_operations (
  operation_key TEXT PRIMARY KEY, request_digest TEXT NOT NULL,
  endpoint TEXT NOT NULL, card TEXT NOT NULL, region TEXT NOT NULL,
  hub TEXT NOT NULL, reason TEXT NOT NULL, rental_id TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)`,
		`CREATE UNIQUE INDEX rental_operation_remote
  ON rental_operations(rental_id) WHERE rental_id <> ''`,
		`CREATE TABLE rentals (
  id TEXT PRIMARY KEY, endpoint TEXT NOT NULL, card TEXT NOT NULL, pod_id TEXT NOT NULL,
  address TEXT NOT NULL, cert_path TEXT NOT NULL, state TEXT NOT NULL, hub TEXT NOT NULL,
  rented_at TEXT NOT NULL, released_at TEXT NOT NULL DEFAULT '')`,
		`INSERT INTO rental_operations VALUES
  ('op-old','sha256:old','acme/h3/v1/generate','NVIDIA H200','provider-dc',
   'https://hub.invalid','original','rnt-old','attached','then','then')`,
		`INSERT INTO rentals VALUES
  ('rnt-old','acme/h3/v1/generate','NVIDIA H200','provider-pod-should-die',
   'worker.invalid:443','/safe/cert.pem','ready','https://hub.invalid','then','')`,
	} {
		if _, err := legacy.Exec(stmt); err != nil {
			legacy.Close()
			t.Fatal(err)
		}
	}
	if err := legacy.Close(); err != nil {
		t.Fatal(err)
	}

	st, e := Open(path)
	if e != nil {
		t.Fatal(e)
	}
	defer st.Close()

	op, e := st.RentalOperation("op-old")
	if e != nil || op == nil {
		t.Fatalf("migrated operation = %#v, %v", op, e)
	}
	if len(op.RequestBody) != 0 {
		t.Fatalf("migrated operation = %#v", op)
	}
	r, e := st.RentalRow("rnt-old")
	if e != nil || r == nil {
		t.Fatalf("migrated rental = %#v, %v", r, e)
	}
	if r.EndpointRef != "acme/h3/v1/generate" || r.AcceleratorModel != "NVIDIA H200" || r.Address == "" {
		t.Fatalf("migrated rental = %#v", r)
	}

	for table, forbidden := range map[string][]string{
		"rental_operations": {"endpoint", "card", "region", "endpoint_ref", "accelerator_model"},
		"rentals":           {"endpoint", "card", "pod_id"},
	} {
		rows, err := st.db.Query(`PRAGMA table_info(` + table + `)`)
		if err != nil {
			t.Fatal(err)
		}
		seen := map[string]bool{}
		for rows.Next() {
			var cid, notnull, pk int
			var name, kind string
			var def any
			if err := rows.Scan(&cid, &name, &kind, &notnull, &def, &pk); err != nil {
				rows.Close()
				t.Fatal(err)
			}
			seen[name] = true
		}
		rows.Close()
		for _, name := range forbidden {
			if seen[name] {
				t.Fatalf("legacy column %s.%s survived hardcut", table, name)
			}
		}
	}
}
