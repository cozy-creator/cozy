package records

import (
	"database/sql"
	"path/filepath"
	"testing"
)

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
	if op.EndpointRef != "acme/h3/v1/generate" || op.AcceleratorModel != "NVIDIA H200" || len(op.RequestBody) != 0 {
		t.Fatalf("migrated operation = %#v", op)
	}
	r, e := st.RentalRow("rnt-old")
	if e != nil || r == nil {
		t.Fatalf("migrated rental = %#v, %v", r, e)
	}
	if r.EndpointRef != op.EndpointRef || r.AcceleratorModel != op.AcceleratorModel || r.Address == "" {
		t.Fatalf("migrated rental = %#v", r)
	}

	for table, forbidden := range map[string][]string{
		"rental_operations": {"endpoint", "card", "region"},
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
