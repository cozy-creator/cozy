package producttest

import (
	"database/sql"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentNewPublicStoreSurvivesMigrationAndReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	st, problem := records.Open(path)
	if problem != nil {
		t.Fatal(problem)
	}
	req := records.Request{ID: "fresh-request", IdemKey: "fresh-key", BodyDigest: "sha256:" + strings.Repeat("a", 64), Package: "proof/model", Entrypoint: "run", Payload: []byte(`{}`), Rental: true, RentalRequired: true, RentNew: true}
	if _, _, problem = st.Submit(req); problem != nil {
		t.Fatal(problem)
	}
	st.Close()
	st, problem = records.Open(path)
	if problem != nil {
		t.Fatal(problem)
	}
	read, problem := st.RequestRow(req.ID)
	if problem != nil || read == nil || !read.RentNew {
		t.Fatalf("fresh intent lost: %+v %v", read, problem)
	}
	st.Close()
	// Reconstruct released schema47 from public SQLite metadata, omitting only
	// the newly introduced column. No production-only test export is needed.
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := db.Query(`SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END,name`)
	if err != nil {
		t.Fatal(err)
	}
	var statements []string
	for rows.Next() {
		var statement string
		if err := rows.Scan(&statement); err != nil {
			t.Fatal(err)
		}
		statements = append(statements, strings.Replace(statement, "  rent_new INTEGER NOT NULL DEFAULT 0 CHECK(rent_new IN (0,1)),\n", "", 1))
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	rows.Close()
	db.Close()
	prior := filepath.Join(t.TempDir(), "prior.sqlite")
	db, err = sql.Open("sqlite", prior)
	if err != nil {
		t.Fatal(err)
	}
	for _, statement := range statements {
		if _, err := db.Exec(statement); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.Exec(`INSERT INTO requests(id,idem_key,body_digest,package,entrypoint,plan_id,payload,state,created_at,rental,rental_required) VALUES('prior-request','prior-key','digest','proof/model','run','plan',x'7b7d','queued','now',1,1); PRAGMA user_version=47`); err != nil {
		t.Fatal(err)
	}
	db.Close()
	st, problem = records.OpenForDaemon(prior, "")
	if problem != nil {
		t.Fatal(problem)
	}
	defer st.Close()
	read, problem = st.RequestRow("prior-request")
	if problem != nil || read == nil || read.RentNew || !read.RentalRequired {
		t.Fatalf("old intent changed: %+v %v", read, problem)
	}
}
