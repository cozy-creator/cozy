package records

import (
	"database/sql"
	"path/filepath"
	"strings"
	"testing"
)

func TestRentNewSurvivesRecordsAndSchemaMigration(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	st, problem := Open(path)
	if problem != nil {
		t.Fatal(problem)
	}
	req := Request{ID: "fresh-request", IdemKey: "fresh-key", BodyDigest: "sha256:" + strings.Repeat("a", 64), Package: "proof/model", Entrypoint: "run", Payload: []byte(`{}`), Rental: true, RentalRequired: true, RentNew: true}
	if _, _, problem = st.Submit(req); problem != nil {
		t.Fatal(problem)
	}
	st.Close()
	st, problem = Open(path)
	if problem != nil {
		t.Fatal(problem)
	}
	read, problem := st.RequestRow(req.ID)
	if problem != nil || read == nil || !read.RentNew {
		t.Fatalf("fresh intent lost: %+v %v", read, problem)
	}
	st.Close()

	// Recreate the exact released v47 schema and preserve its older rental request.
	path = filepath.Join(t.TempDir(), "prior.sqlite")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	for _, statement := range priorStatements(47) {
		if _, err := db.Exec(statement); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.Exec(`INSERT INTO requests(id,idem_key,body_digest,package,entrypoint,plan_id,payload,state,created_at,rental,rental_required) VALUES('fresh-request','prior-key','digest','proof/model','run','plan',x'7b7d','queued','now',1,1); PRAGMA user_version=47`); err != nil {
		t.Fatal(err)
	}
	db.Close()
	st, problem = OpenForDaemon(path, "")
	if problem != nil {
		t.Fatal(problem)
	}
	defer st.Close()
	read, problem = st.RequestRow(req.ID)
	if problem != nil || read == nil || read.RentNew || !read.RentalRequired {
		t.Fatalf("prior rental intent changed: %+v %v", read, problem)
	}
}
