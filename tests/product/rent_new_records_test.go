package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentNewSurvivesReopen(t *testing.T) {
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
}
