package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"database/sql"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func privateAdmissionOwner(t *testing.T) (*orchestrator.PrivateExecutionOwner, string) {
	t.Helper()
	public, key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	_, issuer, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	grant := &pb.ExecutionOwnerGrant{RecordOwnerEpoch: 2, RecordOwnerId: "pod-coordinator", WorkerId: "worker", WorkerBootId: "boot", WorkerTlsCertificateDigest: bytes.Repeat([]byte{1}, 32), InitialCapsuleDigest: bytes.Repeat([]byte{2}, 32), ExecutionPublicKey: public}
	raw, err := canonical.Bytes(grant)
	if err != nil {
		t.Fatal(err)
	}
	digest, err := canonical.Spell(canonical.Digest(raw))
	if err != nil {
		t.Fatal(err)
	}
	return &orchestrator.PrivateExecutionOwner{Authorization: &pb.SignedExecutionOwnerGrant{Grant: grant, Signature: ed25519.Sign(issuer, raw)}, PrivateKey: key}, digest
}

func TestExecutionGrantAdmissionAndImmutableReplay(t *testing.T) {
	path := filepath.Join(t.TempDir(), "records.sqlite")
	store, problem := records.OpenForDaemon(path, "")
	if problem != nil {
		t.Fatal(problem)
	}
	owner, digest := privateAdmissionOwner(t)
	coord, problem := orchestrator.Open(orchestrator.Options{Store: store, PrivateExecution: owner})
	if problem != nil {
		t.Fatal(problem)
	}
	body, _ := canonical.Spell(canonical.Digest([]byte("validated complete capsule")))
	request := orchestrator.Submission{IdemKey: "private-root", RequestID: "job-00112233445566778899aabb", ExecutionGrantDigest: digest, BodyDigest: body, Package: "local/script", Entrypoint: "main", Kind: "job", Payload: []byte("{}"), AttentionKernel: "fa3"}
	for _, tc := range []struct {
		name string
		edit func(*orchestrator.Submission)
	}{
		{"wrong grant", func(s *orchestrator.Submission) { s.ExecutionGrantDigest = "sha256:" + strings.Repeat("f", 64) }},
		{"missing capsule", func(s *orchestrator.Submission) { s.BodyDigest = "" }},
		{"uppercase capsule", func(s *orchestrator.Submission) { s.BodyDigest = strings.ToUpper(body) }},
		{"missing ID", func(s *orchestrator.Submission) { s.RequestID = "" }},
		{"path ID", func(s *orchestrator.Submission) { s.RequestID = "../../other" }},
		{"uppercase ID", func(s *orchestrator.Submission) { s.RequestID = "job-00112233445566778899AABB" }},
		{"non-job root", func(s *orchestrator.Submission) { s.Kind = "serving" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := request
			tc.edit(&s)
			if _, _, p := coord.RecordSubmission(s); p == nil {
				t.Fatal("invalid private admission accepted")
			}
		})
	}
	// Caller mutation cannot change the already frozen authority.
	owner.Authorization.Grant.RecordOwnerEpoch = 99
	got, fresh, problem := coord.RecordSubmission(request)
	if problem != nil || !fresh || got.ID != request.RequestID || got.BodyDigest != body || got.ExecutionGrantDigest != digest {
		t.Fatalf("first admission: %+v fresh=%t error=%v", got, fresh, problem)
	}
	again, fresh, problem := coord.RecordSubmission(request)
	if problem != nil || fresh || again.ID != got.ID {
		t.Fatalf("idempotent replay: %+v fresh=%t error=%v", again, fresh, problem)
	}
	changed := request
	changed.RequestID = "job-ffeeddccbbaa998877665544"
	if _, _, p := coord.RecordSubmission(changed); p == nil {
		t.Fatal("same key accepted changed stable request ID")
	}
	changed = request
	changed.IdemKey = "other-key"
	if _, _, p := coord.RecordSubmission(changed); p == nil {
		t.Fatal("stable request ID admitted under another key")
	}
	coord.Close(0)
	store.Close()

	store, problem = records.Open(path)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	ordinary, problem := orchestrator.Open(orchestrator.Options{Store: store})
	if problem != nil {
		t.Fatal(problem)
	}
	defer ordinary.Close(0)
	if _, _, p := ordinary.RecordSubmission(request); p == nil {
		t.Fatal("laptop injected execution grant")
	}
	changed = request
	changed.ExecutionGrantDigest = ""
	if _, _, p := ordinary.RecordSubmission(changed); p == nil {
		t.Fatal("ordinary caller overrode request ID")
	}
	changed.RequestID = ""
	if _, _, p := ordinary.RecordSubmission(changed); p == nil {
		t.Fatal("ordinary replay erased execution scope")
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var id, storedBody, storedGrant string
	if err := db.QueryRow(`SELECT id,body_digest,execution_grant_digest FROM requests WHERE idem_key=?`, request.IdemKey).Scan(&id, &storedBody, &storedGrant); err != nil || id != request.RequestID || storedBody != body || storedGrant != digest {
		t.Fatalf("reopen lost immutable root: %q %q %q %v", id, storedBody, storedGrant, err)
	}
	var count int
	if err := db.QueryRow(`SELECT count(*) FROM requests`).Scan(&count); err != nil || count != 1 {
		t.Fatalf("refusals inserted requests: count=%d %v", count, err)
	}
}

func TestExecutionGrantSchema39MigrationPreservesUnscopedHistory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "records.sqlite")
	prior, err := os.ReadFile("testdata/records/schema-39.sql")
	if err != nil {
		t.Fatal(err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = db.Exec(string(prior)); err != nil {
		t.Fatal(err)
	}
	body := "sha256:" + strings.Repeat("a", 64)
	for _, r := range []struct {
		id, parent string
		index      int
	}{{"legacy-root", "", -1}, {"legacy-child", "legacy-root", 0}} {
		if _, err = db.Exec(`INSERT INTO requests(id,idem_key,body_digest,package,entrypoint,plan_id,payload,state,created_at,kind,parent_request_id,parent_call_index) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)`, r.id, r.id, body, "local/old", "main", "plan", []byte("{}"), "succeeded", "2026-09-12T00:00:00Z", "job", r.parent, r.index); err != nil {
			t.Fatal(err)
		}
	}
	db.Close()
	if opened, p := records.Open(path); p == nil {
		opened.Close()
		t.Fatal("ordinary reader migrated schema39 without daemon authority")
	}
	store, problem := records.OpenForDaemon(path, "")
	if problem != nil {
		t.Fatal(problem)
	}
	grant := "sha256:" + strings.Repeat("b", 64)
	if _, _, p := store.Submit(records.Request{ID: "tagged-child", IdemKey: "tagged-child", BodyDigest: body, Package: "local/new", Entrypoint: "main", Kind: "job", Payload: []byte("{}"), ParentRequestID: "legacy-root", ExecutionGrantDigest: grant}); p == nil {
		t.Fatal("child created another execution root")
	}
	store.Close()
	reopened, problem := records.Open(path)
	if problem != nil {
		t.Fatal(problem)
	}
	reopened.Close()
	db, err = sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	rows, err := db.Query(`SELECT id,body_digest,execution_grant_digest,parent_request_id FROM requests ORDER BY id`)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for rows.Next() {
		var id, actualBody, scope, parent string
		if err := rows.Scan(&id, &actualBody, &scope, &parent); err != nil {
			t.Fatal(err)
		}
		if actualBody != body || scope != "" || (id == "legacy-child" && parent != "legacy-root") {
			t.Fatalf("migration rewrote history: %q %q %q %q", id, actualBody, scope, parent)
		}
		count++
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	rows.Close()
	if count != 2 {
		t.Fatalf("history count=%d", count)
	}
	if _, err := db.Exec(`UPDATE requests SET execution_grant_digest=? WHERE id='legacy-child'`, grant); err == nil {
		t.Fatal("SQLite admitted a tagged child")
	}
}
