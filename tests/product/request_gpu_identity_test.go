package producttest

import (
	"database/sql"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRequestedGPUsPersistAndDefineSubmissionIdentity(t *testing.T) {
	o := hostOwner(t, "gpu-request-identity")
	for _, kind := range []string{"serving", "job"} {
		sub := orchestrator.Submission{Kind: kind, Package: "proof/model", Entrypoint: "run",
			Payload: []byte("{}"), IdemKey: kind, RequestedGPUs: 2}
		first, fresh, problem := o.c.RecordSubmission(sub)
		fatal(t, problem)
		if !fresh || first.RequestedGPUs != 2 {
			t.Fatalf("GPU count not recorded: %+v", first)
		}
		replay, fresh, problem := o.c.RecordSubmission(sub)
		fatal(t, problem)
		if fresh || replay.ID != first.ID || replay.RequestedGPUs != 2 {
			t.Fatal("same GPU width did not replay")
		}
		sub.RequestedGPUs = 4
		if _, _, problem := o.c.RecordSubmission(sub); problem == nil {
			t.Fatal("idempotency key changed GPU count")
		}
		reader, problem := records.Open(o.l.DB)
		fatal(t, problem)
		stored, problem := reader.RequestRow(first.ID)
		fatal(t, problem)
		reader.Close()
		if stored.RequestedGPUs != 2 {
			t.Fatal("separate records reader lost requested GPU count")
		}
	}
}

func TestRequestedGPUsFlowToChildAndRetainedRetry(t *testing.T) {
	st, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer st.Close()
	parent, _, problem := st.Submit(records.Request{ID: "parent", IdemKey: "parent", BodyDigest: childDigest("a"),
		Package: "local/script", Entrypoint: "main", Kind: "job", Payload: []byte("{}"), RetainWork: true, RequestedGPUs: 2})
	fatal(t, problem)
	fatal(t, st.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/script", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent = offerChildParent(t, st, parent)
	call := records.Request{ID: "child", IdemKey: "child", BodyDigest: childDigest("b"), Package: "local/op", Entrypoint: "run",
		Kind: "job", Payload: []byte("{}"), ParentRequestID: parent.ID, ParentCallIndex: 0,
		ChildIntentDigest: childDigest("c"), ChildTargetDigest: childDigest("d")}
	child, _, problem := st.SubmitChild(call, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	if child.RequestedGPUs != 2 {
		t.Fatal("child lost parent GPU count")
	}
	_, problem = st.BlockRetainedWork(child.ID, "proof", "pause child")
	fatal(t, problem)
	retry := records.Request{ID: "retry", IdemKey: "retry", BodyDigest: childDigest("e"), Package: "local/op", Entrypoint: "run",
		Kind: "job", Payload: []byte("{}"), RetainWork: true, RetryOf: child.ID}
	retried, _, problem := st.Submit(retry)
	fatal(t, problem)
	if retried.RequestedGPUs != 2 {
		t.Fatal("retry lost predecessor GPU count")
	}
	retry.ID, retry.IdemKey, retry.RequestedGPUs = "wide-retry", "wide-retry", 4
	if _, _, problem := st.Submit(retry); problem == nil {
		t.Fatal("retry widened its requested GPU count")
	}
	call.ID, call.IdemKey, call.ParentCallIndex, call.RequestedGPUs = "wide-child", "wide-child", 1, 4
	if _, _, problem := st.SubmitChild(call, 1, childDigest("1"), "private-boot", nil); problem == nil {
		t.Fatal("child widened its parent GPU count")
	}
}

func TestSchema38MigrationPreservesRequestsAndReservation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	st, problem := records.Open(path)
	fatal(t, problem)
	_, _, problem = st.Submit(records.Request{ID: "existing", IdemKey: "existing", BodyDigest: childDigest("a"),
		Package: "proof/model", Entrypoint: "run", Payload: []byte("{}"), RequestedRental: "wanted", Rental: true})
	fatal(t, problem)
	st.Close()
	db, err := sql.Open("sqlite", path)
	must(t, err)
	var create string
	must(t, db.QueryRow(`SELECT sql FROM sqlite_master WHERE name='requests'`).Scan(&create))
	create = strings.Replace(create, ",\n  requested_gpus INTEGER NOT NULL DEFAULT 0 CHECK(requested_gpus>=0 AND requested_gpus<=4294967295)", "", 1)
	_, err = db.Exec(`ALTER TABLE requests DROP COLUMN requested_gpus; PRAGMA user_version=38`)
	must(t, err)
	_, err = db.Exec(`PRAGMA writable_schema=ON; UPDATE sqlite_master SET sql=? WHERE name='requests'; PRAGMA writable_schema=OFF`, create)
	must(t, err)
	db.Close()
	st, problem = records.OpenForDaemon(path, "")
	fatal(t, problem)
	defer st.Close()
	r, problem := st.RequestRow("existing")
	fatal(t, problem)
	if r.RequestedGPUs != 0 || r.RequestedRental != "wanted" {
		t.Fatalf("migration changed request intent: %+v", r)
	}
}
