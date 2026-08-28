package live

import (
	"database/sql"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/records"
)

// TestOutputRetention is the discipline on the only verb in this product that removes a
// user's bytes. Outputs are mirrored onto this host before a terminal is acked and are
// deliberately kept when the worker dies, so this root is the one plane that grows
// forever — and bounding it must never be the reason someone lost work. The producer is
// `cozy-fakeworker --arm output`: a real peer writing a real PNG where the grant said.
func TestOutputRetention(t *testing.T) {
	o := hostOwner(t, "retention")
	root := o.root

	// Two real outputs, over the committed protocol.
	spec := fakeSpec("retention", "7", "--arm", "output", "--cozy-home", root)
	instance, _, e := o.c.EnsureWorker(spec)
	fatal(t, e)
	planID := planIDOf(t, spec)
	fatal(t, o.c.EnsurePlacementReady(instance, planID))
	var mediaIDs, paths []string
	for i := 0; i < 2; i++ {
		requestID, attempt, e := o.c.Submit(submission(planID, "fake/retention",
			"retention-"+string(rune('a'+i)), map[string]any{"n": i}))
		fatal(t, e)
		if _, e := o.c.Await(requestID, attempt, 60*time.Second); e != nil {
			t.Fatalf("attempt %d did not close: %s", i, briefly(e))
		}
		outs, _ := o.store.VisibleOutputs(requestID)
		if len(outs) != 1 {
			t.Fatalf("attempt %d published %d output(s)", i, len(outs))
		}
		mediaIDs = append(mediaIDs, outs[0].MediaID)
		paths = append(paths, outs[0].Path)
	}
	o.c.ShutdownWorker(instance, 10*time.Second)

	// A job publication is a durable result plane, not media. Seed one exact committed
	// row and real file, then prove both planning and performing GC disclose but retain it.
	job := records.Request{
		ID: "job-retained-publication", IdemKey: "job-retained-publication",
		BodyDigest: "sha256:retained-publication", Kind: "job", Payload: []byte("{}"),
	}
	if _, fresh, problem := o.store.Submit(job); problem != nil || !fresh {
		t.Fatalf("seed publication request: fresh=%t problem=%v", fresh, problem)
	}
	publicationRoot := filepath.Join(o.l.Publications, "local", "_job-"+job.ID)
	must(t, os.MkdirAll(publicationRoot, 0o700))
	publicationPath := filepath.Join(publicationRoot, "result.bin")
	publicationBody := []byte("durable bytes")
	must(t, os.WriteFile(publicationPath, publicationBody, 0o600))
	db, err := sql.Open("sqlite", o.l.DB)
	must(t, err)
	_, err = db.Exec(`INSERT INTO publications(
		request_id,attempt,repo,root,status,cause,entries,bytes,committed_at)
		VALUES(?,?,?,?,?,?,?,?,?)`, job.ID, 1, "local/_job-"+job.ID, publicationRoot,
		"SUCCEEDED", "", 1, len(publicationBody), time.Now().UTC().Format(time.RFC3339Nano))
	must(t, err)
	must(t, db.Close())
	o.close() // the root belongs to the `cozy up` process from here on

	svc := startService(t, root)
	if r := svc.call(t, "GET", "/v1/media/"+mediaIDs[0], nil); r.Status != http.StatusOK ||
		len(r.Body) < 8 || string(r.Body[1:4]) != "PNG" {
		t.Fatalf("GET /v1/media/{id} did not serve the bytes with the worker long gone: %s", r.brief())
	}

	// THE HORIZON PROTECTS: a bare `cozy gc` plans zero media and says what it kept.
	code, out := runCozy(t, root, "gc")
	if code != 0 || !strings.Contains(out, "media: 0 output(s)") ||
		!strings.Contains(out, "retained: 2 output(s)") ||
		!strings.Contains(out, "publications: 1 publication(s), 13B retained indefinitely") {
		t.Errorf("the default horizon did not protect both outputs [exit %d]\n%s", code, out)
	}
	// THE PLAN/PERFORM SPLIT: a bare invocation removes nothing, however wide the horizon.
	code, out = runCozy(t, root, "gc", "--keep-media", "0")
	if code != 0 || !strings.Contains(out, "media: 2 output(s)") ||
		!strings.Contains(out, "this is the plan; nothing was removed") {
		t.Errorf("`cozy gc --keep-media 0` was not a plan [exit %d]\n%s", code, out)
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("a planned-only reclamation removed %s", p)
		}
	}
	if _, err := os.Stat(publicationPath); err != nil {
		t.Errorf("planned-only gc removed the durable job publication: %v", err)
	}
	// `--yes` performs: the bytes go, the ROWS stay.
	code, out = runCozy(t, root, "gc", "--keep-media", "0", "--yes")
	if code != 0 || !strings.Contains(out, "media: 2 output(s)") {
		t.Errorf("`cozy gc --keep-media 0 --yes` did not report what it freed [exit %d]\n%s", code, out)
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			t.Errorf("%s survived the reclamation", p)
		}
	}
	if _, err := os.Stat(publicationPath); err != nil {
		t.Errorf("gc removed the durable job publication: %v", err)
	}
	// A reclaimed id is a TOMBSTONE — 410, never 404. "we removed this" and "this never
	// existed" are different facts and a client must be able to tell them apart.
	gone := svc.call(t, "GET", "/v1/media/"+mediaIDs[0], nil)
	if gone.Status != http.StatusGone || gone.code() != "media_reclaimed" {
		t.Errorf("a reclaimed id: %s", gone.brief())
	}
	never := svc.call(t, "GET", "/v1/media/med-000000000000000000000000", nil)
	if never.Status != http.StatusNotFound || never.code() != "not_found" {
		t.Errorf("an id that never existed: %s", never.brief())
	}
	// And it converges: a second pass frees nothing and refuses nothing.
	code, out = runCozy(t, root, "gc", "--keep-media", "0", "--yes")
	if code != 0 || !strings.Contains(out, "media: 0 output(s)") {
		t.Errorf("re-running the reclamation did not converge [exit %d]\n%s", code, out)
	}
}
