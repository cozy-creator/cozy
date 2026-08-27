// Package transfer is the order of operations for moving a canonical checkpoint
// between the local store and the hub (cl-012). It owns no bytes and no protocol:
// TensorFS owns the byte plane (every question crosses internal/tfs) and tensorhub
// owns the declare-first protocol (every call crosses internal/hub). What lives here
// is the SEQUENCE, and the two properties the sequence must have:
//
//   - resumability. There is no client-side journal. A publish resumes because the
//     hub re-plans a re-declared closure server-side, and a fetch resumes because
//     the local store's verification records already say which objects are good.
//     A transfer that kept its own progress file would have a third opinion about
//     what is done, and the third opinion is always the wrong one.
//   - honest accounting. Every progress line separates bytes MOVED from bytes
//     DEDUPED. A publish that sends nothing and prints the artifact's size is
//     claiming credit for work it did not do.
package transfer

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/hub"
	"github.com/cozy-creator/cozy-creator-v2/internal/tfs"
)

// One object write or read at the storage edge is bounded by BYTES MOVING, not by a
// clock — see `mover`. The 30-minute constant that used to live here said in its own
// comment that it was standing in for a byte meter.

// Publish is one declare-first upload, start to finish.
type Publish struct {
	Tool *tfs.Tool
	Hub  *hub.Client
	Ref  hub.Ref
	// Snapshot is the local canonical snapshot being published: `sha256:<64 hex>`.
	Snapshot string
	Session  string
	Reason   string
	DryRun   bool
	// Progress receives one line per phase. It is where the honest accounting is
	// printed as the work happens; the returned Result carries the same numbers.
	Progress func(string)
	// Scratch is where objects are staged out of the store on their way to the wire.
	Scratch string
	// FailAfter is a development kill point (`--crash-after upload:<n>`) used to
	// prove that an interrupted publish converges on a re-run. It is not a product
	// behaviour and every arm that uses it names itself.
	FailAfter int
}

// Result is what a publish did.
type Result struct {
	PublishID string
	Created   bool
	Session   string
	Totals    hub.Totals
	Grants    int
	Multipart int
	Uploaded  int
	Conflicts int
	Verified  int
	// Moved is bytes this invocation actually put on the wire; Deduped is what the
	// hub already held. Their sum is the artifact.
	Moved    int64
	Deduped  int64
	Sources  []string
	Root     hub.Root
	Grade    string
	Verdict  string
	Dup      bool
	MS       map[string]int64
	Reingest int
}

func (p *Publish) say(format string, args ...any) {
	if p.Progress != nil {
		p.Progress(fmt.Sprintf(format, args...))
	}
}

// Run declares, uploads only what the hub lacks, has the hub prove every byte, and
// completes. Every step is the hub's; the only thing this decides is the order.
func (p *Publish) Run(ctx context.Context) (Result, *exit.Error) {
	var res Result
	ms := map[string]int64{}
	res.MS = ms

	// 1. The declaration. Everything in it is a document tfs produced, carried
	//    verbatim: the hub reproduces the closure at completion and compares digests,
	//    so a re-encoded document is a refusal waiting to happen.
	t0 := time.Now()
	if err := os.MkdirAll(p.Scratch, 0o700); err != nil {
		return res, exit.Internalf("cannot create the transfer scratch at %s: %s", p.Scratch, err)
	}
	closure, objects, e := p.Tool.Closure(p.Snapshot, p.Session, filepath.Join(p.Scratch, "closure.json"))
	if e != nil {
		return res, e
	}
	header, e := p.Tool.Header(p.Snapshot)
	if e != nil {
		return res, e
	}
	topology, e := p.Tool.Topology(header, filepath.Join(p.Scratch, "topology.json"))
	if e != nil {
		return res, e
	}
	manifestPath := filepath.Join(p.Scratch, "manifest.bin")
	if e := p.Tool.Extract(p.Snapshot, manifestPath); e != nil {
		return res, e
	}
	manifest, err := os.ReadFile(manifestPath)
	if err != nil {
		return res, exit.Internalf("the manifest tfs extracted is unreadable: %s", err)
	}
	decl := hub.BeginRequest{
		Session:      p.Session,
		Closure:      hub.B64(closure),
		CodeTopology: hub.B64(topology),
		SnapshotID:   p.Snapshot,
		Manifest:     hub.B64(manifest),
		HeaderID:     "sha256:" + header,
	}
	for _, o := range objects {
		decl.Objects = append(decl.Objects, hub.Object{ID: o.ID, Length: o.Length})
	}
	ms["declare"] = since(t0)
	p.say("declared %d objects (%s) from %s", len(decl.Objects), bytesOf(objects), p.Snapshot)

	// 2. Begin. Idempotent on the closure digest: this IS the resume.
	t0 = time.Now()
	begun, e := p.Hub.Begin(ctx, p.Ref, decl, p.Reason)
	if e != nil {
		return res, e
	}
	ms["begin"] = since(t0)
	res.PublishID, res.Created, res.Session = begun.Publish.ID, begun.Created, begun.Publish.Session
	res.Totals = begun.Totals
	res.Deduped = begun.Totals.HeldBytes()
	verb := "resumed"
	if begun.Created {
		verb = "opened"
	}
	p.say("%s publish %s: %d missing (%s to send) · %d held (%s deduped)",
		verb, res.PublishID, len(begun.Missing), size(res.Totals.MissingBytes),
		len(begun.Held), size(res.Deduped))

	if p.DryRun {
		p.say("--dry-run: no grant requested, no byte sent; the session stays open and re-declares to the same plan")
		return res, nil
	}

	// 3-4. Grants and the writes. Nothing is uploaded that the hub did not name.
	if len(begun.Missing) > 0 {
		if e := p.upload(ctx, begun.Missing, &res, ms); e != nil {
			return res, e
		}
	} else {
		p.say("0 bytes to move: the hub already holds every declared object")
	}

	// 5. Completion. The hub re-proves everything — including what it already held.
	t0 = time.Now()
	done, e := p.Hub.Complete(ctx, p.Ref, res.PublishID, p.Reason)
	if e != nil {
		return res, e
	}
	ms["complete"] = since(t0)
	res.Root, res.Dup, res.Reingest = done.Root, done.Duplicate, done.ReingestedHeldObjects
	res.Grade, res.Verdict = done.Verifier.Grade, done.Verifier.Satisfaction
	if res.Grade == "" {
		res.Grade = done.Root.Grade
	}
	for k, v := range done.MS {
		ms["hub_"+k] = v
	}
	if res.Dup {
		p.say("this exact publish already completed: the hub replayed its ledger, same root")
	}
	// The bytes are the hub's now. Note the durability WITHOUT pinning: publishing
	// does not promise the local store keeps a copy.
	if e := p.Tool.Note(p.Snapshot); e != nil {
		return res, e
	}
	p.say("timing: %s", Timing(ms))
	return res, nil
}

// upload requests grants, writes each object at its final content key under every
// condition the grant signed, and then asks the hub to prove them. An expired grant
// comes back as a REPLAN and is answered by asking again — never by failing.
func (p *Publish) upload(ctx context.Context, missing []hub.Missing, res *Result, ms map[string]int64) *exit.Error {
	reports := make([]hub.VerifyReport, 0, len(missing))
	staged := filepath.Join(p.Scratch, "object")
	var grantMS, uploadMS int64
	for i, object := range missing {
		if p.FailAfter > 0 && i >= p.FailAfter {
			return exit.Named(exit.Internal, "crash_after_upload",
				"development kill point: stopped after %d of %d objects", i, len(missing)).
				WithRemedy("re-run the same publish; the hub re-plans the declaration and the uploaded objects are already held")
		}
		t0 := time.Now()
		g, e := p.Hub.Grant(ctx, p.Ref, res.PublishID, object.ID, p.Reason)
		grantMS += since(t0)
		if e != nil {
			return e
		}
		res.Grants++
		if g.Multipart() {
			res.Multipart++
		}
		// The bytes leave the store through a VERIFIED read, so a publisher cannot
		// upload what its own store silently corrupted.
		t0 = time.Now()
		if e := p.Tool.Extract(g.ObjectID, staged); e != nil {
			return e
		}
		conflict, e := p.put(ctx, g, staged, res)
		uploadMS += since(t0)
		if e != nil {
			// A grant whose window closed is a re-plan, not a failure. Ask again
			// once for this same object and continue; a second expiry is a clock
			// problem, not a race.
			if e.Name != "grant.expired_replan" {
				return e
			}
			p.say("a grant window closed mid-upload: re-planning (already-verified objects are the journal)")
			t0 = time.Now()
			fresh, e2 := p.Hub.Grant(ctx, p.Ref, res.PublishID, g.ObjectID, p.Reason)
			grantMS += since(t0)
			if e2 != nil {
				return e2
			}
			t0 = time.Now()
			conflict, e = p.put(ctx, fresh, staged, res)
			uploadMS += since(t0)
			if e != nil {
				return e
			}
		}
		res.Uploaded++
		if conflict {
			res.Conflicts++
		}
		reports = append(reports, hub.VerifyReport{ObjectID: g.ObjectID, Conflict: conflict})
	}
	_ = os.Remove(staged)
	ms["grants"] = grantMS
	ms["upload"] = uploadMS
	p.say("granted and uploaded %d objects one at a time (%d ranged), %s moved "+
		"(%d already resident at the key)", res.Uploaded, res.Multipart, size(res.Moved), res.Conflicts)

	// The hub streams every object BACK and hashes it itself. Nothing this client
	// observed is evidence, which is why the report above carries no checksum.
	t0 := time.Now()
	verdicts, e := p.Hub.VerifyObjects(ctx, p.Ref, res.PublishID, reports, p.Reason)
	if e != nil {
		return e
	}
	ms["verify"] = since(t0)
	sources := map[string]bool{}
	for _, v := range verdicts {
		if v.State == "verified" {
			res.Verified++
		}
		if v.ChecksumSource != "" {
			sources[v.ChecksumSource] = true
		}
	}
	for s := range sources {
		res.Sources = append(res.Sources, s)
	}
	sortStrings(res.Sources)
	p.say("the hub re-hashed %d/%d objects for itself (%s)",
		res.Verified, len(verdicts), strings.Join(res.Sources, ", "))
	return nil
}

// put writes one object at its final content key. Every required header is sent
// verbatim: they are signed into the URL, so stripping one is 403 and a wrong body
// is 400 — neither is a branch a client may take.
func (p *Publish) put(ctx context.Context, g hub.Grant, path string, res *Result) (bool, *exit.Error) {
	if g.Multipart() {
		return p.putRanged(ctx, g, path, res)
	}
	status, _, body, e := send(ctx, http.MethodPut, g.URL, opener(path, 0, g.Length), g.Length, g.Headers)
	if e != nil {
		return false, e
	}
	switch status {
	case http.StatusOK:
		res.Moved += g.Length
		return false, nil
	case http.StatusPreconditionFailed:
		// Someone already holds this exact key. That is not a failure — the key is
		// the digest, so whoever wrote it wrote these bytes. The hub still proves it.
		return true, nil
	default:
		return false, storageRefusal(status, g.ObjectID, body)
	}
}

func (p *Publish) putRanged(ctx context.Context, g hub.Grant, path string, res *Result) (bool, *exit.Error) {
	etags := make([]string, 0, len(g.Parts))
	for _, part := range g.Parts {
		status, headers, body, e := send(ctx, http.MethodPut, part.URL,
			opener(path, part.Offset, part.Length), part.Length, nil)
		if e != nil {
			return false, e
		}
		if status != http.StatusOK {
			return false, storageRefusal(status, fmt.Sprintf("%s part %d", g.ObjectID, part.Number), body)
		}
		etags = append(etags, strings.Trim(headers.Get("ETag"), `"`))
		res.Moved += part.Length
	}
	// The hub assembles: that call carries the no-clobber precondition, and a client
	// that assembled its own could replace verified bytes.
	return p.Hub.FinishMultipart(ctx, p.Ref, res.PublishID, g.ObjectID, etags, p.Reason)
}

// attempts bounds transport retries at the storage edge. A hiccup on a residential
// uplink is not a protocol event: every condition is signed INTO the url, so
// re-sending the same body is the same write, and the no-clobber precondition means
// a partially-received first attempt cannot become half an object. Three tries, then
// the run stops and says so — and a stopped run resumes, because the objects that did
// land are already the hub's.
const attempts = 3

// send is the one storage-edge write in this binary. It talks to object storage,
// never to the hub: the URL is presigned and carries no credential of ours.
func send(ctx context.Context, method, url string, open func() (io.Reader, error), length int64, headers map[string]string) (int, http.Header, []byte, *exit.Error) {
	var last error
	for try := 0; try < attempts; try++ {
		body, err := open()
		if err != nil {
			return 0, nil, nil, exit.Internalf("the staged object is unreadable: %s", err)
		}
		// The body is COUNTED and the context lives while the count moves: an upload
		// that is slow is not an upload that has stopped.
		m := &mover{}
		rctx, cancel := m.context(ctx)
		req, err := http.NewRequestWithContext(rctx, method, url, m.reader(body))
		if err != nil {
			cancel()
			return 0, nil, nil, exit.Internalf("the grant's URL is not usable: %s", err)
		}
		req.ContentLength = length
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			cancel()
			last = err
			time.Sleep(time.Duration(try+1) * time.Second)
			continue
		}
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		resp.Body.Close()
		cancel()
		return resp.StatusCode, resp.Header, raw, nil
	}
	return 0, nil, nil, exit.Unavailablef("object storage is unreachable after %d attempts: %s", attempts, last).
		WithRemedy("re-run to resume: the objects that did land are already held, and the hub re-plans the rest")
}

// storageRefusal renders what the storage layer said. It is not our vocabulary and
// is not translated: BadDigest and SignatureDoesNotMatch mean exactly what they say.
func storageRefusal(status int, subject string, body []byte) *exit.Error {
	said := strings.TrimSpace(string(body))
	if len(said) > 300 {
		said = said[:300] + "…"
	}
	code := exit.Failed
	if status == http.StatusForbidden || status == http.StatusUnauthorized {
		code = exit.Credential
	}
	return exit.Named(code, "storage.refused",
		"object storage answered %d writing %s: %s", status, subject, said).
		WithRemedy("the conditions are signed into the grant; a stripped condition is 403 and a wrong body is 400")
}

// opener hands `send` a FRESH reader per attempt. A retried upload that re-used a
// consumed reader would put an empty body under a signed digest — the exact failure
// retries exist to avoid, dressed up as a corrupt object.
func opener(path string, offset, length int64) func() (io.Reader, error) {
	return func() (io.Reader, error) {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		if offset > 0 {
			if _, err := f.Seek(offset, io.SeekStart); err != nil {
				f.Close()
				return nil, err
			}
		}
		return io.LimitReader(f, length), nil
	}
}

func bytesOf(objs []tfs.Object) string {
	var n int64
	for _, o := range objs {
		n += o.Length
	}
	return size(n)
}

func since(t time.Time) int64 { return time.Since(t).Milliseconds() }
