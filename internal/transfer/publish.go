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
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/tfs"
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
const publishParallelism = 16

type uploadOutcome struct {
	index     int
	report    hub.VerifyReport
	moved     int64
	multipart bool
	verified  bool
	source    string
	primary   bool
	err       *exit.Error
}

func (p *Publish) upload(ctx context.Context, missing []hub.Missing, res *Result, ms map[string]int64) *exit.Error {
	parallelism := publishParallelism
	if p.FailAfter > 0 {
		// The development kill point promises an exact completed-object count. Keep
		// that proof serial without slowing the production path.
		parallelism = 1
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	started := time.Now()
	jobs := make(chan int)
	outcomes := make(chan uploadOutcome, parallelism)
	var workers sync.WaitGroup
	var cancelOnce sync.Once
	for range parallelism {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for index := range jobs {
				outcome := p.uploadOne(ctx, res.PublishID, index, missing[index])
				if outcome.err != nil {
					cancelOnce.Do(func() {
						outcome.primary = true
						cancel()
					})
					outcomes <- outcome
					return
				}
				outcomes <- outcome
			}
		}()
	}
	go func() {
		defer close(outcomes)
		for index := range missing {
			select {
			case jobs <- index:
			case <-ctx.Done():
				close(jobs)
				workers.Wait()
				return
			}
		}
		close(jobs)
		workers.Wait()
	}()

	ordered := make([]*uploadOutcome, len(missing))
	for outcome := range outcomes {
		copy := outcome
		ordered[outcome.index] = &copy
	}
	var primary *exit.Error
	for _, outcome := range ordered {
		if outcome == nil {
			continue
		}
		if outcome.primary {
			primary = outcome.err
		}
	}
	if primary != nil {
		return primary
	}
	for index, outcome := range ordered {
		if outcome == nil {
			return exit.Internalf("upload stopped before object %d/%d produced a verdict", index, len(missing))
		}
	}

	sources := map[string]bool{}
	for _, outcome := range ordered {
		res.Grants++
		res.Uploaded++
		res.Moved += outcome.moved
		if outcome.multipart {
			res.Multipart++
		}
		if outcome.report.Conflict {
			res.Conflicts++
		}
		if outcome.verified {
			res.Verified++
		}
		if outcome.source != "" {
			sources[outcome.source] = true
		}
	}
	for source := range sources {
		res.Sources = append(res.Sources, source)
	}
	sort.Strings(res.Sources)
	ms["grant_upload_verify"] = since(started)
	p.say("granted and uploaded %d objects one at a time (%d ranged), %s moved "+
		"(%d already resident at the key)", res.Uploaded, res.Multipart, size(res.Moved), res.Conflicts)
	p.say("the hub re-hashed %d/%d objects as each upload completed (%s)",
		res.Verified, len(missing), strings.Join(res.Sources, ", "))
	return nil
}

func (p *Publish) uploadOne(ctx context.Context, publishID string, index int, object hub.Missing) uploadOutcome {
	outcome := uploadOutcome{index: index}
	if p.FailAfter > 0 && index >= p.FailAfter {
		outcome.err = exit.Named(exit.Internal, "crash_after_upload",
			"development kill point: stopped after %d objects", index).
			WithRemedy("re-run the same publish; the hub re-plans the declaration and the uploaded objects are already held")
		return outcome
	}
	grant, e := p.Hub.Grant(ctx, p.Ref, publishID, object.ID, p.Reason)
	if e != nil {
		outcome.err = e
		return outcome
	}
	outcome.multipart = grant.Multipart()
	staged := filepath.Join(p.Scratch, fmt.Sprintf("object-%06d", index))
	defer os.Remove(staged)

	// The bytes leave the store through a VERIFIED read, so a publisher cannot
	// upload what its own store silently corrupted.
	local := Result{PublishID: publishID}
	if e := p.Tool.Extract(grant.ObjectID, staged); e != nil {
		outcome.err = e
		return outcome
	}
	conflict, e := p.put(ctx, grant, staged, &local)
	if e != nil && e.Name == "grant.expired_replan" {
		fresh, e2 := p.Hub.Grant(ctx, p.Ref, publishID, grant.ObjectID, p.Reason)
		if e2 != nil {
			outcome.err = e2
			return outcome
		}
		conflict, e = p.put(ctx, fresh, staged, &local)
	}
	if e != nil {
		outcome.err = e
		return outcome
	}
	outcome.moved = local.Moved
	outcome.report = hub.VerifyReport{ObjectID: grant.ObjectID, Conflict: conflict}
	verdicts, verifyErr := p.Hub.VerifyObjects(
		ctx, p.Ref, publishID, []hub.VerifyReport{outcome.report}, p.Reason,
	)
	if verifyErr != nil {
		outcome.err = verifyErr
		return outcome
	}
	if len(verdicts) != 1 || verdicts[0].ObjectID != grant.ObjectID {
		outcome.err = exit.Internalf(
			"verification for %s answered %d rows or another object",
			grant.ObjectID,
			len(verdicts),
		)
		return outcome
	}
	outcome.verified = verdicts[0].State == "verified"
	outcome.source = verdicts[0].ChecksumSource
	if !outcome.verified {
		outcome.err = exit.Internalf(
			"verification for %s returned state %q: %s",
			grant.ObjectID,
			verdicts[0].State,
			verdicts[0].Detail,
		)
	}
	return outcome
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
func send(ctx context.Context, method, url string, open func() (io.ReadCloser, error), length int64, headers map[string]string) (int, http.Header, []byte, *exit.Error) {
	var status int
	var header http.Header
	var raw []byte
	exhausted, err := retryStorage(func(int) (bool, error) {
		status, header, raw = 0, nil, nil
		body, err := open()
		if err != nil {
			return false, exit.Internalf("the staged object is unreadable: %s", err)
		}
		// The body is COUNTED and the context lives while the count moves: an upload
		// that is slow is not an upload that has stopped.
		m := &mover{}
		rctx, cancel := m.context(ctx)
		counted := m.readCloser(body)
		req, err := http.NewRequestWithContext(rctx, method, url, counted)
		if err != nil {
			counted.Close()
			cancel()
			return false, exit.Internalf("the grant's URL is not usable: %s", err)
		}
		req.ContentLength = length
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			counted.Close()
			cancel()
			return true, err
		}
		raw, _ = io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		resp.Body.Close()
		counted.Close()
		cancel()
		status, header = resp.StatusCode, resp.Header
		if retryableStorageStatus(resp.StatusCode) {
			// The recorded answer stands if this was the last try: the caller renders
			// the storage layer's own refusal rather than a generic "unreachable".
			return true, fmt.Errorf("object storage answered %d", resp.StatusCode)
		}
		return false, nil
	})
	if err == nil || (exhausted && status != 0) {
		return status, header, raw, nil
	}
	if !exhausted {
		return 0, nil, nil, err.(*exit.Error)
	}
	return 0, nil, nil, exit.Unavailablef("object storage is unreachable after %d attempts: %s", attempts, err).
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
type limitedReadCloser struct {
	io.Reader
	io.Closer
}

func opener(path string, offset, length int64) func() (io.ReadCloser, error) {
	return func() (io.ReadCloser, error) {
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
		return &limitedReadCloser{Reader: io.LimitReader(f, length), Closer: f}, nil
	}
}

func retryableStorageStatus(status int) bool {
	switch status {
	case http.StatusRequestTimeout, http.StatusTooEarly, http.StatusTooManyRequests,
		http.StatusInternalServerError, http.StatusBadGateway, http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	default:
		return false
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
