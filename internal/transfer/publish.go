// Package transfer is the order of operations for moving a canonical checkpoint
// between the local store and the hub (cl-012). It owns no bytes and no protocol:
// TensorFS owns the byte plane (every question crosses internal/tfs) and tensorhub
// owns the incremental publication protocol (every call crosses internal/hub). What lives here
// is the SEQUENCE, and the two properties the sequence must have:
//
//   - resumability. There is no client-side journal. A publication resumes from the
//     hub's durable transfer rows, and a fetch resumes because
//     the local store's verification records already say which objects are good.
//     A transfer that kept its own progress file would have a third opinion about
//     what is done, and the third opinion is always the wrong one.
//   - honest accounting. Every progress line separates bytes MOVED from bytes
//     DEDUPED. A publish that sends nothing and prints the artifact's size is
//     claiming credit for work it did not do.
package transfer

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
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

// Publish is one incremental model publication, start to finish.
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

// Run opens an incremental publication, claims the exact known-object transfers,
// uploads only what is not already accepted, verifies every settled transfer, and
// seals the TensorFS-authored documents. Tensorhub's transfer rows are the only
// restart journal.
func (p *Publish) Run(ctx context.Context) (Result, *exit.Error) {
	var res Result
	ms := map[string]int64{}
	res.MS = ms

	// 1. Prepare the exact documents and known ObjectRefs. The seal carries the
	//    original bytes; the transfer claim carries only sorted digest/length facts.
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
	seal := hub.SealPublicationRequest{
		Closure:      hub.B64(closure),
		CodeTopology: hub.B64(topology),
		SnapshotID:   p.Snapshot,
		Manifest:     hub.B64(manifest),
		HeaderID:     "sha256:" + header,
		Stamps:       []map[string]any{},
	}
	declared := make([]hub.Object, 0, len(objects))
	for _, o := range objects {
		declared = append(declared, hub.Object{ID: o.ID, Length: o.Length})
	}
	sort.Slice(declared, func(i, j int) bool { return declared[i].ID < declared[j].ID })
	ms["declare"] = since(t0)
	p.say("prepared %d known object transfers (%s) from %s", len(declared), bytesOf(objects), p.Snapshot)

	// 2. Open under the caller-stable operation id. Reopening returns the same durable
	//    publication, including a committed one whose seal result can be replayed.
	t0 = time.Now()
	opened, e := p.Hub.OpenPublication(ctx, p.Ref, p.Session, p.Reason)
	if e != nil {
		return res, e
	}
	ms["open"] = since(t0)
	publication := opened.Publication
	res.PublishID, res.Created, res.Session = publication.ID, opened.Created, publication.Session
	verb := "resumed"
	if opened.Created {
		verb = "opened"
	}
	p.say("%s publication %s in state %s", verb, res.PublishID, publication.State)
	if publication.State != "open" {
		res.Totals.DeclaredObjects = publication.DeclaredObjects
		res.Totals.DeclaredBytes = publication.DeclaredBytes
		res.Totals.HeldObjects = publication.DeclaredObjects
		res.Deduped = publication.DeclaredBytes
		return p.seal(ctx, seal, res, ms)
	}

	// 3. Claim the exact sorted transfer set. Identical claims replay rows with their
	//    current states; changed lengths require another operation id.
	t0 = time.Now()
	transfers, e := p.Hub.ClaimKnownTransfers(ctx, p.Ref, res.PublishID, declared, p.Reason)
	if e != nil {
		return res, e
	}
	ms["claim"] = since(t0)
	if e := exactTransfers(declared, transfers); e != nil {
		return res, e
	}

	toUpload := make([]hub.Transfer, 0, len(transfers))
	toVerify := make([]string, 0, len(transfers))
	for _, transfer := range transfers {
		res.Totals.DeclaredObjects++
		res.Totals.DeclaredBytes += transfer.Length
		switch transfer.State {
		case "accepted":
			res.Totals.HeldObjects++
			res.Deduped += transfer.Length
		case "verifying":
			res.Totals.HeldObjects++
			res.Deduped += transfer.Length
			toVerify = append(toVerify, transfer.TransferID)
		case "claimed", "transferring":
			res.Totals.MissingObjects++
			res.Totals.MissingBytes += transfer.Length
			toUpload = append(toUpload, transfer)
		default:
			return res, exit.New(exit.Conflict,
				"publication transfer %s for %s is %s", transfer.TransferID, transfer.ObjectID, transfer.State).
				WithRemedy("repair the refused immutable snapshot publication before replaying it")
		}
	}
	p.say("publication %s: %d transfers need upload (%s) · %d already resident (%s)",
		res.PublishID, len(toUpload), size(res.Totals.MissingBytes),
		res.Totals.HeldObjects, size(res.Deduped))

	if p.DryRun {
		p.say("--dry-run: transfers claimed, no grant requested and no byte sent")
		return res, nil
	}

	// 4. Resume transfers whose bytes reached storage before the earlier process died.
	if len(toVerify) > 0 {
		t0 = time.Now()
		verdicts, e := p.Hub.VerifyKnownTransfers(ctx, p.Ref, res.PublishID, toVerify, p.Reason)
		if e != nil {
			return res, e
		}
		if e := acceptedVerdicts(verdicts, len(toVerify)); e != nil {
			return res, e
		}
		res.Verified += len(verdicts)
		for _, verdict := range verdicts {
			res.Sources = appendUnique(res.Sources, verdict.ChecksumSource)
		}
		ms["resume_verify"] = since(t0)
	}

	// 5. Grants and writes. Nothing uploads without an exact transfer grant.
	if len(toUpload) > 0 {
		if e := p.upload(ctx, toUpload, &res, ms); e != nil {
			return res, e
		}
	} else {
		p.say("0 bytes to upload: every transfer is already accepted or ready to verify")
	}

	return p.seal(ctx, seal, res, ms)
}

func (p *Publish) seal(ctx context.Context, request hub.SealPublicationRequest,
	res Result, ms map[string]int64,
) (Result, *exit.Error) {
	t0 := time.Now()
	done, e := p.Hub.SealPublication(ctx, p.Ref, res.PublishID, request, p.Reason)
	if e != nil {
		return res, e
	}
	ms["seal"] = since(t0)
	res.Root, res.Dup, res.Reingest = done.Root, done.Duplicate, done.ReingestedHeldObjects
	res.Grade, res.Verdict = done.Verifier.Grade, done.Verifier.Satisfaction
	if res.Grade == "" {
		res.Grade = done.Root.Grade
	}
	for k, v := range done.MS {
		ms["hub_"+k] = v
	}
	if res.Dup {
		p.say("this exact publication already sealed: Tensorhub replayed the same committed root")
	}
	// The bytes are the hub's now. Note the durability WITHOUT pinning: publishing
	// does not promise the local store keeps a copy.
	if e := p.Tool.Note(p.Snapshot); e != nil {
		return res, e
	}
	p.say("timing: %s", Timing(ms))
	return res, nil
}

func exactTransfers(declared []hub.Object, transfers []hub.Transfer) *exit.Error {
	if len(declared) != len(transfers) {
		return exit.Internalf("claim of %d objects answered %d transfers", len(declared), len(transfers))
	}
	want := make(map[string]int64, len(declared))
	for _, object := range declared {
		want[object.ID] = object.Length
	}
	seen := map[string]bool{}
	for _, transfer := range transfers {
		length, ok := want[transfer.ObjectID]
		if !ok || seen[transfer.ObjectID] || transfer.TransferID == "" ||
			transfer.IntakeMode != "known" || length != transfer.Length {
			return exit.Internalf("claim answered a missing, duplicate, or changed transfer for %s", transfer.ObjectID)
		}
		seen[transfer.ObjectID] = true
	}
	return nil
}

func acceptedVerdicts(verdicts []hub.Verdict, want int) *exit.Error {
	if len(verdicts) != want {
		return exit.Internalf("verification of %d transfers answered %d verdicts", want, len(verdicts))
	}
	for _, verdict := range verdicts {
		if verdict.ObjectID == "" || verdict.State != "accepted" {
			return exit.Internalf("verification for %s returned state %q: %s",
				verdict.ObjectID, verdict.State, verdict.Detail)
		}
	}
	return nil
}

func appendUnique(values []string, value string) []string {
	if value == "" {
		return values
	}
	for _, prior := range values {
		if prior == value {
			return values
		}
	}
	return append(values, value)
}

// upload requests grants, writes each object at its final content key under every
// condition the grant signed, and then asks the hub to prove them. An expired grant
// comes back as a REPLAN and is answered by asking again — never by failing.
const publishParallelism = 16

type uploadOutcome struct {
	index     int
	moved     int64
	deduped   int64
	multipart bool
	uploaded  bool
	conflict  bool
	verified  bool
	source    string
	primary   bool
	err       *exit.Error
}

func (p *Publish) upload(ctx context.Context, transfers []hub.Transfer, res *Result, ms map[string]int64) *exit.Error {
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
				outcome := p.uploadOne(ctx, res.PublishID, index, transfers[index])
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
		for index := range transfers {
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

	ordered := make([]*uploadOutcome, len(transfers))
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
			return exit.Internalf("upload stopped before transfer %d/%d produced a verdict", index, len(transfers))
		}
	}

	sources := map[string]bool{}
	for _, outcome := range ordered {
		res.Grants++
		res.Moved += outcome.moved
		res.Deduped += outcome.deduped
		if outcome.uploaded {
			res.Uploaded++
		}
		if outcome.multipart {
			res.Multipart++
		}
		if outcome.conflict {
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
		res.Sources = appendUnique(res.Sources, source)
	}
	sort.Strings(res.Sources)
	ms["grant_upload_verify"] = since(started)
	p.say("requested %d transfer grants and uploaded %d objects (%d ranged): %s moved · %s deduped",
		len(transfers), res.Uploaded, res.Multipart, size(res.Moved), size(res.Deduped))
	p.say("Tensorhub accepted %d transfer verifications (%s)",
		res.Verified, strings.Join(res.Sources, ", "))
	return nil
}

func (p *Publish) uploadOne(ctx context.Context, publicationID string, index int,
	transfer hub.Transfer,
) uploadOutcome {
	outcome := uploadOutcome{index: index}
	if p.FailAfter > 0 && index >= p.FailAfter {
		outcome.err = exit.Named(exit.Internal, "crash_after_upload",
			"development kill point: stopped after %d objects", index).
			WithRemedy("re-run the same model publication; Tensorhub's transfer rows are the restart journal")
		return outcome
	}
	answer, e := p.Hub.GrantKnownTransfer(ctx, p.Ref, publicationID, transfer.TransferID, p.Reason)
	if e != nil {
		outcome.err = e
		return outcome
	}
	if len(answer.Held) == 1 {
		held := answer.Held[0]
		if held.TransferID != transfer.TransferID || held.ObjectID != transfer.ObjectID ||
			held.Length != transfer.Length || held.State != "accepted" {
			outcome.err = exit.Internalf("grant for %s returned a changed held transfer", transfer.TransferID)
			return outcome
		}
		outcome.deduped = transfer.Length
		return outcome
	}
	grant := answer.Grants[0]
	if grant.ObjectID != transfer.ObjectID || grant.Length != transfer.Length {
		outcome.err = exit.Internalf("grant for %s returned another object", transfer.TransferID)
		return outcome
	}
	outcome.multipart = grant.Multipart()
	staged := filepath.Join(p.Scratch, fmt.Sprintf("object-%06d", index))
	defer os.Remove(staged)

	// The bytes leave the store through a VERIFIED read, so a publisher cannot
	// upload what its own store silently corrupted.
	local := Result{PublishID: publicationID}
	if e := p.Tool.Extract(grant.ObjectID, staged); e != nil {
		outcome.err = e
		return outcome
	}
	conflict, e := p.put(ctx, grant, staged, &local)
	if e != nil && e.Name == "grant.expired_replan" {
		freshAnswer, e2 := p.Hub.GrantKnownTransfer(ctx, p.Ref, publicationID,
			transfer.TransferID, p.Reason)
		if e2 != nil {
			outcome.err = e2
			return outcome
		}
		if len(freshAnswer.Held) == 1 {
			outcome.deduped = transfer.Length
			return outcome
		}
		fresh := freshAnswer.Grants[0]
		conflict, e = p.put(ctx, fresh, staged, &local)
	}
	if e != nil {
		outcome.err = e
		return outcome
	}
	outcome.moved = local.Moved
	outcome.uploaded = true
	outcome.conflict = conflict
	if conflict {
		outcome.deduped = transfer.Length
	}
	precondition := "uploaded"
	if conflict {
		precondition = "already_present"
	}
	received, receiveErr := p.Hub.MarkTransfersReceived(ctx, p.Ref, publicationID,
		[]hub.ReceivedTransfer{{TransferID: transfer.TransferID,
			ReceivedBytes: transfer.Length, Precondition: precondition}}, p.Reason)
	if receiveErr != nil {
		outcome.err = receiveErr
		return outcome
	}
	if len(received) != 1 || received[0].TransferID != transfer.TransferID ||
		received[0].State != "verifying" || received[0].ReceivedBytes != transfer.Length {
		outcome.err = exit.Internalf("received acknowledgement for %s was incomplete", transfer.TransferID)
		return outcome
	}
	verdicts, verifyErr := p.Hub.VerifyKnownTransfers(
		ctx, p.Ref, publicationID, []string{transfer.TransferID}, p.Reason,
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
	outcome.verified = verdicts[0].State == "accepted"
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
		WithRemedy("re-run to resume from Tensorhub's durable transfer rows")
}

// UploadPresigned is the shared storage-edge PUT for package-release role grants.
// It sends the exact local file under every signed header and reports whether bytes
// moved. A 412 means another writer won the immutable no-clobber race; Tensorhub still
// verifies the final bytes during finalize.
func UploadPresigned(ctx context.Context, subject, path, url, expectedDigest string, length int64,
	headers map[string]string,
) (bool, *exit.Error) {
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() || info.Size() != length {
		return false, exit.Named(exit.Conflict, "upload.local_bytes_changed",
			"%s is no longer the declared %d-byte regular file", path, length).
			WithRemedy("restart package publication from one unchanged committed source package")
	}
	f, err := os.Open(path)
	if err != nil {
		return false, exit.Internalf("cannot reopen declared %s: %s", subject, err)
	}
	h := sha256.New()
	_, hashErr := io.Copy(h, f)
	f.Close()
	observed := "sha256:" + hex.EncodeToString(h.Sum(nil))
	if hashErr != nil || observed != expectedDigest {
		return false, exit.Named(exit.Conflict, "upload.local_bytes_changed",
			"%s now hashes to %s; its declaration names %s", subject, observed, expectedDigest).
			WithRemedy("restart package publication from one unchanged committed source package")
	}
	status, _, body, problem := send(ctx, http.MethodPut, url, opener(path, 0, length), length, headers)
	if problem != nil {
		return false, problem
	}
	switch status {
	case http.StatusOK, http.StatusCreated, http.StatusNoContent:
		return true, nil
	case http.StatusPreconditionFailed:
		return false, nil
	default:
		return false, storageRefusal(status, subject, body)
	}
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
