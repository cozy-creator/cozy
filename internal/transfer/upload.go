// Package transfer is the order of operations for moving a canonical Manifest
// between the local store and the hub (cl-012). It owns no bytes and no protocol:
// TensorFS owns the byte plane (every question crosses internal/tfs) and tensorhub
// owns the incremental upload protocol (every call crosses internal/hub). What lives here
// is the sequence, with two required properties:
//
//   - resumability. There is no client-side journal. An upload resumes from the
//     hub's durable transfer rows, and a fetch resumes because
//     the local store's verification records already say which objects are good.
//     A transfer that kept its own progress file would have a third opinion about
//     what is done, and the third opinion is always the wrong one.
//   - honest accounting. Every progress line separates bytes MOVED from bytes
//     DEDUPED. An upload that sends nothing and prints the artifact's size is
//     claiming credit for work it did not do.
package transfer

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/tfs"
)

// One object write or read at the storage edge is bounded by BYTES MOVING, not by a
// clock — see `mover`. The 30-minute constant that used to live here said in its own
// comment that it was standing in for a byte meter.

// Upload is one incremental owner-only checkpoint upload, start to finish.
type Upload struct {
	Tool *tfs.Tool
	Hub  *hub.Client
	Ref  hub.Ref
	// ManifestID is the local canonical manifest being published.
	ManifestID string
	Session    string
	Reason     string
	// Progress receives one line per phase. It is where the honest accounting is
	// printed as the work happens; the returned Result carries the same numbers.
	Progress func(string)
	// Scratch is where objects are staged out of the store on their way to the wire.
	Scratch string
}

// Result is what an upload did.
type Result struct {
	PublishID string
	Created   bool
	Session   string
	Totals    hub.Totals
	Grants    int
	Uploaded  int
	Conflicts int
	Verified  int
	// Moved is bytes this invocation actually put on the wire; Deduped is what the
	// hub already held. Their sum is the artifact.
	Moved    int64
	Deduped  int64
	Manifest hub.ManifestRef
	Dup      bool
	MS       map[string]int64
}

func ValidateOpenedPublication(opened hub.OpenPublicationResponse, operation string,
	declared []hub.Object,
) (hub.Totals, *exit.Error) {
	var totals hub.Totals
	publication := opened.Publication
	if publication.Operation != operation ||
		(publication.State != "open" && publication.State != "checkpointed") {
		return totals, exit.Internalf("publication reopened under changed operation or state")
	}
	if e := exactTransfers(declared, publication.Objects); e != nil {
		return totals, e
	}
	for _, row := range publication.Objects {
		totals.DeclaredObjects++
		totals.DeclaredBytes += row.Length
		switch row.State {
		case "accepted", "verifying":
			totals.HeldObjects++
		case "claimed", "transferring":
			totals.MissingObjects++
			totals.MissingBytes += row.Length
		default:
			return totals, exit.New(exit.Conflict,
				"publication object %s is %s", row.ObjectID, row.State)
		}
	}
	return totals, nil
}

func ValidateFinalizedCheckpoint(checkpoint hub.CheckpointPublication, operation,
	manifestID string, manifestLength int64, totals hub.Totals,
) *exit.Error {
	if checkpoint.PublishID != operation || checkpoint.CheckpointID != manifestID ||
		checkpoint.Manifest.SHA256 != strings.TrimPrefix(manifestID, "sha256:") ||
		checkpoint.Manifest.Length != manifestLength ||
		checkpoint.State != "checkpointed" || checkpoint.Objects != totals.DeclaredObjects-1 ||
		checkpoint.Bytes != totals.DeclaredBytes-manifestLength {
		return exit.Internalf("Tensorhub retained a different checkpoint identity or inventory")
	}
	return nil
}

func (p *Upload) say(format string, args ...any) {
	if p.Progress != nil {
		p.Progress(fmt.Sprintf(format, args...))
	}
}

// Run uploads only missing objects and finalizes one owner-only checkpoint.
// Release pointers are a separate operation.
func (p *Upload) Run(ctx context.Context) (Result, *exit.Error) {
	var res Result
	ms := map[string]int64{}
	res.MS = ms

	// 1. Prepare the known ObjectRefs. The Manifest is one of
	//    the transferred objects; finalize names its accepted digest/length and
	//    Tensorhub reads those custody bytes itself.
	t0 := time.Now()
	if err := os.MkdirAll(p.Scratch, 0o700); err != nil {
		return res, exit.Internalf("cannot create the transfer scratch at %s: %s", p.Scratch, err)
	}
	objects, e := p.Tool.ManifestObjects(p.ManifestID, filepath.Join(p.Scratch, "objects.jsonl"))
	if e != nil {
		return res, e
	}
	manifestPath := filepath.Join(p.Scratch, "manifest.bin")
	if e := p.Tool.Manifest(p.ManifestID, manifestPath); e != nil {
		return res, e
	}
	manifestInfo, err := os.Stat(manifestPath)
	if err != nil {
		return res, exit.Internalf("the manifest tfs extracted is unreadable: %s", err)
	}
	res.Manifest = hub.ManifestRef{SHA256: strings.TrimPrefix(p.ManifestID, "sha256:"),
		Length: manifestInfo.Size()}
	finalize := hub.FinalizePublicationRequest{
		ManifestID: p.ManifestID, ManifestLength: manifestInfo.Size(),
	}
	declared := make([]hub.Object, 0, len(objects)+1)
	for _, o := range objects {
		declared = append(declared, hub.Object{ID: o.ID, Length: o.Length})
	}
	declared = append(declared, hub.Object{ID: p.ManifestID, Length: manifestInfo.Size()})
	sort.Slice(declared, func(i, j int) bool { return declared[i].ID < declared[j].ID })
	ms["declare"] = since(t0)
	p.say("declared %d known blob transfers (%s) from %s", len(declared), bytesOf(objects), p.ManifestID)

	// 2. Open under the caller-stable operation id. Reopening returns the same durable
	//    publication, including a checkpointed one whose result can replay.
	t0 = time.Now()
	opened, e := p.Hub.OpenPublication(ctx, p.Ref, p.Session, declared, p.Reason)
	if e != nil {
		return res, e
	}
	ms["open"] = since(t0)
	publication := opened.Publication
	if publication.Operation != p.Session {
		return res, exit.Internalf("publication reopened under a different operation")
	}
	res.PublishID, res.Created, res.Session = publication.Operation, opened.Created, publication.Operation
	verb := "resumed"
	if opened.Created {
		verb = "opened"
	}
	p.say("%s publication %s in state %s", verb, res.PublishID, publication.State)
	if publication.State != "open" && publication.State != "checkpointed" {
		return res, exit.New(exit.Conflict, "upload %s is %s", publication.Operation, publication.State).
			WithRemedy("use a new model upload after repairing or abandoning the refused operation")
	}

	// 3. The idempotent PUT froze and returned the exact object set in every state.
	transfers := publication.Objects
	if e := exactTransfers(declared, transfers); e != nil {
		return res, e
	}

	toUpload := make([]hub.Transfer, 0, len(transfers))
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
		case "claimed", "transferring":
			res.Totals.MissingObjects++
			res.Totals.MissingBytes += transfer.Length
			toUpload = append(toUpload, transfer)
		default:
			return res, exit.New(exit.Conflict,
				"publication object %s is %s", transfer.ObjectID, transfer.State).
				WithRemedy("repair the refused immutable manifest publication before replaying it")
		}
	}
	p.say("publication %s: %d transfers need upload (%s) · %d already resident (%s)",
		res.PublishID, len(toUpload), size(res.Totals.MissingBytes),
		res.Totals.HeldObjects, size(res.Deduped))

	// 4. Grants and writes. Nothing uploads without an exact transfer grant.
	if len(toUpload) > 0 {
		if e := p.upload(ctx, toUpload, &res, ms); e != nil {
			return res, e
		}
	} else {
		p.say("0 bytes to upload: every transfer is ready for final verification")
	}

	return p.finalize(ctx, finalize, res, ms)
}

func (p *Upload) finalize(ctx context.Context, request hub.FinalizePublicationRequest,
	res Result, ms map[string]int64,
) (Result, *exit.Error) {
	t0 := time.Now()
	checkpoint, e := p.Hub.FinalizePublication(ctx, p.Ref, res.PublishID, request, p.Reason)
	if e != nil {
		return res, e
	}
	ms["finalize"] = since(t0)
	if problem := ValidateFinalizedCheckpoint(checkpoint, res.PublishID, p.ManifestID,
		res.Manifest.Length, res.Totals); problem != nil {
		return res, problem
	}
	res.Manifest, res.Verified = checkpoint.Manifest, checkpoint.Objects
	res.Dup = checkpoint.Duplicate
	if checkpoint.Duplicate {
		p.say("this exact checkpoint was already retained")
	}
	p.say("Tensorhub verified %d declared objects and retained checkpoint %s", res.Verified, checkpoint.CheckpointID)

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
		if !ok || seen[transfer.ObjectID] || length != transfer.Length {
			return exit.Internalf("claim answered a missing, duplicate, or changed transfer for %s", transfer.ObjectID)
		}
		seen[transfer.ObjectID] = true
	}
	return nil
}

// upload requests grants in the same bounded batches Tensorhub accepts and writes
// each object at its final content key under every signed condition. Two concurrent
// presigned writes keep a residential uplink and R2's connection churn stable;
// finalization verifies the complete declared set in one act.
const (
	publicationObjectBatch = 128
	publishParallelism     = 2
)

type uploadOutcome struct {
	index    int
	moved    int64
	deduped  int64
	uploaded bool
	conflict bool
	primary  bool
	err      *exit.Error
}

func (p *Upload) upload(ctx context.Context, transfers []hub.Transfer, res *Result, ms map[string]int64) *exit.Error {
	started := time.Now()
	for offset := 0; offset < len(transfers); offset += publicationObjectBatch {
		end := min(offset+publicationObjectBatch, len(transfers))
		if e := p.uploadBatch(ctx, transfers[offset:end], offset, res); e != nil {
			return e
		}
	}
	ms["grant_upload_verify"] = since(started)
	p.say("requested %d transfer grants and uploaded %d objects: %s moved · %s deduped",
		len(transfers), res.Uploaded, size(res.Moved), size(res.Deduped))
	return nil
}

func (p *Upload) uploadBatch(ctx context.Context, transfers []hub.Transfer, indexBase int,
	res *Result,
) *exit.Error {
	objectIDs := make([]string, 0, len(transfers))
	byObject := make(map[string]int, len(transfers))
	for index, transfer := range transfers {
		objectIDs = append(objectIDs, transfer.ObjectID)
		byObject[transfer.ObjectID] = index
	}
	answer, e := p.Hub.GrantKnownTransfers(ctx, p.Ref, res.PublishID, objectIDs, p.Reason)
	if e != nil {
		return e
	}

	ordered := make([]*uploadOutcome, len(transfers))
	grants := make(map[int]hub.Grant, len(answer.Grants))
	for _, held := range answer.Held {
		index, ok := byObject[held.ObjectID]
		if !ok || held.Length != transfers[index].Length || held.State != "accepted" {
			return exit.Internalf("grant returned a changed held object %s", held.ObjectID)
		}
		ordered[index] = &uploadOutcome{index: indexBase + index, deduped: held.Length}
	}
	for _, grant := range answer.Grants {
		index, ok := byObject[grant.ObjectID]
		if !ok || grant.Length != transfers[index].Length || ordered[index] != nil {
			return exit.Internalf("grant returned a missing, duplicate, or changed object %s", grant.ObjectID)
		}
		grants[index] = grant
	}

	uploadCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	jobs := make(chan int)
	outcomes := make(chan uploadOutcome, publishParallelism)
	var workers sync.WaitGroup
	var cancelOnce sync.Once
	for range publishParallelism {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for index := range jobs {
				outcome := p.uploadOne(uploadCtx, indexBase+index, transfers[index], grants[index])
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
			if _, ok := grants[index]; !ok {
				continue
			}
			select {
			case jobs <- index:
			case <-uploadCtx.Done():
				close(jobs)
				workers.Wait()
				return
			}
		}
		close(jobs)
		workers.Wait()
	}()

	for outcome := range outcomes {
		copy := outcome
		ordered[outcome.index-indexBase] = &copy
	}
	var primary *exit.Error
	for _, outcome := range ordered {
		if outcome != nil && outcome.primary {
			primary = outcome.err
		}
	}

	res.Grants += len(transfers)
	for _, outcome := range ordered {
		if outcome == nil {
			if primary != nil {
				continue
			}
			return exit.Internalf("upload stopped before a transfer produced a verdict")
		}
		res.Moved += outcome.moved
		res.Deduped += outcome.deduped
		if outcome.uploaded {
			res.Uploaded++
		}
		if outcome.conflict {
			res.Conflicts++
		}
	}
	return primary
}

func (p *Upload) uploadOne(ctx context.Context, index int, transfer hub.Transfer, grant hub.Grant,
) uploadOutcome {
	outcome := uploadOutcome{index: index}
	staged := filepath.Join(p.Scratch, fmt.Sprintf("object-%06d", index))
	defer os.Remove(staged)

	// The bytes leave the store through a VERIFIED read, so a publisher cannot
	// upload what its own store silently corrupted.
	var e *exit.Error
	if grant.ObjectID == p.ManifestID {
		e = p.Tool.Manifest(grant.ObjectID, staged)
	} else {
		e = p.Tool.Extract(grant.ObjectID, staged)
	}
	if e != nil {
		outcome.err = e
		return outcome
	}
	conflict, e := put(ctx, grant, staged)
	if e != nil {
		outcome.err = e
		return outcome
	}
	if !conflict {
		outcome.moved = grant.Length
	}
	outcome.uploaded = true
	outcome.conflict = conflict
	if conflict {
		outcome.deduped = transfer.Length
	}
	return outcome
}

// put writes one bounded object at its final content key. Every required header is
// sent verbatim: they are signed into the URL, so stripping one is 403 and a wrong
// body is 400 — neither is a branch a client may take.
func put(ctx context.Context, g hub.Grant, path string) (bool, *exit.Error) {
	status, _, body, _, e := send(ctx, http.MethodPut, g.URL, opener(path, 0, g.Length), g.Length, g.Headers)
	if e != nil {
		return false, e
	}
	switch status {
	case http.StatusOK:
		return false, nil
	case http.StatusPreconditionFailed:
		// Someone already holds this exact key. That is not a failure — the key is
		// the digest, so whoever wrote it wrote these bytes. The hub still proves it.
		return true, nil
	default:
		return false, storageRefusal(status, g.ObjectID, body)
	}
}

// attempts bounds transport retries at the storage edge. A hiccup on a residential
// uplink is not a protocol event: every condition is signed INTO the url, so
// re-sending the same body is the same write, and the no-clobber precondition means
// a partially-received first attempt cannot become half an object. Three tries, then
// the run stops and says so — and a stopped run resumes, because the objects that did
// land are already the hub's.
const attempts = 3

var storageIPv4Client = func() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	dialer := &net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}
	transport.DialContext = func(ctx context.Context, _, address string) (net.Conn, error) {
		return dialer.DialContext(ctx, "tcp4", address)
	}
	return &http.Client{Transport: transport}
}()

// send is the one storage-edge write in this binary. It talks to object storage,
// never to the hub: the URL is presigned and carries no credential of ours.
func send(ctx context.Context, method, url string, open func() (io.ReadCloser, error), length int64, headers map[string]string) (int, http.Header, []byte, int64, *exit.Error) {
	var status int
	var header http.Header
	var raw []byte
	var moved int64
	exhausted, err := retryStorage(ctx, func(attempt int) (bool, error) {
		status, header, raw = 0, nil, nil
		body, err := open()
		if err != nil {
			return false, exit.Internalf("the staged object is unreadable: %s", err)
		}
		rctx, cancel := ctx, func() {}
		var counted io.ReadCloser = http.NoBody
		var meter *mover
		if length == 0 {
			body.Close()
		} else {
			// The body is COUNTED and the context lives while the count moves: an upload
			// that is slow is not an upload that has stopped.
			meter = &mover{}
			rctx, cancel = meter.context(ctx)
			counted = meter.readCloser(body)
		}
		defer func() {
			if meter != nil {
				moved += meter.moved.Load()
			}
		}()
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
		client := http.DefaultClient
		if attempt > 0 {
			client = storageIPv4Client
		}
		resp, err := client.Do(req)
		if err != nil {
			client.CloseIdleConnections()
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
		return status, header, raw, moved, nil
	}
	if !exhausted {
		return 0, nil, nil, moved, err.(*exit.Error)
	}
	return 0, nil, nil, moved, exit.Unavailablef("object storage is unreachable after %d attempts: %s", attempts, err).
		WithRemedy("re-run to resume from Tensorhub's durable transfer rows")
}

// UploadPresigned sends one local file to a release-scoped storage grant. Tensorhub
// reads the stored bytes and computes their identity during finalize. The byte
// count is the file length once any PUT attempt read those bytes, capped so
// retries do not count one immutable file twice. A successful PUT whose response
// was lost must not become 0 B when its retry correctly answers that the object
// already exists.
func UploadPresigned(ctx context.Context, subject, path, url string, headers map[string]string) (int64, *exit.Error) {
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() {
		return 0, exit.Named(exit.Conflict, "upload.local_file_unreadable",
			"%s is no longer a regular file", path)
	}
	length := info.Size()
	status, _, body, moved, problem := send(ctx, http.MethodPut, url, opener(path, 0, length), length, headers)
	if problem != nil {
		return min(moved, length), problem
	}
	moved = min(moved, length) // retries do not make one immutable file larger
	switch status {
	case http.StatusOK, http.StatusCreated, http.StatusNoContent:
		return moved, nil
	case http.StatusPreconditionFailed:
		return moved, nil
	default:
		return moved, storageRefusal(status, subject, body)
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
