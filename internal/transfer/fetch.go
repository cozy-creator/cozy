package transfer

// The fetch half: a hub-held checkpoint into the local canonical store.
//
// It is cl-009's transactional install with a different source. The shape is the
// same and so are the guarantees: nothing is visible until it is verified, a killed
// run leaves the prior state runnable, and a re-run converges. What differs is where
// the journal lives — an install's journal is the records database, a fetch's is the
// store's own verification records, which is why this keeps no state of its own.
//
// The walk is DECLARE-FIRST in reverse, and it needs no route that lists a
// checkpoint's objects, because the artifact declares itself:
//
//	round 1  the snapshot manifest, admitted only if it hashes to the id asked for
//	round 2  everything the manifest names directly — header, encoding specs, configs
//	round 3  the transitive closure tfs computes from those documents: the tensors
//
// After round 2 the byte plane can compute the whole object set locally, so round 3
// asks the hub for bytes and never for an inventory. A hub that lied about any of it
// is caught by the same digests the publisher declared.

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/hub"
	"github.com/cozy-creator/cozy-creator-v2/internal/render"
	"github.com/cozy-creator/cozy-creator-v2/internal/tfs"
)

// Fetch is one checkpoint pulled into the local store.
type Fetch struct {
	Tool     *tfs.Tool
	Hub      *hub.Client
	Ref      hub.Ref
	Snapshot string
	DryRun   bool
	Progress func(string)
	Scratch  string
	// FailAfter stops after N objects have been installed, to prove a killed fetch
	// converges on a re-run. Development only, and every arm using it says so.
	FailAfter int

	// seen is what THIS run already handled. The rounds overlap by construction —
	// the closure names the documents round 2 fetched — and counting an object twice
	// would report a fresh fetch as partly deduplicated, which is a lie about the
	// only number anyone reads.
	seen map[string]bool
}

// Fetched is what a fetch did.
type Fetched struct {
	Snapshot string
	HeaderID string
	Objects  int
	Bytes    int64
	// Moved is what came off the wire this run; Held is what the store already had
	// verified and therefore skipped. Their sum is the checkpoint.
	Moved    int64
	Held     int64
	Admitted int
	Skipped  int
	Rounds   int
	Tensors  int
	Parts    int
	Grade    string
	MS       map[string]int64
}

func (f *Fetch) say(format string, args ...any) {
	if f.Progress != nil {
		f.Progress(fmt.Sprintf(format, args...))
	}
}

// Resolve turns a ref into exactly one installed checkpoint. With a snapshot named,
// it confirms the hub holds it; without one, it resolves only when the answer is
// unambiguous — several candidates refuse and list them, because guessing which
// checkpoint someone meant is the one thing a resolver must never do.
func (f *Fetch) Resolve(ctx context.Context) (hub.Checkpoint, *exit.Error) {
	rows, e := f.Hub.Checkpoints(ctx, f.Ref)
	if e != nil {
		return hub.Checkpoint{}, e
	}
	if f.Snapshot != "" {
		for _, r := range rows {
			if r.SnapshotID == f.Snapshot {
				return r, nil
			}
		}
		return hub.Checkpoint{}, exit.New(exit.NotFound,
			"%s holds no checkpoint %s", f.Ref, f.Snapshot).
			WithRemedy("`cozy pull %s` lists what it does hold", f.Ref).
			WithNext("cozy pull " + f.Ref.String() + " --dry-run")
	}
	switch len(rows) {
	case 0:
		return hub.Checkpoint{}, exit.New(exit.NotFound, "%s holds no checkpoint", f.Ref).
			WithRemedy("nothing has been published into this repo yet").
			WithNext("cozy push " + f.Ref.String() + " <sha256:…> --family <f> --reason <why>")
	case 1:
		return rows[0], nil
	default:
		names := make([]string, 0, len(rows))
		for _, r := range rows {
			names = append(names, r.SnapshotID)
		}
		sort.Strings(names)
		return hub.Checkpoint{}, exit.Named(exit.Validation, "ref.ambiguous",
			"%s holds %d checkpoints and this ref names none of them", f.Ref, len(rows)).
			WithRemedy("name one: %s (release addressing — org/repo@release — lands with th-003)",
				strings.Join(short(names), ", ")).
			WithNext("cozy pull " + f.Ref.String() + "@" + names[0])
	}
}

// Run installs the checkpoint transactionally: verified writes, and a local root
// registered only after the whole checkpoint verifies.
func (f *Fetch) Run(ctx context.Context, row hub.Checkpoint) (Fetched, *exit.Error) {
	out := Fetched{Snapshot: row.SnapshotID, HeaderID: row.HeaderID, Grade: row.Grade, MS: map[string]int64{}}
	f.seen = map[string]bool{}
	if err := os.MkdirAll(f.Scratch, 0o700); err != nil {
		return out, exit.Internalf("cannot create the transfer scratch at %s: %s", f.Scratch, err)
	}
	if f.DryRun {
		return f.plan(ctx, row, out)
	}

	// Round 1 — the manifest. It arrives as exact bytes and is admitted under the
	// snapshot id: the document proves itself or it does not enter the store.
	t0 := time.Now()
	doc, e := f.Hub.Manifest(ctx, f.Ref, row.SnapshotID)
	if e != nil {
		return out, e
	}
	path := filepath.Join(f.Scratch, "manifest.bin")
	if err := os.WriteFile(path, doc, 0o644); err != nil {
		return out, exit.Internalf("cannot stage the manifest: %s", err)
	}
	held, e := f.Tool.Held(row.SnapshotID)
	if e != nil {
		return out, e
	}
	if !held {
		if e := f.Tool.Admit(path, row.SnapshotID, int64(len(doc))); e != nil {
			return out, e
		}
		out.Moved += int64(len(doc))
		out.Admitted++
	} else {
		out.Held += int64(len(doc))
		out.Skipped++
	}
	f.seen[row.SnapshotID] = true
	out.MS["manifest"] = since(t0)
	out.Rounds = 1
	f.say("manifest %s admitted (%s)", short1(row.SnapshotID), size(int64(len(doc))))

	// Round 2 — what the manifest names directly: the header, the encoding specs and
	// their vectors, the configs. With these resident the closure below is computable
	// locally, which is why no route needs to list a checkpoint's objects.
	entries, e := f.Tool.Entries(row.SnapshotID)
	if e != nil {
		return out, e
	}
	t0 = time.Now()
	if e := f.round(ctx, row, "documents", entries, &out); e != nil {
		return out, e
	}
	out.MS["documents"] = since(t0)
	out.Rounds = 2

	// Round 3 — the transitive closure, computed by the byte plane from documents it
	// now holds. This is the tensors.
	t0 = time.Now()
	_, objects, e := f.Tool.Closure(row.SnapshotID, "s-fetch", filepath.Join(f.Scratch, "closure.json"))
	if e != nil {
		return out, e
	}
	out.Objects = len(objects)
	for _, o := range objects {
		out.Bytes += o.Length
	}
	if e := f.round(ctx, row, "objects", objects, &out); e != nil {
		return out, e
	}
	out.MS["objects"] = since(t0)
	out.Rounds = 3

	// The proof. Every declared byte, hashed. Only now does the snapshot become a
	// named local root — before this it is objects in a store and nothing points at
	// them, which is what "no partial visibility" means on this side.
	t0 = time.Now()
	tensors, parts, objs, e := f.Tool.Verify(row.SnapshotID)
	if e != nil {
		return out, e
	}
	out.Tensors, out.Parts = tensors, parts
	out.MS["verify"] = since(t0)
	f.say("verified %d objects, %d tensors, %d parts — every declared byte", objs, tensors, parts)

	if e := f.Tool.Register(f.Ref.String(), row.SnapshotID); e != nil {
		return out, e
	}
	f.say("timing: %s", Timing(out.MS))
	return out, nil
}

// plan answers what a fetch WOULD move, and moves nothing at all — not even the
// manifest. What it can be exact about depends on what is already local: with the
// checkpoint's own documents in the store the closure is computable and the answer is
// object-for-object; without them the answer is the hub's row, and it says so. A plan
// that fetched documents to sharpen its own numbers would be a transfer with a
// misleading name.
func (f *Fetch) plan(ctx context.Context, row hub.Checkpoint, out Fetched) (Fetched, *exit.Error) {
	out.Objects, out.Bytes = row.Objects, row.Bytes
	f.say("the hub's row: %d objects, %s, grade %s", row.Objects, size(row.Bytes), row.Grade)

	held, e := f.Tool.Held(row.SnapshotID)
	if e != nil {
		return out, e
	}
	if !held {
		out.Moved = row.Bytes
		f.say("this store holds none of it: the whole checkpoint would move")
		f.say("an object-for-object plan needs the checkpoint's own documents, and fetching those would not be a plan")
		return out, nil
	}
	_, objects, e := f.Tool.Closure(row.SnapshotID, "s-plan", filepath.Join(f.Scratch, "plan-closure.json"))
	if e != nil {
		return out, e
	}
	var need, have int64
	var missing int
	for _, o := range objects {
		h, e := f.Tool.Held(o.ID)
		if e != nil {
			return out, e
		}
		if h {
			have += o.Length
			continue
		}
		need += o.Length
		missing++
	}
	out.Objects, out.Bytes = len(objects), need+have
	out.Moved, out.Held = need, have
	f.say("%d of %d objects to fetch: %s to move, %s already verified here",
		missing, len(objects), size(need), size(have))
	return out, nil
}

// round fetches and installs one set of objects. Presence is not the predicate:
// `tfs fill` skips only objects with a VALID verification record, rehashes a
// present-but-unverified one, and quarantines a corrupt squatter — so a resumed
// fetch converges on verified state rather than on whatever files happen to exist.
func (f *Fetch) round(ctx context.Context, row hub.Checkpoint, name string, objects []tfs.Object, out *Fetched) *exit.Error {
	var want []tfs.Object
	for _, o := range objects {
		if f.seen[o.ID] {
			continue // this run already moved or skipped it; counting it twice would lie
		}
		f.seen[o.ID] = true
		held, e := f.Tool.Held(o.ID)
		if e != nil {
			return e
		}
		if held {
			out.Held += o.Length
			out.Skipped++
			continue
		}
		want = append(want, o)
	}
	if len(want) == 0 {
		f.say("%s: nothing to fetch, this store already holds them", name)
		return nil
	}

	ids := make([]string, 0, len(want))
	for _, o := range want {
		ids = append(ids, o.ID)
	}
	reads, e := f.Hub.Reads(ctx, f.Ref, row.SnapshotID, ids)
	if e != nil {
		return e
	}
	at := map[string]hub.Read{}
	for _, r := range reads {
		at[r.ObjectID] = r
	}

	dir := filepath.Join(f.Scratch, "in")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return exit.Internalf("cannot create the fetch scratch: %s", err)
	}
	var plan strings.Builder
	for i, o := range want {
		if f.FailAfter > 0 && i >= f.FailAfter {
			// Install what has landed so far, then stop. That is exactly what a real
			// interruption leaves behind, and the next run must converge from it.
			if plan.Len() > 0 {
				if e := f.install(name, plan.String(), out); e != nil {
					return e
				}
			}
			return exit.Named(exit.Internal, "crash_after_fetch",
				"development kill point: stopped after %d of %d objects", i, len(want)).
				WithRemedy("re-run the same pull; the store's verification records are the journal")
		}
		r, ok := at[o.ID]
		if !ok {
			return exit.Named(exit.NotFound, "hub.object_unavailable",
				"the hub granted no read for %s", short1(o.ID)).
				WithRemedy("the checkpoint's catalog row and its object custody disagree; the hub owns that reconciliation")
		}
		dst := filepath.Join(dir, strings.TrimPrefix(o.ID, "sha256:"))
		n, e := download(ctx, r.URL, dst, o.Length)
		if e != nil {
			return e
		}
		out.Moved += n
		fmt.Fprintf(&plan, "%s %d %s\n", strings.TrimPrefix(o.ID, "sha256:"), o.Length, dst)
	}
	if e := f.install(name, plan.String(), out); e != nil {
		return e
	}
	_ = os.RemoveAll(dir)
	return nil
}

func (f *Fetch) install(name, plan string, out *Fetched) *exit.Error {
	res, e := f.Tool.Fill([]byte(plan), filepath.Join(f.Scratch, "fill.plan"))
	if e != nil {
		return e
	}
	if res.Refused > 0 {
		return exit.Named(exit.Validation, "fetch.object_refused",
			"the byte plane refused %d of %d fetched objects", res.Refused, res.Put+res.Skipped+res.Refused).
			WithRemedy("a refused object did not hash to the identity the checkpoint declares; nothing was installed under a name it did not earn")
	}
	out.Admitted += res.Put
	out.Skipped += res.Skipped
	f.say("%s: installed %d, skipped %d already verified here", name, res.Put, res.Skipped)
	return nil
}

// download streams one object into a file. It never holds one whole in RAM: an
// object runs to the hub's per-object ceiling and this process is a CLI. A transport
// failure is retried the same bounded way an upload is — the URL is signed, so
// asking again is asking the same question — and the file is truncated on each
// attempt so a half-received body never becomes the input to the next one.
func download(ctx context.Context, url, dst string, length int64) (int64, *exit.Error) {
	var n int64
	exhausted, err := retryStorage(func(int) (bool, error) {
		got, retryable, e := fetchOnce(ctx, url, dst, length)
		if e != nil {
			return retryable, e
		}
		n = got
		return false, nil
	})
	if err == nil {
		return n, nil
	}
	if !exhausted {
		return 0, err.(*exit.Error)
	}
	return 0, exit.Unavailablef("object storage is unreachable after %d attempts: %s", attempts, err).
		WithRemedy("re-run to resume: objects already verified in the store are skipped")
}

func fetchOnce(ctx context.Context, url, dst string, length int64) (int64, bool, *exit.Error) {
	// Bounded by BYTES ARRIVING, not by a clock: a download that is slow is not a
	// download that has stopped, and the two used to be the same 30-minute number.
	m := &mover{}
	rctx, cancel := m.context(ctx)
	defer cancel()
	req, err := http.NewRequestWithContext(rctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, false, exit.Internalf("the read grant's URL is not usable: %s", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, true, exit.Unavailablef("%s", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return 0, retryableStorageStatus(resp.StatusCode),
			storageRefusal(resp.StatusCode, short1(filepath.Base(dst)), raw)
	}
	f, err := os.Create(dst)
	if err != nil {
		return 0, false, exit.Internalf("cannot stage a fetched object: %s", err)
	}
	defer f.Close()
	// The read is bounded by the length the checkpoint DECLARES. A source streaming
	// more than it should is stopped here; whether the bytes hash correctly is the
	// byte plane's question, one step later.
	n, err := io.Copy(f, m.reader(io.LimitReader(resp.Body, length)))
	if err != nil {
		return 0, true, exit.Unavailablef("the transfer broke after %s: %s", size(n), err)
	}
	return n, false, nil
}

func short(ids []string) []string {
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		out = append(out, short1(id))
	}
	return out
}

func short1(id string) string {
	h := strings.TrimPrefix(id, "sha256:")
	if len(h) > 12 {
		return "sha256:" + h[:12] + "…"
	}
	return id
}

func size(n int64) string { return render.Bytes(n) }

// Timing renders a phase breakdown in one line, longest phase last so the eye lands
// on where the time went. Phases are named by what they DID, not by which call was
// made — "upload 19.6s" is the answer to "why did that take 20 seconds".
func Timing(ms map[string]int64) string {
	keys := make([]string, 0, len(ms))
	for k := range ms {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return ms[keys[i]] < ms[keys[j]] })
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		if v := ms[k]; v >= 1000 {
			parts = append(parts, fmt.Sprintf("%s %.1fs", k, float64(v)/1000))
		} else {
			parts = append(parts, fmt.Sprintf("%s %dms", k, v))
		}
	}
	return strings.Join(parts, " · ")
}
