package producttest

import (
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/reclaim"
	"github.com/cozy-creator/cozy/internal/records"
)

// rootEntries is one home's top-level census, sorted, SQLite WAL/SHM siblings folded
// into their database: their appearance is a connection-lifetime detail, not layout.
func rootEntries(t *testing.T, root string) []string {
	t.Helper()
	entries, err := os.ReadDir(root)
	must(t, err)
	seen := map[string]bool{}
	for _, entry := range entries {
		name := entry.Name()
		name = strings.TrimSuffix(strings.TrimSuffix(name, "-wal"), "-shm")
		seen[name] = true
	}
	out := make([]string, 0, len(seen))
	for name := range seen {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// TestFreshHomeIsMinimal proves cl-116's allowlist as a property of a REAL fresh home:
// opening the layout, the records authority and the daemon record creates exactly the
// target entries and nothing else — no triage/uploads/publications/workers/tmp/
// local-packages/companions root exists until real work does.
func TestFreshHomeIsMinimal(t *testing.T) {
	root := filepath.Join(t.TempDir(), "home")
	l, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(l.DB)
	fatal(t, problem)
	defer store.Close()

	held, problem := daemon.Hold(l, "127.0.0.1:0", l.Root+"/worker.sock")
	fatal(t, problem)
	defer held.Release()
	creds, problem := api.Mint(l)
	fatal(t, problem)

	want := []string{"creator.sqlite", "daemon.lock", "installs", "outputs"}
	if got := rootEntries(t, root); strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("fresh home holds %v, want exactly %v", got, want)
	}

	// The token round-trips through the one 0600 record and nowhere else. Digest
	// equality is the comparison the server itself uses; no raw value is read here.
	read, problem := api.ClientCredential(l)
	fatal(t, problem)
	if read.Digest() != creds.CLI.Digest() {
		t.Fatal("the daemon record did not carry the minted client credential back")
	}
	if runtime.GOOS != "windows" {
		for _, path := range []string{l.Daemon, l.DB} {
			info, err := os.Stat(path)
			must(t, err)
			if info.Mode().Perm() != 0o600 {
				t.Fatalf("%s is %v, want 0600", path, info.Mode().Perm())
			}
		}
	}

	// The single-writer lock lives INSIDE installs and never becomes a sweep victim.
	writer, problem := install.Lock(l)
	fatal(t, problem)
	defer writer.Unlock()
	if filepath.Dir(l.Lock) != l.Installs {
		t.Fatalf("writer lock at %s, want under %s", l.Lock, l.Installs)
	}
	swept, problem := install.Sweep(l, store)
	fatal(t, problem)
	if swept.Removed != 0 {
		t.Fatalf("install sweep removed %d entries from a home holding only the writer lock", swept.Removed)
	}
}

// TestHomeMigrationFromPriorShape drives the exact cut a live pre-cl-116 home takes at
// the next daemon start: records.db becomes creator.sqlite in place, the retired
// planes are deleted, publication and rental-secret debris is settled against the
// records authority, empty on-demand roots are pruned — and user outputs, the TensorFS
// store (tfs-047's to move, not ours) and manual backups are untouched.
func TestHomeMigrationFromPriorShape(t *testing.T) {
	root := filepath.Join(t.TempDir(), "home")
	must(t, os.MkdirAll(root, 0o700))

	// The prior shape, exactly as the audit found it.
	seed := func(path string, body string) {
		must(t, os.MkdirAll(filepath.Dir(filepath.Join(root, path)), 0o755))
		must(t, os.WriteFile(filepath.Join(root, path), []byte(body), 0o600))
	}
	prior, err := os.Create(filepath.Join(root, "records.db"))
	must(t, err)
	must(t, prior.Close())
	seed("client.cred", "stale-token\n")
	seed("writer.lock", "")
	seed("uploads/sha256/aaaa", "upload-bytes")
	seed("triage/trb-orphan.json", "{}")
	seed("job-plans/aaaa.json", "{}")
	seed("outputs/cozy-example/deadbeef.png", "user-owned")
	seed("cas/blobs/aa/bb/cc", "tensorfs-owned")
	seed("backups/2026-01-01/records.db", "manual")
	seed("inputs/"+sixtyFour("a"), "historic-input-preserved")
	seed("rentals/pr-gone.media-token", "secret\n")
	seed("rentals/pr-gone.pem", "cert")
	seed("rentals/pending-"+sixtyFour("d")+".media-token", "secret\n")
	for _, dir := range []string{"jit-cache/scope-1", "inputs", "local-packages", "tmp",
		"companions", "workers", "publications/local/_job-req-debris/.staging/a1"} {
		must(t, os.MkdirAll(filepath.Join(root, dir), 0o755))
	}

	l, problem := home.Open(root)
	fatal(t, problem)
	if _, err := os.Stat(filepath.Join(root, "records.db")); !os.IsNotExist(err) {
		t.Fatal("records.db survived the rename")
	}
	store, problem := records.OpenForDaemon(l.DB, filepath.Join(root, "triage"))
	fatal(t, problem)
	defer store.Close()

	// The debris publication belongs to a request that settled without a publication.
	submitSweepRequest(t, store, "req-debris")
	fatal(t, store.SettleRequest("req-debris", "failed"))

	if _, problem := reclaim.Retired(l); problem != nil {
		t.Fatal(problem)
	}
	if _, problem := reclaim.Publications(l, store); problem != nil {
		t.Fatal(problem)
	}
	if _, problem := reclaim.RentalSecrets(l, store); problem != nil {
		t.Fatal(problem)
	}
	reclaim.EmptyRoots(l)

	// daemon.lock is created by the migration's own liveness guard and is a target
	// entry anyway; backups and cas are deliberately untouched (the final backup
	// deletion and the Store move belong to the last cut and tfs-047).
	want := []string{"backups", "cas", "creator.sqlite", "daemon.lock", "inputs", "installs", "outputs"}
	if got := rootEntries(t, root); strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("migrated home holds %v, want exactly %v", got, want)
	}
	if data, err := os.ReadFile(filepath.Join(root, "outputs", "cozy-example", "deadbeef.png")); err != nil ||
		string(data) != "user-owned" {
		t.Fatalf("migration touched a user output: %q, %v", data, err)
	}
	if _, err := os.Stat(filepath.Join(root, "cas", "blobs", "aa", "bb", "cc")); err != nil {
		t.Fatalf("migration touched the TensorFS store: %v", err)
	}
}

// TestTriageBundleLivesInTheAttemptRow proves cl-116's triage cut on the real store
// path: the verified bundle rides the terminal INTO its attempt row, is served back by
// the attempt's opaque key, and no triage directory exists for an orphan file to
// accumulate in.
func TestTriageBundleLivesInTheAttemptRow(t *testing.T) {
	l, problem := home.Open(t.TempDir())
	fatal(t, problem)
	store, problem := records.Open(l.DB)
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "ins-triage",
		Package: "cozy/sweep", WorkerID: "local", Devices: []string{"cpu"}}))
	submitSweepRequest(t, store, "req-triaged")
	session, digest := "session-triage", "sha256:"+sixtyFour("a")
	attempt, problem := store.Dispatch(records.Attempt{RequestID: "req-triaged",
		SessionID: session, InstanceID: "ins-triage",
		InvocationDigest: digest, InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, store.OfferDispatch("req-triaged", attempt, session))
	fatal(t, store.Accepted("req-triaged", attempt, session))
	bundle := []byte(`{"terminal":{"traceback":"boom"}}`)
	if _, problem := store.AcceptTerminal(records.Terminal{RequestID: "req-triaged",
		Attempt: attempt, SessionID: session, InvocationDigest: digest,
		TerminalID: "out-triaged", TerminalDigest: "sha256:" + sixtyFour("f"),
		Status: "FAILED", Cause: "handler_error", TriageSubject: "trb-1",
		TriageDigest: "sha256:" + sixtyFour("b"), TriageLength: int64(len(bundle)),
		TriageBundle: bundle,
		EventType:    "request.failed", EventPayload: map[string]any{},
		RequestState: "failed",
	}); problem != nil {
		t.Fatal(problem)
	}
	attempts, problem := store.Attempts("req-triaged")
	fatal(t, problem)
	if len(attempts) != 1 || !attempts[0].TriageKept {
		t.Fatalf("attempt does not carry its bundle: %+v", attempts)
	}
	subject, kept, problem := store.TriageBundle(attempts[0].AttemptKey)
	fatal(t, problem)
	if subject != "trb-1" || string(kept) != string(bundle) {
		t.Fatalf("served bundle = %q/%q", subject, kept)
	}
	if _, err := os.Stat(filepath.Join(l.Root, "triage")); !os.IsNotExist(err) {
		t.Fatalf("a triage directory exists: %v", err)
	}
}

// TestPublicationDebrisIsReclaimed pins the audit's 51-empty-directories class: a
// settled request that committed no publication loses its whole root, a committed
// publication keeps its bytes minus the settled staging, and the plane itself
// disappears once nothing holds it.
func TestPublicationDebrisIsReclaimed(t *testing.T) {
	l, problem := home.Open(t.TempDir())
	fatal(t, problem)
	store, problem := records.Open(l.DB)
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "ins-attempts",
		Package: "cozy/sweep", WorkerID: "local", Devices: []string{"cpu"}}))

	// One committed publication whose bytes must survive; the row rides the terminal
	// transaction exactly as the job lane writes it.
	publishedRoot := l.PublicationRoot("local", "req-published")
	must(t, os.MkdirAll(filepath.Join(publishedRoot, ".staging", "a1"), 0o755))
	must(t, os.WriteFile(filepath.Join(publishedRoot, "weights.bin"), []byte("published"), 0o644))
	submitSweepRequest(t, store, "req-published")
	session, digest := "session-pub", "sha256:"+sixtyFour("a")
	attempt, problem := store.Dispatch(records.Attempt{RequestID: "req-published",
		SessionID: session, InstanceID: "ins-attempts",
		InvocationDigest: digest, InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, store.OfferDispatch("req-published", attempt, session))
	fatal(t, store.Accepted("req-published", attempt, session))
	if _, problem := store.AcceptTerminal(records.Terminal{RequestID: "req-published",
		Attempt: attempt, SessionID: session, InvocationDigest: digest,
		TerminalID: "out-published", TerminalDigest: "sha256:" + sixtyFour("f"),
		Status: "SUCCEEDED", Cause: "COMPLETED",
		EventType: "request.completed", EventPayload: map[string]any{},
		RequestState: "succeeded",
		Publication: &records.Publication{RequestID: "req-published", Attempt: attempt,
			Repo: home.ScratchRepo("local", "req-published"), Root: publishedRoot,
			Status: "SUCCEEDED", Entries: 1, Bytes: 9},
	}); problem != nil {
		t.Fatal(problem)
	}

	// One canceled job that never committed anything: the audit's debris class.
	submitSweepRequest(t, store, "req-canceled")
	fatal(t, store.SettleRequest("req-canceled", "canceled"))
	must(t, os.MkdirAll(filepath.Join(l.PublicationRoot("local", "req-canceled"), ".staging", "a1"), 0o755))

	// One live request whose stage must NOT be touched.
	submitSweepRequest(t, store, "req-live")
	must(t, os.MkdirAll(filepath.Join(l.PublicationRoot("local", "req-live"), ".staging", "a1"), 0o755))

	swept, problem := reclaim.Publications(l, store)
	fatal(t, problem)
	if swept.Scanned != 3 {
		t.Fatalf("publication sweep = %+v, want 3 scanned", swept)
	}
	if data, err := os.ReadFile(filepath.Join(publishedRoot, "weights.bin")); err != nil || string(data) != "published" {
		t.Fatalf("the committed publication lost its bytes: %q, %v", data, err)
	}
	if _, err := os.Stat(filepath.Join(publishedRoot, ".staging")); !os.IsNotExist(err) {
		t.Fatal("the committed publication kept its settled staging")
	}
	if _, err := os.Stat(l.PublicationRoot("local", "req-canceled")); !os.IsNotExist(err) {
		t.Fatal("the canceled job kept its empty publication root")
	}
	if _, err := os.Stat(filepath.Join(l.PublicationRoot("local", "req-live"), ".staging", "a1")); err != nil {
		t.Fatalf("the live request lost its stage: %v", err)
	}

	// Once the last holder settles and commits nothing, the whole plane goes.
	fatal(t, store.SettleRequest("req-live", "failed"))
	must(t, os.RemoveAll(publishedRoot))
	if _, problem := reclaim.Publications(l, store); problem != nil {
		t.Fatal(problem)
	}
	if _, err := os.Stat(l.Publications); !os.IsNotExist(err) {
		t.Fatal("an empty publication plane remains")
	}
}

// TestDiskCountsEachInodeOnce pins the accounting correction: an intra-tree hardlink
// pair is ONE exclusive inode, not two shared entries, and only an inode with a name
// outside the measured tree is shared.
func TestDiskCountsEachInodeOnce(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("windows reports no link counts")
	}
	base := t.TempDir()
	tree := filepath.Join(base, "install")
	must(t, os.MkdirAll(tree, 0o755))
	inside := filepath.Join(tree, "digest-name")
	must(t, os.WriteFile(inside, make([]byte, 8192), 0o644))
	must(t, os.Link(inside, filepath.Join(tree, "named-view"))) // intra-tree pair
	sharedSource := filepath.Join(base, "outside")
	must(t, os.WriteFile(sharedSource, make([]byte, 4096), 0o644))
	must(t, os.Link(sharedSource, filepath.Join(tree, "uv-hardlink"))) // one name outside

	exclusive, shared := install.Disk(tree)
	if exclusive < 8192 || exclusive >= 8192*2 {
		t.Fatalf("intra-tree hardlink pair measured as %d exclusive bytes, want counted once (~8192)", exclusive)
	}
	if shared < 4096 || shared >= 4096*2 {
		t.Fatalf("externally linked inode measured as %d shared bytes, want counted once (~4096)", shared)
	}
}

// TestEmptyPriorRecordsIsSetAsideAndAPopulatedOneRefuses proves cl-134 on the real
// migration path, both ways.
//
// The failure this ends: an OLD build run once inside an already-migrated root recreates
// records.db, runs its own migrations into it, and writes nothing. Every current binary
// then refused the whole root — the CLI went blind — and the refusal named a decision
// ("remove the one that is not the lifecycle authority") it gave the reader no evidence
// to make. Observed twice on 2026-09-04; the answer needed a sqlite shell to see.
//
// A database carrying no rows in any table carries no lifecycle, so it is set aside and
// the root opens. One carrying rows is a real ambiguity and still refuses — that arm is
// the red arm, and it must stay red.
func TestEmptyPriorRecordsIsSetAsideAndAPopulatedOneRefuses(t *testing.T) {
	// Both databases are made by the REAL store, so both carry the real schema; a
	// hand-rolled file would prove the predicate against a fiction.
	seedStore := func(path string) *records.Store {
		store, problem := records.OpenForDaemon(path, filepath.Join(filepath.Dir(path), "triage"))
		fatal(t, problem)
		return store
	}

	t.Run("empty is set aside", func(t *testing.T) {
		root := filepath.Join(t.TempDir(), "home")
		must(t, os.MkdirAll(root, 0o700))
		authority := seedStore(filepath.Join(root, "creator.sqlite"))
		submitSweepRequest(t, authority, "req-authority")
		authority.Close()
		// The stale binary's contribution: full schema, not one row.
		seedStore(filepath.Join(root, "records.db")).Close()

		l, problem := home.Open(root)
		fatal(t, problem)
		if _, err := os.Stat(filepath.Join(root, "records.db")); !os.IsNotExist(err) {
			t.Fatal("an empty records.db survived beside creator.sqlite")
		}
		aside, err := filepath.Glob(filepath.Join(root, "records.db.superseded-*"))
		must(t, err)
		if len(aside) != 1 {
			t.Fatalf("want exactly one superseded file, got %v", aside)
		}
		// AND THE AUTHORITY IS UNTOUCHED — the point of setting the other one aside.
		store, problem := records.OpenForDaemon(l.DB, filepath.Join(root, "triage"))
		fatal(t, problem)
		defer store.Close()
		fatal(t, store.SettleRequest("req-authority", "failed"))
	})

	t.Run("a populated one still refuses", func(t *testing.T) {
		root := filepath.Join(t.TempDir(), "home")
		must(t, os.MkdirAll(root, 0o700))
		seedStore(filepath.Join(root, "creator.sqlite")).Close()
		prior := seedStore(filepath.Join(root, "records.db"))
		submitSweepRequest(t, prior, "req-prior")
		prior.Close()

		if _, problem := home.Open(root); problem == nil {
			t.Fatal("a records.db carrying rows was silently set aside")
		} else if !strings.Contains(problem.Error(), "records.db") {
			t.Fatalf("refusal does not name the file: %v", problem)
		}
		if _, err := os.Stat(filepath.Join(root, "records.db")); err != nil {
			t.Fatalf("a refused migration moved the database anyway: %v", err)
		}
	})
}
