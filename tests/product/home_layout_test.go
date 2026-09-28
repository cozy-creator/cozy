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
		EventType: "run.completed", EventPayload: map[string]any{},
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
