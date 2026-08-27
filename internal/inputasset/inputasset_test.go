package inputasset

import (
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
)

func TestSweepKeepsOwnedObjectsAndDropsCrashOrphans(t *testing.T) {
	layout, e := home.Open(t.TempDir())
	if e != nil {
		t.Fatal(e)
	}
	store, e := records.Open(layout.DB)
	if e != nil {
		t.Fatal(e)
	}
	defer store.Close()
	owned := "sha256:" + strings.Repeat("1", 64)
	orphan := "sha256:" + strings.Repeat("2", 64)
	for _, path := range []string{layout.InputAsset(owned), layout.InputAsset(orphan), layout.Inputs + "/.asset-crash"} {
		if err := os.WriteFile(path, []byte("bytes"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, e := store.Submit(records.Request{ID: "req-owned", IdemKey: "idem-owned",
		BodyDigest: "sha256:body", Endpoint: "org/model", Entrypoint: "generate",
		PlanID: "plan", Payload: []byte(`{}`), Assets: []records.AssetBinding{{
			FieldPath: "first_frame", LocalPath: layout.InputAsset(owned), Digest: owned, Length: 5,
		}}}); e != nil {
		t.Fatal(e)
	}
	if e := Sweep(layout, store); e != nil {
		t.Fatal(e)
	}
	if _, err := os.Stat(layout.InputAsset(owned)); err != nil {
		t.Fatalf("owned object removed: %v", err)
	}
	for _, path := range []string{layout.InputAsset(orphan), layout.Inputs + "/.asset-crash"} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("orphan %s remains: %v", path, err)
		}
	}
}

func TestLargeStageUsesBoundedMemory(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	source := layout.Root + "/large.mp4"
	file, err := os.Create(source)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Truncate(64 << 20); err != nil {
		t.Fatal(err)
	}
	_ = file.Close()
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	staged, problem := Stage(layout, records.AssetBinding{FieldPath: "video", LocalPath: source},
		128<<20)
	runtime.ReadMemStats(&after)
	if problem != nil {
		t.Fatal(problem)
	}
	if staged.Length != 64<<20 || staged.LocalPath != layout.InputAsset(staged.Digest) {
		t.Fatalf("staged=%#v", staged)
	}
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 8<<20 {
		t.Fatalf("64 MiB stage allocated %d B", allocated)
	}
	if problem := Verify(staged, 128<<20); problem != nil {
		t.Fatal(problem)
	}
}
