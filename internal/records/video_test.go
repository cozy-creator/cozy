package records

import (
	"sync"
	"sync/atomic"
	"testing"
)

func TestVideoCompositionRetainsPathFreeAssetIdentity(t *testing.T) {
	store := testStore(t)
	asset := CompositionAsset{Step: 1, FieldPath: "references.0.image",
		Digest: workflowDigest, Length: 4, MediaType: "image/png", Kind: "image"}
	row := VideoComposition{SourceDigest: workflowDigest,
		CreativePlanDigest: workflowDigest, CreativePlan: []byte(`{"format":"creative"}`),
		Assets: []CompositionAsset{asset}}
	created, fresh, problem := store.RecordVideoComposition(row)
	if problem != nil || !fresh || created.CreatedAt == "" {
		t.Fatalf("created=%#v fresh=%v problem=%v", created, fresh, problem)
	}
	replayed, fresh, problem := store.RecordVideoComposition(row)
	if problem != nil || fresh || replayed.CreativePlanDigest != workflowDigest {
		t.Fatalf("replayed=%#v fresh=%v problem=%v", replayed, fresh, problem)
	}
	loaded, problem := store.VideoCompositionByCreativePlan(workflowDigest)
	if problem != nil || loaded == nil || len(loaded.Assets) != 1 || loaded.Assets[0] != asset {
		t.Fatalf("loaded=%#v problem=%v", loaded, problem)
	}
	used, problem := store.AssetInUse(workflowDigest)
	if problem != nil || !used {
		t.Fatalf("composition asset ownership=%v problem=%v", used, problem)
	}
}

func TestSameSourceDigestMayResolveDifferentCreativePlan(t *testing.T) {
	store := testStore(t)
	other := "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
	for _, creative := range []string{workflowDigest, other} {
		_, fresh, problem := store.RecordVideoComposition(VideoComposition{
			SourceDigest: workflowDigest, CreativePlanDigest: creative,
			CreativePlan: []byte(creative), Assets: []CompositionAsset{},
		})
		if problem != nil || !fresh {
			t.Fatalf("creative=%s fresh=%v problem=%v", creative, fresh, problem)
		}
	}
}

func TestConcurrentEqualCompositionConverges(t *testing.T) {
	store := testStore(t)
	row := VideoComposition{SourceDigest: workflowDigest,
		CreativePlanDigest: workflowDigest, CreativePlan: []byte(`{"format":"creative"}`)}
	start := make(chan struct{})
	problems := make(chan any, 2)
	var fresh atomic.Int64
	var wait sync.WaitGroup
	for range 2 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			<-start
			_, created, problem := store.RecordVideoComposition(row)
			if problem != nil {
				problems <- problem
			}
			if created {
				fresh.Add(1)
			}
		}()
	}
	close(start)
	wait.Wait()
	close(problems)
	if fresh.Load() != 1 || len(problems) != 0 {
		t.Fatalf("fresh=%d problems=%v", fresh.Load(), <-problems)
	}
}
