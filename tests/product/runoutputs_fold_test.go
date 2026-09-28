package producttest

import (
	"crypto/sha256"
	"testing"

	"github.com/cozy-creator/cozy/internal/runoutputs"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func foldRef(data string) *pb.Ref {
	sum := sha256.Sum256([]byte(data))
	return &pb.Ref{Digest: sum[:], Length: uint64(len(data))}
}

// film is revision k of a growing fMP4: an init and k fragments of 0.5 s.
func foldFilm(k int) *pb.RunProduct {
	parts := []*pb.RunProductPart{{Content: foldRef("init")}}
	whole := "init"
	for f := 1; f <= k; f++ {
		fragment := string(rune('a'+f)) + "-fragment"
		parts = append(parts, &pb.RunProductPart{Content: foldRef(fragment), DurationUs: 500_000})
		whole += fragment
	}
	return &pb.RunProduct{Output: "video", Op: pb.RunProductOp_RUN_PRODUCT_OP_SET, Content: foldRef(whole), MediaType: "video/mp4", Parts: parts}
}

func foldImage(data string) *pb.RunProduct {
	return &pb.RunProduct{Output: "image", Op: pb.RunProductOp_RUN_PRODUCT_OP_SET, Content: foldRef(data), MediaType: "image/png"}
}

func foldReference(index uint32, data string) *pb.RunProduct {
	return &pb.RunProduct{Output: "references", Op: pb.RunProductOp_RUN_PRODUCT_OP_APPEND, Index: index, Content: foldRef(data), MediaType: "image/png"}
}

func foldAdd(t *testing.T, fold *runoutputs.Fold, sequence uint64, product *pb.RunProduct) (runoutputs.Item, bool) {
	t.Helper()
	item, added, err := fold.Add(sequence, product)
	if err != nil {
		t.Fatal(err)
	}
	return item, added
}

// A growing video appends: each revision's bytes begin with the last one's, from its length.
func TestAGrowingVideoAppendsFromTheLastLength(t *testing.T) {
	fold := runoutputs.New("1564")
	first, added := foldAdd(t, fold, 3, foldFilm(1))
	if !added || first.ID != "1564/video" || first.Type != "video" || first.Current.Rev != 1 || first.Current.AppendedFrom != nil {
		t.Fatalf("first revision: %+v", first)
	}
	second, added := foldAdd(t, fold, 5, foldFilm(2))
	if added || second.Current.Rev != 2 || second.Current.AppendedFrom == nil || *second.Current.AppendedFrom != first.Current.Length {
		t.Fatalf("second revision did not append from %d: %+v", first.Current.Length, second.Current)
	}
	third, _ := foldAdd(t, fold, 9, foldFilm(3))
	if *third.Current.AppendedFrom != second.Current.Length || third.Current.DurationUs != 1_500_000 || third.Current.ETag() != `"r3"` {
		t.Fatalf("third revision: %+v", third.Current)
	}
	if one, _ := third.Revision(1); !one.PrefixOf(third.Current) || third.Name("1564") != "1564-video" {
		t.Fatal("revision 1 is not a prefix of revision 3")
	}
}

// A revision whose bytes do not extend the last one's replaces them.
func TestADifferentRevisionReplaces(t *testing.T) {
	fold := runoutputs.New("7")
	foldAdd(t, fold, 1, foldImage("first"))
	second, _ := foldAdd(t, fold, 2, foldImage("second"))
	one, _ := second.Revision(1)
	if second.Current.Rev != 2 || second.Current.AppendedFrom != nil || one.PrefixOf(second.Current) || second.Current.DurationUs != 0 {
		t.Fatalf("a replaced image: %+v", second.Current)
	}
	// A growing video that starts over is a replace too.
	foldAdd(t, fold, 3, foldFilm(2))
	restart := foldFilm(1)
	restart.Parts[1].Content = foldRef("other-fragment")
	restart.Content = foldRef("initother-fragment")
	again, _ := foldAdd(t, fold, 4, restart)
	if again.Current.AppendedFrom != nil {
		t.Fatal("a rewritten video appended")
	}
}

// Each list element is its own item, 1-based, in the order items were first added.
func TestListElementsAreItemsInOrder(t *testing.T) {
	fold := runoutputs.New("1564")
	foldAdd(t, fold, 1, foldReference(0, "traveler"))
	foldAdd(t, fold, 2, foldFilm(1))
	second, added := foldAdd(t, fold, 3, foldReference(1, "harbor"))
	if !added || second.ID != "1564/references/2" || second.Index != 2 || !second.List || second.OutputIndex != 2 || second.Name("1564") != "1564-references-2" {
		t.Fatalf("the second reference: %+v", second)
	}
	items := fold.Items()
	if len(items) != 3 || items[0].ID != "1564/references/1" || items[1].ID != "1564/video" {
		t.Fatalf("items out of order: %+v", items)
	}
	if item, ok := fold.Item("references", 1); !ok || item.Current.Rev != 1 {
		t.Fatal("the first reference is not found by its output and index")
	}
}

// A replayed entry changes nothing; a malformed one is refused.
func TestReplaysAndMalformedEntries(t *testing.T) {
	fold := runoutputs.New("1")
	foldAdd(t, fold, 4, foldFilm(1))
	if item, added, err := fold.Add(4, foldFilm(2)); err != nil || added || item.ID != "" {
		t.Fatalf("a replay changed the fold: %+v %v", item, err)
	}
	if item, _ := fold.Item("video", 0); item.Current.Rev != 1 {
		t.Fatal("a replay made a revision")
	}
	broken := foldFilm(2)
	broken.Content.Length++
	if _, _, err := fold.Add(5, broken); err == nil {
		t.Fatal("parts that are not the whole bytes were accepted")
	}
}
