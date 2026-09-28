package runoutputs

import (
	"crypto/sha256"
	"testing"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func ref(data string) *pb.Ref {
	sum := sha256.Sum256([]byte(data))
	return &pb.Ref{Digest: sum[:], Length: uint64(len(data))}
}

// film is revision k of a growing fMP4: an init and k fragments of 0.5 s.
func film(k int) *pb.RunProduct {
	parts := []*pb.RunProductPart{{Content: ref("init")}}
	whole := "init"
	for f := 1; f <= k; f++ {
		fragment := string(rune('a'+f)) + "-fragment"
		parts = append(parts, &pb.RunProductPart{Content: ref(fragment), DurationUs: 500_000})
		whole += fragment
	}
	return &pb.RunProduct{Output: "video", Op: pb.RunProductOp_RUN_PRODUCT_OP_SET, Content: ref(whole), MediaType: "video/mp4", Parts: parts}
}

func image(data string) *pb.RunProduct {
	return &pb.RunProduct{Output: "image", Op: pb.RunProductOp_RUN_PRODUCT_OP_SET, Content: ref(data), MediaType: "image/png"}
}

func reference(index uint32, data string) *pb.RunProduct {
	return &pb.RunProduct{Output: "references", Op: pb.RunProductOp_RUN_PRODUCT_OP_APPEND, Index: index, Content: ref(data), MediaType: "image/png"}
}

func add(t *testing.T, fold *Fold, sequence uint64, product *pb.RunProduct) (Item, bool) {
	t.Helper()
	item, added, err := fold.Add(sequence, product)
	if err != nil {
		t.Fatal(err)
	}
	return item, added
}

// A growing video appends: each revision's bytes begin with the last one's, from its length.
func TestAGrowingVideoAppendsFromTheLastLength(t *testing.T) {
	fold := New("1564")
	first, added := add(t, fold, 3, film(1))
	if !added || first.ID != "1564/video" || first.Type != "video" || first.Current.Rev != 1 || first.Current.AppendedFrom != nil {
		t.Fatalf("first revision: %+v", first)
	}
	second, added := add(t, fold, 5, film(2))
	if added || second.Current.Rev != 2 || second.Current.AppendedFrom == nil || *second.Current.AppendedFrom != first.Current.Length {
		t.Fatalf("second revision did not append from %d: %+v", first.Current.Length, second.Current)
	}
	third, _ := add(t, fold, 9, film(3))
	if *third.Current.AppendedFrom != second.Current.Length || third.Current.DurationUs != 1_500_000 || third.Current.ETag() != `"r3"` {
		t.Fatalf("third revision: %+v", third.Current)
	}
	if one, _ := third.Revision(1); !one.PrefixOf(third.Current) || third.Name("1564") != "1564-video" {
		t.Fatal("revision 1 is not a prefix of revision 3")
	}
}

// A revision whose bytes do not extend the last one's replaces them.
func TestADifferentRevisionReplaces(t *testing.T) {
	fold := New("7")
	add(t, fold, 1, image("first"))
	second, _ := add(t, fold, 2, image("second"))
	one, _ := second.Revision(1)
	if second.Current.Rev != 2 || second.Current.AppendedFrom != nil || one.PrefixOf(second.Current) || second.Current.DurationUs != 0 {
		t.Fatalf("a replaced image: %+v", second.Current)
	}
	// A growing video that starts over is a replace too.
	add(t, fold, 3, film(2))
	restart := film(1)
	restart.Parts[1].Content = ref("other-fragment")
	restart.Content = ref("initother-fragment")
	again, _ := add(t, fold, 4, restart)
	if again.Current.AppendedFrom != nil {
		t.Fatal("a rewritten video appended")
	}
}

// Each list element is its own item, 1-based, in the order items were first added.
func TestListElementsAreItemsInOrder(t *testing.T) {
	fold := New("1564")
	add(t, fold, 1, reference(0, "traveler"))
	add(t, fold, 2, film(1))
	second, added := add(t, fold, 3, reference(1, "harbor"))
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
	fold := New("1")
	add(t, fold, 4, film(1))
	if item, added, err := fold.Add(4, film(2)); err != nil || added || item.ID != "" {
		t.Fatalf("a replay changed the fold: %+v %v", item, err)
	}
	if item, _ := fold.Item("video", 0); item.Current.Rev != 1 {
		t.Fatal("a replay made a revision")
	}
	broken := film(2)
	broken.Content.Length++
	if _, _, err := fold.Add(5, broken); err == nil {
		t.Fatal("parts that are not the whole bytes were accepted")
	}
}
