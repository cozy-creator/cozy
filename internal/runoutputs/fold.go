// Package runoutputs folds a run's output log into its items: the one place that says what an
// output's current bytes are, which revision they are, and how they changed.
//
// A run's outputs are items with stable ids, `<run>/<output>` for a single output and
// `<run>/<output>/<i>` (1-based) for a list element. Each `product` entry the machine journals
// is a new revision of its item, rewritten in place: it appends when its parts extend the
// previous revision's parts, and otherwise replaces them. `rev` is the item's 1-based entry
// ordinal and the ETag is `"r<rev>"`. The content digest is exposed on the final revision only
// (progressive-outputs.md §1–2).
package runoutputs

import (
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy/internal/resultfiles"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Part is one stored piece of an item's bytes; the bytes are the parts in order.
type Part struct {
	Digest     string
	Length     int64
	DurationUs uint64
	// Source is the machine hold the part is read through (ReadByteTreeObject).
	Source *pb.NativeByteRetentionRequest
}

// Revision is an item's bytes as one log entry left them.
type Revision struct {
	Rev      uint32
	Sequence uint64
	Length   int64
	// AppendedFrom is set exactly when this revision appends: the offset its new bytes start
	// at, which is the previous revision's length. A replace leaves it nil.
	AppendedFrom *int64
	// DurationUs is the media duration so far; 0 for bytes without a timeline.
	DurationUs uint64
	// Digest is the sha256 of the whole bytes. Show it only on the final revision.
	Digest    string
	MediaType string
	Label     string
	// Parts are the current bytes in order; a single-source product is one part.
	Parts []Part
}

// Item is one output of a run at its current revision.
type Item struct {
	ID string
	// OutputIndex is the item's 0-based position in the order items were first added.
	OutputIndex int
	Output      string
	// Index is the list element's 1-based position; 0 for a single output.
	Index   uint32
	List    bool
	Type    string
	Current Revision
	// History is the item's revisions' lengths and parts: enough to answer an If-Range for
	// any earlier rev without holding its bytes.
	History []Revision
}

// Name is the item's file stem: `<run>-<output>` or `<run>-<output>-<i>`.
func (i Item) Name(run string) string {
	if i.List {
		return fmt.Sprintf("%s-%s-%d", run, i.Output, i.Index)
	}
	return run + "-" + i.Output
}

// TypeOf is an item's Responses-style type, from its media type.
func TypeOf(mediaType string) string {
	switch {
	case mediaType == resultfiles.TreeMediaType:
		return "tree"
	case strings.HasPrefix(mediaType, "video/"):
		return "video"
	case strings.HasPrefix(mediaType, "image/"):
		return "image"
	case strings.HasPrefix(mediaType, "audio/"):
		return "audio"
	}
	return "file"
}
