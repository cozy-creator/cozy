package records

import (
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
)

// ProductType is the event a run's product becomes once this client holds its bytes.
const ProductType = "request.product"

// Product is one revision of a run's output item (progressive-outputs.md §1): the machine's
// `product` entry as the shared fold (internal/runoutputs) reads it, and the stable file this
// client keeps it in. An item is rewritten in place: each revision appends to its bytes or
// replaces them.
type Product struct {
	Sequence uint64 `json:"sequence"`
	// Item is the stable id, `<run>/<output>` or `<run>/<output>/<i>` (i 1-based).
	Item   string `json:"item"`
	Output string `json:"output"`
	Op     string `json:"op"`
	// Index is the machine's 0-based list position; 0 for a single output.
	Index uint32 `json:"index"`
	// Rev is the item's 1-based revision; its ETag is "r<rev>".
	Rev uint32 `json:"rev"`
	// Digest is the sha256 of the item's whole bytes at this revision. Clients see it only on
	// the final revision.
	Digest string `json:"digest"`
	Length int64  `json:"length"`
	// AppendedFrom is set exactly when this revision appends: where its new bytes start.
	AppendedFrom *int64 `json:"appended_from,omitempty"`
	// DurationUs is the media duration so far; 0 without a timeline.
	DurationUs uint64        `json:"duration_us,omitempty"`
	MediaType  string        `json:"media_type"`
	Label      string        `json:"label,omitempty"`
	Parts      []ProductPart `json:"parts,omitempty"`
	// ContentBytes is a tree item's member bytes: its manifest is the item's bytes.
	ContentBytes int64 `json:"content_bytes,omitempty"`
	// Path is the item's stable file in the run's outputs folder, the same at every revision.
	Path string `json:"path,omitempty"`
}

// ProductPart is one part of an item's bytes: the bytes are the parts concatenated.
type ProductPart struct {
	Digest     string `json:"digest"`
	Length     int64  `json:"length"`
	DurationUs uint64 `json:"duration_us,omitempty"`
}

const (
	ProductSet    = "set"
	ProductAppend = "append"
)

// Products are a run's products in log order.
func (s *Store) Products(request string) ([]Product, *exit.Error) {
	rows, err := s.db.Query(`SELECT payload FROM request_events WHERE request_id=? AND type=? ORDER BY seq`, request, ProductType)
	if err != nil {
		return nil, exit.Internalf("cannot read the run's products: %s", err)
	}
	defer rows.Close()
	var result []Product
	for rows.Next() {
		var raw []byte
		var product Product
		if rows.Scan(&raw) != nil || json.Unmarshal(raw, &product) != nil {
			return nil, exit.Internalf("a recorded product is unreadable")
		}
		result = append(result, product)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish reading the run's products: %s", err)
	}
	return result, nil
}

// Fold is the run's result as its log states it: each item at its last revision, in the
// order the items were first added.
func Fold(products []Product) []Product {
	latest := map[string]int{}
	var result []Product
	for _, product := range products {
		if at, ok := latest[product.Item]; ok {
			result[at] = product
			continue
		}
		latest[product.Item] = len(result)
		result = append(result, product)
	}
	return result
}
