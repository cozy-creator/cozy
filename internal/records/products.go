package records

import (
	"cmp"
	"encoding/json"
	"slices"

	"github.com/cozy-creator/cozy/internal/exit"
)

// ProductType is the event a run's product becomes once this client holds its bytes.
const ProductType = "request.product"

// Product is one entry of a run's output log (worker-protocol RunProduct) as this client
// holds it: bytes verified into the run's product store, and the file written for people.
// A single output's product is replaced by each SET; a list output grows by one APPEND.
type Product struct {
	Sequence  uint64        `json:"sequence"`
	Output    string        `json:"output"`
	Op        string        `json:"op"`
	Index     uint32        `json:"index"`
	Digest    string        `json:"digest"`
	Length    int64         `json:"length"`
	MediaType string        `json:"media_type"`
	Label     string        `json:"label,omitempty"`
	Parts     []ProductPart `json:"parts,omitempty"`
	// ContentBytes is a tree product's member bytes: its manifest is the product.
	ContentBytes int64 `json:"content_bytes,omitempty"`
	// Path is where the product was written for people: a list product's own file, or the
	// output's `.partial` file while the run is going. The final fold is the export.
	Path string `json:"path,omitempty"`
}

// ProductPart is one part of a composite product: its bytes are the parts concatenated.
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

// Fold is the run's result as its log states it: each single output's last SET and every
// list output's APPENDs, by output then index.
func Fold(products []Product) []Product {
	latest := map[string]Product{}
	var result []Product
	for _, product := range products {
		if product.Op == ProductAppend {
			result = append(result, product)
		} else {
			latest[product.Output] = product
		}
	}
	for _, product := range latest {
		result = append(result, product)
	}
	slices.SortFunc(result, func(a, b Product) int {
		return cmp.Or(cmp.Compare(a.Output, b.Output), cmp.Compare(a.Index, b.Index))
	})
	return result
}
