package records

import (
	"database/sql"
	"encoding/json"
	"slices"

	"github.com/cozy-creator/cozy/internal/exit"
)

// A run's outputs as OpenAI Responses-style events (progressive-outputs.md §2). Each item is
// added once, gets one delta per revision, and is done when it can no longer change: a list
// element at once, a single output at the run's terminal.
const (
	OutputItemAdded = "output_item.added"
	OutputItemDelta = "output_item.delta"
	OutputItemDone  = "output_item.done"
)

// Product is one revision of a run's output item (progressive-outputs.md §1): the machine's
// `product` entry as the shared fold (internal/runoutputs) reads it, and the stable file this
// client keeps it in. An item is rewritten in place: each revision appends to its bytes or
// replaces them.
type Product struct {
	Sequence uint64 `json:"sequence"`
	// Item is the stable id, `<run>/<output>` or `<run>/<output>/<i>` (i 1-based).
	Item string `json:"item_id"`
	// OutputIndex is the item's 0-based position in the order items were first added.
	OutputIndex int `json:"output_index"`
	// Type is the item's kind: video, image, audio, file or tree.
	Type   string `json:"type"`
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

// Products are a run's output revisions in log order: its output_item.delta events.
func (s *Store) Products(request string) ([]Product, *exit.Error) {
	return products(s.db, request)
}

func products(q interface {
	Query(string, ...any) (*sql.Rows, error)
}, request string) ([]Product, *exit.Error) {
	rows, err := q.Query(`SELECT payload FROM request_events WHERE request_id=? AND type=? ORDER BY seq`, request, OutputItemDelta)
	if err != nil {
		return nil, exit.Internalf("cannot read the run's outputs: %s", err)
	}
	defer rows.Close()
	var result []Product
	for rows.Next() {
		var raw []byte
		var product Product
		if rows.Scan(&raw) != nil || json.Unmarshal(raw, &product) != nil {
			return nil, exit.Internalf("a recorded output revision is unreadable")
		}
		result = append(result, product)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish reading the run's outputs: %s", err)
	}
	return result, nil
}

// OutputItem is an item as clients see it: output_item.added/done's `item`, and each entry
// of a run's `output`. Its sha256 is shown only once the item is done.
type OutputItem struct {
	ID          string `json:"id"`
	Type        string `json:"type"`
	Name        string `json:"name"`
	Index       uint32 `json:"index,omitempty"`
	Label       string `json:"label,omitempty"`
	MediaType   string `json:"media_type"`
	Status      string `json:"status"`
	Rev         uint32 `json:"rev,omitempty"`
	Length      int64  `json:"length,omitempty"`
	DurationUs  uint64 `json:"duration_us,omitempty"`
	Sha256      string `json:"sha256,omitempty"`
	Path        string `json:"path,omitempty"`
	OutputIndex int    `json:"-"`
}

// OutputDelta is output_item.delta as clients see it: a revision, whose bytes are fetched.
type OutputDelta struct {
	ItemID       string `json:"item_id"`
	OutputIndex  int    `json:"output_index"`
	Rev          uint32 `json:"rev"`
	Length       int64  `json:"length"`
	AppendedFrom *int64 `json:"appended_from,omitempty"`
	DurationUs   uint64 `json:"duration_us,omitempty"`
	Label        string `json:"label,omitempty"`
}

// Delta is this revision as its output_item.delta.
func (p Product) Delta() OutputDelta {
	return OutputDelta{ItemID: p.Item, OutputIndex: p.OutputIndex, Rev: p.Rev, Length: p.Length,
		AppendedFrom: p.AppendedFrom, DurationUs: p.DurationUs, Label: p.Label}
}

// View is the item at this revision with `status`; a done item (completed or incomplete)
// shows its sha256.
func (p Product) View(status string) OutputItem {
	item := OutputItem{ID: p.Item, Type: p.Type, Name: p.Output, Label: p.Label, MediaType: p.MediaType, Status: status,
		Rev: p.Rev, Length: p.Length, DurationUs: p.DurationUs, Path: p.Path, OutputIndex: p.OutputIndex}
	if p.Op == ProductAppend {
		item.Index = p.Index + 1
	}
	if status != "in_progress" {
		item.Sha256 = p.Digest
	}
	return item
}

// Output is the run's output: every item at its last revision. A list element is completed
// once added; a single output is in progress until the run settles in `state`. An item its
// failed export never put in its folder is undelivered.
func (s *Store) Output(request, state string) ([]OutputItem, *exit.Error) {
	revisions, problem := s.Products(request)
	if problem != nil {
		return nil, problem
	}
	export, problem := s.OutputExportOf(request)
	if problem != nil {
		return nil, problem
	}
	status := "in_progress"
	switch state {
	case "succeeded":
		status = "completed"
	case "failed", "canceled":
		status = "incomplete"
	}
	var output []OutputItem
	for _, product := range Fold(revisions) {
		item := product.View(status)
		if product.Op == ProductAppend {
			item = product.View("completed")
		}
		if export != nil && export.State == "failed" && product.Path != "" && !slices.Contains(export.PublishedPaths, product.Path) {
			item.Status = "undelivered"
		}
		output = append(output, item)
	}
	return output, nil
}

// outputItemEvents are the events one revision records: added on its item's first revision,
// its delta, and done at once for a list element, which never changes.
func outputItemEvents(p Product) []itemEvent {
	var events []itemEvent
	if p.Rev == 1 {
		events = append(events, itemEvent{OutputItemAdded, outputItemEnvelope{OutputIndex: p.OutputIndex, Item: p.View("in_progress")}})
	}
	events = append(events, itemEvent{OutputItemDelta, p})
	if p.Op == ProductAppend {
		events = append(events, itemEvent{OutputItemDone, outputItemEnvelope{OutputIndex: p.OutputIndex, Item: p.View("completed")}})
	}
	return events
}

type itemEvent struct {
	kind    string
	payload any
}

// outputItemEnvelope is output_item.added and output_item.done's payload.
type outputItemEnvelope struct {
	OutputIndex int        `json:"output_index"`
	Item        OutputItem `json:"item"`
}

// finishOutputs records output_item.done for each single output at the run's terminal, and
// answers the run's output: every item at its last revision.
func finishOutputs(tx *sql.Tx, request string, attempt uint64, status string) ([]OutputItem, *exit.Error) {
	revisions, problem := products(tx, request)
	if problem != nil {
		return nil, problem
	}
	var output []OutputItem
	for _, product := range Fold(revisions) {
		item := product.View(status)
		if product.Op == ProductAppend {
			item = product.View("completed")
		} else {
			payload, _ := json.Marshal(outputItemEnvelope{OutputIndex: product.OutputIndex, Item: item})
			if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,?,?,?,?)`,
				request, OutputItemDone, attempt, string(payload), now()); err != nil {
				return nil, exit.Internalf("cannot record a finished output: %s", err)
			}
		}
		output = append(output, item)
	}
	return output, nil
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
