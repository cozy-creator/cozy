// Package render is the one output layer: compact structured text by default,
// `--json` for the full typed result. Truncation, aggregates, empty states and
// `next:` lines live here, per cozy-runtime-cli.md "Output".
package render

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"

	"github.com/cozy-creator/cozy-creator/internal/exit"
)

const (
	rowCap = 20 // lists cap at 20 rows without --full
	// Two caps, because the constraints differ (AXI 3). A list CELL is bounded by the
	// column it sits in; a record/lines VALUE is bounded only by what an agent can read,
	// and 72 guillotines a job error or a result blob. 900 sits in AXI's 500–1500 band.
	cellCap   = 72
	fieldCap  = 900
	maxNext   = 2 // at most two next: lines
	sepAggreg = " · "
)

// Document is one emittable surface. WithDefaultNext lets the ONE emit seam hand a
// document its verb's manifest-declared next steps (AXI 9) without a handler that
// computed its own state-dependent ones losing them.
type Document interface {
	Emit(w io.Writer, m Mode) error
	WithDefaultNext(next []string) Document
}

// Mode carries the global presentation flags.
type Mode struct {
	JSON   bool
	Full   bool
	Fields []string
}

type Field struct {
	K string
	V any
}

// Record is aligned `key: value` output (status, version, one object).
type Record struct {
	Kind   string
	Fields []Field
	Notes  []string
	Next   []string
}

// Lines is a one-item-per-line surface (capabilities tokens).
type Lines struct {
	Kind  string
	Key   string
	Items []string
	Empty string
	Extra []Field
	Notes []string
	Next  []string
}

// List is a columnar listing with footer aggregates.
type List struct {
	Kind       string
	Fields     []string // default columns (3-4)
	AllFields  []string // every available column
	Rows       []map[string]string
	Aggregates []Field
	Empty      string
	Notes      []string
	Next       []string
}

func (r Record) WithDefaultNext(n []string) Document {
	if len(r.Next) == 0 {
		r.Next = n
	}
	return r
}

func (l Lines) WithDefaultNext(n []string) Document {
	if len(l.Next) == 0 {
		l.Next = n
	}
	return l
}

func (l List) WithDefaultNext(n []string) Document {
	if len(l.Next) == 0 {
		l.Next = n
	}
	return l
}

func trimNext(n []string) []string {
	if len(n) > maxNext {
		return n[:maxNext]
	}
	return n
}

func text(v any) string {
	switch t := v.(type) {
	case nil:
		return "unknown"
	case string:
		return t
	case bool:
		if t {
			return "true"
		}
		return "false"
	case []string:
		if len(t) == 0 {
			return "none"
		}
		return strings.Join(t, ", ")
	default:
		return fmt.Sprint(v)
	}
}

// Bytes is the ONE byte-count rendering every cozy surface shares.
func Bytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%dB", n)
	}
	div, exp := int64(unit), 0
	for v := n / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f%ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

// Elide shortens a value to `limit` RUNES and reports its TOTAL length, so the agent knows
// the size of what it is missing (AXI 3) rather than a remainder it cannot act on. Counting
// is rune-based throughout: the old mix of len() bytes with rune indexing both over-reported
// and over-cut a multibyte value. The hint fires only when something was actually cut.
func Elide(s string, limit int, full bool) string {
	total := utf8.RuneCountInString(s)
	if full || total <= limit {
		return s
	}
	kept := string([]rune(s)[:limit-1])
	out := fmt.Sprintf("%s… (truncated, %d chars total, --full)", kept, total)
	if utf8.RuneCountInString(out) >= total {
		return s // eliding would print more, not less
	}
	return out
}

func writeTail(w io.Writer, notes, next []string) {
	for _, n := range notes {
		fmt.Fprintf(w, "note: %s\n", n)
	}
	for _, n := range trimNext(next) {
		fmt.Fprintf(w, "next: %s\n", n)
	}
}

func writeJSON(w io.Writer, ordered []Field) error {
	var b strings.Builder
	seen := make(map[string]bool, len(ordered))
	b.WriteByte('{')
	for i, f := range ordered {
		// A document key emitted twice is a silent lie to every machine consumer:
		// most JSON readers keep the LAST value, so `.kind` would answer with the
		// wrong one and nothing would say so. Found live on cl-011's repo document,
		// whose own "kind" collided with the envelope's. Refuse instead.
		if seen[f.K] {
			return fmt.Errorf("document key %q is emitted twice; a JSON reader would keep only one", f.K)
		}
		seen[f.K] = true
		if i > 0 {
			b.WriteByte(',')
		}
		k, err := json.Marshal(f.K)
		if err != nil {
			return err
		}
		v, err := json.Marshal(f.V)
		if err != nil {
			return err
		}
		b.Write(k)
		b.WriteByte(':')
		b.Write(v)
	}
	b.WriteByte('}')
	_, err := fmt.Fprintln(w, b.String())
	return err
}

func tail(f []Field, notes, next []string) []Field {
	if len(notes) > 0 {
		f = append(f, Field{"notes", notes})
	}
	f = append(f, Field{"next", trimNext(orEmpty(next))})
	return f
}

func orEmpty(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

// selected narrows a record to `--fields`, in the order asked for. A record is the same
// surface a listing is, one row wide: a flag that parses and prints everything anyway is
// the bug cl-010 named, so an unknown name refuses here exactly as it does for a List.
func (r Record) selected(m Mode) ([]Field, error) {
	if len(m.Fields) == 0 {
		return r.Fields, nil
	}
	have := map[string]Field{}
	names := make([]string, 0, len(r.Fields))
	for _, f := range r.Fields {
		have[f.K] = f
		names = append(names, f.K)
	}
	out := make([]Field, 0, len(m.Fields))
	for _, want := range m.Fields {
		f, ok := have[want]
		if !ok {
			return nil, exit.Usagef("unknown field %q for `%s`", want, r.Kind).
				WithRemedy("available fields: %s", strings.Join(names, ", "))
		}
		out = append(out, f)
	}
	return out, nil
}

func (r Record) Emit(w io.Writer, m Mode) error {
	fields, err := r.selected(m)
	if err != nil {
		return err
	}
	if m.JSON {
		f := []Field{{"kind", r.Kind}}
		f = append(f, fields...)
		return writeJSON(w, tail(f, r.Notes, r.Next))
	}
	width := 0
	for _, f := range fields {
		if len(f.K) > width {
			width = len(f.K)
		}
	}
	for _, f := range fields {
		fmt.Fprintf(w, "%-*s %s\n", width+1, f.K+":", Elide(text(f.V), fieldCap, m.Full))
	}
	writeTail(w, r.Notes, r.Next)
	return nil
}

func (l Lines) Emit(w io.Writer, m Mode) error {
	if m.JSON {
		f := []Field{{"kind", l.Kind}, {l.Key, orEmpty(l.Items)}, {"count", len(l.Items)}}
		f = append(f, l.Extra...)
		return writeJSON(w, tail(f, l.Notes, l.Next))
	}
	if len(l.Items) == 0 {
		fmt.Fprintln(w, l.Empty)
	}
	shown := l.Items
	if !m.Full && len(shown) > rowCap {
		shown = shown[:rowCap]
	}
	for _, it := range shown {
		fmt.Fprintln(w, it)
	}
	if len(shown) < len(l.Items) {
		fmt.Fprintf(w, "+%d more — --full\n", len(l.Items)-len(shown))
	}
	for _, e := range l.Extra {
		fmt.Fprintf(w, "%s: %s\n", e.K, Elide(text(e.V), fieldCap, m.Full))
	}
	writeTail(w, l.Notes, l.Next)
	return nil
}

// Columns resolves the columns to print: --fields wins, else the defaults.
// An unknown field name is a typed usage refusal.
func (l List) Columns(m Mode) ([]string, error) {
	if len(m.Fields) == 0 {
		if m.Full {
			return l.AllFields, nil
		}
		return l.Fields, nil
	}
	for _, f := range m.Fields {
		if !contains(l.AllFields, f) {
			return nil, exit.Usagef("unknown field %q for `%s`", f, l.Kind).
				WithRemedy("available fields: %s", strings.Join(l.AllFields, ", ")).
				WithNext("cozy " + l.Kind + " --full")
		}
	}
	return m.Fields, nil
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

func (l List) Emit(w io.Writer, m Mode) error {
	cols, err := l.Columns(m)
	if err != nil {
		return err
	}
	if m.JSON {
		rows := l.Rows
		if rows == nil {
			rows = []map[string]string{}
		}
		agg := map[string]string{}
		for _, a := range l.Aggregates {
			agg[a.K] = text(a.V)
		}
		f := []Field{
			{"kind", l.Kind},
			{"count", len(rows)},
			{"fields", l.AllFields},
			{"rows", rows},
			{"aggregates", agg},
		}
		return writeJSON(w, tail(f, l.Notes, l.Next))
	}
	if len(l.Rows) == 0 {
		fmt.Fprintln(w, l.Empty)
		l.footer(w, m)
		return nil
	}
	shown := l.Rows
	if !m.Full && len(shown) > rowCap {
		shown = shown[:rowCap]
	}
	width := make([]int, len(cols))
	for i, c := range cols {
		width[i] = len(c)
	}
	cells := make([][]string, 0, len(shown))
	for _, r := range shown {
		row := make([]string, len(cols))
		for i, c := range cols {
			row[i] = Elide(r[c], cellCap, m.Full)
			if len(row[i]) > width[i] {
				width[i] = len(row[i])
			}
		}
		cells = append(cells, row)
	}
	printRow(w, cols, width)
	for _, row := range cells {
		printRow(w, row, width)
	}
	if len(shown) < len(l.Rows) {
		fmt.Fprintf(w, "+%d more — --full\n", len(l.Rows)-len(shown))
	}
	l.footer(w, m)
	return nil
}

func (l List) footer(w io.Writer, m Mode) {
	if len(l.Aggregates) > 0 {
		parts := make([]string, 0, len(l.Aggregates))
		for _, a := range l.Aggregates {
			parts = append(parts, fmt.Sprintf("%s: %s", a.K, text(a.V)))
		}
		fmt.Fprintln(w, strings.Join(parts, sepAggreg))
	}
	writeTail(w, l.Notes, l.Next)
}

func printRow(w io.Writer, cells []string, width []int) {
	out := make([]string, len(cells))
	for i, c := range cells {
		if i == len(cells)-1 {
			out[i] = c
		} else {
			out[i] = fmt.Sprintf("%-*s", width[i], c)
		}
	}
	fmt.Fprintln(w, strings.TrimRight(strings.Join(out, "  "), " "))
}

// EmitError renders a typed error onto the SAME stream the data would have used —
// stdout — in both modes (AXI 6). A refusal is structured output the agent must read and
// act on; on stderr it left an agent capturing stdout with an empty buffer and a bare
// exit code. stderr carries progress and diagnostics, and this layer never writes there.
func EmitError(w io.Writer, e *exit.Error, m Mode) {
	if m.JSON {
		_ = writeJSON(w, []Field{
			{"error", map[string]any{
				"code":    int(e.Code),
				"name":    e.ErrName(),
				"message": e.Message,
				"remedy":  e.Remedy,
			}},
			{"next", trimNext(orEmpty(e.Next))},
		})
		return
	}
	fmt.Fprintf(w, "error(%s): %s\n", e.ErrName(), e.Message)
	if e.Remedy != "" {
		fmt.Fprintf(w, "remedy: %s\n", e.Remedy)
	}
	for _, n := range trimNext(e.Next) {
		fmt.Fprintf(w, "next: %s\n", n)
	}
}
