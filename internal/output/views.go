package output

import (
	"fmt"
	"io"
	"strings"
	"unicode/utf8"

	"github.com/cozy-creator/cozy-creator/internal/units"
)

const (
	rowCap   = 20
	cellCap  = 72
	fieldCap = 900
)

// Field, Record, and List keep handlers declarative while both surfaces collapse
// into the same Result at the output boundary.
type Field struct {
	K string
	V any
}

type Record struct {
	Kind   string
	Fields []Field
	Notes  []string
	Next   []string
}

type List struct {
	Kind       string
	Fields     []string
	AllFields  []string
	AllRows    bool
	Rows       []map[string]string
	Aggregates []Field
	Empty      string
	Notes      []string
	Next       []string
}

type ListData struct {
	Fields     []string            `json:"fields"`
	Rows       []map[string]string `json:"rows"`
	Count      int                 `json:"count"`
	Omitted    int                 `json:"omitted,omitempty"`
	Empty      string              `json:"empty,omitempty"`
	Aggregates map[string]any      `json:"aggregates,omitempty"`
}

func (r Record) WithDefaultNext(next []string) Document {
	if len(r.Next) == 0 {
		r.Next = trimNext(next)
	}
	return r
}

func (l List) WithDefaultNext(next []string) Document {
	if len(l.Next) == 0 {
		l.Next = trimNext(next)
	}
	return l
}

func (r Record) Emit(w io.Writer, mode Mode) error {
	fields, err := r.selected(mode)
	if err != nil {
		return err
	}
	data, err := fieldMap(fields, mode.Full)
	if err != nil {
		return err
	}
	result := Success(r.Kind, data)
	result.Notes, result.Next = r.Notes, trimNext(r.Next)
	return Write(w, result, mode)
}

func (l List) Emit(w io.Writer, mode Mode) error {
	columns, err := l.Columns(mode)
	if err != nil {
		return err
	}
	shown := l.Rows
	if !mode.Full && !l.AllRows && len(shown) > rowCap {
		shown = shown[:rowCap]
	}
	rows := make([]map[string]string, 0, len(shown))
	for _, source := range shown {
		row := make(map[string]string, len(columns))
		for _, column := range columns {
			row[column] = Elide(source[column], cellCap, mode.Full)
		}
		rows = append(rows, row)
	}
	aggregates, err := fieldMap(l.Aggregates, mode.Full)
	if err != nil {
		return err
	}
	empty := ""
	if len(l.Rows) == 0 {
		empty = l.Empty
		if empty == "" {
			empty = "No " + l.Kind + "."
		}
	}
	data := ListData{
		Fields: copyStrings(columns), Rows: rows, Count: len(l.Rows),
		Omitted: len(l.Rows) - len(rows), Empty: empty, Aggregates: aggregates,
	}
	result := Success(l.Kind, data)
	result.Notes, result.Next = l.Notes, trimNext(l.Next)
	return Write(w, result, mode)
}

func (r Record) selected(mode Mode) ([]Field, error) {
	if len(mode.Fields) == 0 {
		return r.Fields, nil
	}
	have := make(map[string]Field, len(r.Fields))
	names := make([]string, 0, len(r.Fields))
	for _, field := range r.Fields {
		have[field.K] = field
		names = append(names, field.K)
	}
	selected := make([]Field, 0, len(mode.Fields))
	for _, name := range mode.Fields {
		field, ok := have[name]
		if !ok {
			return nil, NewError(Usage, "output.field_unknown", fmt.Sprintf("unknown field %q for %s", name, r.Kind)).
				WithRemedy("available fields: " + strings.Join(names, ", "))
		}
		selected = append(selected, field)
	}
	return selected, nil
}

// Columns resolves --fields first, then --full, then the compact defaults.
func (l List) Columns(mode Mode) ([]string, error) {
	if len(mode.Fields) == 0 {
		if mode.Full {
			return l.AllFields, nil
		}
		return l.Fields, nil
	}
	for _, field := range mode.Fields {
		if !contains(l.AllFields, field) {
			return nil, NewError(Usage, "output.field_unknown", fmt.Sprintf("unknown field %q for %s", field, l.Kind)).
				WithRemedy("available fields: " + strings.Join(l.AllFields, ", "))
		}
	}
	return mode.Fields, nil
}

func fieldMap(fields []Field, full bool) (map[string]any, error) {
	if len(fields) == 0 {
		return nil, nil
	}
	result := make(map[string]any, len(fields))
	for _, field := range fields {
		if _, exists := result[field.K]; exists {
			return nil, fmt.Errorf("output field %q occurs twice", field.K)
		}
		value := field.V
		if text, ok := value.(string); ok {
			value = Elide(text, fieldCap, full)
		}
		result[field.K] = value
	}
	return result, nil
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func copyStrings(values []string) []string {
	result := make([]string, len(values))
	copy(result, values)
	return result
}

// Bytes is the shared binary byte-count rendering.
func Bytes(n int64) string {
	return units.Bytes(n)
}

// Elide bounds a string by runes and reports the original size.
func Elide(value string, limit int, full bool) string {
	total := utf8.RuneCountInString(value)
	if full || total <= limit || limit < 2 {
		return value
	}
	short := fmt.Sprintf("%s… (truncated, %d chars total, --full)", string([]rune(value)[:limit-1]), total)
	if utf8.RuneCountInString(short) >= total {
		return value
	}
	return short
}
