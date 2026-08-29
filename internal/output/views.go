package output

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"

	"github.com/cozy-creator/cozy/internal/units"
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
	Fields    []Field
	AllFields []Field
	Notes     []string
	Next      []string
}

type List struct {
	Name       string
	Fields     []string
	AllFields  []string
	Rows       []map[string]string
	Total      int
	Aggregates []Field
	Notes      []string
	Next       []string
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
	if data == nil {
		data = map[string]any{}
	}
	if len(r.Notes) > 0 {
		data["notes"] = copyStrings(r.Notes)
	}
	if next := trimNext(r.Next); len(next) > 0 {
		data["next"] = next
	}
	if mode.Human && !mode.JSON {
		return writeHumanRecord(w, fields, data, r.Notes, r.Next, mode.Full)
	}
	return Write(w, data, mode)
}

func (l List) Emit(w io.Writer, mode Mode) error {
	if strings.TrimSpace(l.Name) == "" {
		return fmt.Errorf("output list name is required")
	}
	columns, err := l.Columns(mode)
	if err != nil {
		return err
	}
	shown := l.Rows
	if !mode.Full && len(shown) > rowCap {
		shown = shown[:rowCap]
	}
	document := map[string]any{}
	if len(columns) == 1 {
		values := make([]string, 0, len(shown))
		for _, source := range shown {
			values = append(values, Elide(source[columns[0]], cellCap, mode.Full))
		}
		document[l.Name] = values
	} else {
		rows := make([]map[string]string, 0, len(shown))
		for _, source := range shown {
			row := make(map[string]string, len(columns))
			for _, column := range columns {
				row[column] = Elide(source[column], cellCap, mode.Full)
			}
			rows = append(rows, row)
		}
		document[l.Name] = rows
	}
	total := l.Total
	if total < len(l.Rows) {
		total = len(l.Rows)
	}
	if omitted := total - len(shown); omitted > 0 {
		document["omitted"] = omitted
	}
	for _, aggregate := range l.Aggregates {
		if aggregate.K == l.Name || aggregate.K == "count" || aggregate.K == "results" {
			continue
		}
		if _, exists := document[aggregate.K]; exists {
			return fmt.Errorf("output field %q occurs twice", aggregate.K)
		}
		value := aggregate.V
		if text, ok := value.(string); ok {
			value = Elide(text, fieldCap, mode.Full)
		}
		document[aggregate.K] = value
	}
	if len(l.Notes) > 0 {
		document["notes"] = copyStrings(l.Notes)
	}
	if next := trimNext(l.Next); len(next) > 0 {
		document["next"] = next
	}
	if mode.Human && !mode.JSON {
		return writeHumanList(w, l, columns, shown, total, document, mode.Full)
	}
	return Write(w, document, mode)
}

func writeHumanRecord(w io.Writer, fields []Field, data map[string]any, notes, next []string, full bool) error {
	var rendered strings.Builder
	width := 0
	for _, field := range fields {
		width = max(width, utf8.RuneCountInString(field.K))
	}
	for _, field := range fields {
		if values, ok := data[field.K].([]string); ok {
			rendered.WriteString(field.K)
			rendered.WriteByte(':')
			if len(values) == 0 {
				rendered.WriteString(strings.Repeat(" ", width-utf8.RuneCountInString(field.K)+1))
				rendered.WriteString("-\n")
				continue
			}
			rendered.WriteByte('\n')
			for _, value := range values {
				rendered.WriteString("  - ")
				rendered.WriteString(humanValue(value, full))
				rendered.WriteByte('\n')
			}
			continue
		}
		rendered.WriteString(field.K)
		rendered.WriteByte(':')
		rendered.WriteString(strings.Repeat(" ", width-utf8.RuneCountInString(field.K)+1))
		rendered.WriteString(humanValue(data[field.K], full))
		rendered.WriteByte('\n')
	}
	writeHumanGuidance(&rendered, notes, next)
	_, err := io.WriteString(w, rendered.String())
	return err
}

func writeHumanList(w io.Writer, list List, columns []string, shown []map[string]string,
	total int, document map[string]any, full bool,
) error {
	var rendered strings.Builder
	if len(shown) == 0 {
		fmt.Fprintf(&rendered, "No %s found.\n", list.Name)
	} else if len(columns) == 1 {
		for _, row := range shown {
			rendered.WriteString("- ")
			rendered.WriteString(orDash(Elide(row[columns[0]], cellCap, full)))
			rendered.WriteByte('\n')
		}
	} else {
		writeHumanTable(&rendered, columns, shown, full)
	}
	if omitted := total - len(shown); omitted > 0 {
		fmt.Fprintf(&rendered, "%d more not shown. Use --full to show all.\n", omitted)
	}
	wroteAggregate := false
	for _, aggregate := range list.Aggregates {
		if aggregate.K == list.Name || aggregate.K == "count" || aggregate.K == "results" {
			continue
		}
		if value, ok := document[aggregate.K]; ok {
			if !wroteAggregate && rendered.Len() > 0 {
				rendered.WriteByte('\n')
			}
			fmt.Fprintf(&rendered, "%s: %s\n", aggregate.K, humanValue(value, full))
			wroteAggregate = true
		}
	}
	writeHumanGuidance(&rendered, list.Notes, list.Next)
	_, err := io.WriteString(w, rendered.String())
	return err
}

func writeHumanTable(rendered *strings.Builder, columns []string, rows []map[string]string, full bool) {
	widths := make([]int, len(columns))
	for i, column := range columns {
		widths[i] = utf8.RuneCountInString(strings.ToUpper(column))
	}
	values := make([][]string, len(rows))
	for i, row := range rows {
		values[i] = make([]string, len(columns))
		for j, column := range columns {
			value := orDash(Elide(row[column], cellCap, full))
			values[i][j] = value
			widths[j] = max(widths[j], utf8.RuneCountInString(value))
		}
	}
	headings := make([]string, len(columns))
	for i, column := range columns {
		headings[i] = strings.ToUpper(column)
	}
	writeHumanRow(rendered, headings, widths)
	for _, row := range values {
		writeHumanRow(rendered, row, widths)
	}
}

func writeHumanRow(rendered *strings.Builder, values []string, widths []int) {
	for i, value := range values {
		rendered.WriteString(value)
		if i < len(values)-1 {
			rendered.WriteString(strings.Repeat(" ", widths[i]-utf8.RuneCountInString(value)+2))
		}
	}
	rendered.WriteByte('\n')
}

func writeHumanGuidance(rendered *strings.Builder, notes, next []string) {
	next = trimNext(next)
	if len(notes) == 0 && len(next) == 0 {
		return
	}
	if rendered.Len() > 0 {
		rendered.WriteByte('\n')
	}
	for _, note := range notes {
		fmt.Fprintf(rendered, "Note: %s\n", strings.TrimSpace(note))
	}
	for _, command := range next {
		fmt.Fprintf(rendered, "Next: %s\n", strings.TrimSpace(command))
	}
}

func humanValue(value any, full bool) string {
	if value == nil {
		return "-"
	}
	var text string
	switch scalar := value.(type) {
	case string:
		text = scalar
	case bool, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64,
		float32, float64, json.Number:
		text = fmt.Sprint(scalar)
	default:
		encoded, err := json.Marshal(value)
		if err != nil {
			text = fmt.Sprint(value)
		} else {
			text = string(encoded)
		}
	}
	return orDash(Elide(strings.ReplaceAll(text, "\n", " "), fieldCap, full))
}

func orDash(value string) string {
	if strings.TrimSpace(value) == "" {
		return "-"
	}
	return value
}

func (r Record) selected(mode Mode) ([]Field, error) {
	available := r.AllFields
	if len(available) == 0 {
		available = r.Fields
	}
	if len(mode.Fields) == 0 {
		if mode.Full {
			return available, nil
		}
		return r.Fields, nil
	}
	have := make(map[string]Field, len(available))
	names := make([]string, 0, len(available))
	for _, field := range available {
		have[field.K] = field
		names = append(names, field.K)
	}
	selected := make([]Field, 0, len(mode.Fields))
	for _, name := range mode.Fields {
		field, ok := have[name]
		if !ok {
			return nil, NewError(Usage, "output.field_unknown", fmt.Sprintf("unknown output field %q", name)).
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
			return nil, NewError(Usage, "output.field_unknown", fmt.Sprintf("unknown output field %q", field)).
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
