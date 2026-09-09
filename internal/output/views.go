package output

import (
	"encoding/json"
	"fmt"
	"io"
	"strconv"
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
	Name      string
	Fields    []string
	AllFields []string
	Rows      []map[string]string
	// TypedFields, TypedAllFields, and TypedRows optionally provide the logical
	// machine document separately from the terminal table. A table cell may be a
	// formatted duration, percentage, or dash; JSON/TOON must instead carry the
	// underlying number or omit an unavailable fact. When present, TypedRows has
	// exactly one row for every Rows entry and is used for every non-human mode.
	TypedFields    []string
	TypedAllFields []string
	TypedRows      []map[string]any
	// Bytes names the columns whose cells are byte counts, written as decimal integers
	// (`Int`): the terminal shows binary units, JSON carries the integer. An empty cell is
	// an absent fact in both.
	Bytes []string
	// Machine names columns every JSON document carries and the terminal never shows:
	// facts a program wants that would only widen a table.
	Machine    []string
	Total      int
	Aggregates []Field
	// Lead lines stand above the table: the facts a human reads first. A list that has a
	// lead and no rows shows the lead alone — the lead already says there is nothing.
	// Trail lines stand below it, plain, before the guidance.
	Lead  []string
	Trail []string
	Notes []string
	Next  []string
}

// Table separates the fixed parts of a live list from its scrollable data rows.
// Unlike a snapshot, a live table keeps its heading even when it has no rows.
type Table struct {
	Lead   []string
	Header string
	Rows   []string
	Footer []string
}

func (l List) Table(mode Mode) (Table, error) {
	columns, err := l.Columns(mode)
	if err != nil {
		return Table{}, err
	}
	var footer strings.Builder
	lines := humanTableLines(columns, humanBytes(l.Bytes, l.Rows), mode.Full)
	document := make(map[string]any, len(l.Aggregates))
	for _, aggregate := range l.Aggregates {
		document[aggregate.K] = aggregate.V
	}
	writeHumanListFooter(&footer, l, len(l.Rows), len(l.Rows), document, mode.Full)
	table := Table{Header: lines[0], Rows: lines[1:]}
	if len(l.Lead) > 0 {
		table.Lead = strings.Split(strings.Join(l.Lead, "\n"), "\n")
		table.Lead = append(table.Lead, "")
	}
	if text := strings.TrimSuffix(footer.String(), "\n"); text != "" {
		table.Footer = append([]string{""}, strings.Split(text, "\n")...)
	}
	return table, nil
}

// Human is a value with its own terminal sentence; JSON carries the value itself. An
// empty sentence keeps an aggregate out of the terminal and shows a record field as `-`.
type Human interface {
	Human() string
}

// Int writes a byte count for a `Bytes` column.
func Int(n int64) string {
	return strconv.FormatInt(n, 10)
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
	human := mode.Human && !mode.JSON
	typed := !human && l.TypedRows != nil
	if typed && len(l.TypedRows) != len(l.Rows) {
		return fmt.Errorf("output list typed rows (%d) do not match display rows (%d)",
			len(l.TypedRows), len(l.Rows))
	}
	var columns []string
	var err error
	if typed {
		columns, err = l.typedColumns(mode)
	} else {
		columns, err = l.Columns(mode)
	}
	if err != nil {
		return err
	}
	if !human && !typed && len(mode.Fields) == 0 {
		for _, column := range l.Machine {
			if !contains(columns, column) {
				columns = append(columns, column)
			}
		}
	}
	shown := l.Rows
	if !mode.Full && len(shown) > rowCap {
		shown = shown[:rowCap]
	}
	document := map[string]any{}
	if typed {
		shownTyped := l.TypedRows[:len(shown)]
		if len(columns) == 1 {
			values := make([]any, 0, len(shownTyped))
			for _, source := range shownTyped {
				values = append(values, source[columns[0]])
			}
			document[l.Name] = values
		} else {
			rows := make([]map[string]any, 0, len(shownTyped))
			for _, source := range shownTyped {
				row := make(map[string]any, len(columns))
				for _, column := range columns {
					if value, present := source[column]; present {
						row[column] = value
					}
				}
				rows = append(rows, row)
			}
			document[l.Name] = rows
		}
	} else if len(columns) == 1 {
		values := make([]string, 0, len(shown))
		for _, source := range shown {
			values = append(values, Elide(source[columns[0]], cellCap, mode.Full))
		}
		document[l.Name] = values
	} else {
		rows := make([]map[string]any, 0, len(shown))
		for _, source := range shown {
			row := make(map[string]any, len(columns))
			for _, column := range columns {
				if contains(l.Bytes, column) {
					if source[column] == "" {
						continue
					}
					n, err := strconv.ParseInt(source[column], 10, 64)
					if err != nil {
						return fmt.Errorf("output column %q carries %q, not a byte count", column, source[column])
					}
					row[column] = n
					continue
				}
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
	if human {
		return writeHumanList(w, l, columns, shown, total, document, mode.Full)
	}
	return Write(w, document, mode)
}

func writeHumanRecord(w io.Writer, fields []Field, data map[string]any, notes, next []string, full bool) error {
	var rendered strings.Builder
	width := 0
	for _, field := range fields {
		width = max(width, utf8.RuneCountInString(humanFieldName(field.K)))
	}
	for _, field := range fields {
		label := humanFieldName(field.K)
		if values, ok := data[field.K].([]string); ok {
			rendered.WriteString(label)
			rendered.WriteByte(':')
			if len(values) == 0 {
				rendered.WriteString(strings.Repeat(" ", width-utf8.RuneCountInString(label)+1))
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
		rendered.WriteString(label)
		rendered.WriteByte(':')
		rendered.WriteString(strings.Repeat(" ", width-utf8.RuneCountInString(label)+1))
		rendered.WriteString(humanValue(data[field.K], full))
		rendered.WriteByte('\n')
	}
	writeHumanGuidance(&rendered, notes, next)
	_, err := io.WriteString(w, rendered.String())
	return err
}

func humanFieldName(name string) string {
	if name == "queue_position" {
		return "queue position"
	}
	return name
}

func writeHumanList(w io.Writer, list List, columns []string, shown []map[string]string,
	total int, document map[string]any, full bool,
) error {
	var rendered strings.Builder
	for _, line := range list.Lead {
		rendered.WriteString(line)
		rendered.WriteByte('\n')
	}
	if len(list.Lead) > 0 && len(shown) > 0 {
		rendered.WriteByte('\n')
	}
	if len(shown) == 0 {
		if len(list.Lead) == 0 {
			fmt.Fprintf(&rendered, "No %s found.\n", list.Name)
		}
	} else if len(columns) == 1 {
		for _, row := range shown {
			rendered.WriteString("- ")
			rendered.WriteString(orDash(Elide(row[columns[0]], cellCap, full)))
			rendered.WriteByte('\n')
		}
	} else {
		writeHumanTable(&rendered, columns, humanBytes(list.Bytes, shown), full)
	}
	writeHumanListFooter(&rendered, list, len(shown), total, document, full)
	_, err := io.WriteString(w, rendered.String())
	return err
}

func writeHumanListFooter(rendered *strings.Builder, list List, shown, total int,
	document map[string]any, full bool,
) {
	if omitted := total - shown; omitted > 0 {
		fmt.Fprintf(rendered, "%d more not shown. Use --full to show all.\n", omitted)
	}
	wroteAggregate := false
	for _, aggregate := range list.Aggregates {
		if aggregate.K == list.Name || aggregate.K == "count" || aggregate.K == "results" {
			continue
		}
		if value, ok := document[aggregate.K]; ok {
			text := ""
			if spoken, ok := value.(Human); ok {
				text = spoken.Human()
			} else {
				text = humanValue(value, full)
			}
			if text == "" {
				continue
			}
			if !wroteAggregate && rendered.Len() > 0 {
				rendered.WriteByte('\n')
			}
			fmt.Fprintf(rendered, "%s: %s\n", aggregate.K, text)
			wroteAggregate = true
		}
	}
	if len(list.Trail) > 0 && rendered.Len() > 0 {
		rendered.WriteByte('\n')
	}
	for _, line := range list.Trail {
		rendered.WriteString(line)
		rendered.WriteByte('\n')
	}
	if len(list.Trail) > 0 {
		// The trail already sits apart from the table; the guidance follows it directly.
		writeGuidanceLines(rendered, list.Notes, list.Next)
	} else {
		writeHumanGuidance(rendered, list.Notes, list.Next)
	}
}

// humanBytes renders every byte column of every row in binary units; other cells pass.
func humanBytes(byteColumns []string, rows []map[string]string) []map[string]string {
	if len(byteColumns) == 0 {
		return rows
	}
	out := make([]map[string]string, 0, len(rows))
	for _, source := range rows {
		row := make(map[string]string, len(source))
		for column, value := range source {
			if contains(byteColumns, column) && value != "" {
				if n, err := strconv.ParseInt(value, 10, 64); err == nil {
					value = Bytes(n)
				}
			}
			row[column] = value
		}
		out = append(out, row)
	}
	return out
}

func writeHumanTable(rendered *strings.Builder, columns []string, rows []map[string]string, full bool) {
	for _, line := range humanTableLines(columns, rows, full) {
		rendered.WriteString(line)
		rendered.WriteByte('\n')
	}
}

func humanTableLines(columns []string, rows []map[string]string, full bool) []string {
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
	lines := []string{humanTableRow(headings, widths)}
	for _, row := range values {
		lines = append(lines, humanTableRow(row, widths))
	}
	return lines
}

func humanTableRow(values []string, widths []int) string {
	var rendered strings.Builder
	for i, value := range values {
		rendered.WriteString(value)
		if i < len(values)-1 {
			rendered.WriteString(strings.Repeat(" ", widths[i]-utf8.RuneCountInString(value)+2))
		}
	}
	return rendered.String()
}

func writeHumanGuidance(rendered *strings.Builder, notes, next []string) {
	next = trimNext(next)
	if len(notes) == 0 && len(next) == 0 {
		return
	}
	if rendered.Len() > 0 {
		rendered.WriteByte('\n')
	}
	writeGuidanceLines(rendered, notes, next)
}

func writeGuidanceLines(rendered *strings.Builder, notes, next []string) {
	for _, note := range notes {
		fmt.Fprintf(rendered, "Note: %s\n", strings.TrimSpace(note))
	}
	for _, command := range trimNext(next) {
		fmt.Fprintf(rendered, "Next: %s\n", strings.TrimSpace(command))
	}
}

func humanValue(value any, full bool) string {
	if value == nil {
		return "-"
	}
	var text string
	switch scalar := value.(type) {
	case Human:
		text = scalar.Human()
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

func (l List) typedColumns(mode Mode) ([]string, error) {
	available := l.TypedAllFields
	if len(available) == 0 {
		available = l.TypedFields
	}
	if len(mode.Fields) == 0 {
		if mode.Full {
			return available, nil
		}
		return l.TypedFields, nil
	}
	for _, field := range mode.Fields {
		if !contains(available, field) {
			return nil, NewError(Usage, "output.field_unknown", fmt.Sprintf("unknown output field %q", field)).
				WithRemedy("available fields: " + strings.Join(available, ", "))
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

// Elide bounds a string by runes and reports the original size. A string
// carrying an OSC 8 hyperlink is never elided: cutting inside its escapes
// would corrupt the terminal.
func Elide(value string, limit int, full bool) string {
	if strings.Contains(value, "\x1b]8;") {
		return value
	}
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
