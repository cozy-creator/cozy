// Package output is the CLI's one result boundary. Commands build domain data;
// JSON is an encoding switch, while progress remains on stderr.
package output

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	toon "github.com/toon-format/toon-go"
)

const maxNext = 2

// Mode carries presentation choices that do not change command semantics.
type Mode struct {
	JSON  bool
	Human bool
	// Color allows SGR styling: a live terminal whose environment does not set NO_COLOR.
	Color bool
	// TTY records that the result writer is a terminal; it gates output only a
	// terminal can render, such as OSC 8 hyperlinks.
	TTY bool
	// Live says progress may redraw in place: a terminal that is not TERM=dumb.
	Live   bool
	Full   bool
	Fields []string
}

// Class determines the shell projection of a structured error.
type Class string

const (
	Usage       Class = "usage"
	Config      Class = "config"
	Operational Class = "operational"
)

// Error is the stable, machine-readable failure carried by a Result.
type Error struct {
	Class   Class          `json:"class"`
	Code    string         `json:"code"`
	Message string         `json:"message"`
	Remedy  string         `json:"remedy,omitempty"`
	Details map[string]any `json:"details,omitempty"`
	Next    []string       `json:"-"`
}

func (e *Error) Error() string {
	if e == nil {
		return ""
	}
	return e.Message
}

// NewError makes a typed refusal. Code is the detailed stable identifier;
// Class has only the three shell-relevant meanings above.
func NewError(class Class, code, message string) *Error {
	return &Error{Class: class, Code: code, Message: message}
}

func (e *Error) WithRemedy(remedy string) *Error {
	e.Remedy = remedy
	return e
}

func (e *Error) WithNext(next ...string) *Error {
	e.Next = trimNext(next)
	return e
}

// Document is one domain-shaped success document.
type Document interface {
	Emit(io.Writer, Mode) error
}

// Write preserves the document's authored JSON types. TOON is the default rendering;
// JSON must not pass through TOON's float64 reader before reaching its consumer.
func Write(w io.Writer, document any, mode Mode) error {
	if mode.JSON {
		return writeJSONValue(w, document, mode)
	}
	_, toonBytes, err := normalize(document)
	if err != nil {
		// A document TOON cannot round-trip (a traceback whose lines read as list items,
		// say) is still written: as its JSON value, indented for a person.
		return writeJSONValue(w, document, mode)
	}
	if _, err := w.Write(toonBytes); err != nil {
		return err
	}
	if !bytes.HasSuffix(toonBytes, []byte("\n")) {
		_, err = io.WriteString(w, "\n")
	}
	return err
}

// normalize prepares the TOON rendering after honoring domain JSON tags.
// Its reader validates that rendering; explicit JSON preserves the original values.
func normalize(document any) (any, []byte, error) {
	encoded, err := json.Marshal(document)
	if err != nil {
		return nil, nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, nil, err
	}
	toonBytes, err := toon.Marshal(value, toon.WithLengthMarkers(true))
	if err != nil {
		return nil, nil, err
	}
	logical, err := toon.Decode(toonBytes)
	if err != nil {
		return nil, nil, err
	}
	return logical, toonBytes, nil
}

func writeJSONValue(w io.Writer, document any, mode Mode) error {
	encoder := json.NewEncoder(w)
	encoder.SetEscapeHTML(false)
	if !mode.JSON {
		encoder.SetIndent("", "  ")
	}
	return encoder.Encode(document)
}

type failure struct {
	Error *Error   `json:"error"`
	Next  []string `json:"next,omitempty"`
}

// EmitError writes one structured failure to stdout.
func EmitError(w io.Writer, problem *Error, mode Mode) error {
	if problem == nil {
		return errors.New("output error is required")
	}
	if problem.Class != Usage && problem.Class != Config && problem.Class != Operational {
		return fmt.Errorf("unknown error class %q", problem.Class)
	}
	if strings.TrimSpace(problem.Code) == "" || strings.TrimSpace(problem.Message) == "" {
		return errors.New("error code and message are required")
	}
	if mode.Human && !mode.JSON {
		return writeHumanError(w, problem, mode)
	}
	return Write(w, failure{Error: problem, Next: trimNext(problem.Next)}, mode)
}

func writeHumanError(w io.Writer, problem *Error, mode Mode) error {
	var rendered strings.Builder
	next := trimNext(problem.Next)
	if mode.Color {
		rendered.WriteString("\x1b[1;31m")
	}
	rendered.WriteString("Error: ")
	rendered.WriteString(strings.TrimSpace(problem.Message))
	if mode.Color {
		rendered.WriteString("\x1b[0m")
	}
	rendered.WriteByte('\n')
	if strings.TrimSpace(problem.Remedy) != "" || len(next) > 0 {
		rendered.WriteByte('\n')
	}
	if remedy := strings.TrimSpace(problem.Remedy); remedy != "" {
		rendered.WriteString("Try: ")
		rendered.WriteString(remedy)
		rendered.WriteByte('\n')
	}
	for _, next := range next {
		rendered.WriteString("Next: ")
		rendered.WriteString(next)
		rendered.WriteByte('\n')
	}
	_, err := io.WriteString(w, rendered.String())
	return err
}

// ShellCode projects all detail onto AXI's three shell outcomes.
func ShellCode(err error) int {
	if err == nil {
		return 0
	}
	var problem *Error
	if errors.As(err, &problem) && (problem.Class == Usage || problem.Class == Config) {
		return 2
	}
	return 1
}

// Progress writes one progress line to the caller-provided stderr stream.
func Progress(w io.Writer, message string) error {
	_, err := fmt.Fprintln(w, message)
	return err
}

func trimNext(next []string) []string {
	if len(next) > maxNext {
		next = next[:maxNext]
	}
	return append([]string(nil), next...)
}
