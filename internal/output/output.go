// Package output is the CLI's one result boundary. Commands build one logical
// document; JSON is an encoding switch, while progress remains on stderr.
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
	JSON   bool
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
	Class   Class    `json:"class"`
	Code    string   `json:"code"`
	Message string   `json:"message"`
	Remedy  string   `json:"remedy,omitempty"`
	Next    []string `json:"-"`
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

// Result is the sole logical stdout document for both success and failure.
type Result struct {
	OK    bool     `json:"ok"`
	Kind  string   `json:"kind"`
	Data  any      `json:"data,omitempty"`
	Error *Error   `json:"error,omitempty"`
	Notes []string `json:"notes,omitempty"`
	Next  []string `json:"next,omitempty"`
}

// Success builds a successful logical document.
func Success(kind string, data any) Result {
	return Result{OK: true, Kind: kind, Data: data}
}

// Failure builds the logical error document written to stdout.
func Failure(problem *Error) Result {
	result := Result{Kind: "error", Error: problem}
	if problem != nil {
		result.Next = trimNext(problem.Next)
	}
	return result
}

// Document is the small compatibility surface used by command handlers.
type Document interface {
	Emit(io.Writer, Mode) error
	WithDefaultNext([]string) Document
}

func (r Result) WithDefaultNext(next []string) Document {
	if len(r.Next) == 0 {
		r.Next = trimNext(next)
	}
	return r
}

func (r Result) Emit(w io.Writer, mode Mode) error {
	return Write(w, r, mode)
}

// Write emits exactly one logical document. TOON is the default; JSON changes
// only the encoding. Both formats pass through the same TOON data model first.
func Write(w io.Writer, result Result, mode Mode) error {
	if err := result.valid(); err != nil {
		return err
	}
	logical, toonBytes, err := normalize(result)
	if err != nil {
		return err
	}
	if mode.JSON {
		encoder := json.NewEncoder(w)
		encoder.SetEscapeHTML(false)
		return encoder.Encode(logical)
	}
	if _, err := w.Write(toonBytes); err != nil {
		return err
	}
	if !bytes.HasSuffix(toonBytes, []byte("\n")) {
		_, err = io.WriteString(w, "\n")
	}
	return err
}

func (r Result) valid() error {
	if strings.TrimSpace(r.Kind) == "" {
		return errors.New("output kind is required")
	}
	if r.OK && r.Error != nil {
		return errors.New("successful output cannot carry an error")
	}
	if !r.OK && r.Error == nil {
		return errors.New("failed output must carry an error")
	}
	if r.Error == nil {
		return nil
	}
	if r.Data != nil {
		return errors.New("failed output cannot carry success data")
	}
	if r.Error.Class != Usage && r.Error.Class != Config && r.Error.Class != Operational {
		return fmt.Errorf("unknown error class %q", r.Error.Class)
	}
	if strings.TrimSpace(r.Error.Code) == "" || strings.TrimSpace(r.Error.Message) == "" {
		return errors.New("error code and message are required")
	}
	return nil
}

// normalize makes the TOON library's JSON-like model authoritative, then uses
// that same value for JSON. This also honors json tags on domain payloads.
func normalize(result Result) (any, []byte, error) {
	encoded, err := json.Marshal(result)
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

// EmitError writes a structured failure to stdout.
func EmitError(w io.Writer, problem *Error, mode Mode) error {
	return Write(w, Failure(problem), mode)
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
