package producttest

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/output"
)

// TestFileURLEncoding pins the file:// spelling: this host's name in the
// authority, and RFC 8089/3986 percent-encoding of the path — spaces and
// non-ASCII included — so every OSC 8 emitter agrees on one URL.
func TestFileURLEncoding(t *testing.T) {
	host, err := os.Hostname()
	if err != nil {
		t.Fatalf("hostname: %v", err)
	}
	for _, row := range [][2]string{
		{"/home/u/outputs/img.png", "file://" + host + "/home/u/outputs/img.png"},
		{"/tmp/a b/c d.png", "file://" + host + "/tmp/a%20b/c%20d.png"},
		{"/tmp/héllo/画像.png", "file://" + host + "/tmp/h%C3%A9llo/%E7%94%BB%E5%83%8F.png"},
	} {
		if got := output.FileURL(row[0]); got != row[1] {
			t.Errorf("FileURL(%q) = %q, want %q", row[0], got, row[1])
		}
	}
}

// savedRecord composes the run record the way renderRun composes its human
// `saved:` lines, so the arms below drive the REAL Record renderer.
func savedRecord(mode output.Mode, path string) output.Record {
	saved := mode.Hyperlink(path) + " (12 KiB)"
	return output.Record{Fields: []output.Field{
		{K: "target", V: "org/pkg/fn"},
		{K: "status", V: "succeeded"},
		{K: "saved", V: []string{saved}},
	}}
}

// TestHyperlinkOnTTY renders to a TTY-shaped human mode and requires the OSC 8
// escapes to wrap the displayed path exactly once, URL and all.
func TestHyperlinkOnTTY(t *testing.T) {
	mode := output.Mode{Human: true, TTY: true}
	path := "/home/u/outputs/img 1.png"
	var buf bytes.Buffer
	if err := savedRecord(mode, path).Emit(&buf, mode); err != nil {
		t.Fatalf("emit: %v", err)
	}
	rendered := buf.String()
	wrapped := "\x1b]8;;" + output.FileURL(path) + "\x1b\\" + path + "\x1b]8;;\x1b\\"
	if strings.Count(rendered, wrapped) != 1 {
		t.Fatalf("TTY rendering does not wrap the path exactly once:\n%q", rendered)
	}
	if strings.Count(rendered, "\x1b]8;;") != 2 {
		t.Fatalf("TTY rendering carries stray OSC 8 escapes:\n%q", rendered)
	}
}

// TestPipedAndJSONUnchanged is the invariant that matters: with the writer not
// a terminal, human and JSON renderings are byte-identical to a record that
// never heard of hyperlinks — the escapes exist on a TTY and nowhere else.
func TestPipedAndJSONUnchanged(t *testing.T) {
	path := "/home/u/outputs/img 1.png"
	for _, mode := range []output.Mode{
		{Human: true},                         // piped human output
		{JSON: true, Human: false},            // --json to a pipe
		{JSON: true, Human: false, TTY: true}, // --json even on a terminal
	} {
		var linked, plain bytes.Buffer
		if err := savedRecord(mode, path).Emit(&linked, mode); err != nil {
			t.Fatalf("emit: %v", err)
		}
		before := output.Record{Fields: []output.Field{
			{K: "target", V: "org/pkg/fn"},
			{K: "status", V: "succeeded"},
			{K: "saved", V: []string{path + " (12 KiB)"}},
		}}
		if err := before.Emit(&plain, mode); err != nil {
			t.Fatalf("emit: %v", err)
		}
		if !bytes.Equal(linked.Bytes(), plain.Bytes()) {
			t.Errorf("mode %+v output differs from the pre-hyperlink bytes:\n%q\n%q",
				mode, linked.Bytes(), plain.Bytes())
		}
		if bytes.Contains(linked.Bytes(), []byte{0x1b}) {
			t.Errorf("mode %+v output carries an escape byte:\n%q", mode, linked.Bytes())
		}
	}
}
