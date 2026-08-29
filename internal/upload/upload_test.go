package upload

import (
	"io"
	"os"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy-creator/internal/home"
)

func TestPutPublishesOneOpaqueContentIdentity(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem.Message)
	}
	body := "cozy-upload-bytes"
	first, problem := Put(layout, strings.NewReader(body), int64(len(body)), "text/plain; charset=utf-8")
	if problem != nil {
		t.Fatal(problem.Message)
	}
	second, problem := Put(layout, strings.NewReader(body), -1, "text/plain")
	if problem != nil {
		t.Fatal(problem.Message)
	}
	if first.ID != second.ID || first.Digest != second.Digest || first.Length != int64(len(body)) {
		t.Fatalf("deduplication changed content identity: %#v / %#v", first, second)
	}
	if strings.Contains(first.ID, "/") || first.URL != "/v1/uploads/"+first.ID {
		t.Fatalf("upload exposed a path-shaped identity: %#v", first)
	}

	file, digest, problem := Open(layout, first.ID)
	if problem != nil {
		t.Fatal(problem.Message)
	}
	defer file.Close()
	got, err := io.ReadAll(file)
	if err != nil || string(got) != body || digest != first.Digest {
		t.Fatalf("readback changed bytes or identity: bytes=%q digest=%q err=%v", got, digest, err)
	}
	info, err := file.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("upload mode = %#o, want 0600", info.Mode().Perm())
	}
}

func TestPutRejectsDeclaredOverLimitWithoutReading(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem.Message)
	}
	reader := &countingReader{}
	_, problem = Put(layout, reader, MaxBytes+1, "application/octet-stream")
	if problem == nil || problem.ErrName() != "upload_too_large" {
		t.Fatalf("oversize declaration returned %#v", problem)
	}
	if reader.reads != 0 {
		t.Fatalf("oversize declaration consumed the body %d time(s)", reader.reads)
	}
}

func TestOpenRejectsNonOpaqueIDs(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem.Message)
	}
	for _, id := range []string{"../../etc/passwd", "upl-../records.db", "upl-" + strings.Repeat("A", 64)} {
		file, _, problem := Open(layout, id)
		if file != nil {
			file.Close()
		}
		if problem == nil || problem.ErrName() != "upload_not_found" {
			t.Errorf("Open(%q) returned %#v", id, problem)
		}
	}
	entries, err := os.ReadDir(layout.Uploads)
	if err != nil || len(entries) != 0 {
		t.Fatalf("invalid ids changed the upload store: entries=%d err=%v", len(entries), err)
	}
}

type countingReader struct{ reads int }

func (r *countingReader) Read([]byte) (int, error) {
	r.reads++
	return 0, io.EOF
}
