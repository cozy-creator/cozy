package producttest

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"io"
	"os"
	"slices"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestPublishedSourceArchiveIsDeterministicAndComplete(t *testing.T) {
	project := weightlessProject(t)
	build := func() []byte {
		pack, problem := packagepublish.PrepareFrom(project)
		if problem != nil {
			t.Fatal(problem)
		}
		defer pack.Close()
		if problem := pack.BuildForPublish(context.Background()); problem != nil {
			t.Fatal(problem)
		}
		if pack.SourceArchive == "" {
			t.Fatal("publish build produced no source archive")
		}
		raw, err := os.ReadFile(pack.SourceArchive)
		if err != nil {
			t.Fatal(err)
		}
		return raw
	}
	first, second := build(), build()
	if !bytes.Equal(first, second) {
		t.Fatal("published source archive is not deterministic")
	}
	gz, err := gzip.NewReader(bytes.NewReader(first))
	if err != nil {
		t.Fatal(err)
	}
	tr := tar.NewReader(gz)
	var names []string
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		names = append(names, header.Name)
	}
	for _, required := range []string{"package.toml", "pyproject.toml", "uv.lock"} {
		if !slices.Contains(names, required) {
			t.Fatalf("source archive omits %s; members=%v", required, names)
		}
	}
}
