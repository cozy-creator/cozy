package producttest

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestPublishedSourceArchiveIsStandardSdist(t *testing.T) {
	project := weightlessProject(t)
	pack, problem := packagepublish.PrepareFrom(project)
	if problem != nil {
		t.Fatal(problem)
	}
	defer pack.Close()
	if problem := pack.BuildForPublish(context.Background(), packagepublish.Namespace{Hub: "http://127.0.0.1:1", Account: "proof"}); problem != nil {
		t.Fatal(problem)
	}
	if pack.SourceArchive == "" {
		t.Fatal("publish build produced no source distribution")
	}
	raw, err := os.ReadFile(pack.SourceArchive)
	if err != nil {
		t.Fatal(err)
	}
	gz, err := gzip.NewReader(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	tr := tar.NewReader(gz)
	var root string
	seenPKGInfo, seenPyproject := false, false
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		parts := strings.SplitN(header.Name, "/", 2)
		if len(parts) != 2 || parts[0] == "" {
			t.Fatalf("source distribution member %q is not under one root directory", header.Name)
		}
		if root == "" {
			root = parts[0]
		} else if root != parts[0] {
			t.Fatalf("source distribution has multiple roots: %q and %q", root, parts[0])
		}
		switch parts[1] {
		case "PKG-INFO":
			seenPKGInfo = true
		case "pyproject.toml":
			seenPyproject = true
		}
	}
	if !seenPKGInfo || !seenPyproject {
		t.Fatalf("source distribution is missing standard PKG-INFO or pyproject.toml (root %q)", root)
	}
}
