package packagepublish

import (
	"archive/tar"
	"compress/gzip"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestSourceArchiveIsDeterministicAndContainsFilteredTree(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{}
	for name, contents := range map[string]string{
		"pyproject.toml": "[project]\nname='demo'\nversion='1.0.0'\n",
		"package.toml":   "[application]\nobject='demo:app'\n",
		"src/demo.py":    "def app(): pass\n",
	} {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
			t.Fatal(err)
		}
		files[name] = path
	}
	first, err := sourceArchiveFile(root, "demo", "1.0.0", files)
	if err != nil {
		t.Fatal(err)
	}
	secondRoot := t.TempDir()
	second, err := sourceArchiveFile(secondRoot, "demo", "1.0.0", files)
	if err != nil {
		t.Fatal(err)
	}
	firstBytes, err := os.ReadFile(first)
	if err != nil {
		t.Fatal(err)
	}
	secondBytes, err := os.ReadFile(second)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(firstBytes, secondBytes) {
		t.Fatal("source archive is not deterministic")
	}

	file, err := os.Open(first)
	if err != nil {
		t.Fatal(err)
	}
	gz, err := gzip.NewReader(file)
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
	want := []string{"package.toml", "pyproject.toml", "src/demo.py"}
	if !slices.Equal(names, want) {
		t.Fatalf("archive members = %v, want %v", names, want)
	}
}
